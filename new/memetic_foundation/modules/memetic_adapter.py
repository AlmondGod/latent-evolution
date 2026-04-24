from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankDelta(nn.Module):
    """Small ES-friendly low-rank update.

    This module is meant to sit on top of a frozen backbone projection. The
    backbone produces a base projection; LowRankDelta produces a trainable /
    evolvable additive offset.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    @property
    def scale(self) -> float:
        return self.alpha / float(self.rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x)) * self.scale


class DenseDelta(nn.Module):
    """Dense additive update used for ablations against low-rank deltas."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.alpha


class MemeticStateCell(nn.Module):
    """Persistent memetic state update.

    z is the fast, socially-shaped latent state. h is the slower task memory
    learned by MAPPO. Communication modifies z, while h stays comparatively
    stable.
    """

    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        c_dim: int,
        rank: int = 4,
        alpha: float = 1.0,
        update_mode: str = "low_rank",
    ) -> None:
        super().__init__()
        if update_mode not in {"low_rank", "dense"}:
            raise ValueError(f"Unsupported update_mode: {update_mode}")
        if update_mode == "low_rank" and rank <= 0:
            raise ValueError("rank must be positive")
        mix_in = z_dim + h_dim + c_dim
        self.rank = rank
        self.update_mode = update_mode
        delta_cls = LowRankDelta if update_mode == "low_rank" else DenseDelta
        delta_kwargs = {"alpha": alpha}
        if update_mode == "low_rank":
            delta_kwargs["rank"] = rank
        self.update_gate = delta_cls(mix_in, z_dim, **delta_kwargs)
        self.candidate = delta_cls(mix_in, z_dim, **delta_kwargs)
        self.update_gate_bias = nn.Parameter(torch.zeros(z_dim))
        self.candidate_bias = nn.Parameter(torch.zeros(z_dim))
        self.eta_logit = nn.Parameter(torch.tensor(-1.5))

    def forward(self, z: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mix = torch.cat([z, h, c], dim=-1)
        gate = torch.sigmoid(self.update_gate(mix) + self.update_gate_bias)
        eta = torch.sigmoid(self.eta_logit)
        proposal = torch.tanh(self.candidate(mix) + self.candidate_bias)
        z_next = (1.0 - eta * gate) * z + (eta * gate) * proposal
        return F.layer_norm(z_next, z_next.shape[-1:])


class MemeticCommAdapter(nn.Module):
    """Evolvable communication + memetic-state scaffold.

    This module is designed for the phase-2 setup:
      1. train a strong MAPPO backbone
      2. freeze most of it
      3. evolve only the communication adapter and memetic-state update

    The adapter can run either:
      - standalone, using only its own low-rank projections
      - on top of a frozen backbone, by adding deltas to base q/k/v/o tensors
    """

    def __init__(
        self,
        h_dim: int,
        u_dim: int,
        z_dim: int = 16,
        attn_dim: int = 64,
        rank: int = 4,
        alpha: float = 1.0,
        use_z: bool = True,
        dense_state_update: bool = False,
    ) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.u_dim = u_dim
        self.use_z = bool(use_z and z_dim > 0)
        self.z_dim = z_dim if self.use_z else 0
        self.attn_dim = attn_dim
        self.rank = rank
        self.dense_state_update = dense_state_update

        q_in_dim = h_dim + self.z_dim
        self.q_delta = LowRankDelta(q_in_dim, attn_dim, rank=rank, alpha=alpha)
        self.k_delta = LowRankDelta(u_dim, attn_dim, rank=rank, alpha=alpha)
        self.v_delta = LowRankDelta(u_dim, attn_dim, rank=rank, alpha=alpha)
        self.o_delta = LowRankDelta(attn_dim, u_dim, rank=rank, alpha=alpha)
        self.state_cell = None
        if self.use_z:
            self.state_cell = MemeticStateCell(
                z_dim=self.z_dim,
                h_dim=h_dim,
                c_dim=u_dim,
                rank=rank,
                alpha=alpha,
                update_mode="dense" if dense_state_update else "low_rank",
            )

    def initial_state(
        self,
        n_agents: int,
        device: torch.device,
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> Optional[torch.Tensor]:
        if not self.use_z:
            return None
        shape = (*batch_shape, n_agents, self.z_dim) if batch_shape else (n_agents, self.z_dim)
        return torch.zeros(shape, device=device)

    def q_features(self, h: torch.Tensor, z: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.use_z:
            return h
        if z is None:
            raise ValueError("z must be provided when use_z=True")
        return torch.cat([h, z], dim=-1)

    def next_state(self, z: Optional[torch.Tensor], h: torch.Tensor, c: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_z or self.state_cell is None:
            return z
        if z is None:
            raise ValueError("z must be provided when use_z=True")
        return self.state_cell(z=z, h=h, c=c)

    def forward(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        z: torch.Tensor,
        base_q: Optional[torch.Tensor] = None,
        base_k: Optional[torch.Tensor] = None,
        base_v: Optional[torch.Tensor] = None,
        base_o: Optional[torch.Tensor] = None,
        mask_self: bool = True,
    ) -> dict[str, torch.Tensor]:
        q_in = self.q_features(h, z)
        q = self.q_delta(q_in)
        k = self.k_delta(u)
        v = self.v_delta(u)

        if base_q is not None:
            q = q + base_q
        if base_k is not None:
            k = k + base_k
        if base_v is not None:
            v = v + base_v

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attn_dim)
        if mask_self:
            eye = torch.eye(scores.shape[-1], device=scores.device, dtype=torch.bool)
            while eye.dim() < scores.dim():
                eye = eye.unsqueeze(0)
            scores = scores.masked_fill(eye, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        c_latent = torch.matmul(attn, v)

        c = self.o_delta(c_latent)
        if base_o is not None:
            c = c + base_o
        z_next = self.next_state(z=z, h=h, c=c)

        return {
            "c": c,
            "z_next": z_next,
            "attn": attn,
            "q_delta": q,
            "k_delta": k,
            "v_delta": v,
        }

    def genotype_size(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flatten_genotype(self) -> torch.Tensor:
        return nn.utils.parameters_to_vector(self.parameters()).detach().clone()

    def load_genotype(self, vector: torch.Tensor) -> None:
        nn.utils.vector_to_parameters(vector.to(next(self.parameters()).device), self.parameters())


def freeze_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()
    return module
