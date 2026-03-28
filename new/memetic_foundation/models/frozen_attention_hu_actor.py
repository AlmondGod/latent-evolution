from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..modules.memory_cells import GRUMemory
from ..modules.obs_encoder import ObsEncoder
from ..modules.memetic_adapter import MemeticCommAdapter


class AttentionCommHUCompat(nn.Module):
    """Checkpoint-compatible attention block for attention_hu_actor."""

    def __init__(self, mem_dim: int, enc_dim: int, attn_dim: int = 64) -> None:
        super().__init__()
        self.mem_dim = mem_dim
        self.enc_dim = enc_dim
        self.attn_dim = attn_dim
        self.query = nn.Linear(mem_dim, attn_dim)
        self.key = nn.Linear(enc_dim, attn_dim)
        self.value = nn.Linear(enc_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, enc_dim)


class FrozenAttentionHUActorBackbone(nn.Module):
    """Frozen, checkpoint-compatible attention_hu_actor backbone.

    This exists because the evaluated attention_hu_actor checkpoints no longer
    line up exactly with the current main training-time network definition. The
    phase-2 memetic-selection runner needs a reliable inference-time wrapper
    around those checkpoints, without changing the main PPO stack.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        n_agents: int,
        hidden_dim: int = 128,
        mem_dim: int = 128,
        enc_dim: int = 128,
        attn_dim: int = 64,
        mem_decay: float = 0.005,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim
        self.enc_dim = enc_dim
        self.attn_dim = attn_dim

        self.encoder = ObsEncoder(obs_dim, enc_dim)
        self.memory = GRUMemory(n_agents=n_agents, input_dim=enc_dim, mem_dim=mem_dim, mem_decay=mem_decay)
        self.attn_comm_hu = AttentionCommHUCompat(mem_dim=mem_dim, enc_dim=enc_dim, attn_dim=attn_dim)
        self.comm_scale_logit = nn.Parameter(torch.tensor(-3.0))
        self.actor = nn.Sequential(
            nn.Linear(enc_dim + mem_dim + enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.last_comm_attn: Optional[torch.Tensor] = None

    @staticmethod
    def _rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        n_agents: int,
        map_location: str = "cpu",
    ) -> "FrozenAttentionHUActorBackbone":
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state = ckpt["policy_state_dict"]

        enc_dim = int(state["encoder.net.0.weight"].shape[0])
        hidden_dim = int(state["actor.0.weight"].shape[0])
        mem_dim = int(state["memory.h_init"].shape[0])
        attn_dim = int(state["attn_comm_hu.query.weight"].shape[0])
        backbone = cls(
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            mem_dim=mem_dim,
            enc_dim=enc_dim,
            attn_dim=attn_dim,
            mem_decay=0.005,
        )
        filtered = {
            key: value
            for key, value in state.items()
            if key in backbone.state_dict()
        }
        missing, unexpected = backbone.load_state_dict(filtered, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys after filtering: {unexpected}")
        required_missing = [k for k in missing if not k.endswith("hidden_state")]
        if required_missing:
            raise RuntimeError(f"Missing required checkpoint keys: {required_missing}")
        return backbone

    def freeze(self) -> "FrozenAttentionHUActorBackbone":
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def reset_memory(self) -> None:
        self.memory.reset_state()

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def _base_comm_tensors(self, h: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.attn_comm_hu.query(h), self.attn_comm_hu.key(u), self.attn_comm_hu.value(u)

    def step_with_adapter(
        self,
        obs: torch.Tensor,
        avail_actions: Optional[torch.Tensor],
        adapter: Optional[MemeticCommAdapter] = None,
        z: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        intervene_comm_silence: bool = False,
        intervene_comm_shift: bool = False,
    ) -> dict[str, torch.Tensor]:
        u = self.encode(obs)
        h = self.memory.step(u)

        if adapter is not None:
            if z is None:
                z = adapter.initial_state(n_agents=u.shape[0], device=u.device)
            q_base, k_base, v_base = self._base_comm_tensors(h, u)
            q = q_base + adapter.q_delta(torch.cat([h, z], dim=-1))
            k = k_base + adapter.k_delta(u)
            v = v_base + adapter.v_delta(u)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attn_dim)
            mask = torch.eye(scores.shape[-1], device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(mask, float("-inf"))
            comm_attn = torch.softmax(scores, dim=-1)
            comm_attn = torch.nan_to_num(comm_attn, nan=0.0)
            c_latent = torch.matmul(comm_attn, v)
            c = self.attn_comm_hu.out_proj(c_latent) + adapter.o_delta(c_latent)
            z_next = adapter.state_cell(z=z, h=h, c=c)
        else:
            q, k, v = self._base_comm_tensors(h, u)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attn_dim)
            mask = torch.eye(scores.shape[-1], device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(mask, float("-inf"))
            comm_attn = torch.softmax(scores, dim=-1)
            comm_attn = torch.nan_to_num(comm_attn, nan=0.0)
            c_latent = torch.matmul(comm_attn, v)
            c = self.attn_comm_hu.out_proj(c_latent)
            z_next = z

        if intervene_comm_shift:
            c = torch.roll(c, shifts=1, dims=0)
        if intervene_comm_silence:
            c = torch.zeros_like(c)

        c_scaled = self._rms_normalize(c) * torch.sigmoid(self.comm_scale_logit)
        actor_input = torch.cat([u, h, c_scaled], dim=-1)
        logits = self.actor(actor_input)
        if avail_actions is not None:
            logits = logits.masked_fill(avail_actions == 0, -1e10)

        dist = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        self.last_comm_attn = comm_attn.detach()

        return {
            "u": u,
            "h": h,
            "z": z,
            "z_next": z_next,
            "c": c_scaled,
            "actions": actions,
            "log_probs": log_probs,
            "logits": logits,
            "comm_attn": comm_attn,
        }
