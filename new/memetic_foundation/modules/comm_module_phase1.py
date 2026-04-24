"""
comm_module_phase1.py — restored communication modules used by the Phase-1
attention_hu_actor experiments.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommGate(nn.Module):
    """IC3Net-style per-agent communication gate."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.gate_fc = nn.Linear(input_dim, 1)
        nn.init.orthogonal_(self.gate_fc.weight, gain=0.1)
        nn.init.constant_(self.gate_fc.bias, 1.0)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate_fc(z)
        soft = torch.sigmoid(logits)
        hard = (soft > 0.5).float()
        gate = hard - soft.detach() + soft
        return gate, soft


class AttentionCommU(nn.Module):
    """Attention over current observation encodings."""

    def __init__(self, enc_dim: int, comm_dim: int = 64):
        super().__init__()
        self.scale = math.sqrt(comm_dim)
        self.query = nn.Linear(enc_dim, comm_dim)
        self.key = nn.Linear(enc_dim, comm_dim)
        self.value = nn.Linear(enc_dim, comm_dim)
        self.out_proj = nn.Linear(comm_dim, enc_dim)

        for m in (self.query, self.key, self.value):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, u: torch.Tensor):
        n_agents = u.shape[0]
        q = self.query(u)
        k = self.key(u)
        v = self.value(u)

        scores = torch.mm(q, k.t()) / self.scale
        mask = torch.eye(n_agents, device=u.device, dtype=torch.bool)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        agg = torch.mm(attn, v)
        c = self.out_proj(agg)
        return c, attn


class AttentionCommHU(nn.Module):
    """Query from hidden state, keys/values from current obs encoding."""

    def __init__(self, mem_dim: int, enc_dim: int, comm_dim: int = 64):
        super().__init__()
        self.scale = math.sqrt(comm_dim)
        self.query = nn.Linear(mem_dim, comm_dim)
        self.key = nn.Linear(enc_dim, comm_dim)
        self.value = nn.Linear(enc_dim, comm_dim)
        self.out_proj = nn.Linear(comm_dim, enc_dim)

        for m in (self.query, self.key, self.value):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _rms_normalize(x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    def project_query(self, h_query: torch.Tensor) -> torch.Tensor:
        h_query = self._rms_normalize(h_query)
        q = self.query(h_query)
        return self._rms_normalize(q)

    def project_key(self, u_kv: torch.Tensor) -> torch.Tensor:
        k = self.key(u_kv)
        return self._rms_normalize(k)

    def project_value(self, u_kv: torch.Tensor) -> torch.Tensor:
        return self.value(u_kv)

    def forward(self, h_query: torch.Tensor, u_kv: torch.Tensor):
        n_agents = u_kv.shape[0]
        q = self.project_query(h_query)
        k = self.project_key(u_kv)
        v = self.project_value(u_kv)

        scores = torch.mm(q, k.t()) / self.scale
        mask = torch.eye(n_agents, device=u_kv.device, dtype=torch.bool)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        agg = torch.mm(attn, v)
        c = self.out_proj(agg)
        return c, attn


class TargetedComm(nn.Module):
    """TarMAC-style targeted communication with optional IC3Net gating."""

    def __init__(
        self,
        hidden_dim: int = 128,
        mem_dim: int = 128,
        comm_dim: int = 128,
        use_gate: bool = True,
    ):
        super().__init__()
        input_dim = hidden_dim + mem_dim
        self.comm_dim = comm_dim
        self.scale = math.sqrt(comm_dim)
        self.use_gate = use_gate

        self.key_head = nn.Linear(input_dim, comm_dim)
        self.value_head = nn.Linear(input_dim, comm_dim)
        self.query_head = nn.Linear(input_dim, comm_dim)

        if use_gate:
            self.gate = CommGate(mem_dim)
        else:
            self.gate = None

        self._init_weights()

    def _init_weights(self):
        for m in [self.key_head, self.value_head, self.query_head]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)

    def forward(
        self,
        u: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_agents = u.shape[0]
        s = torch.cat([u, z], dim=-1)

        keys = self.key_head(s)
        values = self.value_head(s)

        gate_entropy = torch.tensor(0.0, device=u.device)
        if self.use_gate and self.gate is not None:
            gate, gate_soft = self.gate(z)
            keys = keys * gate
            values = values * gate
            eps = 1e-8
            p = gate_soft.clamp(eps, 1.0 - eps)
            gate_entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()

        queries = self.query_head(s)
        scores = torch.mm(queries, keys.t()) / self.scale
        mask = torch.eye(n_agents, device=u.device, dtype=torch.bool)
        scores = scores.masked_fill(mask, float("-inf"))
        comm_attn = F.softmax(scores, dim=-1)
        comm_attn = torch.nan_to_num(comm_attn, nan=0.0)

        m_bar = torch.mm(comm_attn, values)
        return m_bar, comm_attn, gate_entropy
