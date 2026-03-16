"""
comm_module.py — TarMAC-style targeted multi-agent communication with IC3Net gating.

Each agent generates a message (key, value) from its current state [u; z],
and each receiver attends over all senders' messages to aggregate
incoming information.

Sender:
    g_i = STEstimator( sigmoid( W_gate · z_i ) )   ← IC3Net gate
    k_i = g_i · W_k · [u_i ; z_i]
    v_i = g_i · W_v · [u_i ; z_i]

Receiver:
    q_j = W_q^comm · [u_j ; z_j]
    a_{j←i} = softmax_i( q_j^T · k_i / √d )
    m̄_j = Σ_i a_{j←i} · v_i

The gate g_i ∈ {0,1} lets agent i silence itself when communication
would add noise. During training a straight-through estimator passes
gradients through the hard threshold; at inference the gate is
deterministically 0 or 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommGate(nn.Module):
    """IC3Net-style per-agent communication gate.

    Produces a binary gate g_i ∈ {0,1} per agent indicating whether
    it should broadcast a message this timestep.

    Training: straight-through estimator — forward pass uses hard threshold,
    backward pass flows gradients through the sigmoid (allows gate to learn).
    Inference: hard threshold (gate is exactly 0 or 1).

    Also returns soft gate values for entropy regularization:
        gate_entropy = -p*log(p) - (1-p)*log(1-p)  where p = sigmoid(logit)
    Adding a small gate_entropy_coef * gate_entropy loss encourages the gate
    to be decisive (p near 0 or 1) rather than always open (p ≈ 0.73 from bias).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.gate_fc = nn.Linear(input_dim, 1)
        nn.init.orthogonal_(self.gate_fc.weight, gain=0.1)
        nn.init.constant_(self.gate_fc.bias, 1.0)  # biased open early

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (N, input_dim) — per-agent state (memory h or obs encoding u)
        Returns:
            gate:       (N, 1) — binary gate, straight-through during training
            gate_soft:  (N, 1) — soft sigmoid values (for entropy regularization)
        """
        logits     = self.gate_fc(z)          # (N, 1)
        soft       = torch.sigmoid(logits)    # (N, 1) — carries gradients
        hard       = (soft > 0.5).float()     # (N, 1) — hard forward value
        # Straight-through: forward=hard, backward flows through soft
        gate       = hard - soft.detach() + soft
        return gate, soft                     # (N,1), (N,1)


class TargetedComm(nn.Module):
    """TarMAC-style targeted communication with optional IC3Net gating.

    Each agent sends a key-value message derived from [obs_encoding ; memory_summary].
    Each receiver computes attention over all senders (self-masked) and
    produces an aggregated incoming message.

    When use_gate=True, each agent also runs a CommGate that zeros its key/value
    when the gate is closed, selectively suppressing noisy senders.
    """

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

        # Sender: message key and value
        self.key_head   = nn.Linear(input_dim, comm_dim)
        self.value_head = nn.Linear(input_dim, comm_dim)

        # Receiver: communication query
        self.query_head = nn.Linear(input_dim, comm_dim)

        # Optional gate (IC3Net)
        if use_gate:
            self.gate = CommGate(mem_dim)  # gate reads from z (memory state)
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
        """
        Args:
            u: (N, hidden_dim) — observation encodings
            z: (N, mem_dim)    — memory summaries (or projected obs)

        Returns:
            m_bar:        (N, comm_dim) — aggregated incoming messages per agent
            comm_attn:    (N, N)        — attention weights (post-gate)
            gate_entropy: scalar        — gate entropy for regularization (0 if no gate)
        """
        N = u.shape[0]
        s = torch.cat([u, z], dim=-1)  # (N, hidden_dim + mem_dim)

        # Sender side
        keys   = self.key_head(s)    # (N, comm_dim)
        values = self.value_head(s)  # (N, comm_dim)

        # Apply gate: zero out keys/values of silent agents
        gate_entropy = torch.tensor(0.0, device=u.device)
        if self.use_gate and self.gate is not None:
            gate, gate_soft = self.gate(z)      # (N,1) hard, (N,1) soft
            keys   = keys   * gate              # silent agents contribute nothing
            values = values * gate
            # Entropy: H(p) = -p log p - (1-p) log(1-p)
            eps = 1e-6
            p = gate_soft.clamp(eps, 1 - eps)
            gate_entropy = (-p * p.log() - (1 - p) * (1 - p).log()).mean()

        # Receiver side
        queries = self.query_head(s)  # (N, comm_dim)

        # Attention: (N, N) — row=receiver, col=sender
        scores = torch.mm(queries, keys.t()) / self.scale  # (N, N)

        # Self-masking: agent cannot attend to its own message
        mask = torch.eye(N, device=u.device, dtype=torch.bool)
        scores = scores.masked_fill(mask, float("-inf"))

        comm_attn = F.softmax(scores, dim=-1)  # (N, N)
        comm_attn = torch.nan_to_num(comm_attn, nan=0.0)  # handle N=1

        # Aggregate incoming messages
        m_bar = torch.mm(comm_attn, values)  # (N, comm_dim)

        return m_bar, comm_attn, gate_entropy
