"""
memory_cells.py — GRU-based persistent per-agent memory.

Each agent maintains a single GRU hidden state h of dimension mem_dim.
The GRU unifies read and write: its hidden state IS the memory, and
the GRU update IS the write operation.

For the full variant, GRU input is [u; m̄_prev] — observation encoding
concatenated with the previous timestep's received communication.
Communication at time t only affects h at time t+1.

For memory_only, GRU input is just u (observation encoding).

Memory persists across episodes by default, detached between rollouts.
Optional decay gently pulls h toward zero for self-stabilization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GRUMemory(nn.Module):
    """GRU-based persistent per-agent memory.

    Parameters (learned via gradient descent):
        gru_cell: nn.GRUCell — the recurrent update mechanism
        h_init: (mem_dim,) — learned initial hidden state template

    State (runtime, not optimized directly):
        hidden_state: (n_agents, mem_dim) — actual evolving memory

    The hidden_state is registered as a buffer so it moves with .to(device)
    and is included in state_dict, but is NOT a trainable parameter.
    """

    def __init__(
        self,
        n_agents: int,
        input_dim: int,
        mem_dim: int = 128,
        mem_decay: float = 0.005,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.mem_dim = mem_dim
        self.mem_decay = mem_decay

        # GRU cell — the learned recurrent update
        self.gru_cell = nn.GRUCell(input_dim, mem_dim)

        # Learned initial hidden state template
        self.h_init = nn.Parameter(torch.zeros(mem_dim))

        # Runtime hidden state — actual evolving per-agent memory
        self.register_buffer(
            "hidden_state",
            self.h_init.data.unsqueeze(0).expand(n_agents, -1).clone(),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init on GRU weights for stable training."""
        nn.init.orthogonal_(self.gru_cell.weight_ih)
        nn.init.orthogonal_(self.gru_cell.weight_hh)
        nn.init.zeros_(self.gru_cell.bias_ih)
        nn.init.zeros_(self.gru_cell.bias_hh)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Run one GRU step and update hidden state.

        Args:
            x: (N, input_dim) — GRU input (e.g. [u; m̄_prev] or just u)

        Returns:
            h_new: (N, mem_dim) — updated hidden state
        """
        h_new = self.gru_cell(x, self.hidden_state)

        # Apply decay: h <- (1 - λ) * h
        if self.mem_decay > 0:
            h_new = h_new * (1.0 - self.mem_decay)

        # Update buffer in-place (preserves registered buffer)
        self.hidden_state.data.copy_(h_new.data)

        return h_new

    def reset_state(self) -> None:
        """Re-copy from learned h_init into runtime state.

        Use at: eval starts, new environments, ablation-controlled resets.
        """
        with torch.no_grad():
            self.hidden_state.copy_(
                self.h_init.data.unsqueeze(0).expand(self.n_agents, -1)
            )

    def detach_state(self) -> None:
        """Detach hidden state from computation graph.

        Call at rollout boundaries to truncate gradient flow
        while preserving memory values.
        """
        self.hidden_state.detach_()

    def get_state(self) -> torch.Tensor:
        """Return a detached clone of current hidden state."""
        return self.hidden_state.detach().clone()

    def set_state(self, state: torch.Tensor) -> None:
        """Restore hidden state from a saved tensor."""
        with torch.no_grad():
            self.hidden_state.copy_(state)

    def forward(self) -> torch.Tensor:
        """Return current hidden state: (n_agents, mem_dim)."""
        return self.hidden_state

    def extra_repr(self) -> str:
        return (
            f"n_agents={self.n_agents}, input_dim={self.input_dim}, "
            f"mem_dim={self.mem_dim}, mem_decay={self.mem_decay}"
        )
