"""
agent_network.py — Unified actor-critic for Memetic Foundation (GRU version).

Single GRU per agent replaces the slot-based memory bank. The GRU hidden
state h serves as both memory representation and recurrent state.

Communication modes (comm_mode):
  - 'ic3net':                 Binary gate (straight-through) + m̄_prev feeds INTO GRU.
                              Current default. Gate entropy loss encourages selectivity.
  - 'attention_integrated':   Soft attention (no gate, no entropy loss) + m̄_prev feeds
                              INTO GRU. h encodes personal obs AND received social info.
  - 'attention_separated':    Soft attention (no gate); GRU sees ONLY personal obs.
                              h stays a pure individual meme; actor gets [u; h; c]
                              where c = attention-aggregated social context.
                              Social memory and personal memory interact only at actor.

Forward pass per comm_mode:

  ic3net / attention_integrated:
    1. u = encoder(obs)
    2. h = GRU([u; m̄_prev], h)    ← received comm feeds into personal memory
    3. m̄ = comm(u, h)
    4. a ~ π([u; h])
    5. store m̄

  attention_separated:
    1. u = encoder(obs)
    2. h = GRU(u, h)               ← h is purely personal (no social input)
    3. c = Attention(u, h)         ← social context from pure personal memes
    4. a ~ π([u; h; c])            ← action sees personal + social memory
    (no m̄_prev buffer needed — comm is synchronous, not delayed)

Ablation variants:
  - Full:        h = GRU([u; m̄_prev], h), comm active
  - Memory only: h = GRU(u, h), no comm
  - Comm only:   no GRU, a ~ π([u; m̄]), instant coordination
  - Baseline:    no GRU, no comm, a ~ π(u)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..modules.memory_cells import GRUMemory
from ..modules.obs_encoder import ObsEncoder
from ..modules.comm_module import TargetedComm


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class MemeticFoundationAC(nn.Module):
    """Unified actor-critic with GRU memory and targeted communication.

    Supports 4 ablation variants via use_memory and use_comm flags.
    Parameter equalization: when modules are ablated, encoder and actor
    are widened to compensate for lost parameters.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        n_agents: int,
        hidden_dim: int = 128,
        mem_dim: int = 128,
        comm_dim: int = 128,
        n_mem_cells: int = 8,  # kept for backward compat, unused
        use_memory: bool = True,
        use_comm: bool = True,
        use_gate: bool = True,
        mem_decay: float = 0.005,
        comm_mode: str = "ic3net",  # 'ic3net' | 'attention_integrated' | 'attention_separated'
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim
        self.comm_dim = comm_dim
        self.use_memory = use_memory
        self.use_comm = use_comm
        self.use_gate = use_gate and (comm_mode == "ic3net")  # gate only for ic3net
        self.comm_mode = comm_mode

        # --- Parameter equalization ---
        ablated_params = self._estimate_ablated_params(
            hidden_dim, mem_dim, comm_dim, n_agents, use_memory, use_comm,
        )
        expanded_hidden = self._compute_expanded_hidden(
            obs_dim, n_actions, hidden_dim, mem_dim, comm_dim, ablated_params,
            use_memory, use_comm,
        )
        enc_dim = expanded_hidden if ablated_params > 0 else hidden_dim
        actor_hidden = expanded_hidden if ablated_params > 0 else hidden_dim
        self.enc_dim = enc_dim

        # --- Core modules ---
        self.encoder = ObsEncoder(obs_dim, enc_dim)

        # GRU memory (only if use_memory)
        # attention_separated: GRU takes only obs (no comm), so no m_bar_prev buffer
        self._gru_takes_comm = use_comm and (comm_mode != "attention_separated")
        if use_memory:
            if self._gru_takes_comm:
                gru_input_dim = enc_dim + comm_dim  # [u; m̄_prev]
            else:
                gru_input_dim = enc_dim  # just u (separated or no comm)
            self.memory = GRUMemory(n_agents, gru_input_dim, mem_dim, mem_decay)
            # Buffer for previous timestep's communication (not needed for separated)
            if self._gru_takes_comm:
                self.register_buffer(
                    "m_bar_prev", torch.zeros(n_agents, comm_dim)
                )
        else:
            self.memory = None

        # Communication (only if use_comm)
        # Gate is only used in ic3net mode; attention variants use pure softmax
        _use_gate = use_gate and (comm_mode == "ic3net")
        if use_comm:
            self.comm = TargetedComm(enc_dim, mem_dim, comm_dim, use_gate=_use_gate)
            if not use_memory:
                # No memory → need projection for comm input
                self.comm_z_proj = nn.Linear(enc_dim, mem_dim)
            else:
                self.comm_z_proj = None
        else:
            self.comm = None
            self.comm_z_proj = None

        # --- Actor head ---
        if use_memory and use_comm and comm_mode == "attention_separated":
            actor_in = enc_dim + mem_dim + comm_dim  # [u; h; c] — personal + social
        elif use_memory:
            actor_in = enc_dim + mem_dim      # [u; h]
        elif use_comm:
            actor_in = enc_dim + comm_dim     # [u; m̄]
        else:
            actor_in = enc_dim                # just u

        self.actor = nn.Sequential(
            nn.Linear(actor_in, actor_hidden),
            nn.ReLU(),
            nn.Linear(actor_hidden, n_actions),
        )

        # --- Critic (centralized) ---
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_actor_critic_weights()
        self.last_comm_attn: Optional[torch.Tensor] = None

    def _estimate_ablated_params(
        self, hidden_dim, mem_dim, comm_dim, n_agents, use_memory, use_comm,
    ) -> int:
        """Estimate parameters lost by ablation relative to full model."""
        lost = 0

        if not use_memory:
            # GRUCell(enc_dim + comm_dim, mem_dim):
            #   weight_ih: 3*mem_dim*(enc+comm), weight_hh: 3*mem*mem, biases: 6*mem
            # But enc_dim isn't known yet (it's what we're computing).
            # Use hidden_dim as the reference (full model uses hidden_dim as enc_dim)
            gru_input = hidden_dim + comm_dim
            lost += 3 * mem_dim * gru_input   # weight_ih
            lost += 3 * mem_dim * mem_dim     # weight_hh
            lost += 6 * mem_dim               # biases
            lost += mem_dim                   # h_init parameter

        if not use_comm:
            # TargetedComm: key, value, query heads
            input_dim = hidden_dim + mem_dim
            lost += input_dim * comm_dim + comm_dim  # key_head
            lost += input_dim * comm_dim + comm_dim  # value_head
            lost += input_dim * comm_dim + comm_dim  # query_head

        if use_comm and not use_memory:
            # comm_z_proj is ADDED (not lost)
            lost -= hidden_dim * mem_dim + mem_dim

        if use_memory and not use_comm:
            # GRU input shrinks from (enc+comm) to enc
            # Lost: 3*mem_dim*comm_dim from weight_ih
            lost += 3 * mem_dim * comm_dim
            # Also lost: m_bar_prev buffer (not a param, doesn't count)

        return max(lost, 0)

    def _compute_expanded_hidden(
        self, obs_dim, n_actions, hidden_dim, mem_dim, comm_dim,
        extra_params, use_memory, use_comm,
    ) -> int:
        """Compute expanded hidden dim to compensate for lost parameters."""
        if extra_params <= 0:
            return hidden_dim

        new_dim = hidden_dim
        while True:
            new_dim += 1
            # Encoder growth
            enc_new = obs_dim * new_dim + new_dim + new_dim * new_dim + new_dim
            enc_old = obs_dim * hidden_dim + hidden_dim + hidden_dim * hidden_dim + hidden_dim

            # Actor input depends on variant
            if use_memory:
                act_in_new = new_dim + mem_dim
                act_in_old = hidden_dim + mem_dim
            elif use_comm:
                act_in_new = new_dim + comm_dim
                act_in_old = hidden_dim + comm_dim
            else:
                act_in_new = new_dim
                act_in_old = hidden_dim

            act_new = act_in_new * new_dim + new_dim + new_dim * n_actions + n_actions
            act_old = act_in_old * hidden_dim + hidden_dim + hidden_dim * n_actions + n_actions

            gained = (enc_new - enc_old) + (act_new - act_old)

            # Account for surviving module growth
            if use_comm and not use_memory:
                gained += (new_dim - hidden_dim) * comm_dim * 3  # comm heads
                gained += (new_dim - hidden_dim) * mem_dim       # comm_z_proj

            if use_memory and not use_comm:
                # GRU weight_ih grows with enc_dim
                gained += 3 * mem_dim * (new_dim - hidden_dim)

            if gained >= extra_params:
                break
            if new_dim > hidden_dim * 4:
                break

        return new_dim

    def _init_actor_critic_weights(self):
        for module in [self.actor, self.critic]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def forward_step(
        self,
        obs: torch.Tensor,
        avail_actions: Optional[torch.Tensor] = None,
        intervene_comm_silence: bool = False,
        intervene_write_block: bool = False,
        deterministic: bool = False,
    ) -> dict:
        """Complete forward pass for one timestep.

        1. Encode obs → u
        2. GRU step (if memory): h = GRU([u; m̄_prev], h) or GRU(u, h)
        3. Communicate (if comm): m̄ = comm(u, h)
        4. Act from [u; h] or [u; m̄] or u
        5. Store m̄ for next step
        """
        u = self.encode(obs)  # (N, enc_dim)

        # --- GRU memory update ---
        h = None
        if self.use_memory and self.memory is not None:
            if self._gru_takes_comm:
                gru_input = torch.cat([u, self.m_bar_prev], dim=-1)
            else:
                gru_input = u  # separated: GRU only sees personal obs

            if not intervene_write_block:
                h = self.memory.step(gru_input)
            else:
                h = self.memory()

        # --- Communication ---
        m_bar = None
        comm_attn = None
        if self.use_comm and self.comm is not None:
            if h is not None:
                m_bar, comm_attn, _ = self.comm(u, h)
            else:
                z_proj = self.comm_z_proj(u)
                m_bar, comm_attn, _ = self.comm(u, z_proj)

            if intervene_comm_silence:
                m_bar = torch.zeros_like(m_bar)

            self.last_comm_attn = comm_attn.detach()

        # --- Action selection ---
        if self.comm_mode == "attention_separated" and h is not None and m_bar is not None:
            # Separated: actor sees personal memory AND social context independently
            actor_input = torch.cat([u, h, m_bar], dim=-1)
        elif h is not None:
            actor_input = torch.cat([u, h], dim=-1)
        elif m_bar is not None:
            actor_input = torch.cat([u, m_bar], dim=-1)
        else:
            actor_input = u

        logits = self.actor(actor_input)
        if avail_actions is not None:
            logits = logits.masked_fill(avail_actions == 0, -1e10)

        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # --- Store m̄ for next timestep (integrated only) ---
        if (self._gru_takes_comm and m_bar is not None and not intervene_comm_silence):
            self.m_bar_prev.data.copy_(m_bar.data)

        return {
            "u": u,
            "h": h,
            "actions": actions,
            "log_probs": log_probs,
            "logits": logits,
            "m_bar": m_bar,
            "comm_attn": comm_attn,
        }

    # ------------------------------------------------------------------
    # Critic and evaluation (for PPO)
    # ------------------------------------------------------------------

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Evaluate stored actions for PPO update.

        Uses current GRU hidden state as fixed context (no GRU step during
        PPO evaluation — same approach as the slot-based version).

        Returns: (log_probs, entropy, values, aux_loss, norms_dict)
        """
        u = self.encode(obs)
        B_total = u.shape[0]
        N = self.n_agents

        norms_dict = {
            "memory": 0.0,
            "memory_delta": 0.0,
            "message_out": 0.0,
            "message_in": 0.0,
            "gate_open_frac": 1.0,  # fraction of agents with gate open
        }

        # --- Build actor input ---
        if self.use_memory and self.memory is not None:
            h = self.memory()  # (N, mem_dim)
            norms_dict["memory"] = h.norm().item()

            if B_total != N:
                n_repeats = (B_total + N - 1) // N
                h_expanded = h.repeat(n_repeats, 1)[:B_total]
            else:
                h_expanded = h

            if self.comm_mode == "attention_separated" and self.use_comm and self.comm is not None:
                # Separated: compute comm from pure h, concatenate separately
                m_bar_eval, _, _ = self.comm(u[:N], h)
                if B_total != N:
                    m_bar_expanded = m_bar_eval.repeat(n_repeats, 1)[:B_total]
                else:
                    m_bar_expanded = m_bar_eval
                actor_input = torch.cat([u, h_expanded, m_bar_expanded], dim=-1)
                norms_dict["message_in"] = m_bar_eval.norm().item()
            else:
                actor_input = torch.cat([u, h_expanded], dim=-1)
        elif self.use_comm and self.comm is not None:
            # comm_only: run comm on first N agents, tile
            z_proj = self.comm_z_proj(u[:N])
            m_bar, _, _ = self.comm(u[:N], z_proj)
            if B_total != N:
                n_repeats = (B_total + N - 1) // N
                m_bar_expanded = m_bar.repeat(n_repeats, 1)[:B_total]
            else:
                m_bar_expanded = m_bar
            actor_input = torch.cat([u, m_bar_expanded], dim=-1)
            norms_dict["message_in"] = m_bar.norm().item()
            norms_dict["message_out"] = norms_dict["message_in"]
        else:
            actor_input = u

        logits = self.actor(actor_input)
        logits = logits.masked_fill(avail_actions == 0, -1e10)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.get_value(state)

        # --- Auxiliary loss & norms ---
        # Run a GRU step on the first N obs to provide gradient signal
        # to the GRU cell parameters. This is a "shadow step" that doesn't
        # update the actual hidden state but lets gradients flow.
        aux_loss = torch.tensor(0.0, device=obs.device)

        if B_total >= N and self.use_memory and self.memory is not None:
            u_group = u[:N]
            h_current = self.memory()  # (N, mem_dim) — buffer, detached

            if self.use_comm:
                # Build GRU input with m_bar_prev
                gru_input = torch.cat([u_group, self.m_bar_prev], dim=-1)
            else:
                gru_input = u_group

            # Shadow GRU step (with gradients, but doesn't update state)
            h_shadow = self.memory.gru_cell(gru_input, h_current)
            aux_loss = aux_loss + 0.001 * h_shadow.pow(2).mean()
            norms_dict["memory_delta"] = (h_shadow - h_current).norm().item()

        if B_total >= N and self.use_comm and self.comm is not None:
            u_group = u[:N]
            if self.use_memory and self.memory is not None:
                h_group = self.memory()
                m_bar_aux, _, gate_ent = self.comm(u_group, h_group)
            else:
                z_proj = self.comm_z_proj(u_group)
                m_bar_aux, _, gate_ent = self.comm(u_group, z_proj)

            aux_loss = aux_loss + 0.001 * m_bar_aux.pow(2).mean()
            # Gate entropy only for ic3net; attention modes use softmax (no entropy needed)
            if self.comm_mode == "ic3net":
                aux_loss = aux_loss + 0.05 * gate_ent
            norms_dict["message_in"] = m_bar_aux.norm().item()
            norms_dict["message_out"] = norms_dict["message_in"]

            # Track gate open fraction
            if self.use_gate and self.comm.gate is not None:
                z_gate = self.memory().detach() if (self.use_memory and self.memory) else self.comm_z_proj(u_group).detach()
                gate_hard, _ = self.comm.gate(z_gate)  # returns (gate, gate_soft)
                norms_dict["gate_open_frac"] = gate_hard.mean().item()

        return log_probs, entropy, values, aux_loss, norms_dict

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def detach_memory(self) -> None:
        if self.memory is not None:
            self.memory.detach_state()
        if hasattr(self, "m_bar_prev"):
            self.m_bar_prev.detach_()

    def reset_memory(self) -> None:
        if self.memory is not None:
            self.memory.reset_state()
        if self._gru_takes_comm and hasattr(self, "m_bar_prev"):
            self.m_bar_prev.zero_()

    def get_memory_state(self) -> Optional[torch.Tensor]:
        if self.memory is not None:
            return self.memory.get_state()
        return None

    def set_memory_state(self, state: torch.Tensor) -> None:
        if self.memory is not None:
            self.memory.set_state(state)

    def get_variant_name(self) -> str:
        if self.use_memory and self.use_comm:
            mode_suffix = f"_{self.comm_mode}" if self.comm_mode != "ic3net" else ""
            return f"full{mode_suffix}"
        elif self.use_comm:
            return "comm_only"
        elif self.use_memory:
            return "memory_only"
        else:
            return "baseline"

    def extra_repr(self) -> str:
        return (
            f"variant={self.get_variant_name()}, "
            f"n_agents={self.n_agents}, "
            f"enc_dim={self.enc_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"mem_dim={self.mem_dim}, "
            f"comm_dim={self.comm_dim}"
        )
