"""
agent_network.py — Unified actor-critic for Memetic Foundation (GRU version).

Single GRU per agent replaces the slot-based memory bank. The GRU hidden
state h serves as both memory representation and recurrent state.

Communication modes (comm_mode):
  - 'ic3net':                 Binary gate (straight-through) + m̄_prev feeds INTO GRU
                              via concatenation [u; m̄_prev]. Gate entropy loss.
  - 'attention_integrated':   Soft attention, m̄_prev feeds INTO GRU via concatenation.
  - 'attention_separated':    Soft attention; GRU sees ONLY personal obs.
                              h stays a pure individual meme; actor gets [u; h; c].
  - 'commnet':                CommNet-style (Sukhbaatar 2016) / IC3Net-style addition.
                              Message = h itself (no projection). Aggregation = mean of
                              neighbors' h. Integration = ADDITIVE: gru_input = u + proj(c).
                              GRU input dim stays enc_dim (no concatenation).
                              This avoids the noise-corruption problem of cat([u; m̄_prev])
                              early in training when messages are uninformative.

Forward pass per comm_mode:

  ic3net / attention_integrated (forward_step):
    1. u = encoder(obs)
    2. h = GRU([u; m̄_prev], h)    ← received comm feeds into personal memory
    3. m̄ = comm(u, h)
    4. a ~ π([u; h])
    5. store m̄

  attention_separated (forward_step):
    1. u = encoder(obs)
    2. h = GRU(u, h)               ← h is purely personal (no social input)
    3. c = Attention(u, h)         ← social context from pure personal memes
    4. a ~ π([u; h; c])            ← action sees personal + social memory
    (no m̄_prev buffer needed — comm is synchronous, not delayed)

PPO evaluate_actions uses a "shadow GRU step" for differentiable comm:
  - GRU and comm weights receive policy gradient via h_shadow (not detached buffer)
  - ic3net/integrated: h_shadow = GRU([u; m̄], h_buf), actor sees [u; h_shadow]
  - separated:         h_shadow = GRU(u, h_buf), m̄ = comm(u, h_shadow),
                       actor sees [u; h_shadow; m̄]
  This is the key fix: previously comm only received gradient from a tiny L2
  auxiliary loss, not from the policy objective.

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
        comm_mode: str = "commnet",
        param_eq: bool = False,  # disabled: all variants use hidden_dim, no widening
        persistent_memory: bool = False,  # if True, reset_memory() is a no-op
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
        self.use_gate = use_gate and (comm_mode == "ic3net")
        self.comm_mode = comm_mode
        self.persistent_memory = persistent_memory

        # Classify mode
        _concat_comm = use_comm and use_memory and comm_mode in ("ic3net", "attention_integrated")
        _additive_comm = use_comm and use_memory and comm_mode == "commnet"
        _sep_comm = use_comm and use_memory and comm_mode in ("attention_separated", "commnet_sep")
        # comm_only commnet: additive comm based on neighbors' encoded obs (no GRU)
        _additive_comm_ff = use_comm and not use_memory and comm_mode == "commnet"

        # --- Encoder dim: fixed at hidden_dim unless param_eq enabled ---
        if param_eq:
            ablated_params = self._estimate_ablated_params(
                hidden_dim, mem_dim, comm_dim, n_agents, use_memory, use_comm,
                comm_mode=comm_mode,
            )
            expanded_hidden = self._compute_expanded_hidden(
                obs_dim, n_actions, hidden_dim, mem_dim, comm_dim, ablated_params,
                use_memory, use_comm, comm_mode=comm_mode,
            )
            enc_dim = expanded_hidden if ablated_params > 0 else hidden_dim
        else:
            enc_dim = hidden_dim
        self.enc_dim = enc_dim
        actor_hidden = hidden_dim  # always hidden_dim for actor MLP width

        # --- Encoder ---
        self.encoder = ObsEncoder(obs_dim, enc_dim)

        # --- GRU memory ---
        # Concat modes: GRU takes [u; m̄_prev], needs m_bar_prev buffer
        # Additive (commnet): GRU takes u, comm added before GRU via addition
        # Separated: GRU takes only u, comm goes to actor
        self._gru_takes_comm = _concat_comm  # only concat modes use m_bar_prev
        if use_memory:
            gru_input_dim = (enc_dim + comm_dim) if _concat_comm else enc_dim
            self.memory = GRUMemory(n_agents, gru_input_dim, mem_dim, mem_decay)
            if _concat_comm:
                self.register_buffer("m_bar_prev", torch.zeros(n_agents, comm_dim))
        else:
            self.memory = None

        # --- Communication modules ---
        _use_gate = use_gate and (comm_mode == "ic3net")
        self.comm = None
        self.comm_z_proj = None
        self.comm_mean_proj = None

        if _additive_comm or _sep_comm or _additive_comm_ff:
            # commnet / commnet_sep: mean of neighbors' h → linear proj → additive or actor
            # comm_only commnet: mean of neighbors' u → linear proj → additive (no GRU)
            # Zero-init so at step 0 comm contributes nothing (starts as memory_only/baseline)
            if _additive_comm:
                proj_in, proj_out = mem_dim, enc_dim   # h → u space
            elif _sep_comm:
                proj_in, proj_out = mem_dim, mem_dim   # h → h space
            else:  # _additive_comm_ff
                proj_in, proj_out = enc_dim, enc_dim   # u → u space
            self.comm_mean_proj = nn.Linear(proj_in, proj_out)
            nn.init.zeros_(self.comm_mean_proj.weight)
            nn.init.zeros_(self.comm_mean_proj.bias)
        elif use_comm:
            # ic3net / attention variants: TargetedComm attention module
            self.comm = TargetedComm(enc_dim, mem_dim, comm_dim, use_gate=_use_gate)
            if not use_memory:
                self.comm_z_proj = nn.Linear(enc_dim, mem_dim)

        # --- Actor input dim ---
        if use_memory and use_comm:
            if comm_mode == "commnet":
                actor_in = enc_dim + mem_dim          # [u; h] — comm baked into h
            elif comm_mode in ("commnet_sep", "attention_separated"):
                proj_out = mem_dim if comm_mode == "commnet_sep" else comm_dim
                actor_in = enc_dim + mem_dim + proj_out  # [u; h; c]
            else:
                actor_in = enc_dim + mem_dim          # [u; h] — integrated
        elif use_memory:
            actor_in = enc_dim + mem_dim
        elif use_comm and comm_mode == "commnet":
            actor_in = enc_dim                        # u + proj(mean_u_others) — additive, stays enc_dim
        elif use_comm:
            actor_in = enc_dim + comm_dim
        else:
            actor_in = enc_dim

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
        comm_mode: str = "commnet",
    ) -> int:
        """Estimate parameters lost by ablation relative to full model.

        Reference model is commnet (mem + additive comm). Lost params are
        estimated relative to that architecture so that param_eq widens
        ablated variants to match commnet's size.
        """
        lost = 0
        _commnet_style = comm_mode in ("commnet", "commnet_sep")

        if not use_memory:
            # GRUCell(hidden_dim, mem_dim) — commnet GRU takes enc_dim input
            gru_input = hidden_dim  # no comm concatenation in commnet
            lost += 3 * mem_dim * gru_input   # weight_ih
            lost += 3 * mem_dim * mem_dim     # weight_hh
            lost += 6 * mem_dim               # biases
            lost += mem_dim                   # h_init parameter

        if not use_comm:
            if _commnet_style:
                # commnet comm = single Linear(mem_dim, enc_dim or mem_dim)
                proj_out = hidden_dim  # enc_dim for commnet, mem_dim for commnet_sep
                lost += mem_dim * proj_out + proj_out
            else:
                # TargetedComm: key, value, query heads
                input_dim = hidden_dim + mem_dim
                lost += input_dim * comm_dim + comm_dim  # key_head
                lost += input_dim * comm_dim + comm_dim  # value_head
                lost += input_dim * comm_dim + comm_dim  # query_head

        if use_comm and not use_memory and not _commnet_style:
            # TargetedComm comm_only: comm_z_proj is ADDED (not lost)
            lost -= hidden_dim * mem_dim + mem_dim
        # commnet comm_only: comm_mean_proj stays Linear(enc_dim, enc_dim) ≈ same size as
        # full commnet's Linear(mem_dim, enc_dim) when enc_dim==mem_dim — no adjustment needed

        return max(lost, 0)

    def _compute_expanded_hidden(
        self, obs_dim, n_actions, hidden_dim, mem_dim, comm_dim,
        extra_params, use_memory, use_comm, comm_mode: str = "commnet",
    ) -> int:
        """Compute expanded hidden dim to compensate for lost parameters."""
        if extra_params <= 0:
            return hidden_dim

        _commnet_style = comm_mode in ("commnet", "commnet_sep")
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
            elif use_comm and _commnet_style:
                # commnet comm_only: additive, actor still sees enc_dim
                act_in_new = new_dim
                act_in_old = hidden_dim
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
            if use_comm and not use_memory and not _commnet_style:
                gained += (new_dim - hidden_dim) * comm_dim * 3  # TargetedComm heads
                gained += (new_dim - hidden_dim) * mem_dim       # comm_z_proj
            elif use_comm and not use_memory and _commnet_style:
                # comm_mean_proj: Linear(new_dim, new_dim) grows quadratically
                gained += new_dim * new_dim - hidden_dim * hidden_dim  # weight
                gained += new_dim - hidden_dim                          # bias

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
            N = u.shape[0]
            if self.comm_mode == "commnet" and self.comm_mean_proj is not None:
                # Additive: gru_input = u + proj(mean of others' h from last step)
                h_buf = self.memory()
                total = h_buf.sum(dim=0, keepdim=True)
                c = (total - h_buf) / max(N - 1, 1)
                gru_input = u + self.comm_mean_proj(c)     # enc_dim + enc_dim (additive)
            elif self._gru_takes_comm:
                gru_input = torch.cat([u, self.m_bar_prev], dim=-1)
            else:
                gru_input = u  # commnet_sep / separated: GRU sees only personal obs

            if not intervene_write_block:
                h = self.memory.step(gru_input)
            else:
                h = self.memory()

        # --- Communication ---
        m_bar = None
        comm_attn = None
        if self.comm_mode in ("commnet_sep", "attention_separated") and h is not None \
                and self.comm_mean_proj is not None:
            # commnet_sep: mean of others' CURRENT h → project → goes to actor
            N = u.shape[0]
            total = h.sum(dim=0, keepdim=True)
            c = (total - h) / max(N - 1, 1)
            m_bar = self.comm_mean_proj(c)          # (N, mem_dim)
            self.last_comm_attn = None
        elif not self.use_memory and self.comm_mode == "commnet" \
                and self.comm_mean_proj is not None:
            # comm_only commnet: mean of neighbors' encoded obs → additive to u
            N = u.shape[0]
            total = u.sum(dim=0, keepdim=True)
            c = (total - u) / max(N - 1, 1)
            m_bar = self.comm_mean_proj(c)          # (N, enc_dim) — additive
        elif self.use_comm and self.comm is not None:
            if h is not None:
                m_bar, comm_attn, _ = self.comm(u, h)
            else:
                z_proj = self.comm_z_proj(u)
                m_bar, comm_attn, _ = self.comm(u, z_proj)

            if intervene_comm_silence:
                m_bar = torch.zeros_like(m_bar)

            self.last_comm_attn = comm_attn.detach()

        # --- Action selection ---
        if self.comm_mode in ("attention_separated", "commnet_sep") \
                and h is not None and m_bar is not None:
            actor_input = torch.cat([u, h, m_bar], dim=-1)   # [u; personal_meme; social_meme]
        elif h is not None:
            actor_input = torch.cat([u, h], dim=-1)
        elif not self.use_memory and self.comm_mode == "commnet" and m_bar is not None:
            actor_input = u + m_bar                           # additive: u + proj(mean_u_others)
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
        """Evaluate stored actions for PPO update with differentiable communication.

        The key fix over the previous version: communication is recomputed with
        current parameters and flows through to the actor via a shadow GRU step,
        giving the comm module a real policy gradient signal.

        Gradient paths per mode:
          ic3net / attention_integrated:
            loss → actor → h_shadow → GRU_cell([u; m_bar]) → comm_weights + encoder
          attention_separated:
            loss → actor → [h_shadow; m_bar] → GRU_cell(u) + comm_weights → encoder
          memory_only:
            loss → actor → h_shadow → GRU_cell(u) → encoder
          baseline:
            loss → actor → u → encoder

        The shadow GRU step approximates h_t using end-of-rollout h as init.
        This avoids storing per-timestep hidden states while still giving the
        GRU and comm modules real policy gradient.

        Returns: (log_probs, entropy, values, aux_loss, norms_dict)
        """
        u = self.encode(obs)
        B_total = u.shape[0]
        N = self.n_agents
        u_group = u[:N]  # (N, enc_dim) — one copy per real agent

        norms_dict = {
            "memory": 0.0,
            "memory_delta": 0.0,
            "message_out": 0.0,
            "message_in": 0.0,
            "gate_open_frac": 1.0,
        }
        aux_loss = torch.tensor(0.0, device=obs.device)

        # Helper: tile (N, d) → (B_total, d)
        def _tile(t):
            if B_total == N:
                return t
            n_reps = (B_total + N - 1) // N
            return t.repeat(n_reps, 1)[:B_total]

        # Helper: shadow GRU step (differentiable, does NOT update buffer)
        def _shadow_gru(x, h_buf):
            h_s = self.memory.gru_cell(x, h_buf)
            if self.memory.mem_decay > 0:
                h_s = h_s * (1.0 - self.memory.mem_decay)
            return h_s

        # --- Build actor input with differentiable comm ---
        if self.use_memory and self.memory is not None:
            h = self.memory()  # (N, mem_dim) — buffer, no grad
            norms_dict["memory"] = h.norm().item()

            if self.comm_mode == "commnet" and self.comm_mean_proj is not None:
                # Additive: gru_input = u + proj(mean_h), full gradient path
                total = h.sum(dim=0, keepdim=True)
                c = (total - h) / max(N - 1, 1)
                c_proj = self.comm_mean_proj(c)         # (N, enc_dim)
                norms_dict["message_in"] = c_proj.norm().item()
                h_shadow = _shadow_gru(u_group + c_proj, h)
                norms_dict["memory_delta"] = (h_shadow - h).norm().item()
                actor_input = torch.cat([u, _tile(h_shadow)], dim=-1)

            elif self.comm_mode == "commnet_sep" and self.comm_mean_proj is not None:
                # GRU sees only u; social meme = mean of others' h → actor
                # Gradient paths: loss → actor → h_shadow → GRU(u) → encoder
                #                 loss → actor → c_proj → comm_mean_proj
                h_shadow = _shadow_gru(u_group, h)
                total = h_shadow.sum(dim=0, keepdim=True)
                c = (total - h_shadow) / max(N - 1, 1)
                c_proj = self.comm_mean_proj(c)         # (N, mem_dim)
                norms_dict["message_in"] = c_proj.norm().item()
                norms_dict["memory_delta"] = (h_shadow - h).norm().item()
                actor_input = torch.cat([u, _tile(h_shadow), _tile(c_proj)], dim=-1)

            elif self.use_comm and self.comm is not None:
                if self.comm_mode in ("attention_separated",):
                    # GRU stays pure (no comm input).
                    # Shadow GRU: loss → actor → h_shadow → GRU_cell(u) → encoder
                    h_shadow = _shadow_gru(u_group, h)
                    # Comm from fresh h_shadow so both GRU and comm get policy gradient.
                    # loss → actor → m_bar → comm(u, h_shadow) → GRU → encoder
                    m_bar, _, _ = self.comm(u_group, h_shadow)
                    norms_dict["message_in"] = m_bar.norm().item()
                    norms_dict["memory_delta"] = (h_shadow - h).norm().item()
                    actor_input = torch.cat([u, _tile(h_shadow), _tile(m_bar)], dim=-1)

                else:
                    # ic3net / attention_integrated: social info merges into GRU via concat.
                    # Full path: loss → actor → h_shadow → GRU([u; m_bar]) → comm + encoder
                    m_bar, _, gate_ent = self.comm(u_group, h)
                    norms_dict["message_in"] = m_bar.norm().item()
                    norms_dict["message_out"] = m_bar.norm().item()
                    gru_input = torch.cat([u_group, m_bar], dim=-1)
                    h_shadow = _shadow_gru(gru_input, h)
                    norms_dict["memory_delta"] = (h_shadow - h).norm().item()
                    actor_input = torch.cat([u, _tile(h_shadow)], dim=-1)

                    if self.comm_mode == "ic3net":
                        aux_loss = aux_loss + 0.05 * gate_ent
                        if self.use_gate and self.comm.gate is not None:
                            gate_hard, _ = self.comm.gate(h)
                            norms_dict["gate_open_frac"] = gate_hard.mean().item()

            else:
                # Memory only: loss → actor → h_shadow → GRU_cell(u) → encoder
                h_shadow = _shadow_gru(u_group, h)
                norms_dict["memory_delta"] = (h_shadow - h).norm().item()
                actor_input = torch.cat([u, _tile(h_shadow)], dim=-1)

        elif self.comm_mode == "commnet" and self.comm_mean_proj is not None:
            # comm_only commnet: mean of neighbors' u → additive → actor
            # Gradient: loss → actor → (u + c_proj) → comm_mean_proj + encoder
            total = u_group.sum(dim=0, keepdim=True)
            c = (total - u_group) / max(N - 1, 1)
            c_proj = self.comm_mean_proj(c)             # (N, enc_dim)
            norms_dict["message_in"] = c_proj.norm().item()
            actor_input = u + _tile(c_proj)             # additive, stays enc_dim

        elif self.use_comm and self.comm is not None:
            # Comm only (no GRU): loss → actor → m_bar → comm + encoder
            z_proj = self.comm_z_proj(u_group)
            m_bar, _, gate_ent = self.comm(u_group, z_proj)
            norms_dict["message_in"] = m_bar.norm().item()
            norms_dict["message_out"] = m_bar.norm().item()
            actor_input = torch.cat([u, _tile(m_bar)], dim=-1)
            if self.comm_mode == "ic3net":
                aux_loss = aux_loss + 0.05 * gate_ent

        else:
            # Baseline: loss → actor → u → encoder
            actor_input = u

        logits = self.actor(actor_input)
        logits = logits.masked_fill(avail_actions == 0, -1e10)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.get_value(state)

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
        if self.persistent_memory:
            return  # persistent: memory carries across episode boundaries
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
        persist = "_persistent" if self.persistent_memory else ""
        if self.use_memory and self.use_comm:
            return f"full_{self.comm_mode}{persist}"
        elif self.use_comm and not self.use_memory:
            return f"comm_only_{self.comm_mode}"
        elif self.use_memory:
            return f"memory_only{persist}"
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
