"""
trainer_phase1.py — thin trainer wrapper that restores the Phase-1 network path.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from ..models.agent_network_phase1 import MemeticFoundationAC
from .trainer import MemeticFoundationTrainer


class MemeticFoundationTrainerPhase1(MemeticFoundationTrainer):
    """Same trainer logic as MemeticFoundationTrainer, but with the restored model."""

    def __init__(
        self,
        env,
        device: str = "cpu",
        lr: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 10.0,
        update_epochs: int = 5,
        num_mini_batches: int = 1,
        hidden_dim: int = 128,
        mem_dim: int = 128,
        comm_dim: int = 128,
        n_mem_cells: int = 8,
        use_memory: bool = True,
        use_comm: bool = True,
        use_gate: bool = True,
        mem_decay: float = 0.005,
        comm_mode: str = "ic3net",
        param_eq: bool = False,
        persistent_memory: bool = False,
    ):
        self.env = env
        self.vec_env = None
        self.device = torch.device(device)

        env_info = env.get_env_info()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.obs_shape = env_info["obs_shape"]
        self.state_shape = env_info["state_shape"]

        self.policy = MemeticFoundationAC(
            obs_dim=self.obs_shape,
            state_dim=self.state_shape,
            n_actions=self.n_actions,
            n_agents=self.n_agents,
            hidden_dim=hidden_dim,
            mem_dim=mem_dim,
            comm_dim=comm_dim,
            n_mem_cells=n_mem_cells,
            use_memory=use_memory,
            use_comm=use_comm,
            use_gate=use_gate,
            mem_decay=mem_decay,
            comm_mode=comm_mode,
            param_eq=param_eq,
            persistent_memory=persistent_memory,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.lr_init = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_mini_batches = num_mini_batches

        self._started = False
        self._episode_reward = 0.0

        variant = self.policy.get_variant_name()
        total_params = sum(p.numel() for p in self.policy.parameters())
        print(f"  Variant:    {variant}")
        print(f"  Parameters: {total_params:,}")
        print(f"  Enc dim:    {self.policy.enc_dim}")
        print(f"  Device:     {self.device}")
