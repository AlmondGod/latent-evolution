# Tiny shared rollout helpers used across SMACv2 phase-2 scripts (probe + analysis).
# Kept dependency-light so probe/analysis scripts can import without pulling in
# every env wrapper (VMAS, MPE, RWARE, etc.).

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def tensorize_obs(obs_list, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(obs_list, dtype=np.float32), device=device)


def tensorize_avail(env, env_info: dict, device: torch.device) -> torch.Tensor:
    avail = np.zeros((env_info["n_agents"], env_info["n_actions"]), dtype=np.float32)
    for aid in range(env_info["n_agents"]):
        avail[aid] = np.asarray(env.get_avail_agent_actions(aid), dtype=np.float32)
    return torch.tensor(avail, device=device)


def make_smacv2_env(race: str, n_units: int, n_enemies: Optional[int] = None):
    from new.memetic_foundation.training.env_utils import make_env

    return make_env(race=race, n_units=n_units, n_enemies=n_enemies if n_enemies is not None else n_units)
