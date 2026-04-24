from __future__ import annotations

import numpy as np
import torch
from vmas import make_env


class VMASWrapper:
    """VMAS wrapper using fixed-workload cooperative scenarios.

    Supported scenarios:
      - transport: keep `n_packages` fixed while varying `n_agents`
      - discovery: keep `n_targets` fixed while varying `n_agents`
    """

    def __init__(
        self,
        scenario_name: str,
        n_agents: int,
        max_steps: int = 100,
        device: str = "cpu",
        n_packages: int = 1,
        n_targets: int = 4,
        agents_per_target: int = 1,
        targets_respawn: bool = False,
        shared_reward: bool = True,
    ) -> None:
        self.scenario_name = scenario_name
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.device = torch.device(device)

        kwargs = {"n_agents": n_agents}
        if scenario_name == "transport":
            kwargs["n_packages"] = n_packages
        elif scenario_name == "discovery":
            kwargs["n_targets"] = n_targets
            kwargs["agents_per_target"] = agents_per_target
            kwargs["targets_respawn"] = targets_respawn
            kwargs["shared_reward"] = shared_reward
        else:
            raise ValueError(f"Unsupported VMAS scenario: {scenario_name}")

        self.env = make_env(
            scenario=scenario_name,
            num_envs=1,
            device=self.device,
            continuous_actions=False,
            max_steps=max_steps,
            **kwargs,
        )
        self.last_obs = None

    def _obs_to_numpy(self, obs_list):
        return [o.squeeze(0).detach().cpu().numpy().astype(np.float32) for o in obs_list]

    def get_env_info(self):
        if self.last_obs is None:
            self.last_obs = self.env.reset()
        obs_shape = int(self.last_obs[0].shape[-1])
        n_actions = int(self.env.get_agent_action_space(self.env.agents[0]).n)
        return {
            "n_agents": self.n_agents,
            "n_actions": n_actions,
            "obs_shape": obs_shape,
            "state_shape": obs_shape * self.n_agents,
            "episode_limit": self.max_steps,
        }

    def reset(self, seed=None):
        if seed is not None:
            seed = int(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.last_obs = self.env.reset()
        return self.get_obs(), self.get_state()

    def get_obs(self):
        return self._obs_to_numpy(self.last_obs)

    def get_state(self):
        return np.concatenate(self.get_obs(), axis=0).astype(np.float32)

    def get_avail_agent_actions(self, agent_id: int):
        return np.ones(self.env.get_agent_action_space(self.env.agents[agent_id]).n, dtype=np.float32)

    def _transport_info(self):
        packages = getattr(self.env.scenario, "packages", [])
        if not packages:
            return {}
        dists = []
        on_goal = 0
        for package in packages:
            dist = torch.linalg.vector_norm(
                package.state.pos - package.goal.state.pos, dim=1
            )[0].item()
            dists.append(float(dist))
            on_goal += int(bool(package.on_goal[0].item()))
        return {
            "mean_goal_dist": float(np.mean(dists)),
            "packages_on_goal": float(on_goal),
            "success": bool(on_goal == len(packages)),
        }

    def _discovery_info(self, infos):
        first = infos[0] if infos else {}
        out = {}
        if "targets_covered" in first:
            covered = float(first["targets_covered"][0].item())
            out["targets_covered"] = covered
            n_targets = float(getattr(self.env.scenario, "n_targets", 1))
            out["coverage_frac"] = covered / max(n_targets, 1.0)
        all_time = getattr(self.env.scenario, "all_time_covered_targets", None)
        if all_time is not None:
            completed = float(all_time[0].sum().item())
            total = float(all_time.shape[-1])
            out["targets_completed"] = completed
            out["completion_frac"] = completed / max(total, 1.0)
            out["success"] = bool(all_time[0].all().item())
        return out

    def step(self, actions):
        action_tensors = [
            torch.tensor([int(a)], device=self.device, dtype=torch.long) for a in actions
        ]
        obs, rewards, dones, infos = self.env.step(action_tensors)
        self.last_obs = obs
        team_reward = float(torch.stack(rewards).mean().item())
        terminated = bool(dones[0].item())
        if self.scenario_name == "transport":
            info = self._transport_info()
        else:
            info = self._discovery_info(infos)
        return team_reward, terminated, info

    def close(self):
        try:
            self.env.close()
        except AttributeError:
            pass
