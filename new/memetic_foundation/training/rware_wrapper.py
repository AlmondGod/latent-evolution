from __future__ import annotations

import numpy as np
import rware
from rware.warehouse import ObservationType, RewardType, Warehouse


class RWAREWrapper:
    """Thin RWARE wrapper matching the project env interface.

    The important scaling knob here is `request_queue_size`: keep it fixed while
    varying `n_agents` to create a more positive-sum workload than the default
    RWARE registrations, which scale queue size with agent count.
    """

    def __init__(
        self,
        n_agents: int,
        shelf_rows: int = 2,
        shelf_columns: int = 3,
        column_height: int = 8,
        request_queue_size: int = 2,
        sensor_range: int = 1,
        max_steps: int = 100,
        reward_type: str = "global",
    ) -> None:
        reward_enum = {
            "global": RewardType.GLOBAL,
            "individual": RewardType.INDIVIDUAL,
            "twostage": RewardType.TWO_STAGE,
        }[reward_type.lower()]

        self.n_agents = n_agents
        self.max_steps = max_steps
        self.request_queue_size = request_queue_size
        self.reward_type = reward_enum
        self.env = Warehouse(
            shelf_columns=shelf_columns,
            column_height=column_height,
            shelf_rows=shelf_rows,
            n_agents=n_agents,
            msg_bits=0,
            sensor_range=sensor_range,
            request_queue_size=request_queue_size,
            max_inactivity_steps=None,
            max_steps=max_steps,
            reward_type=reward_enum,
            observation_type=ObservationType.FLATTENED,
        )
        self.last_obs = None

    def get_env_info(self):
        obs_shape = int(self.env.observation_space[0].shape[0])
        n_actions = int(self.env.action_space[0].n)
        return {
            "n_agents": self.n_agents,
            "n_actions": n_actions,
            "obs_shape": obs_shape,
            "state_shape": obs_shape * self.n_agents,
            "episode_limit": self.max_steps,
        }

    def reset(self, seed=None):
        if seed is None:
            obs, _info = self.env.reset()
        else:
            obs, _info = self.env.reset(seed=int(seed))
        self.last_obs = tuple(np.asarray(o, dtype=np.float32) for o in obs)
        return self.get_obs(), self.get_state()

    def get_obs(self):
        return [np.asarray(o, dtype=np.float32) for o in self.last_obs]

    def get_state(self):
        return np.concatenate(self.get_obs(), axis=0).astype(np.float32)

    def get_avail_agent_actions(self, agent_id: int):
        return np.ones(self.env.action_space[agent_id].n, dtype=np.float32)

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        self.last_obs = tuple(np.asarray(o, dtype=np.float32) for o in obs)
        team_reward = float(np.mean(rewards))
        out_info = dict(info)
        out_info["deliveries_proxy"] = team_reward
        return team_reward, bool(terminated or truncated), out_info

    def close(self):
        self.env.close()
