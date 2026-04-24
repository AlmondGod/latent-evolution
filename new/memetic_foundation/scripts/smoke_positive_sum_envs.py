#!/opt/homebrew/bin/python3.9
from __future__ import annotations

import argparse
import json
import numpy as np
from statistics import mean

from new.memetic_foundation.training.rware_wrapper import RWAREWrapper
from new.memetic_foundation.training.vmas_wrapper import VMASWrapper


def run_episode(env):
    env.reset()
    info = {}
    terminated = False
    ep_reward = 0.0
    steps = 0
    while not terminated:
        actions = []
        env_info = env.get_env_info()
        for agent_id in range(env_info["n_agents"]):
            avail = env.get_avail_agent_actions(agent_id)
            avail_idx = np.flatnonzero(avail)
            actions.append(int(np.random.choice(avail_idx)))
        reward, terminated, info = env.step(actions)
        ep_reward += reward
        steps += 1
    return {"reward": ep_reward, "steps": steps, **info}


def build_env(kind: str, n_agents: int):
    if kind == "rware":
        return RWAREWrapper(
            n_agents=n_agents,
            shelf_rows=2,
            shelf_columns=3,
            column_height=8,
            request_queue_size=2,
            sensor_range=1,
            max_steps=50,
            reward_type="global",
        )
    if kind == "transport":
        return VMASWrapper(
            scenario_name="transport",
            n_agents=n_agents,
            max_steps=50,
            n_packages=1,
        )
    if kind == "discovery":
        return VMASWrapper(
            scenario_name="discovery",
            n_agents=n_agents,
            max_steps=50,
            n_targets=4,
            agents_per_target=1,
            targets_respawn=False,
            shared_reward=True,
        )
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--agent-counts", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()

    results = {}
    for kind in ["rware", "transport", "discovery"]:
        results[kind] = {}
        for n in args.agent_counts:
            env = build_env(kind, n)
            rows = [run_episode(env) for _ in range(args.episodes)]
            env.close()
            reward_mean = mean(r["reward"] for r in rows)
            out = {"mean_reward": reward_mean, "mean_steps": mean(r["steps"] for r in rows)}
            for key in ["success", "mean_goal_dist", "targets_covered", "coverage_frac", "deliveries_proxy"]:
                vals = [r[key] for r in rows if key in r]
                if vals:
                    out[f"mean_{key}"] = mean(vals)
            results[kind][str(n)] = out
            print(json.dumps({"env": kind, "n_agents": n, **out}))


if __name__ == "__main__":
    main()
