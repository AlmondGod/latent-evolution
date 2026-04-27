"""
smacv2_vec_env.py — Subprocess-pooled vector env for SMACv2.

n_envs SC2 instances run in worker processes. Main process batches obs across
envs, runs one policy forward over (n_envs * n_agents), and dispatches actions
back. CUDA stays in the main process; workers are CPU-only. Done envs auto-
reset and the post-reset obs is returned alongside the terminal transition,
so the trainer never sees a "stale" env state.

Worker IPC (Pipe send/recv):
  cmd = "step",                data = list[int]                -> (obs, state, avail, reward, terminated, info, next_obs, next_state, next_avail)
  cmd = "reset",               data = None                     -> None
  cmd = "get_obs_state_avail", data = None                     -> (obs, state, avail)
  cmd = "get_env_info",        data = None                     -> dict
  cmd = "close",               data = None                     -> (worker exits)

Only baseline policies (no recurrent state) are supported here. For memory/
comm variants we'd need to track per-env hidden states in the main process
and adapt the policy's mean-of-others pooling to a per-env axis.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Tuple

import numpy as np

from .env_utils import make_env


def _worker(remote: mp.connection.Connection, race: str, n_units: int, n_enemies: int) -> None:
    env = make_env(race=race, n_units=n_units, n_enemies=n_enemies, render=False)
    env.reset()
    info_cache = env.get_env_info()
    n_agents: int = info_cache["n_agents"]

    while True:
        cmd, data = remote.recv()

        if cmd == "step":
            reward, terminated, info = env.step(data)
            obs = env.get_obs()
            state = env.get_state()
            avail = [env.get_avail_agent_actions(i) for i in range(n_agents)]
            next_obs = next_state = next_avail = None
            if terminated:
                env.reset()
                next_obs = env.get_obs()
                next_state = env.get_state()
                next_avail = [env.get_avail_agent_actions(i) for i in range(n_agents)]
            remote.send((obs, state, avail, reward, terminated, info, next_obs, next_state, next_avail))

        elif cmd == "reset":
            env.reset()
            remote.send(None)

        elif cmd == "get_obs_state_avail":
            obs = env.get_obs()
            state = env.get_state()
            avail = [env.get_avail_agent_actions(i) for i in range(n_agents)]
            remote.send((obs, state, avail))

        elif cmd == "get_env_info":
            remote.send(info_cache)

        elif cmd == "close":
            env.close()
            remote.close()
            return


class SMACv2VecEnv:
    """Vector env over n_envs SC2 instances using a process pool."""

    def __init__(self, n_envs: int, race: str, n_units: int, n_enemies: int) -> None:
        self._closed = True  # so __del__ is a no-op if init fails midway
        self.n_envs = n_envs
        # spawn so each worker gets a clean Python interpreter (CUDA-safe)
        ctx = mp.get_context("spawn")
        parent_pipes, worker_pipes = zip(*[ctx.Pipe(duplex=True) for _ in range(n_envs)])
        self._remotes = list(parent_pipes)
        self._processes = []
        for worker_pipe in worker_pipes:
            p = ctx.Process(target=_worker, args=(worker_pipe, race, n_units, n_enemies), daemon=True)
            p.start()
            self._processes.append(p)
            worker_pipe.close()

        self._remotes[0].send(("get_env_info", None))
        self._env_info: dict = self._remotes[0].recv()
        self.n_agents: int = self._env_info["n_agents"]
        self.n_actions: int = self._env_info["n_actions"]
        self.obs_shape: int = self._env_info["obs_shape"]
        self.state_shape: int = self._env_info["state_shape"]

        self._closed = False

    def get_env_info(self) -> dict:
        return dict(self._env_info)

    def reset_all(self) -> None:
        for r in self._remotes:
            r.send(("reset", None))
        for r in self._remotes:
            r.recv()

    def get_obs_state_avail(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for r in self._remotes:
            r.send(("get_obs_state_avail", None))
        results = [r.recv() for r in self._remotes]
        obs_b = np.stack([np.asarray(r[0], dtype=np.float32) for r in results])
        state_b = np.stack([np.asarray(r[1], dtype=np.float32) for r in results])
        avail_b = np.stack([np.asarray(r[2], dtype=np.float32) for r in results])
        return obs_b, state_b, avail_b

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, np.ndarray]:
        # actions: (n_envs, n_agents) int array. Returns batched arrays + post-reset obs in
        # place of terminal obs so the trainer can keep stepping without explicit resets.
        # Per-env terminal flag is preserved so the value bootstrap can still mask correctly.
        for r, a in zip(self._remotes, actions):
            r.send(("step", a.tolist()))
        results = [r.recv() for r in self._remotes]

        obs_list, state_list, avail_list, reward_list, done_list, info_list = [], [], [], [], [], []
        for r in results:
            obs, state, avail, reward, terminated, info, next_obs, next_state, next_avail = r
            if terminated:
                # Ship post-reset obs/state/avail forward so the next iteration sees a fresh env.
                obs_list.append(np.asarray(next_obs, dtype=np.float32))
                state_list.append(np.asarray(next_state, dtype=np.float32))
                avail_list.append(np.asarray(next_avail, dtype=np.float32))
            else:
                obs_list.append(np.asarray(obs, dtype=np.float32))
                state_list.append(np.asarray(state, dtype=np.float32))
                avail_list.append(np.asarray(avail, dtype=np.float32))
            reward_list.append(float(reward))
            done_list.append(bool(terminated))
            info_list.append(info)

        obs_b = np.stack(obs_list)
        state_b = np.stack(state_list)
        avail_b = np.stack(avail_list)
        rewards_b = np.asarray(reward_list, dtype=np.float32)
        dones_b = np.asarray(done_list, dtype=np.float32)
        # The current-step obs (BEFORE auto-reset) is what was used to pick the action.
        # We don't need to return it — the trainer already cached it from the previous
        # get_obs_state_avail / step call. Returning the *next* obs after auto-reset is
        # what matters for the next iteration.
        return obs_b, state_b, avail_b, rewards_b, dones_b, info_list, np.stack([np.asarray(r[0], dtype=np.float32) for r in results])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for r in self._remotes:
            r.send(("close", None))
        for p in self._processes:
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()

    def __del__(self) -> None:
        self.close()
