"""Hanabi Learning Environment wrapper matching the project env interface.

Hanabi is a turn-based cooperative game (only one player acts per timestep),
but the trainer expects a simultaneous-action interface where every agent
provides an action at every step.  This wrapper bridges the gap:

  - ``get_obs()`` returns the vectorized observation for *every* agent,
    using each agent's most-recent ``player_observation``.
  - ``get_avail_agent_actions(i)`` returns the legal-action mask.  Only the
    current player has a non-trivial mask; other agents receive an all-ones
    mask (their "action" is ignored by ``step``).
  - ``step(actions)`` uses only the current player's action, advances the
    game, and returns (team_reward, terminated, info).
  - The team reward is the *incremental* score change (fireworks placed since
    the last step), matching standard RL conventions.

Game variants (configurable via constructor):
  - ``Hanabi-Full``   : 5 colors × 5 ranks  (default, max score 25)
  - ``Hanabi-Small``  : 2 colors × 5 ranks  (max score 10)
  - ``Hanabi-Very-Small`` : 1 color × 5 ranks (max score 5)
"""

from __future__ import annotations

import numpy as np

from hanabi_learning_environment import rl_env


class HanabiWrapper:
    """Wraps hanabi_learning_environment.rl_env to match the project env interface.

    Parameters
    ----------
    num_players : int
        Number of Hanabi players (2–5).  This maps directly to ``n_agents``.
    game_type : str
        One of ``"Hanabi-Full"``, ``"Hanabi-Small"``, ``"Hanabi-Very-Small"``.
    max_steps : int
        Maximum environment steps per episode.  Hanabi games naturally end when
        the deck runs out and each player takes a final turn, but this provides
        a hard safety cap.
    """

    def __init__(
        self,
        num_players: int = 2,
        game_type: str = "Hanabi-Full",
        max_steps: int = 80,
    ) -> None:
        self.num_players = num_players
        self.game_type = game_type
        self.max_steps = max_steps

        self.env = rl_env.make(game_type, num_players=num_players)
        self.n_actions = self.env.game.max_moves()

        # Determine obs shape from a dummy reset
        obs = self.env.reset()
        sample_vec = np.array(
            obs["player_observations"][0]["vectorized"], dtype=np.float32
        )
        self.obs_dim = sample_vec.shape[0]

        # Cache latest per-agent observations and the raw obs dict
        self._obs: dict = obs
        self._per_agent_obs: list[np.ndarray] = self._extract_all_obs(obs)
        self._step_count = 0
        self._prev_score = 0
        self._needs_reset = False

    # ------------------------------------------------------------------
    # Project env interface
    # ------------------------------------------------------------------

    def get_env_info(self) -> dict:
        return {
            "n_agents": self.num_players,
            "n_actions": self.n_actions,
            "obs_shape": self.obs_dim,
            "state_shape": self.obs_dim * self.num_players,
            "episode_limit": self.max_steps,
        }

    def reset(self, seed: int | None = None) -> tuple:
        if seed is not None:
            np.random.seed(int(seed))
        self._obs = self.env.reset()
        self._per_agent_obs = self._extract_all_obs(self._obs)
        self._step_count = 0
        self._prev_score = 0
        self._needs_reset = False
        return self.get_obs(), self.get_state()

    def get_obs(self) -> list[np.ndarray]:
        """Return list of float32 arrays, one per agent."""
        return [o.copy() for o in self._per_agent_obs]

    def get_state(self) -> np.ndarray:
        """Global state = concatenation of all agent observations."""
        return np.concatenate(self._per_agent_obs, axis=0).astype(np.float32)

    def get_avail_agent_actions(self, agent_id: int) -> np.ndarray:
        """Return a binary mask of shape (n_actions,) for the given agent.

        Only the current player has a meaningful mask (1 for legal moves,
        0 for illegal).  Non-current players get all-ones — their action
        is discarded by ``step()`` anyway.
        """
        mask = np.zeros(self.n_actions, dtype=np.float32)
        cur_player = self._obs["current_player"]
        if agent_id == cur_player:
            legal = self._obs["player_observations"][cur_player][
                "legal_moves_as_int"
            ]
            for a in legal:
                mask[a] = 1.0
        else:
            # Non-active agents: all actions "available" (ignored by step)
            mask[:] = 1.0
        return mask

    def step(self, actions: list[int]) -> tuple[float, bool, dict]:
        """Execute one Hanabi step using the current player's action.

        Parameters
        ----------
        actions : list[int]
            One action per agent.  Only ``actions[current_player]`` is used.

        Returns
        -------
        team_reward : float
            Incremental score change (0 or 1 per step in standard Hanabi).
        terminated : bool
            Whether the game has ended.
        info : dict
            Contains ``score``, ``life_tokens``, ``max_score``, and ``success``.
        """
        cur_player = self._obs["current_player"]
        action = int(actions[cur_player])

        # Validate action is legal; fall back to first legal move if not
        legal = self._obs["player_observations"][cur_player][
            "legal_moves_as_int"
        ]
        if action not in legal:
            action = legal[0] if legal else 0

        obs, reward, done, env_info = self.env.step(action)
        self._obs = obs
        self._per_agent_obs = self._extract_all_obs(obs)
        self._step_count += 1

        # Compute incremental reward (reward from env is the raw score delta)
        team_reward = float(reward)

        # Compute current score from fireworks
        fireworks = obs["player_observations"][0]["fireworks"]
        score = sum(fireworks.values())

        # Check for max-steps truncation
        terminated = done or (self._step_count >= self.max_steps)

        # Build info dict
        max_score = self.env.game.num_colors() * self.env.game.num_ranks()
        info = {
            "score": score,
            "life_tokens": obs["player_observations"][0]["life_tokens"],
            "max_score": max_score,
            "success": score == max_score,
        }

        if terminated:
            self._needs_reset = True

        return team_reward, terminated, info

    def close(self) -> None:
        """No-op; Hanabi env has no resources to release."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_all_obs(self, obs: dict) -> list[np.ndarray]:
        """Extract vectorized observations for all agents.

        Each agent sees its own partial observation (can see others' cards
        but not their own).  The vectorized encoding already handles this.
        """
        result = []
        for pid in range(self.num_players):
            po = obs["player_observations"][pid]
            vec = np.array(po["vectorized"], dtype=np.float32)
            result.append(vec)
        return result
