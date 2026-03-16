import numpy as np
import copy
from pettingzoo.mpe import simple_tag_v2
from pettingzoo.mpe import simple_spread_v2

class MPEWrapper:
    """Wraps PettingZoo MPE scenarios to match StarCraftCapabilityEnvWrapper API.

    Optional partial observability:
        obs_radius (float): if set, agent observations of other agents/prey
            are masked to zero when those entities are further than obs_radius
            world units away. This forces agents to hold genuinely different
            memory states (they see different subsets of the world), which is
            necessary for meaningful memetic differentiation.
            Set obs_radius=None (default) for full observability.
    """
    def __init__(self, scenario_name="simple_tag_v2", num_good=1, num_adversaries=3,
                 num_obstacles=2, max_cycles=100, obs_radius=None, **kwargs):
        self.scenario_name = scenario_name
        self.max_cycles = max_cycles
        self.obs_radius = obs_radius  # None = full obs; float = partial obs radius
        
        if self.scenario_name == "simple_tag_v2":
            # For 1M parameter logic
            self.env = simple_tag_v2.parallel_env(
                num_good=num_good, 
                num_adversaries=num_adversaries, 
                num_obstacles=num_obstacles, 
                max_cycles=max_cycles, 
                continuous_actions=False
            )
            self.num_agents = num_adversaries
            self.agents = [f"adversary_{i}" for i in range(num_adversaries)]
            self.prey_agent = "agent_0"
            self.num_landmarks = num_obstacles
        elif self.scenario_name == "simple_spread_v2":
            n_agents = kwargs.get('N', num_adversaries)
            self.env = simple_spread_v2.parallel_env(
                N=n_agents, 
                local_ratio=0.5, 
                max_cycles=max_cycles, 
                continuous_actions=False
            )
            self.num_agents = n_agents
            self.agents = [f"agent_{i}" for i in range(n_agents)]
            self.prey_agent = None
            self.num_landmarks = n_agents
        elif self.scenario_name == "simple_spread":
            n_agents = kwargs.get('N', num_adversaries)
            # Fallback if v3 is used or needed
            import importlib
            simple_spread = importlib.import_module("pettingzoo.mpe.simple_spread_v3")
            self.env = simple_spread.parallel_env(
                N=n_agents, 
                local_ratio=0.5, 
                max_cycles=max_cycles, 
                continuous_actions=False
            )
            self.num_agents = n_agents
            self.agents = [f"agent_{i}" for i in range(n_agents)]
            self.prey_agent = None
            self.num_landmarks = n_agents
        else:
            raise ValueError(f"Unknown scenario {self.scenario_name}")
            
        # Will be populated on reset
        self.last_obs = None
    
    def get_env_info(self):
        # We need to initialize the environment spaces
        self.env.reset()
        obs_shape = self.env.observation_space(self.agents[0]).shape[0]
        n_actions = self.env.action_space(self.agents[0]).n
        try:
            state_shape = self.env.state().shape[0]
        except Exception:
            state_shape = obs_shape * self.num_agents
        
        return {
            "n_agents": self.num_agents,
            "n_actions": n_actions,
            "obs_shape": obs_shape,
            "state_shape": state_shape,
            "episode_limit": self.max_cycles,
        }

    def reset(self):
        res = self.env.reset()
        if isinstance(res, tuple):
            self.last_obs = res[0]
        else:
            self.last_obs = res
        return self.get_obs(), self.get_state()

    def _prey_heuristic(self, obs_dict):
        # Simple heuristic: move away from average predator position
        prey_obs = obs_dict.get(self.prey_agent)
        if prey_obs is None:
            return 0  # noop if prey dead or missing
        
        # In simple_tag, agents 0..N-1 are adversaries, the last is the prey.
        # The prey's observation format ends with relative positions of the adversaries
        # [..., adv1_x, adv1_y, adv2_x, adv2_y, ...]
        adv_rel_pos = prey_obs[-self.num_agents*2:].reshape(self.num_agents, 2)
        avg_rel = np.mean(adv_rel_pos, axis=0)
        dx, dy = avg_rel[0], avg_rel[1]
        
        # 0: noop, 1: right (+x), 2: left (-x), 3: up (+y), 4: down (-y)
        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 1
        else:
            return 4 if dy > 0 else 3

    def step(self, actions):
        """
        actions: list of integers [a_0, a_1, ..., a_{N-1}] for the predators.
        """
        dict_actions = {}
        for i, agent in enumerate(self.agents):
            dict_actions[agent] = actions[i]
            
        # Add prey action
        if self.prey_agent and self.prey_agent in self.last_obs:
            dict_actions[self.prey_agent] = self._prey_heuristic(self.last_obs)
            
        step_res = self.env.step(dict_actions)
        # Handle v2 (obs, rew, done, info) vs v3 (obs, rew, term, trunc, info)
        if len(step_res) == 4:
            self.last_obs, rewards, dones, infos = step_res
        else:
            self.last_obs, rewards, terminations, truncations, infos = step_res
            dones = {k: terminations[k] or truncations.get(k, False) for k in terminations}
            
        # Predators get shared reward from collisions (adversary rewards are usually identical in simple_tag)
        team_reward = rewards.get(self.agents[0], 0.0)
        
        # Terminate if any agent is done
        terminated = any(dones.values())
        
        # Fill in missing agents in last_obs if they died (MPE might remove them from dict)
        all_entities = list(self.agents)
        if self.prey_agent:
            all_entities.append(self.prey_agent)
            
        for agent in all_entities:
            if agent not in self.last_obs:
                self.last_obs[agent] = np.zeros(self.env.observation_space(self.agents[0]).shape)
                
        # Parse MPE Custom Metrics
        info = {}
        if self.scenario_name in ["simple_spread_v2", "simple_spread"]:
            # Custom simple_spread metrics
            # Agent obs ends with [..., landmark_1_rel_x, landmark_1_rel_y, ..., other_agent_1_rel_x, ...]
            # The structure is: [self_vel(2), self_pos(2), landmark_rel_pos(N*2), other_agent_rel_pos((N-1)*2), comm_states]
            
            # Since PettingZoo doesn't easily expose global state positions in parallel_env without accessing the underlying AEC env,
            # we can infer everything from agent_0's observation!
            obs0 = self.last_obs[self.agents[0]]
            
            # 1. Landmark distances (from agent 0)
            ld_start = 4 # vel(2) + pos(2)
            ld_end = ld_start + (self.num_landmarks * 2)
            landmarks_rel_to_0 = obs0[ld_start:ld_end].reshape(self.num_landmarks, 2)
            
            # 2. Other agent distances (from agent 0)
            other_start = ld_end
            other_end = other_start + ((self.num_agents - 1) * 2)
            others_rel_to_0 = obs0[other_start:other_end].reshape(self.num_agents - 1, 2)
            
            # Reconstruct absolute positions (assuming agent_0 is at 0,0 for relative math)
            agent_positions = [np.array([0.0, 0.0])]
            for i in range(self.num_agents - 1):
                agent_positions.append(others_rel_to_0[i])
            agent_positions = np.array(agent_positions)
            
            landmark_positions = landmarks_rel_to_0 # Already relative to agent 0
            
            # Compute Min Distances to Landmarks
            min_dists = []
            is_covered = []
            for l_pos in landmark_positions:
                dists = np.linalg.norm(agent_positions - l_pos, axis=1)
                min_dist = np.min(dists)
                min_dists.append(min_dist)
                is_covered.append(min_dist < 0.1) # Covered threshold
                
            info['min_dist'] = float(np.mean(min_dists))
            info['success'] = all(is_covered)
            
            # Compute Collisions
            collisions = 0
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    dist = np.linalg.norm(agent_positions[i] - agent_positions[j])
                    # Agents have size 0.15 each -> 0.3 total for collision
                    if dist < 0.3:
                        collisions += 1
            info['collisions'] = collisions

        return team_reward, terminated, info

    def _apply_obs_radius(self, obs_list):
        """Mask out entities beyond obs_radius from each agent's observation.

        In simple_tag, the adversary observation format is:
          [self_vel(2), self_pos(2), obstacle_rel_pos(num_obs*2),
           other_adv_rel_pos((N-1)*2), prey_rel_pos(2), prey_vel(2)]

        We zero out other adversaries and prey positions when their distance
        (inferred from rel_pos) exceeds obs_radius.
        """
        if self.obs_radius is None:
            return obs_list

        masked = []
        for i, obs in enumerate(obs_list):
            obs = obs.copy()
            if self.scenario_name == "simple_tag_v2":
                # Layout: vel(2) pos(2) obs_rel(num_obs*2) other_adv_rel((N-1)*2) prey_rel(2) prey_vel(2)
                n_obs = self.num_landmarks  # obstacles
                N = self.num_agents         # adversaries
                adv_start = 4 + n_obs * 2
                # Mask other adversaries
                for j in range(N - 1):
                    idx = adv_start + j * 2
                    rel = obs[idx:idx+2]
                    dist = float(np.linalg.norm(rel))
                    if dist > self.obs_radius:
                        obs[idx:idx+2] = 0.0
                # Mask prey
                prey_start = adv_start + (N - 1) * 2
                prey_rel = obs[prey_start:prey_start+2]
                if float(np.linalg.norm(prey_rel)) > self.obs_radius:
                    obs[prey_start:prey_start+4] = 0.0  # zero rel_pos + vel

            elif self.scenario_name in ("simple_spread_v2", "simple_spread"):
                # Layout: vel(2) pos(2) landmark_rel(N*2) other_agent_rel((N-1)*2) [comm]
                # Partial obs: mask landmarks and other agents beyond obs_radius.
                # This forces each agent to focus on nearby landmarks → role specialization.
                N = self.num_agents
                n_landmarks = self.num_landmarks  # == N for spread
                landmark_start = 4  # after vel(2) + pos(2)
                # Mask landmarks
                for l in range(n_landmarks):
                    idx = landmark_start + l * 2
                    rel = obs[idx:idx+2]
                    dist = float(np.linalg.norm(rel))
                    if dist > self.obs_radius:
                        obs[idx:idx+2] = 0.0
                # Mask other agents
                agent_start = landmark_start + n_landmarks * 2
                for j in range(N - 1):
                    idx = agent_start + j * 2
                    rel = obs[idx:idx+2]
                    dist = float(np.linalg.norm(rel))
                    if dist > self.obs_radius:
                        obs[idx:idx+2] = 0.0
            masked.append(obs)
        return masked

    def get_obs(self):
        raw = [np.array(self.last_obs[agent], dtype=np.float32) for agent in self.agents]
        return self._apply_obs_radius(raw)

    def get_state(self):
        try:
            state = self.env.state()
            if state is not None:
                return np.array(state, dtype=np.float32)
        except Exception:
            pass
        return np.concatenate(self.get_obs())

    def get_avail_agent_actions(self, agent_id):
        n_actions = self.env.action_space(self.agents[0]).n
        return [1] * n_actions

    def close(self):
        self.env.close()
