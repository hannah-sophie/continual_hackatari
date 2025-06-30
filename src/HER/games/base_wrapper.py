from abc import ABC, abstractmethod
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize

from src.HER.atari_data import get_human_score

class BaseHerWrapper(ABC):
    def __init__(self, envs, env_id, num_envs):
        self.envs = envs
        self.game_name = env_id.lower()
        self.num_envs = num_envs

    @abstractmethod
    def reward_fct(self, actual_goal, original_reward):
        pass

    @abstractmethod
    def extract_actual_goal(self, obs, info):
        pass

    @abstractmethod
    def set_new_goal(self, new_goal):
        pass

    @abstractmethod
    def sample_new_goal(self):
        pass

    @abstractmethod
    def append_goal_frame(self, obs, goal=None):
        pass

    @abstractmethod
    def goal_space_shape(self):
        pass


## game unspecific her wrapper (based on rewards)
class HerWrapper(BaseHerWrapper):
    def __init__(self, envs, env_id, num_envs):
        super().__init__(envs, env_id, num_envs)
        self.max_reward = get_human_score(self.game_name)
        self.goal = np.full(self.num_envs, self.max_reward)

    def sample_new_goal(self):
        self.goal = np.full(self.num_envs, self.max_reward)
        return self.goal

    def set_new_goal(self, new_goal):
        if isinstance(new_goal, np.ndarray):
            self.goal = new_goal
        else:
            self.goal = np.full(self.num_envs, new_goal)

    def extract_actual_goal(self, obs, infos):
        actual_goals = [0] * self.num_envs
        for i, info in enumerate(infos):
            if "episode" in info:
                actual_goals[i] = info["episode"]["r"]
        return actual_goals

    def reward_fct(self, actual_goal, original_reward):
        original_reward = self.envs.get_original_reward() if hasattr(self.envs, "get_original_reward") else original_reward
        idx = np.array((self.goal == actual_goal).nonzero())
        original_reward[tuple(idx.T)] = (original_reward[tuple(idx.T)] + 1) * 2
        return self.envs.normalize_reward(original_reward) if hasattr(self.envs, "get_original_reward") else original_reward

    def goal_space_shape(self):
        return ()

    def append_goal_frame(self, obs, goal=None):
        B, C, H, W = obs.shape
        num_envs = self.num_envs
        rollout_length = B // num_envs
        env_ids = (
            torch.arange(num_envs, device=obs.device)
            .unsqueeze(1)
            .expand(num_envs, rollout_length)
            .reshape(-1)
        )

        max_scaled = 1.5 * self.max_reward
        goal = torch.tensor(self.goal) if goal is None else goal.detach().clone()
        clipped = torch.clip(goal, 0.0, max_scaled)
        normed = clipped / max_scaled
        goal = torch.round(normed * 255.0).to(torch.int32)
        goal = goal[env_ids].to(obs.device)
        goal_plane = goal.view(B, 1, 1, 1).expand(B, 1, H, W)
        return torch.cat([obs, goal_plane], dim=1)