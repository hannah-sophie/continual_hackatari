import numpy as np
import torch

from src.HER.games.base_wrapper import BaseHerWrapper


class HerWrapper(BaseHerWrapper):

    def __init__(self, env, env_id, num_envs):
        super(HerWrapper, self).__init__(env, env_id, num_envs)
        self.igloo_position = np.array([123, 0])
        self.sample_new_goal()

    def extract_actual_goal(self, obs, info):
        ram_states = self.envs.env_method("get_ram")
        actual_goal = np.stack(ram_states)[:, (102, 100)]
        return actual_goal

    def goal_space_shape(self):
        return (2,)

    def set_new_goal(self, new_goal):
        if isinstance(new_goal, np.ndarray):
            self.goal = new_goal
        else:
            self.goal = np.full((self.num_envs, 2), new_goal)

    def number_of_input_planes(self):
        return self.envs.observation_space.shape[0] + 2

    def normalize_goal(self, goal):
        max_position = torch.Tensor(np.full(goal.shape, [150, 131]))
        min_position = torch.zeros_like(goal)
        clipped = torch.clip(goal, min_position, max_position)
        normed = clipped / max_position
        goal = torch.round(normed * 255.0).to(torch.int32)
        return goal

    def sample_new_goal(self):
        self.goal = np.full((self.num_envs, 2), self.igloo_position)
        return self.goal

    def goal_equivalence_mask(self, actual_goal):
        mask = super().goal_equivalence_mask(actual_goal)
        mask_no_water = actual_goal[:, 1] > 27
        return mask & mask_no_water
