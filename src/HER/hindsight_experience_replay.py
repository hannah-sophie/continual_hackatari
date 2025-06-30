import importlib

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnvWrapper

from src.HER.games.base_wrapper import HerWrapper

try:
    from termcolor import colored
except ImportError:

    def colored(text, color):
        return text

    print("Warning: termcolor not installed. Colored output will not be available.")


class GoalConditionedEnv(VecEnvWrapper):
    def __init__(self, venv, num_envs, env_id, game_specific_goals: bool):
        super().__init__(
            venv,
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )
        self.her_wrapper = HerWrapper(self, env_id, num_envs)
        self.game_name = env_id.lower()
        if game_specific_goals:
            try:
                her_wrapper_module = importlib.import_module(
                    f"src.HER.games.{self.game_name}"
                )
                self.her_wrapper = her_wrapper_module.HerWrapper(self, env_id, num_envs)

            except ModuleNotFoundError as e:
                print(
                    colored(
                        f"Error: {e}. No HER wrapper available for {self.game_name}.",
                        "yellow",
                    )
                )

    def reset(self):
        obs = self.venv.reset()
        self.her_wrapper.sample_new_goal()
        infos = self.reset_infos
        actual_goal = self.her_wrapper.extract_actual_goal(obs, infos)
        for i, inf in enumerate(infos):
            inf["original_reward"] = 0
            inf["actual_goal"] = actual_goal[i]
        return obs

    def append_goal_frame(self, obs, goal=None):
        return self.her_wrapper.append_goal_frame(obs, goal)

    def get_goal_space_shape(self):
        return self.her_wrapper.goal_space_shape()

    def step_wait(self):
        obs, original_reward, done, info = self.venv.step_wait()
        # compute goal‐conditioned reward
        actual_goal = self.her_wrapper.extract_actual_goal(obs, info)
        for i, inf in enumerate(info):
            inf["original_reward"] = original_reward[i]
            inf["actual_goal"] = actual_goal[i]
        r = self.her_wrapper.reward_fct(actual_goal, original_reward)
        return obs, r, done, info

    def relabel_trajectories(
        self,
        agent,
        obs,
        actions,
        rewards,
        dones,
        logprobs,
        values,
        actual_goals,
        device,
    ):
        def set_new_goal(indices):
            if len(indices) > 0:
                current_index = indices.pop(0)
                self.her_wrapper.set_new_goal(actual_goals[current_index])
            else:
                current_index = np.inf
                self.her_wrapper.set_new_goal(current_goal)
            return current_index

        current_goal = self.her_wrapper.goal
        obs_her = obs.detach().clone()
        actions_her = actions.detach().clone()
        logprobs_her = torch.zeros_like(logprobs).to(device)
        rewards_her = torch.zeros_like(rewards).to(device)
        dones_her = dones.detach().clone()
        values_her = torch.zeros_like(values).to(device)
        desired_goals_her = torch.zeros_like(actual_goals).to(device)
        indices = (dones == 1).nonzero(as_tuple=False)
        indices = [indices[k][0] for k in range(len(indices))]
        current_index = set_new_goal(indices)
        for i in range(len(obs)):
            ob = obs_her[i]
            if i > current_index:
                current_index = set_new_goal(indices)
            r = self.her_wrapper.reward_fct(actual_goals[i], rewards[i])
            rewards_her[i] = torch.tensor(r).to(device).view(-1)
            x = self.append_goal_frame(ob)
            action, logprob, _, value = agent.get_action_and_value(x, actions_her[i])
            logprobs_her[i] = logprob
            values_her[i] = value.flatten()
            desired_goals_her[i] = (
                torch.tensor(self.her_wrapper.goal).to(device).view(-1)
            )
        self.her_wrapper.set_new_goal(current_goal)

        return (
            obs_her,
            actions_her,
            rewards_her,
            dones_her,
            logprobs_her,
            values_her,
            desired_goals_her,
        )
