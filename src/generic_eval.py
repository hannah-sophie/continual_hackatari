from typing import Callable

import torch
from stable_baselines3.common.vec_env import SubprocVecEnv


def evaluate(agent, make_env: Callable, eval_episodes: int, device, **env_kwargs):
    env = SubprocVecEnv([make_env(idx=0, **env_kwargs)])
    agent.eval()
    obs = env.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, next_done, infos = env.step(actions.cpu().numpy())
        if 1 in next_done:
            for info in infos:
                if "episode" not in info:
                    continue
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns
