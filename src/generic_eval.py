import torch
from src.training_helpers import Args
from dataclasses import dataclass


@dataclass
class EvalArgs(Args):
    agent_path: str = ""
    """Path to the agent to be evaluated"""
    eval_episodes: int = 21
    """Number of episodes for evaluation"""


def evaluate(agent, env, eval_episodes: int, device,  her:bool=False):
    agent.eval()
    obs = env.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        x = torch.Tensor(obs).to(device)
        if her:
            x = env.append_goal_frame(x)
        actions, _, _, _ = agent.get_action_and_value(x)
        next_obs, _, next_done, infos = env.step(actions.cpu().numpy())
        if 1 in next_done:
            for info in infos:
                if "episode" not in info:
                    continue
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns
