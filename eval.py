import numpy as np
import torch
import tyro
from hackatari.core import HackAtari
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.training_helpers import make_agent, make_env
from src.generic_eval import evaluate, EvalArgs

has_agent = False

if __name__ == "__main__":
    args = tyro.cli(EvalArgs)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    modifs_list = [i for i in args.modifs.split(" ") if i]

    env = SubprocVecEnv(
        [
            make_env(
                idx=0,
                env_id=args.env_id,
                capture_video=args.capture_video,
                run_dir="eval",
                args=args,
                modifs=args.modifs,
            )
        ]
    )

    obs = env.reset()
    agent = make_agent(env, args, device)

    ckpt = torch.load(args.agent_path, map_location=torch.device("cpu"))
    agent.load_state_dict(ckpt["model_weights"])

    episode_rewards = evaluate(agent, env, args.eval_episodes, device)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"{mean_reward} +- {std_reward}")
    env.close()
