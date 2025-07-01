import glob
import os

import numpy as np
import torch
import tyro
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.training_helpers import make_agent, make_env, init_wandb
from src.generic_eval import evaluate, EvalArgs

has_agent = False

if __name__ == "__main__":
    args = tyro.cli(EvalArgs)
    if args.exp_name == "":
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    modifs_list = [i for i in args.modifs.split(" ") if i]
    _, writer_dir, postfix = init_wandb(args)

    env = SubprocVecEnv(
        [
            make_env(
                idx=0,
                env_id=args.env_id,
                capture_video=args.capture_video,
                run_dir=writer_dir,
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
    if args.track:
        import wandb

        wandb.log({"FinalReward": mean_reward})
        if args.capture_video:
            list_of_videos = glob.glob(f"{writer_dir}/media/videos/*.mp4")
            latest_video = max(list_of_videos, key=os.path.getctime)
            wandb.log({"video_eval": wandb.Video(latest_video)})
        wandb.finish()
    env.close()
