import glob
import os
import re

import numpy as np
import torch
import tyro
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.HER.hindsight_experience_replay import GoalConditionedEnv
from src.training_helpers import make_agent, make_env, init_wandb
from src.generic_eval import evaluate, EvalArgs

has_agent = False

if __name__ == "__main__":
    args = tyro.cli(EvalArgs)
    if args.exp_name == "":
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    modifs_list = [i for i in args.modifs.split(" ") if i]
    if args.wandb_run_id is None:
        m = re.search(r'run_id([A-Za-z0-9]+)(?=[^A-Za-z0-9]|$)', args.agent_path)
        if m:
            args.wandb_run_id = m.group(1)
    ckpt = torch.load(args.agent_path, map_location=torch.device("cpu"))
    model_args = ckpt["args"]
    args.her = model_args["her"]
    args.num_envs = 1
    args.wandb_dir = model_args["wandb_dir"]
    args.wandb_project_name = model_args["wandb_project_name"]
    args.wandb_entity = model_args["wandb_entity"]
    args.wandb_run_name = model_args["wandb_run_name"]
    _, writer_dir, postfix = init_wandb(args, job_type="eval")

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
    if args.her and args.backend not in ["OCAtari", "HackAtari"]:
        raise ValueError("Her and backend must be either 'OCAtari' or 'HackAtari'")

    if args.her:
        env = GoalConditionedEnv(
            env, args.num_envs, args.env_id, args.game_specific_goals
        )

    obs = env.reset()
    agent = make_agent(env, args, device)
    agent.load_state_dict(ckpt["model_weights"])

    episode_rewards = evaluate(agent, env, args.eval_episodes, device,her=args.her)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"{mean_reward} +- {std_reward}")
    if args.track:
        import wandb

        wandb.log({f"{args.modifs}/FinalReward_eval": mean_reward})
        if args.capture_video:
            list_of_videos = glob.glob(f"{writer_dir}/media/videos/*.mp4")
            latest_video = max(list_of_videos, key=os.path.getctime)
            wandb.log({f"{args.modifs}/video_eval": wandb.Video(latest_video)})
        wandb.finish()
    env.close()
