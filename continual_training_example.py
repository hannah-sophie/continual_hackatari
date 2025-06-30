# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import glob
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from rtpt import RTPT
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb
from architectures.ppo import shrink_perturb_agent_weights
from src.HER.hindsight_experience_replay import GoalConditionedEnv
from src.generic_eval import evaluate  # noqa
from src.modification_factories import get_modification_factory
from src.training_helpers import (
    TrainArgs,
    init_wandb,
    make_agent,
    make_env,
    save_agent,
    compute_gae,
)

# Suppress warnings to avoid cluttering output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set CUDA environment variable for determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Add custom paths if OC_ATARI_DIR is set (optional integration for extended functionality)
oc_atari_dir = os.getenv("OC_ATARI_DIR")
if oc_atari_dir is not None:
    oc_atari_path = os.path.join(Path(__file__), oc_atari_dir)
    sys.path.insert(1, oc_atari_path)

# Add the evaluation directory to the Python path to import custom evaluation functions
eval_dir = os.path.join(Path(__file__).parent.parent, "cleanrl_utils/evals/")
sys.path.insert(1, eval_dir)

# Global variable to hold parsed arguments
global args

if __name__ == "__main__":
    # Parse command-line arguments using Tyro
    args = tyro.cli(TrainArgs)
    if args.exp_name == "":
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    with open(args.config) as file:
        config = json.load(file)
    modification_factory = get_modification_factory(
        config["modification_factory"], config["modification_factory_kwargs"]
    )
    args.total_timesteps = modification_factory.get_total_timesteps()
    # Load configuration from file if provided
    # Generate run name based on environment, experiment, seed, and timestamp
    run_name = f"{args.env_id}_s{args.seed}__{args.exp_name}__{args.architecture}{'_shrink_perturb__' if args.shrink_and_perturb else '__'}{int(time.time())}"

    # Initialize tracking with Weights and Biases if enabled
    run, writer_dir, postfix = init_wandb(args)

    # Initialize Tensorboard SummaryWriter to log metrics and hyperparameters
    writer = SummaryWriter(writer_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    learning_rate = args.learning_rate
    # Create RTPT object to monitor progress with estimated time remaining

    rtpt = RTPT(
        name_initials=args.author,
        experiment_name="OCALM",
        max_iterations=args.num_iterations,
    )
    rtpt.start()  # Start RTPT tracking
    modif = modification_factory.get_modification(0)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = SubprocVecEnv(
        [
            make_env(args.env_id, i, args.capture_video, writer_dir, args, modif)
            for i in range(0, args.num_envs)
        ]
    )
    envs = VecNormalize(envs, norm_obs=False, norm_reward=True)

    if args.her and args.backend not in ["OCAtari", "HackAtari"]:
        raise ValueError("Her and backend must be either 'OCAtari' or 'HackAtari'")

    if args.her:
        envs = GoalConditionedEnv(
            envs, args.num_envs, args.env_id, args.game_specific_goals
        )

    # Seeding the environment and PyTorch for reproducibility
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.use_deterministic_algorithms(args.torch_deterministic)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_random_seed(args.seed, args.cuda)
    envs.seed(args.seed)
    envs.action_space.seed(args.seed)
    agent = make_agent(envs, args, device)
    # TODO what if agent depends on env

    # Initialize optimizer for training
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # Allocate storage for observations, actions, rewards, etc.
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.observation_space.shape
    ).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(
        device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)

    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    if args.her:
        original_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        actual_goals = torch.zeros(
            (args.num_steps, args.num_envs) + envs.get_goal_space_shape()
        ).to(device)
        desired_goals = torch.zeros(
            (args.num_steps, args.num_envs) + envs.get_goal_space_shape()
        ).to(device)
    # Start training loop
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    infos = envs.reset_infos
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    pbar = tqdm(range(1, args.num_iterations + 1), postfix=postfix)
    for iteration in pbar:  # Anneal learning rate if specified
        if args.anneal_lr:  # todo what to do with this? look at literature
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        elength = 0
        eorgr = 0
        enewr = 0
        count = 0

        done_in_episode = False
        new_modif = modification_factory.get_modification(global_step)
        if new_modif != modif:
            if args.shrink_and_perturb:
                shrink_perturb_agent_weights(
                    agent, args.shrink_factor, args.noise_scale
                )
            if args.save_agent_with_switch:
                save_agent(args, agent, modif, run, writer_dir)
                if args.capture_video:
                    import glob

                    list_of_videos = glob.glob(f"{writer_dir}/media/videos/*.mp4")
                    if len(list_of_videos) > 0:
                        latest_video = max(list_of_videos, key=os.path.getctime)
                        modif_name = modif if modif != "" else "no_modif"
                        modif_name = modif_name.replace(" ", "_")
                        if args.track:
                            wandb.log(
                                {f"video_{modif_name}": wandb.Video(latest_video)}
                            )
            modif = new_modif
            envs = SubprocVecEnv(
                [
                    make_env(
                        args.env_id, i, args.capture_video, writer_dir, args, modif
                    )
                    for i in range(0, args.num_envs)
                ]
            )
            envs = VecNormalize(envs, norm_obs=False, norm_reward=True)
            if args.her:
                envs = GoalConditionedEnv(
                    envs, args.num_envs, args.env_id, args.game_specific_goals
                )
            next_obs = envs.reset()
            infos = envs.reset_infos
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(args.num_envs).to(device)
            # TODO: seeding
        # Perform rollout in each environment
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # Get action and value from agent
            with torch.no_grad():
                x = next_obs
                if args.her:
                    x = envs.append_goal_frame(x)
                action, logprob, _, value = agent.get_action_and_value(x)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            if args.her:
                actual_goals[step] = (
                    torch.tensor([info["actual_goal"] for info in infos])
                    .to(device)
                    .view(-1)
                )
                desired_goals[step] = (
                    torch.tensor(envs.her_wrapper.goal).to(device).view(-1)
                )
            # Execute the game and store reward, next observation, and done flag
            next_obs, reward, next_done, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            if args.her:
                original_rewards[step] = (
                    torch.tensor([info["original_reward"] for info in infos])
                    .to(device)
                    .view(-1)
                )

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            # Track episode-level statistics if a game is done
            if 1 in next_done:

                for i, info in enumerate(infos):
                    if "episode" in info:
                        count += 1
                        done_in_episode = True
                        if args.new_rf:
                            enewr += info["episode"]["r"]
                            eorgr += info["org_return"]
                        else:
                            eorgr += info["episode"]["r"]
                        elength += info["episode"]["l"]
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)

        with torch.no_grad():
            advantages, returns = compute_gae(
                agent,
                envs,
                next_obs,
                next_done,
                desired_goals[-1] if args.her else None,
                rewards,
                dones,
                values,
                args,
                device,
            )
            advantages.detach()
            returns.detach()
            if args.her:
                (
                    obs_her,
                    actions_her,
                    rewards_her,
                    dones_her,
                    logprobs_her,
                    values_her,
                    desired_goals_her,
                ) = envs.relabel_trajectories(
                    agent,
                    obs,
                    actions,
                    original_rewards,
                    dones,
                    logprobs,
                    values,
                    actual_goals,
                    device,
                )
                advantages_her, returns_her = compute_gae(
                    agent,
                    envs,
                    next_obs.detach().clone(),
                    next_done.detach().clone(),
                    desired_goals_her[-1] if args.her else None,
                    rewards_her,
                    dones_her,
                    values_her,
                    args,
                    device,
                )
                advantages_her.detach()
                returns_her.detach()

        # Flatten the batch for optimization
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        if args.her:
            b_obs_her = obs_her.reshape((-1,) + envs.observation_space.shape)
            b_actions_her = actions_her.reshape((-1,) + envs.action_space.shape)
            b_logprobs_her = logprobs_her.reshape(-1)
            b_values_her = values_her.reshape(-1)
            b_advantages_her = advantages_her.reshape(-1)
            b_returns_her = returns_her.reshape(-1)
            b_goal_her = desired_goals_her.reshape((-1,) + envs.get_goal_space_shape())
            b_goal = desired_goals.reshape((-1,) + envs.get_goal_space_shape())

            b_obs = torch.cat([b_obs, b_obs_her], dim=0)
            b_actions = torch.cat([b_actions, b_actions_her], dim=0)
            b_logprobs = torch.cat([b_logprobs, b_logprobs_her], dim=0)
            b_values = torch.cat([b_values, b_values_her], dim=0)
            b_advantages = torch.cat([b_advantages, b_advantages_her], dim=0)
            b_returns = torch.cat([b_returns, b_returns_her], dim=0)
            # if your policy/V takes goals as input:
            b_goal = torch.cat([b_goal, b_goal_her], dim=0)
        # Optimize the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                x = b_obs[mb_inds]
                if args.her:
                    x = envs.append_goal_frame(x, b_goal[mb_inds])
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    x, b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approximate KL divergence
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    # Normalize advantages
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss (for exploration)
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Backpropagation and optimizer step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Compute explained variance (diagnostic measure for value function fit quality)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log episode statistics for Tensorboard
        if done_in_episode:
            if args.new_rf:
                writer.add_scalar(
                    "charts/Episodic_New_Reward", enewr / count, global_step
                )
            writer.add_scalar(
                "charts/Episodic_Original_Reward", eorgr / count, global_step
            )
            modif_name = modif if modif != "" else "no_modif"
            modif_name = modif_name.replace(" ", "_")
            writer.add_scalar(
                f"episodic_reward_by_modif/{modif_name}",
                eorgr / count,
                global_step,
            )
            writer.add_scalar("charts/Episodic_Length", elength / count, global_step)

            pbar.set_description(
                f"Reward: {eorgr if isinstance(eorgr, float) else eorgr.item()/ count:.1f}"
            )

        # Log other statistics
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        writer.add_text(
            "charts/Modifications",
            modif,
            global_step,
        )

        # Update RTPT for progress tracking
        rtpt.step()

    if args.capture_video:
        import glob

        list_of_videos = glob.glob(f"{writer_dir}/media/videos/*.mp4")
        latest_video = max(list_of_videos, key=os.path.getctime)
        modif_name = modif if modif != "" else "no_modif"
        modif_name = modif_name.replace(" ", "_")
        if args.track:
            wandb.log({f"video_{modif_name}": wandb.Video(latest_video)})

    # Save the trained model to disk
    save_agent(args, agent, modif, run, writer_dir)
    save_agent(args, agent, modif, run, writer_dir, True)
    # Log final model and performance with Weights and Biases if enabled
    if args.track:
        # Evaluate agent's performance
        args.new_rf = ""
        env = SubprocVecEnv(
            [
                make_env(
                    idx=0,
                    env_id=args.env_id,
                    capture_video=args.capture_video,
                    run_dir=writer_dir,
                    args=args,
                )
            ]
        )
        if args.her:
            env = GoalConditionedEnv(
                env, args.num_envs, args.env_id, args.game_specific_goals
            )
        rewards = evaluate(
            agent,
            env,
            args.eval_episodes,
            device=device,
            her=args.her,
        )

        wandb.log({"FinalReward": np.mean(rewards)})

        if args.test_modifs != "":
            args.modifs = args.test_modifs
            args.backend = "HackAtari"
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
            if args.her:
                env = GoalConditionedEnv(
                    env, args.num_envs, args.env_id, args.game_specific_goals
                )
            rewards = evaluate(
                agent,
                env,
                args.eval_episodes,
                device=device,
            )

            wandb.log({"HackAtariReward": np.mean(rewards)})

        # Log video of agent's performance
        if args.capture_video:
            list_of_videos = glob.glob(f"{writer_dir}/media/videos/*.mp4")
            latest_video = max(list_of_videos, key=os.path.getctime)
            wandb.log({"video_eval": wandb.Video(latest_video)})

        wandb.finish()

    # Close environments and writer after training is complete
    envs.close()
    writer.close()
