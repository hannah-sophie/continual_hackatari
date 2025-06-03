# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import json
import logging
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
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

from src.training_helpers import make_env, make_agent
from src.modification_factories import ModificationFactory, get_modification_factory
from src.generic_eval import evaluate  # noqa

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


# Command line argument configuration using dataclass
@dataclass
class Args:
    # config file
    config: str = ""
    """Path to the config file, if set it will override command line arguments"""

    # General
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Environment parameters
    env_id: str = "ALE/Pong-v5"
    """the id of the environment"""
    obs_mode: str = "dqn"
    """observation mode for OCAtari"""
    feature_func: str = ""
    """the object features to use as observations"""
    buffer_window_size: int = 4
    """length of history in the observations"""
    backend: str = "HackAtari"
    """Which Backend should we use"""
    modifs: str = ""
    """Modifications for Hackatari"""
    new_rf: str = ""
    """Path to a new reward functions for OCALM and HACKATARI"""
    frameskip: int = 4
    """the frame skipping option of the environment"""

    # Tracking (Logging and monitoring configurations)
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "continual_hackatari"
    """the wandb's project name"""
    wandb_entity: str = "lrteam"
    """the entity (team) of wandb's project"""
    wandb_dir: str = None
    """the wandb directory"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    ckpt: str = ""
    """Path to a checkpoint to a model to start training from"""
    author: str = "JB"
    """Initials of the author"""

    # Algorithm-specific arguments
    architecture: str = "PPO"
    """ Specifies the used architecture"""

    total_timesteps: int = 20_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Transformer parameters
    emb_dim: int = 128
    """input embedding size of the transformer"""
    num_heads: int = 64
    """number of multi-attention heads"""
    num_blocks: int = 1
    """number of transformer blocks"""
    patch_size: int = 12
    """ViT patch size"""

    # PPObj network parameters
    encoder_dims: list[int] = (256, 512, 1024, 512)
    """layer dimensions before nn.Flatten()"""
    decoder_dims: list[int] = (512,)
    """layer dimensions after nn.Flatten()"""

    # HackAtari testing
    test_modifs: str = ""
    """Modifications for Hackatari"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# Global variable to hold parsed arguments
global args


if __name__ == "__main__":
    # Parse command-line arguments using Tyro
    args = tyro.cli(Args)
    with open(args.config) as file:
        config = json.load(file)
    modification_factory = get_modification_factory(
        config["modification_factory"], config["modification_factory_kwargs"]
    )
    args.total_timesteps = modification_factory.get_total_timesteps()
    # Load configuration from file if provided
    # Generate run name based on environment, experiment, seed, and timestamp
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Initialize tracking with Weights and Biases if enabled
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,  # True
            save_code=True,
            dir=args.wandb_dir,
        )
        writer_dir = run.dir
        postfix = dict(url=run.url)
    else:
        global_dir = f"{args.wandb_dir }/" if args.wandb_dir is not None else ""
        writer_dir = f"{global_dir}runs/{run_name}"
        postfix = None

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

    # Start training loop
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
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
            next_obs = envs.reset()
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
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and store reward, next observation, and done flag
            next_obs, reward, next_done, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            # Track episode-level statistics if a game is done
            if 1 in next_done:
                for info in infos:
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
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # Flatten the batch for optimization
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
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

    # Save the trained model to disk
    model_path = f"{writer_dir}/{args.exp_name}_{modif}.cleanrl_model"
    model_data = {
        "model_weights": agent.state_dict(),
        "args": vars(args),
    }
    torch.save(model_data, model_path)
    if args.capture_video:
        import glob

        list_of_videos = glob.glob(f"{writer_dir}/media/videos/*.mp4")
        latest_video = max(list_of_videos, key=os.path.getctime)
        modif_name = modif if modif != "" else "no_modif"
        wandb.log({f"video_{modif_name}": wandb.Video(latest_video)})

    # Save the trained model to disk
    model_path = f"{writer_dir}/{args.exp_name}.cleanrl_model"
    model_data = {
        "model_weights": agent.state_dict(),
        "args": vars(args),
    }
    torch.save(model_data, model_path)
    # Log final model and performance with Weights and Biases if enabled
    if args.track:
        # Evaluate agent's performance
        args.new_rf = ""
        rewards = evaluate(
            agent,
            make_env,
            10,
            env_id=args.env_id,
            capture_video=args.capture_video,
            run_dir=writer_dir,
            device=device,
            args=args,
        )

        wandb.log({"FinalReward": np.mean(rewards)})

        if args.test_modifs != "":
            args.modifs = args.test_modifs
            args.backend = "HackAtari"
            rewards = evaluate(
                agent,
                make_env,
                10,
                env_id=args.env_id,
                capture_video=args.capture_video,
                run_dir=writer_dir,
                device=device,
                args=args,
                modifs=args.modifs,
            )

            wandb.log({"HackAtariReward": np.mean(rewards)})

        # Log model to Weights and Biases
        name = f"{args.exp_name}_s{args.seed}"
        run.log_model(model_path, name)  # noqa: cannot be undefined

        # Log video of agent's performance
        if args.capture_video:
            import glob

            list_of_videos = glob.glob(f"{writer_dir}/media/videos/*.mp4")
            latest_video = max(list_of_videos, key=os.path.getctime)
            wandb.log({"video_eval": wandb.Video(latest_video)})

        wandb.finish()

    # Close environments and writer after training is complete
    envs.close()
    writer.close()
