from gymnasium.wrappers.transform_observation import GrayscaleObservation
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
)
import ale_py
import gymnasium as gym
from dataclasses import dataclass
import os
import time


# Command line argument configuration using dataclass
@dataclass
class Args:
    # config file
    config: str = ""
    """Path to the config file, if set it will override command line arguments"""

    # General
    exp_name: str = ""
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


@dataclass
class TrainArgs(Args):
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

    # HackAtari testing
    test_modifs: str = ""
    """Modifications for Hackatari"""
    eval_episodes: int = 10
    """number of episodes to evaluate the agent"""
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# Function to create a gym environment with the specified settings
def make_env(env_id, idx, capture_video, run_dir, args, modifs=""):
    """
    Creates a gym environment with the specified settings.
    """

    def thunk():
        # Setup environment based on backend type (HackAtari, OCAtari, Gym)
        if args.backend == "HackAtari":
            from hackatari.core import HackAtari

            modifs_list = [i for i in modifs.split(" ") if i]
            env = HackAtari(
                env_id,
                modifs=modifs_list,
                rewardfunc_path=args.new_rf,
                obs_mode=args.obs_mode,
                hud=False,
                render_mode="rgb_array",
                frameskip=args.frameskip,
            )
            env.ale = env._env.unwrapped.ale
        elif args.backend == "OCAtari":
            from ocatari.core import OCAtari

            env = OCAtari(
                env_id,
                hud=False,
                render_mode="rgb_array",
                obs_mode=args.obs_mode,
                frameskip=args.frameskip,
            )
            env.ale = env._env.unwrapped.ale
        elif args.backend == "Gym":
            # Use Gym backend with image preprocessing wrappers
            env = gym.make(env_id, render_mode="rgb_array", frameskip=args.frameskip)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = GrayscaleObservation(env)
            env = gym.wrappers.FrameStackObservation(env, args.buffer_window_size)
        else:
            raise ValueError("Unknown Backend")

        # Capture video if required
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env, f"{run_dir}/media/videos", disable_logger=True
            )

        # Apply standard Atari environment wrappers
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = EpisodicLifeEnv(env)
        action_meanings = [
            ale_py.Action(a).name for a in env.unwrapped.ale.getMinimalActionSet()
        ]
        if "FIRE" in action_meanings:
            env = FireResetEnv(env)

        # If architecture is OCT, apply OCWrapper to environment
        if args.architecture == "OCT":
            from ocrltransformer.wrappers import OCWrapper

            env = OCWrapper(env)

        return env

    return thunk


def make_agent(envs, args, device):
    # Define the agent's architecture based on command line arguments
    if args.architecture == "OCT":
        from architectures.transformer import OCTransformer as Agent

        agent = Agent(envs, args.emb_dim, args.num_heads, args.num_blocks, device).to(
            device
        )
    elif args.architecture == "VIT":
        from architectures.transformer import VIT as Agent

        agent = Agent(
            envs,
            args.emb_dim,
            args.num_heads,
            args.num_blocks,
            args.patch_size,
            args.buffer_window_size,
            device,
        ).to(device)
    elif args.architecture == "VIT2":
        from architectures.transformer import SimpleViT2 as Agent

        agent = Agent(
            envs,
            args.emb_dim,
            args.num_heads,
            args.num_blocks,
            args.patch_size,
            args.buffer_window_size,
            device,
        ).to(device)
    elif args.architecture == "MobileVit":
        from architectures.transformer import MobileVIT as Agent

        agent = Agent(envs, args.emb_dim, device).to(device)
    elif args.architecture == "MobileVit2":
        from architectures.transformer import MobileViT2 as Agent

        agent = Agent(
            envs,
            args.emb_dim,
            args.num_heads,
            args.num_blocks,
            args.patch_size,
            args.buffer_window_size,
            device,
        ).to(device)
    elif args.architecture == "PPO":
        from architectures.ppo import PPODefault as Agent

        agent = Agent(envs, device).to(device)
    elif args.architecture == "PPO_OBJ":
        from architectures.ppo import PPObj as Agent

        agent = Agent(envs, device, args.encoder_dims, args.decoder_dims).to(device)
    else:
        raise NotImplementedError(f"Architecture {args.architecture} does not exist!")
    return agent


def init_wandb(args):
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
        global_dir = f"{args.wandb_dir}/" if args.wandb_dir is not None else ""
        writer_dir = f"{global_dir}runs/{run_name}"
        postfix = None
    return writer_dir, postfix
