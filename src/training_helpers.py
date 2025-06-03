from gymnasium.wrappers.transform_observation import GrayscaleObservation
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
)
import ale_py
import gymnasium as gym


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
