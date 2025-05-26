# %%
import os

import gymnasium as gym
import numpy as np
import torch
from hackatari.core import HackAtari

has_agent = False
# %%
env_id = "Freeway"
obs_mode = "dqn"
pth = "./wandb/run-20250523_150710-et572dag/files/continual_training_example.cleanrl_model"
architecture = "PPO"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
from ocatari.core import OCAtari

# env = OCAtari(
#     env_id,
#     hud=False,
#     render_mode="rgb_array",
#     render_oc_overlay=False,
#     obs_mode=obs_mode,
#     # logger=logger, feature_func=feature_func,
#     # buffer_window_size=window_size
# )
modifs_list = ["stop_all_cars_edge", "re"]
env = HackAtari(
    env_id,
    hud=False,
    render_mode="rgb_array",
    render_oc_overlay=False,
    obs_mode=obs_mode,
    modifs=modifs_list,
)
os.makedirs("./eval_videos/", exist_ok=True)

env = gym.wrappers.RecordVideo(
    env=env,
    video_folder="./eval_videos/",
    name_prefix=f"eval-{env_id}-{'_'.join(modifs_list)}",
)

obs, info = env.reset()
# %%
if architecture == "OCT":
    from architectures.transformer import OCTransformer as Agent

    agent = Agent(env, emb_dim, num_heads, num_blocks, device).to(device)
elif architecture == "VIT":
    from architectures.transformer import VIT as Agent

    agent = Agent(
        env, emb_dim, num_heads, num_blocks, patch_size, buffer_window_size, device
    ).to(device)
elif architecture == "VIT2":
    from architectures.transformer import SimpleViT2 as Agent

    agent = Agent(
        env, emb_dim, num_heads, num_blocks, patch_size, buffer_window_size, device
    ).to(device)
elif architecture == "MobileVit":
    from architectures.transformer import MobileVIT as Agent

    agent = Agent(
        env, emb_dim, num_heads, num_blocks, patch_size, buffer_window_size, device
    ).to(device)
elif architecture == "MobileVit2":
    from architectures.transformer import MobileViT2 as Agent

    agent = Agent(
        env, emb_dim, num_heads, num_blocks, patch_size, buffer_window_size, device
    ).to(device)
elif architecture == "PPO":
    from architectures.ppo import PPODefault as Agent

    agent = Agent(env, device).to(device)
else:
    from architectures.ppo import PPO_Obj as Agent

    agent = Agent(env, device).to(device)
# %%
ckpt = torch.load(pth, map_location=torch.device("cpu"))
agent.load_state_dict(ckpt["model_weights"])
has_agent = True
# %%
episode_rewards = []
for i in range(21):
    done = False
    crew = 0
    j = 0
    while not done:
        if has_agent:
            obs = torch.from_numpy(obs).to(device)
            obs = obs.unsqueeze(0)
            action, _, _, _ = agent.get_action_and_value(obs)
        else:
            action = env.action_space.sample()  # random moves

        obs, reward, terminated, truncated, info = env.step(action)
        crew += reward
        j += 1

        if terminated or truncated:
            print(
                f"{env_id}: Reward: {crew}, Framses:",
                info["episode_frame_number"],
                f"w/o Frameskip: {j}, Episode: {i}",
            )
            # run.log({f"{opts.game}_reward": crew, f"{opts.game}_episode_length": info["episode_frame_number"]})
            observation, info = env.reset()
            done = True
            episode_rewards.append(crew)

print()
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
modifs_list = ["stop_all_cars_edge", "re"]
modif_string = "_".join(modifs_list)
print(f"{mean_reward} +- {std_reward}")
env.close()

# %%
