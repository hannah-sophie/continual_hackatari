# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
# %%

data = pd.read_csv("data/eval_data.csv")
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
data_melted = data.melt(
    id_vars=["Group"], value_vars=numeric_columns, var_name="Metric", value_name="Value"
)
data_melted["Group"].replace(
    {
        "Pong-v5 parallel_enemy  + random_perturbation_enemy": "Pong-v5 parallel_enemy + random_perturbation_enemy"
    },
    inplace=True,
)
# %%


def plot_bars(groups, metrics, metrics_names, figsize, xlim, figname=None):
    plt.figure(figsize=figsize)
    sns.barplot(
        data=data_melted[
            (data_melted["Group"].isin(groups)) & (data_melted["Metric"].isin(metrics))
        ],
        hue_order=groups,
        x="Value",
        y="Metric",
        errorbar="sd",
        orient="h",
        hue="Group",
        palette=[colors[g] for g in groups],
    )

    plt.ylabel("Modifications (Eval)", fontsize=12, fontweight="bold")
    plt.xlabel("Evaluation Reward", fontsize=12, fontweight="bold")
    if len(metrics) == 1:
        plt.yticks(
            ticks=np.arange(len(metrics)),
            labels=metrics_names,
            fontsize=10,
            va="center",
            rotation=90,
        )
    else:
        plt.yticks(ticks=np.arange(len(metrics)), labels=metrics_names, fontsize=10)
    plt.xlim(xlim)
    plt.legend(title="Training Condition", fontsize=7)
    plt.tight_layout()
    if figname:
        plt.savefig(
            f"figures/{figname}.pdf",
        )
    plt.show()


# %%
# Mappings
colors = {
    "Freeway-v5 baseline": "#7f7f7f",
    "Freeway-v5 Colors": "#ff7f0e",
    "Freeway-v5 disable_cars": "#2ca02c",
    "Freeway-v5 vary_car_speeds": "#9467bd",
    "Freeway-v5 Colors and vary_car_speeds": "#1f77b4",
    "Freeway-v5 HER": "#bcbd22",
    "Pong-v5 baseline": "#7f7f7f",
    "Pong-v5 lazy_enemy": "#ff7f0e",
    "Pong-v5 parallel_enemy": "#2ca02c",
    "Pong-v5 random_pertubation_enemy": "#9467bd",
    "Pong-v5 parallel_enemy + lazy_enemy": "#1f77b4",
    "Pong-v5 lazy_enemy + random_perturbation_enemy": "#bcbd22",
    "Pong-v5 parallel_enemy + random_perturbation_enemy": "#8c564b",
    "Breakout-v5 baseline": "#7f7f7f",
    "Breakout-v5 Colors": "#ff7f0e",
    "Breakout-v5 Color blocks": "#2ca02c",
}
# %%
# Freeway Baseline
eval_cols = [
    "FinalReward",
    "all_black_cars/FinalReward_eval",
    "stop_all_cars/FinalReward_eval",
    "stop_all_cars_edge/FinalReward_eval",
    "disable_cars/FinalReward_eval",
]
eval_names = [
    "Baseline",
    "All black cars",
    "Stop all cars",
    "Stop all cars on edge",
    "Disable cars",
]
plot_bars(
    ["Freeway-v5 baseline"], eval_cols, eval_names, (6, 3), (0, 41), "freeway_baseline"
)
# %%
# Freeway Baseline vs Colors
eval_cols = [
    "FinalReward",
]
eval_names = [
    "Baseline",
]
groups = ["Freeway-v5 baseline", "Freeway-v5 Colors"]
plot_bars(groups, eval_cols, eval_names, (6, 3), (0, 41), "freeway_colors_baseline")
eval_cols = [
    "all_blue_cars/FinalReward_eval",
    "all_green_cars/FinalReward_eval",
    "all_pink_cars/FinalReward_eval",
    "all_white_cars/FinalReward_eval",
]
eval_names = [
    "All blue cars",
    "All green cars",
    "All pink cars",
    "All white cars",
]
plot_bars(groups, eval_cols, eval_names, (6, 3), (0, 41), "freeway_colors_train_colors")

eval_cols = [
    "all_black_cars/FinalReward_eval",
    "all_red_cars/FinalReward_eval",
    "strobo_mode/FinalReward_eval",
]
eval_names = [
    "All black cars",
    "All red cars",
    "Strobo mode",
]
plot_bars(groups, eval_cols, eval_names, (6, 3), (0, 41), "freeway_colors_eval_colors")
eval_cols = [
    "stop_all_cars/FinalReward_eval",
    "stop_all_cars_edge/FinalReward_eval",
    "disable_cars/FinalReward_eval",
]
eval_names = [
    "Stop all cars",
    "Stop all cars on edge",
    "Disable cars",
]
plot_bars(groups, eval_cols, eval_names, (6, 3), (0, 41), "freeway_colors_other_mods")
# %%
# Freeway Baseline vs Disable cars
eval_cols = [
    "FinalReward",
]
eval_names = [
    "Baseline",
]
groups = ["Freeway-v5 baseline", "Freeway-v5 disable_cars"]
plot_bars(
    groups, eval_cols, eval_names, (6, 3), (0, 41), "freeway_disable_cars_baseline"
)
eval_cols = [
    "disable_cars/FinalReward_eval",
]
eval_names = [
    "Disable cars",
]
plot_bars(groups, eval_cols, eval_names, (6, 3), (0, 41), "freeway_disable_cars_eval")
eval_cols = [
    "all_black_cars/FinalReward_eval",
    "stop_all_cars/FinalReward_eval",
    "stop_all_cars_edge/FinalReward_eval",
]
eval_names = [
    "All black cars",
    "Stop all cars",
    "Stop all cars on edge",
]
plot_bars(
    groups, eval_cols, eval_names, (6, 3), (0, 41), "freeway_disable_cars_other_mods"
)
# %%
# Freeway Baseline vs Colors vs Vary Car Speeds
eval_cols = [
    "FinalReward",
]
eval_names = [
    "Baseline",
]
groups = [
    "Freeway-v5 baseline",
    "Freeway-v5 Colors",
    "Freeway-v5 vary_car_speeds",
    "Freeway-v5 Colors and vary_car_speeds",
]
plot_bars(groups, eval_cols, eval_names, (6, 3), (0, 41))
eval_cols = [
    "vary_car_speeds/FinalReward_eval",
]
eval_names = [
    "Vary car speeds",
]
plot_bars(groups, eval_cols, eval_names, (6, 3), (0, 41))
eval_cols = [
    "all_black_cars/FinalReward_eval",
    "all_red_cars/FinalReward_eval",
]
eval_names = [
    "All black cars",
    "All red cars",
]
plot_bars(groups, eval_cols, eval_names, (6, 3), (0, 41))
eval_cols = [
    "stop_all_cars/FinalReward_eval",
    "stop_all_cars_edge/FinalReward_eval",
    "disable_cars/FinalReward_eval",
]
eval_names = [
    "Stop all cars",
    "Stop all cars on edge",
    "Disable cars",
]
plot_bars(groups, eval_cols, eval_names, (6, 3), (0, 41))
# %%
# Pong Baseline
eval_cols = [
    "FinalReward",
    "lazy_enemy/FinalReward_eval",
    "parallel_enemy/FinalReward_eval",
    "random_perturbation_enemy/FinalReward_eval",
]
eval_names = [
    "Baseline",
    "Lazy enemy",
    "Parallel enemy",
    "Lazy enemy",
]
plot_bars(
    ["Pong-v5 baseline"], eval_cols, eval_names, (6, 3), (-21, 21), "pong_baseline"
)


# %%
# Pong enemy behavior
eval_cols = [
    "FinalReward",
]
eval_names = [
    "Baseline",
]
groups = [
    "Pong-v5 baseline",
    "Pong-v5 lazy_enemy",
    "Pong-v5 parallel_enemy",
    "Pong-v5 random_pertubation_enemy",
]
plot_bars(groups, eval_cols, eval_names, (6, 3), (-21, 21), "pong_enemy_baseline")
eval_cols = [
    "lazy_enemy/FinalReward_eval",
]
eval_names = [
    "Lazy enemy",
]
plot_bars(groups, eval_cols, eval_names, (6, 3), (-21, 21), "pong_enemy_lazy_enemy")
eval_cols = [
    "parallel_enemy/FinalReward_eval",
]
eval_names = [
    "Parallel enemy",
]
plot_bars(groups, eval_cols, eval_names, (6, 3), (-21, 21), "pong_enemy_parallel_enemy")
eval_cols = [
    "random_perturbation_enemy/FinalReward_eval",
]
eval_names = [
    "Random pertubation enemy",
]
plot_bars(
    groups,
    eval_cols,
    eval_names,
    (6, 3),
    (-21, 21),
    "pong_enemy_random_pertubation_enemy",
)
# %%
# Pong Combinations of enemy behavior
groups = [
    "Pong-v5 baseline",
    "Pong-v5 parallel_enemy + lazy_enemy",
    "Pong-v5 lazy_enemy + random_perturbation_enemy",
    "Pong-v5 parallel_enemy + random_perturbation_enemy",
]
eval_cols = [
    "FinalReward",
]
eval_names = [
    "Baseline",
]
plot_bars(
    groups, eval_cols, eval_names, (6, 3), (-21, 21), "pong_enemy_combinations_baseline"
)
eval_cols = [
    "lazy_enemy/FinalReward_eval",
]
eval_names = [
    "Lazy enemy",
]
plot_bars(
    groups,
    eval_cols,
    eval_names,
    (6, 3),
    (-21, 21),
    "pong_enemy_combinations_lazy_enemy",
)
eval_cols = [
    "parallel_enemy/FinalReward_eval",
]
eval_names = [
    "Parallel enemy",
]
plot_bars(
    groups,
    eval_cols,
    eval_names,
    (6, 3),
    (-21, 21),
    "pong_enemy_combinations_parallel_enemy",
)
eval_cols = [
    "random_perturbation_enemy/FinalReward_eval",
]
eval_names = [
    "Random pertubation enemy",
]
plot_bars(
    groups,
    eval_cols,
    eval_names,
    (6, 3),
    (-21, 21),
    "pong_enemy_combinations_random_pertubation_enemy",
)

# %%
# Breakout Baseline
eval_cols = [
    "FinalReward",
    "color_all_blocks_red/FinalReward_eval",
    "strobo_mode_blocks_no_black/FinalReward_eval",
    "color_player_and_ball_red/FinalReward_eval",
    "strobo_mode_player_and_ball_no_black/FinalReward_eval",
]
eval_names = [
    "Baseline",
    "All blocks red",
    "Strobo mode blocks no black",
    "Player and ball red",
    "Strobo mode player and ball no black",
]
plot_bars(
    ["Breakout-v5 baseline"],
    eval_cols,
    eval_names,
    (6, 3),
    (0, 400),
    "breakout_baseline",
)

# %%
# Breakout
groups = ["Breakout-v5 baseline", "Breakout-v5 Colors", "Breakout-v5 Color blocks"]
eval_cols = [
    "/FinalReward_eval",
]
eval_names = [
    "Baseline",
]
plot_bars(groups, eval_cols, eval_names, (6, 3), (0, 400), "breakout_baseline")
eval_cols = [
    "color_player_and_ball_blue/FinalReward_eval",
    "color_player_and_ball_green/FinalReward_eval",
    "color_player_and_ball_white/FinalReward_eval",
]
eval_names = [
    "Player and ball blue",
    "Player and ball green",
    "Player and ball white",
]
plot_bars(
    groups, eval_cols, eval_names, (6, 3), (0, 400), "breakout_colors_player_ball_train"
)
eval_cols = [
    "color_player_and_ball_red/FinalReward_eval",
    "strobo_mode_player_and_ball/FinalReward_eval",
    "strobo_mode_player_and_ball_no_black/FinalReward_eval",
]
eval_names = [
    "Player and ball red",
    "Strobo mode player and ball",
    "Strobo mode player and ball no black",
]
plot_bars(
    groups, eval_cols, eval_names, (6, 3), (0, 400), "breakout_colors_player_ball_eval"
)
eval_cols = [
    "color_all_blocks_blue/FinalReward_eval",
    "color_all_blocks_green/FinalReward_eval",
    "color_all_blocks_white/FinalReward_eval",
]
eval_names = [
    "All blocks blue",
    "All blocks green",
    "All blocks white",
]
plot_bars(
    groups, eval_cols, eval_names, (6, 3), (0, 400), "breakout_colors_blocks_train"
)
eval_cols = [
    "color_all_blocks_red/FinalReward_eval",
    "color_all_blocks_yellow/FinalReward_eval",
    "strobo_mode_blocks/FinalReward_eval",
    "strobo_mode_blocks_no_black/FinalReward_eval",
]
eval_names = [
    "All blocks red",
    "All blocks yellow",
    "Strobo mode blocks",
    "Strobo mode blocks no black",
]
plot_bars(
    groups, eval_cols, eval_names, (6, 3), (0, 400), "breakout_colors_blocks_eval"
)
# %%
# Freeway HER
groups = ["Freeway-v5 baseline", "Freeway-v5 HER"]
eval_cols = [
    "FinalReward",
    "all_black_cars/FinalReward_eval",
    "stop_all_cars/FinalReward_eval",
    "stop_all_cars_edge/FinalReward_eval",
    "disable_cars/FinalReward_eval",
]
eval_names = [
    "Baseline",
    "All black cars",
    "Stop all cars",
    "Stop all cars on edge",
    "Disable cars",
]
plot_bars(groups, eval_cols, eval_names, (6, 3), (0, 41), "freeway_her_comparison")
# %%
