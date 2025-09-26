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
# %%


def plot_bars(groups, metrics, metrics_names, figsize, xlim):
    plt.figure(figsize=figsize)
    sns.barplot(
        data=data_melted[(data_melted["Group"].isin(groups)) & (data_melted["Metric"].isin(metrics))],
        hue_order=groups,
        x="Value",
        y="Metric",
        errorbar="sd",
        orient="h",
        hue="Group"
    )

    plt.ylabel("Modifications (Eval)", fontsize=12, fontweight="bold")
    plt.xlabel("Evaluation Reward", fontsize=12, fontweight="bold")
    plt.yticks(ticks=np.arange(len(metrics)), labels=metrics_names)
    plt.xlim(xlim)
    plt.legend(title="Training Condition")
    plt.tight_layout()
    plt.show()
# %%
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
plot_bars(["Freeway-v5 baseline"], eval_cols, eval_names, (6, 3), (0,40))
# %%
eval_cols = [
    "FinalReward",
]
eval_names = [
    "Baseline",
]
groups = ["Freeway-v5 baseline", "Freeway-v5 Colors"]
plot_bars(groups, eval_cols, eval_names, (6, 3), (0,40))
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
plot_bars(groups, eval_cols, eval_names, (6, 3), (0,40))

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
plot_bars(groups, eval_cols, eval_names, (6, 3), (0,40))
# %%
