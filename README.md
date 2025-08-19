# Continual Hackatari

> Continual RL for **Atari** with environment **modifications**, **goal‑conditioned learning (HER)**, and **Continual Backpropagation (CBP)**. Train robust agents that adapt as the game dynamics or visuals change.

---

## 1) General

This repository provides:

- a **training script** (`continual_training_example.py`) that supports **Modification Factories** (non‑stationary curricula), optional **HER**, **CBP** via a PPO‑based agent (`PPO_CBP`), and a **Shrink & Perturb** mechanism at regime switches; and
- an **evaluation script** (`eval.py`) to test saved checkpoints under specific modifications.

Built on Gymnasium/Atari, HackAtari, OCAtari, Stable‑Baselines3 PPO using CleanRL as inspiration.

---

## 2) Installation

Clone this repository and run the following command:
```bash
git clone https://github.com/hannah-sophie/continual_hackatari
pip install -e .
```
---

## 3) Usage

### 3.1 `continual_training_example.py`

#### Quickstart

```bash
python continual_training_example.py \
  --config run_config.json \
  --env_id ALE/Freeway-v5 \
  --backend HackAtari \
  --architecture PPO \
  --track --capture_video
```

#### Modification Factories

Simulates **domain shifts/curricula** (color changes, speed flips, disabled objects, etc.) so agents learn invariances and adapt fast.

A factory is a function `f(step) → "modif string"` that tells the training loop **which environment modifications** to apply at each step (e.g., `"all_green_cars reverse_car_speed_top"`). When the string **changes**, a *switch* of the environment happens. Therefore, factories let you script or randomize shifts so the agent (a) sees diversity and (b) learns to recover at change-points. Epsilon variants revisit earlier regimes to potentially reduce temporal overfitting and forgetting.


**Built‑in factories (overview)**

The following factories pre-define how environment **modifications** change over time.  

- Use **Sequential** for curricula in a controlled sequential order; add `Eps…` to occasionally revisit and avoid overfitting to temporal order.
- Use **Random** for curricula displaying random mixtures of modifications.
- Use **AllCombinationsRandom** to cover **factorized variation** without enumerating every combo.
- Use **Combined/EpsCombined** to chain phases (e.g., warm‑up → randomization) and sometimes replay earlier regimes (with `epsilon`).


A) NoModificationFactory

*A static baseline with no environment changes; always returns an empty modification string (`""`).*

| Argument             | Type | default        | Description                                    |
|------------------|------|----------------|----------------------------------------------|
| num_total_steps  | int  | — (required)   | total steps to hold the empty modif `""`.    |

---

B) SequentialModificationFactory / EpsSequentialModificationFactory

*A fixed curriculum that steps through a list of modifications in order. The epsilon variant occasionally revisits previously seen modifications to reduce overfitting to a strict order.*

| Argument                | Type      | default        | Description                                                                                                                                         |
|----------------|-----------|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| num_total_steps | int       | — (required)   | total steps for this factory.                                                                                                                     |
| modifications  | list[str] | — (required)   | ordered list of modif strings.                                                                                                                    |
| switching_thresholds | list[int] | — (required)   | strictly increasing and length must equal `modifications`. Each threshold is a step (within this factory) where the **next** modification begins. |
| epsilon        | float     | —              | **EpsSequential only**: probability to **revisit** a previously seen modification instead of the next one.                                        |

---

C) RandomModificationFactory

*A stationary mixture over a fixed set of modification strings. The factory draws one modification according to a defined probability distribution (uniform if omitted), then holds that choice unchanged for a fixed number of repetition steps*  

| Argument             | Type        | default      | Description                                                   |
|------------|-------------|--------------|---------------------------------------------------------------|
| num_total_steps | int         | — (required) | total steps for this factory.                                 |
| modifications | list[str]   | — (required) | choices to sample from.                                       |
| probabilities | list[float] | uniform      | *(optional)* same length as `modifications`, **sums to 1**.   |
| num_repetitions | int         | 1            | hold the sampled modification for N steps before re-sampling. |

---

D) AllCombinationsRandomModificationFactory

*A factorized mixture: samples from the Cartesian product of factor lists (e.g., color × speed), then joins tokens into a single modification string.*

| Argument        | Type            | default     | explanation                                                                                                                                                        |
|-----------------|-----------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| num_total_steps | int             | — (required) | total steps for this factory.                                                                                                                                      |
| modifications   | list[list[str]] | — (required) | lists of **factors**; the factory samples from the **Cartesian product** and joins tokens with spaces (e.g., `color × speed` → `"all_green_cars reverse_car_speed_top"`). |
| probabilities   | list[float]     | uniform     | *(optional)* weights over the **joint combinations** (length equals product of factor sizes).                                                                     |
| num_repetitions | int             | 1           | hold period.                                                                                                                                                       |

---

E) CombinedModificationFactory / EpsCombinedModificationFactory

*A multi-phase schedule: runs several sub-factories back to back (e.g., warm-up → randomized). The epsilon variant occasionally replays a modification from earlier phases.*

| Argument                        | Type   | default        | Description                                                                                                                                               |
|-----------------------------|--------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| modification_factory_kwargs | object | — (required)   | mapping of **sub-factory name → kwargs**; each sub-factory must define `num_total_steps` and its own `modifications`/`thresholds` as needed. |
| epsilon                     | float  | —              | **EpsCombined only**: probability to **replay a previously completed** modification from any earlier sub-factory.                            |


**Custom Modification Factories**

To add your own curriculum/scheduling mechansim:

1) **Implement** a class in `modification_factories.py` that:
   - accepts `num_total_steps` (plus any kwargs) in `__init__`,
   - exposes `get_modification(step: int) -> str` returning a **single, space-separated** modification string (or `""` for no change),

2) **Register** it in the `modification_factory_mapping` (key = class name, value = class).

3) **Reference** it from `run_config.json`:
   ```json
   {
     "modification_factory": "MyFactory",
     "modification_factory_kwargs": { "num_total_steps": 200000, "...": "..." }
   }
   
**At switches** you can:

- **checkpoint** (`--save_agent_with_switch`), and/or
- apply **Shrink & Perturb** (see below) to encourage quick re‑adaptation.

To train on a curriculum with a specific modification factory **configure** (JSON via `--config`). 

A minimal schema you should include:

- `modification_factory` *(string)*: factory class name. Must match an entry in the internal `modification_factory_mapping`.
- `modification_factory_kwargs` *(object)*: kwargs for that factory.
  - For **Combined/EpsCombined** factories, include an inner `modification_factory_kwargs` mapping **sub‑factory name → kwargs**.

> Provide each factory’s parameters exactly under `"modification_factory_kwargs"` in `run_config.json` as shown by their argument tables.

Example from `run_config.json` for `EpsCombinedModificationFactory`:
```json
{
  "modification_factory": "EpsCombinedModificationFactory",
  "modification_factory_kwargs": {
    "modification_factory_kwargs": {
      "SequentialModificationFactory": {
        "num_total_steps": 1000,
        "modifications": ["disable_cars"],
        "switching_thresholds": []
      },
      "AllCombinationsRandomModificationFactory": {
        "num_total_steps": 1000,
        "modifications": [
          ["all_green_cars","all_pink_cars","all_white_cars","all_blue_cars"],
          ["","reverse_car_speed_bottom","reverse_car_speed_top"]
        ],
        "num_repetitions": 20
      }
    },
    "epsilon": 0.005
  }
}
```

 **Note:** Any extra keys you add to `run_config.json` will **override CLI flags** with the same name (e.g., `env_id`, `backend`, `architecture`, `total_timesteps`).

#### Hindsight Experience Replay (HER)

Re‑label transitions with **achieved goals** (i.e., `final` strategy of HER) → stronger learning signal for **sparse‑reward** Atari tasks.
Enable HER only with **HackAtari** or **OCAtari** (the wrapper augments observations and may query object signals).

**Enable**: `--her`. Optional: `--game_specific_goals` for per‑game goal encodings/rewards when available.

When `--game_specific_goals`** is OFF**: goal sampling falls back to a **generic scheme**. For supported games (e.g., *Freeway*), that means using a **specified target position** of the game for goal sampling.

#### Continual Backpropagation (CBP)

Select `--architecture PPO_CBP` to use the PPO‑based CBP agent.
Mitigate **catastrophic forgetting** under distribution shifts by **replacing low‑utility units** over time while protecting new units until mature.

**Key parameters** (names as in the CLI/args):

- `--replacement_rate` – how often to replace units over time.
- `--init` – initializer for freshly created units (e.g., `kaiming`).
- `--maturity_threshold` – minimum age before a unit can be replaced.
- `--decay_rate` – decay of the utility estimator.

Use alongside standard PPO args (learning rate, clip, GAE, entropy/value coeffs, etc.).

#### Shrink & Perturb (optional, at switches)

At each factory **switch**, multiplicatively **shrink** policy weights and add small **Gaussian noise** — a gentle reset that keeps structure but improves plasticity.

**Enable**: `--shrink_and_perturb`

**Key Parameters**


`shrink_and_perturb`: bool = False  – if toggled, the agent's weights will be shrunk and perturbed at each change of modification

`shrink_factor`: float = 0.7   – the factor by which the agent's weights will be shrunk

`noise_scale`: float = 0.01  – the scale of the noise to be added to the agent's weights

---

### 3.2 `eval.py`

#### Quickstart

```bash
python eval.py \
  --agent_path runs/.../model.cleanrl_model \
  --env_id ALE/Freeway-v5 \
  --backend HackAtari \
  --eval_episodes 50 \
  --capture_video
```

**Key parameters**

- `--agent_path` – checkpoint to evaluate (`.cleanrl_model`).
- `--env_id`, `--backend` – evaluation environment/backend.
- `--eval_episodes` – how many episodes to average (e.g., 21 or 50).
- `--her`, `--game_specific_goals` – must match how the agent was trained.
- `--modifs` – (HackAtari) evaluate under a specific modification string.
- Logging: `--track`, `--wandb_project_name`, `--wandb_entity`, `--wandb_run_id`, `--capture_video`.

**Examples**

- Evaluate under a **specific modification** (HackAtari):
  ```bash
  python eval.py --agent_path runs/.../model.cleanrl_model \
    --env_id ALE/Freeway-v5 --backend HackAtari \
    --modifs "all_green_cars reverse_car_speed_top"
  ```

### Tracking

**Prerequisites**
- `pip install wandb`
- Authenticate once: `wandb login` (or set `WANDB_API_KEY` in your environment)

**Enable**`: --track` to either script to turn on W&B logging.

**Common flags**
- `--wandb_project_name <project>` – project to log into
- `--wandb_entity <team_or_user>` – your W&B workspace
- `--wandb_dir <path>` – local cache/artifact directory
- `--capture_video` – upload rollout videos
- **Training:** `--save_agent_wandb` – upload model checkpoints as artifacts
- **Eval:** `--wandb_run_id <id>` – (optional) attach eval to an existing run

### What gets logged
- **Scalars:** episodic return/length, losses, value/entropy, KL, LR
- **Config:** full CLI args, environment/backend, factory name & kwargs (if loaded from `run_config.json`)
- **Media:** videos when `--capture_video` is set
- **Artifacts:** checkpoints when `--save_agent_wandb` is set
---

