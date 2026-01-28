import csv
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO

# ---- CONFIG ----
MODEL_PATH = Path("runs/ppo_test/agent9.zip")
OUT_CSV    = Path("data/ppo_expert_dataset_v3.csv")
N_EPISODES = 30
MAX_STEPS  = 5000
# --------------

def flatten_obs_64(obs: dict) -> np.ndarray:
    """
    Convert env observation dict into a stable 64-dim vector:
      obs["1"] -> 10
      obs["speedlimits"] -> 27x2 = 54
      total = 64
    """
    x = np.array(obs["1"], dtype=np.float32).reshape(-1)  # (10,)
    sl = np.array(obs.get("speedlimits", np.zeros((27, 2))), dtype=np.float32).reshape(-1)  # (54,)
    feats = np.concatenate([x, sl], axis=0)
    assert feats.shape[0] == 64, f"Expected 64 features, got {feats.shape[0]}"
    return feats


def make_feature_names() -> list:
    base = [
        "last_action",
        "speed",
        "distance_to_destination",
        "acceleration",
        "time_error",
        "jerk",
        "distance_covered",
        "current_gradient",
        "distance_next_gradient",
        "next_gradient",
    ]
    speedlimit_cols = []
    for i in range(27):
        speedlimit_cols.append(f"speedlimit_{i}_value")
        speedlimit_cols.append(f"speedlimit_{i}_length")

    names = base + speedlimit_cols
    assert len(names) == 64
    return names


def safe_get(info: dict, key: str, default=0.0):
    v = info.get(key, default)
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # IMPORTANT: use your local modified environment (no registration needed)
    import gym_train.envs.train_env as te
    print("Using file:", te.__file__)
    from gym_train.envs.train_env import TrainEnv

    env = TrainEnv()
    model = PPO.load(str(MODEL_PATH))

    # Probe feature length
    obs = env.reset()
    feat0 = flatten_obs_64(obs)
    assert feat0.shape[0] == 64

    feat_names = make_feature_names()

    # Columns for dashboard + WP1 analysis
    extra_cols = [
        "episode_id",
        "step",
        "action",
        "reward",
        "done",
        "termination_reason",

        # original reward components
        "r_safety",
        "r_energy",
        "r_comfort",
        "r_punctuality",
        "r_parking",

        # WP1 new reward terms
        "r_progress",
        "r_timewaste",
        "r_terminal",

        # raw metrics (from info)
        "speed_limit_raw",
        "speed_raw",
        "position_raw",
        "distance_to_destination_raw",
        "delta_time_left_raw",
        "max_jerk_raw",
        "cummulated_energy_raw",

        # gradients (from info)
        "current_gradient_raw",
        "next_gradient_raw",
        "distance_next_gradient_raw",
    ]

    header = feat_names + extra_cols
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        total_rows = 0

        for ep in range(N_EPISODES):
            obs = env.reset()
            for t in range(MAX_STEPS):
                feats = flatten_obs_64(obs)

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(int(action))

                row = list(feats) + [
                    ep,
                    t,
                    int(action),
                    float(reward),
                    int(done),
                    safe_get(info, "termination_reason", "Unknown"),

                    safe_get(info, "r_safety", 0.0),
                    safe_get(info, "r_energy", 0.0),
                    safe_get(info, "r_comfort", 0.0),
                    safe_get(info, "r_punctuality", 0.0),
                    safe_get(info, "r_parking", 0.0),

                    safe_get(info, "r_progress", 0.0),
                    safe_get(info, "r_timewaste", 0.0),
                    safe_get(info, "r_terminal", 0.0),

                    safe_get(info, "speed_limit", 0.0),
                    safe_get(info, "speed", 0.0),
                    safe_get(info, "position", 0.0),
                    safe_get(info, "distance_to_destination", 0.0),
                    safe_get(info, "delta_time_left", 0.0),
                    safe_get(info, "max_jerk", 0.0),
                    safe_get(info, "cummulated_energy", 0.0),

                    safe_get(info, "current_gradient", 0.0),
                    safe_get(info, "next_gradient", 0.0),
                    safe_get(info, "distance_next_gradient", 0.0),
                ]

                writer.writerow(row)
                total_rows += 1

                if done:
                    break

            print(f"Episode {ep+1}/{N_EPISODES} done, rows so far: {total_rows}")

    env.close()
    print(f"\nSaved dataset: {OUT_CSV} (rows={total_rows})")


if __name__ == "__main__":
    main()
