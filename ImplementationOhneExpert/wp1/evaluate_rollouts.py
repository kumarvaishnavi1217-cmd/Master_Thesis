import pandas as pd
from pathlib import Path

DATASET = Path("data/ppo_expert_dataset_v2.csv")

def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)

    # --- detect columns safely ---
    has_speed = "speed" in df.columns
    has_jerk = "max_jerk" in df.columns
    has_term = "termination_reason" in df.columns

    # Steps per episode = number of rows in that episode
    df["_row_in_ep"] = df.groupby("episode_id").cumcount()

    agg_dict = {
        "_row_in_ep": "max",
        "reward": ["sum", "mean"],
    }

    if has_speed:
        agg_dict["speed"] = "mean"
    if has_jerk:
        agg_dict["max_jerk"] = "max"
    if has_term:
        agg_dict["termination_reason"] = "last"

    ep = df.groupby("episode_id").agg(agg_dict)
    ep.columns = ["_".join([c for c in col if c]) for col in ep.columns.to_flat_index()]
    ep = ep.reset_index()

    # Rename nicer
    ep = ep.rename(columns={
        "_row_in_ep_max": "steps",
        "reward_sum": "total_reward",
        "reward_mean": "mean_reward",
        "speed_mean": "mean_speed",
        "max_jerk_max": "max_jerk",
        "termination_reason_last": "termination_reason"
    })

    print("Episodes:", len(ep))
    print(" Avg steps:", round(ep["steps"].mean(), 2))

    if "termination_reason" in ep.columns:
        success_rate = (ep["termination_reason"] == "AtDestination").mean() * 100
        term_dist = ep["termination_reason"].value_counts(normalize=True) * 100

        print(" Success rate (AtDestination):", round(success_rate, 2), "%")
        print("\n Termination distribution (%):")
        print(term_dist.round(2).to_string())

    print("\n Top 5 episodes by total reward:")
    print(ep.sort_values("total_reward", ascending=False).head(5).to_string(index=False))

    print("\n✅ Bottom 5 episodes by total reward:")
    print(ep.sort_values("total_reward", ascending=True).head(5).to_string(index=False))

if __name__ == "__main__":
    main()
