import pandas as pd
from pathlib import Path

DATASET = Path("data/wp1_rollout_dataset.csv")

def main():
    df = pd.read_csv(DATASET)

    # Episode-level summary
    ep = df.groupby("episode_id").agg(
        steps=("step_id", "max"),
        total_reward=("reward", "sum"),
        mean_reward=("reward", "mean"),
        mean_speed=("speed", "mean"),
        max_jerk=("max_jerk", "max"),
        termination_reason=("termination_reason", "last"),
    ).reset_index()

    success_rate = (ep["termination_reason"] == "AtDestination").mean() * 100
    term_dist = ep["termination_reason"].value_counts(normalize=True) * 100

    print("Episodes:", len(ep))
    print("Success rate (AtDestination):", round(success_rate, 2), "%")
    print("\nTermination distribution (%):")
    print(term_dist.round(2))
    print("\nTop 5 best episodes by reward:")
    print(ep.sort_values("total_reward", ascending=False).head(5))

if __name__ == "__main__":
    main()
