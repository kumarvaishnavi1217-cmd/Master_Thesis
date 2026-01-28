import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path


st.set_page_config(page_title="Train RL/ML Environment Dashboard", layout="wide")


def infer_episode_id(df: pd.DataFrame) -> pd.DataFrame:
    """Infer episode_id if not provided, using 'done' if available."""
    df = df.copy()

    if "episode_id" in df.columns:
        return df

    if "done" in df.columns:
        done = df["done"].astype(bool).fillna(False)
        # episode increments after a done=True row
        df["episode_id"] = done.shift(fill_value=False).cumsum()
        return df

    # fallback: treat everything as one episode
    df["episode_id"] = 0
    return df


def infer_step(df: pd.DataFrame) -> pd.DataFrame:
    """Infer step within each episode if not provided."""
    df = df.copy()
    if "step" in df.columns:
        return df
    df["step"] = df.groupby("episode_id").cumcount()
    return df


def safe_numeric(df: pd.DataFrame, cols):
    """Coerce selected columns to numeric if present."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize common boolean forms
    if "done" in df.columns:
        if df["done"].dtype != bool:
            df["done"] = df["done"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return df


st.title("🚆 Train Environment — Reward & Behavior Dashboard")

with st.sidebar:
    st.header("Data input")
    default_path = "data/wp1_expert_free_dataset.csv"  # change if your file differs
    csv_path = st.text_input("CSV path (local)", value=default_path)

    uploaded = st.file_uploader("…or upload a CSV", type=["csv"])

    st.divider()
    st.header("Display")
    max_rows = st.slider("Max rows to load (for speed)", 10_000, 2_000_000, 200_000, step=10_000)
    st.caption("If your CSV is huge, reduce max rows for a faster dashboard.")


# Load data
df = None
load_error = None

try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        if not Path(csv_path).exists():
            st.warning(f"CSV not found at: {csv_path}")
        else:
            df = load_csv(csv_path)
except Exception as e:
    load_error = str(e)

if load_error:
    st.error(load_error)
    st.stop()

if df is None:
    st.info("Provide a CSV path or upload a CSV to begin.")
    st.stop()

# Trim rows for performance
if len(df) > max_rows:
    df = df.iloc[:max_rows].copy()

# Basic cleanup / inference
df = infer_episode_id(df)
df = infer_step(df)

numeric_cols = [
    "reward",
    "r_safety", "r_energy", "r_comfort", "r_punctuality", "r_parking", "r_progress",
    "speed", "speed_limit", "position", "delta_time_left", "max_jerk", "cummulated_energy",
    "action", "delta_pos_action",
]
df = safe_numeric(df, numeric_cols)

# A few convenience columns
if "termination_reason" in df.columns:
    df["termination_reason"] = df["termination_reason"].astype(str)
else:
    df["termination_reason"] = "Unknown"

# Episode-level aggregates
episode_last = df.sort_values(["episode_id", "step"]).groupby("episode_id").tail(1).copy()
episode_stats = df.groupby("episode_id").agg(
    steps=("step", "max"),
    total_reward=("reward", "sum") if "reward" in df.columns else ("step", "max"),
    mean_speed=("speed", "mean") if "speed" in df.columns else ("step", "max"),
    mean_energy=("r_energy", "mean") if "r_energy" in df.columns else ("step", "max"),
    max_jerk=("max_jerk", "max") if "max_jerk" in df.columns else ("step", "max"),
).reset_index()

episode_last = episode_last[["episode_id", "termination_reason"]].merge(episode_stats, on="episode_id", how="left")

# KPIs
colA, colB, colC, colD = st.columns(4)
colA.metric("Rows", f"{len(df):,}")
colB.metric("Episodes", f"{df['episode_id'].nunique():,}")

# success rate: define success as AtDestination if present
if "termination_reason" in episode_last.columns and (episode_last["termination_reason"] != "Unknown").any():
    success = (episode_last["termination_reason"] == "AtDestination").mean()
    colC.metric("Success rate (AtDestination)", f"{success*100:.1f}%")
else:
    colC.metric("Success rate", "n/a")

if "reward" in df.columns:
    colD.metric("Mean step reward", f"{df['reward'].mean():.3f}")
else:
    colD.metric("Mean step reward", "n/a")

st.divider()

# Episode selector
with st.sidebar:
    st.header("Episode selection")
    ep_ids = df["episode_id"].unique()
    ep_ids_sorted = np.sort(ep_ids)
    selected_ep = st.selectbox("Choose episode_id", ep_ids_sorted.tolist(), index=0)

ep = df[df["episode_id"] == selected_ep].sort_values("step").copy()

# --- Layout: Left = episode plots, Right = distributions ---
left, right = st.columns([2, 1])

with left:
    st.subheader(f"Episode {selected_ep} — behavior over time")

    # Speed vs speed limit
    if "speed" in ep.columns:
        fig = px.line(ep, x="step", y="speed", title="Speed over steps")
        if "speed_limit" in ep.columns:
            fig2 = px.line(ep, x="step", y="speed_limit")
            # overlay second trace
            for tr in fig2.data:
                fig.add_trace(tr)
            fig.update_layout(legend_title_text="series")
            fig.data[0].name = "speed"
            if len(fig.data) > 1:
                fig.data[1].name = "speed_limit"
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'speed' column found; skipping speed plot.")

    # Action plot
    if "action" in ep.columns:
        fig = px.line(ep, x="step", y="action", title="Action (Discrete) over steps")
        st.plotly_chart(fig, use_container_width=True)

    # Reward components (time series)
    reward_cols = [c for c in ["reward", "r_safety", "r_energy", "r_comfort", "r_punctuality", "r_parking", "r_progress"] if c in ep.columns]
    if reward_cols:
        melted = ep.melt(id_vars=["step"], value_vars=reward_cols, var_name="component", value_name="value")
        fig = px.line(melted, x="step", y="value", color="component", title="Reward components over steps")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No reward component columns found; skipping reward breakdown plot.")

    # Jerk & time error
    c1, c2 = st.columns(2)
    with c1:
        if "max_jerk" in ep.columns:
            fig = px.line(ep, x="step", y="max_jerk", title="Max jerk per step")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'max_jerk' column found.")
    with c2:
        if "delta_time_left" in ep.columns:
            fig = px.line(ep, x="step", y="delta_time_left", title="Punctuality error (delta_time_left)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'delta_time_left' column found.")

with right:
    st.subheader("Across episodes — summary")

    # Termination reasons
    term_counts = episode_last["termination_reason"].value_counts().reset_index()
    term_counts.columns = ["termination_reason", "count"]
    fig = px.pie(term_counts, names="termination_reason", values="count", title="Termination reasons")
    st.plotly_chart(fig, use_container_width=True)

    # Total reward distribution
    if "total_reward" in episode_last.columns:
        fig = px.histogram(episode_last, x="total_reward", nbins=40, title="Episode total reward distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Steps per episode
    if "steps" in episode_last.columns:
        fig = px.histogram(episode_last, x="steps", nbins=40, title="Episode length (steps) distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Mean energy per episode (if present)
    if "mean_energy" in episode_last.columns and "r_energy" in df.columns:
        fig = px.histogram(episode_last, x="mean_energy", nbins=40, title="Mean r_energy per episode")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# Data preview + export
st.subheader("Data preview")
st.dataframe(df.head(200), use_container_width=True)

st.subheader("Episode table")
st.dataframe(episode_last.sort_values("episode_id").head(200), use_container_width=True)

# Export filtered episode
st.subheader("Export selected episode")
csv_bytes = ep.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"Download episode {selected_ep} CSV",
    data=csv_bytes,
    file_name=f"episode_{selected_ep}.csv",
    mime="text/csv",
)
