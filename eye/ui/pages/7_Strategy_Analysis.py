"""Strategy Analysis — compare trained RL models saved in the database."""
from collections import defaultdict

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from eye.services.scenario import ScenarioService

st.set_page_config(page_title="Strategy Analysis — ACES", layout="wide")
st.title("Strategy Analysis")

if "scenario_service" not in st.session_state:
    st.session_state.scenario_service = ScenarioService()
    st.session_state.scenario_service.ensure_seed_data()

svc: ScenarioService = st.session_state.scenario_service
sid = st.session_state.get("active_scenario_id")

if not sid:
    st.warning("Select a scenario on the home page.")
    st.stop()

scenarios = svc.list_scenarios()
sc_map = {s["id"]: s["name"] for s in scenarios}
st.subheader(f"Scenario: {sc_map.get(sid, sid)}")

models = svc.list_saved_models(sid)

if not models:
    st.info("No trained models found for this scenario. Train some in **Mission Planner** first.")
    st.stop()

# ------------------------------------------------------------------ Model table

st.subheader("Trained Models")
st.dataframe(
    [
        {
            "Name": m["name"],
            "Algorithm": m["algorithm"],
            "Training Steps": f"{m['training_steps']:,}",
            "Mean Reward": f"{m['mean_reward']:.2f}",
            "Created": m["created_at"][:10],
        }
        for m in models
    ],
    use_container_width=True,
)

# ------------------------------------------------------------------ Performance bar chart

st.subheader("Mean Reward by Model")
fig_bar = px.bar(
    x=[m["name"] for m in models],
    y=[m["mean_reward"] for m in models],
    color=[m["algorithm"] for m in models],
    labels={"x": "Model", "y": "Mean Reward", "color": "Algorithm"},
    title="Model Performance Comparison",
)
fig_bar.update_layout(height=400, xaxis_tickangle=-30)
st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------------------------------ Best model

best = max(models, key=lambda m: m["mean_reward"])
st.success(
    f"Best model: **{best['name']}** ({best['algorithm']}) "
    f"— Mean Reward {best['mean_reward']:.2f} over {best['training_steps']:,} steps"
)

# ------------------------------------------------------------------ Algorithm group stats

st.subheader("Stats by Algorithm")
algo_groups: dict[str, list[float]] = defaultdict(list)
for m in models:
    algo_groups[m["algorithm"]].append(m["mean_reward"])

algo_stats = []
for algo, rewards in sorted(algo_groups.items()):
    algo_stats.append(
        {
            "Algorithm": algo,
            "Models Trained": len(rewards),
            "Best Reward": f"{max(rewards):.2f}",
            "Avg Reward": f"{sum(rewards) / len(rewards):.2f}",
            "Worst Reward": f"{min(rewards):.2f}",
        }
    )
st.dataframe(algo_stats, use_container_width=True)

# ------------------------------------------------------------------ Reward vs training steps scatter

if len(models) > 1:
    st.subheader("Reward vs Training Steps")
    fig_scatter = px.scatter(
        x=[m["training_steps"] for m in models],
        y=[m["mean_reward"] for m in models],
        color=[m["algorithm"] for m in models],
        text=[m["name"] for m in models],
        labels={"x": "Training Steps", "y": "Mean Reward", "color": "Algorithm"},
        title="Sample Efficiency Comparison",
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------------------------------------------ Algorithm radar (if 3+ algorithms)

unique_algos = list(algo_groups.keys())
if len(unique_algos) >= 3:
    st.subheader("Algorithm Comparison Radar")
    categories = ["Best Reward", "Avg Reward", "Models Trained"]
    fig_radar = go.Figure()
    max_reward = max(m["mean_reward"] for m in models)
    max_models = max(len(v) for v in algo_groups.values())
    for algo in unique_algos:
        rewards = algo_groups[algo]
        scores = [
            max(rewards) / max_reward if max_reward > 0 else 0,
            (sum(rewards) / len(rewards)) / max_reward if max_reward > 0 else 0,
            len(rewards) / max_models if max_models > 0 else 0,
        ]
        fig_radar.add_trace(
            go.Scatterpolar(r=scores + [scores[0]], theta=categories + [categories[0]],
                            fill="toself", name=algo)
        )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400,
        title="Relative Performance (normalized)",
    )
    st.plotly_chart(fig_radar, use_container_width=True)
