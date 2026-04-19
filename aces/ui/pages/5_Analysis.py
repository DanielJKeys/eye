"""Analysis — post-mission analytics dashboard.

Replaces the Word/Excel report output from v4.
Renders supply timelines, asset attrition, delivery metrics, and destruction events
directly in the browser. No kaleido, no python-docx required.
"""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aces.services.scenario import ScenarioService
from aces.services.simulation import SimulationService

st.set_page_config(page_title="Analysis — ACES", layout="wide")
st.title("Mission Analysis")

if "scenario_service" not in st.session_state:
    st.session_state.scenario_service = ScenarioService()
    st.session_state.scenario_service.ensure_seed_data()

svc: ScenarioService = st.session_state.scenario_service

# ------------------------------------------------------------------ Source selection
st.info("Analysis uses the most recent simulation run. Complete a mission in the **Debug Console** or **Live Dashboard** first.")

# Try to grab history from either simulation service
sim_candidates = [
    st.session_state.get("sim_service"),
    st.session_state.get("debug_sim"),
]
active_sim: SimulationService | None = None
for candidate in sim_candidates:
    if candidate and candidate.is_initialized() and candidate.env and candidate.env._history:
        active_sim = candidate
        break

if not active_sim:
    st.warning("No completed simulation found. Run a mission first.")
    st.stop()

history = active_sim.env._history
scenario = active_sim.env.scenario
sid = st.session_state.get("active_scenario_id", "")

if not history:
    st.warning("No history recorded. Run at least one step.")
    st.stop()

# ------------------------------------------------------------------ Summary metrics
times = [h["time_hr"] for h in history]
final = history[-1]

n_assets_start = len(history[0]["assets"])
n_assets_end = sum(1 for a in final["assets"] if not a["is_destroyed"])
n_destroyed = n_assets_start - n_assets_end
bases_final = final["bases"]
avg_runway = sum(b["runway_health_pct"] for b in bases_final) / max(len(bases_final), 1)

# Delivery: sum munitions delivered to target bases
total_munitions_at_target = sum(
    b["supplies"].get("Munitions", 0)
    for b in bases_final
    if b.get("is_target")
)
initial_munitions_at_target = sum(
    b["supplies"].get("Munitions", 0)
    for b in history[0]["bases"]
    if b.get("is_target")
)

st.subheader("Mission Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mission Duration", f"{times[-1]:.1f} hr")
c2.metric("Steps Simulated", len(history))
c3.metric("Assets Survived", f"{n_assets_end} / {n_assets_start}")
c4.metric("Avg Runway Health", f"{avg_runway:.0f}%")
c5.metric("Munitions at Target", f"{total_munitions_at_target:,.0f} lbs")

# ------------------------------------------------------------------ Supply timelines
st.subheader("Supply Levels Over Time")

supply_keys = ["Munitions", "AvGas", "Food", "Water", "Parts", "RunwayMaterial"]
base_names = [b["name"] for b in history[0]["bases"]]

selected_bases = st.multiselect(
    "Select Bases", base_names, default=base_names
)
selected_supplies = st.multiselect(
    "Select Supplies", supply_keys, default=["Munitions", "AvGas"]
)

if selected_bases and selected_supplies:
    fig = go.Figure()
    for b_idx, bname in enumerate(base_names):
        if bname not in selected_bases:
            continue
        for supply in selected_supplies:
            vals = [h["bases"][b_idx]["supplies"].get(supply, 0) for h in history]
            fig.add_trace(go.Scatter(
                x=times, y=vals,
                name=f"{bname} — {supply}",
                mode="lines",
            ))
    fig.update_layout(
        xaxis_title="Mission Time (hr)", yaxis_title="lbs",
        height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ Runway health
st.subheader("Runway Health Over Time")
fig_rw = go.Figure()
for b_idx, bname in enumerate(base_names):
    if bname not in (selected_bases or base_names):
        continue
    rh = [h["bases"][b_idx]["runway_health_pct"] for h in history]
    fig_rw.add_trace(go.Scatter(x=times, y=rh, name=bname, mode="lines"))
fig_rw.update_layout(
    xaxis_title="Mission Time (hr)", yaxis_title="Runway Health (%)",
    yaxis=dict(range=[0, 105]), height=300,
)
st.plotly_chart(fig_rw, use_container_width=True)

# ------------------------------------------------------------------ Asset status
st.subheader("Asset Attrition")

asset_names = [a["name"] for a in history[0]["assets"]]

fig_assets = go.Figure()
for a_idx, aname in enumerate(asset_names):
    fuel_over_time = [h["assets"][a_idx].get("fuel_pct", 0) for h in history]
    destroyed_at = next(
        (h["time_hr"] for h in history if h["assets"][a_idx].get("is_destroyed")), None
    )
    fig_assets.add_trace(go.Scatter(
        x=times, y=fuel_over_time,
        name=f"{aname} fuel%", mode="lines",
        line=dict(dash="dot" if destroyed_at else "solid"),
    ))
    if destroyed_at:
        fig_assets.add_vline(x=destroyed_at, line_color="red", line_dash="dash",
                              annotation_text=f"{aname} destroyed")

fig_assets.update_layout(
    xaxis_title="Mission Time (hr)", yaxis_title="Fuel %",
    height=350, legend=dict(orientation="h"),
)
st.plotly_chart(fig_assets, use_container_width=True)

# ------------------------------------------------------------------ Destruction / strike log
st.subheader("Event Log")
events = []
for h in history:
    for a in h["assets"]:
        if a.get("is_destroyed"):
            events.append({"Time (hr)": h["time_hr"], "Event": f"Asset DESTROYED: {a['name']}"})
    for b in h["bases"]:
        if b.get("num_craters", 0) > 0:
            events.append({"Time (hr)": h["time_hr"], "Event": f"{b['name']}: {b['num_craters']} craters, runway {b['runway_health_pct']:.0f}%"})

seen = set()
unique_events = []
for e in events:
    key = (e["Time (hr)"], e["Event"])
    if key not in seen:
        seen.add(key)
        unique_events.append(e)

if unique_events:
    st.dataframe(unique_events, use_container_width=True)
else:
    st.success("No destruction or strike events recorded.")

# ------------------------------------------------------------------ Export
st.subheader("Export")
if st.button("Export to CSV"):
    import pandas as pd
    import io
    rows = []
    for h in history:
        row = {"time_hr": h["time_hr"]}
        for b in h["bases"]:
            prefix = b["name"].replace(" ", "_")[:12]
            row[f"{prefix}_runway_pct"] = b["runway_health_pct"]
            for k, v in b["supplies"].items():
                row[f"{prefix}_{k}"] = v
        for a in h["assets"]:
            aprefix = a["name"].replace(" ", "_")[:10]
            row[f"{aprefix}_fuel_pct"] = a.get("fuel_pct", 0)
            row[f"{aprefix}_destroyed"] = int(a.get("is_destroyed", False))
        rows.append(row)
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(
        "Download CSV", data=buf.getvalue(),
        file_name="aces_mission_data.csv", mime="text/csv"
    )
