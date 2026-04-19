"""Live Dashboard — real-time map and charts during an automated simulation run."""
import time

import plotly.graph_objects as go
import streamlit as st

from aces.services.scenario import ScenarioService
from aces.services.simulation import SimulationService

st.set_page_config(page_title="Live Dashboard — ACES", layout="wide")
st.title("Live Dashboard")

if "scenario_service" not in st.session_state:
    st.session_state.scenario_service = ScenarioService()
    st.session_state.scenario_service.ensure_seed_data()
if "sim_service" not in st.session_state:
    st.session_state.sim_service = SimulationService()

svc: ScenarioService = st.session_state.scenario_service
sim: SimulationService = st.session_state.sim_service
sid = st.session_state.get("active_scenario_id")

if not sid:
    st.warning("Select a scenario on the home page.")
    st.stop()

# ------------------------------------------------------------------ Controls
col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)

with col_ctrl1:
    if st.button("Initialize / Reset", type="primary"):
        scenario = svc.get_scenario_config(sid)
        mc = st.session_state.get("custom_mission_config")
        rc = st.session_state.get("custom_reward_config")
        if mc:
            scenario.mission_config = mc
        if rc:
            scenario.reward_config = rc
        seed = st.session_state.get("run_seed", 42)
        snap = sim.initialize(scenario, seed=seed)
        st.session_state.dashboard_running = False
        st.session_state.dashboard_snap = snap
        st.rerun()

with col_ctrl2:
    agent_mode = st.selectbox("Agent", ["Heuristic Greedy", "Loaded Model", "Random"])

with col_ctrl3:
    speed = st.slider("Steps per second", 1, 20, 5)

with col_ctrl4:
    seed_val = st.number_input("Seed", 0, 99999, 42, key="run_seed")

# ------------------------------------------------------------------ Run controls
if sim.is_initialized():
    run_col, stop_col = st.columns(2)
    with run_col:
        if not st.session_state.get("dashboard_running") and not sim.is_terminated():
            if st.button("Run to Completion"):
                st.session_state.dashboard_running = True
    with stop_col:
        if st.button("Pause"):
            st.session_state.dashboard_running = False

# ------------------------------------------------------------------ Auto-advance
if st.session_state.get("dashboard_running") and sim.is_initialized() and not sim.is_terminated():
    if agent_mode == "Heuristic Greedy":
        sim.set_heuristic_agent()
    snap = sim.step()
    st.session_state.dashboard_snap = snap
    time.sleep(1.0 / max(speed, 1))
    if sim.is_terminated():
        st.session_state.dashboard_running = False
    st.rerun()

# ------------------------------------------------------------------ Render current state
snap = st.session_state.get("dashboard_snap")

if not snap:
    st.info("Click **Initialize / Reset** to begin.")
    st.stop()

# Header metrics
time_hr = snap.get("time_hr", 0)
step_n = snap.get("step", 0)
cum_r = snap.get("cumulative_reward", 0)
bases_data = snap.get("bases", [])
assets_data = snap.get("assets", [])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Mission Time", f"{time_hr:.1f} hr")
m2.metric("Step", step_n)
m3.metric("Cumulative Reward", f"{cum_r:.1f}")
m4.metric("Assets Alive", sum(1 for a in assets_data if not a.get("is_destroyed")))

if snap.get("terminated"):
    st.success("Mission Complete")

# Map
st.subheader("Theater Map")
try:
    import folium
    from streamlit_folium import st_folium

    center_lat = sum(b["latitude"] for b in bases_data) / max(len(bases_data), 1)
    center_lon = sum(b["longitude"] for b in bases_data) / max(len(bases_data), 1)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB dark_matter")

    # Threat zones
    for tz in snap.get("threat_zones", []):
        folium.Circle(
            location=[tz["latitude"], tz["longitude"]],
            radius=tz["radius_nm"] * 1852,  # nm to meters
            color="red", fill=True, fill_opacity=0.1,
            popup=f"Threat: {tz['peak_threat_pct']:.0f}%",
        ).add_to(m)

    # Bases
    for b in bases_data:
        color = "green" if b["runway_health_pct"] > 50 else ("orange" if b["runway_health_pct"] > 0 else "red")
        icon = "star" if b["is_target"] else "home"
        alert_str = f" ALERT {b['missile_alert']*100:.0f}%" if b["missile_alert"] > 0 else ""
        folium.Marker(
            location=[b["latitude"], b["longitude"]],
            popup=(
                f"<b>{b['name']}</b>{alert_str}<br>"
                f"Runway: {b['runway_health_pct']:.0f}%<br>"
                f"Craters: {b['num_craters']}<br>"
                f"Personnel: {b['personnel_count']}"
            ),
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
        ).add_to(m)

    # Assets
    for a in assets_data:
        if a.get("is_destroyed"):
            continue
        color = "blue" if a.get("in_transit") else "gray"
        folium.CircleMarker(
            location=[a["latitude"], a["longitude"]],
            radius=8, color=color, fill=True, fill_opacity=0.8,
            popup=f"<b>{a['name']}</b> [{a['type']}]<br>Fuel: {a['fuel_pct']:.0f}%",
        ).add_to(m)

    st_folium(m, height=450, use_container_width=True)
except ImportError:
    st.warning("Install streamlit-folium for the map view: pip install streamlit-folium")

# Supply charts
st.subheader("Base Supply Levels")
history = sim.env._history if sim.env else []
if len(history) > 1:
    times = [h["time_hr"] for h in history]
    fig = go.Figure()
    for b_idx, base_data_init in enumerate(history[0]["bases"]):
        bname = base_data_init["name"]
        munitions = [h["bases"][b_idx]["supplies"].get("Munitions", 0) for h in history]
        avgas = [h["bases"][b_idx]["supplies"].get("AvGas", 0) for h in history]
        fig.add_trace(go.Scatter(x=times, y=munitions, name=f"{bname} Munitions", mode="lines"))
        fig.add_trace(go.Scatter(x=times, y=avgas, name=f"{bname} AvGas", mode="lines", line=dict(dash="dot")))
    fig.update_layout(
        title="Supply Levels Over Time", xaxis_title="Mission Time (hr)",
        yaxis_title="lbs", height=350, legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

# Asset table
st.subheader("Asset Status")
if assets_data:
    st.dataframe(
        [{
            "Asset": a["name"], "Type": a["type"],
            "Status": "DESTROYED" if a["is_destroyed"] else ("Transit" if a["in_transit"] else "Ground"),
            "Fuel %": f"{a.get('fuel_pct', 0):.0f}%",
            "Cargo %": f"{a.get('cargo_pct', 0):.0f}%",
            "Broken Parts (lbs)": a.get("broken_parts_lbs", 0),
        } for a in assets_data],
        use_container_width=True,
    )
