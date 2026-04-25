"""Decision Support Dashboard — real-time recommendations from live simulation."""
import plotly.graph_objects as go
import streamlit as st

from eye.config import SUPPLY_TYPES, SupplyType
from eye.services.scenario import ScenarioService

st.set_page_config(page_title="Decision Support — ACES", layout="wide")
st.title("Decision Support Dashboard")

if "scenario_service" not in st.session_state:
    st.session_state.scenario_service = ScenarioService()
    st.session_state.scenario_service.ensure_seed_data()

svc: ScenarioService = st.session_state.scenario_service
sid = st.session_state.get("active_scenario_id")

# Pull simulation state from whichever service is active
active_sim = st.session_state.get("sim_service") or st.session_state.get("debug_sim")

if not active_sim or not active_sim.is_initialized():
    st.info("No active simulation. Initialize one in **Live Dashboard** or **Debug Console** first.")
    st.stop()

state = active_sim.get_state()
bases_data = state.get("bases", [])
assets_data = state.get("assets", [])
time_hr = state.get("time_hr", 0)
cum_r = state.get("cumulative_reward", 0)

# ------------------------------------------------------------------ Mission status

st.subheader("Mission Status")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Mission Time", f"{time_hr:.1f} hr")
alive = sum(1 for a in assets_data if not a.get("is_destroyed"))
m2.metric("Assets Alive", f"{alive} / {len(assets_data)}")
operational = sum(1 for b in bases_data if b.get("runway_health_pct", 0) > 0)
m3.metric("Bases Operational", f"{operational} / {len(bases_data)}")
m4.metric("Cumulative Reward", f"{cum_r:.1f}")

if state.get("terminated"):
    st.success("Mission Complete.")

# ------------------------------------------------------------------ Missile alerts

missile_alerts = [b for b in bases_data if b.get("missile_alert", 0) > 0]
if missile_alerts:
    st.subheader("Missile Alerts")
    for b in missile_alerts:
        pct = b["missile_alert"] * 100
        st.error(f"INCOMING: **{b['name']}** — strike {pct:.0f}% imminent")

# ------------------------------------------------------------------ Supply alerts

st.subheader("Supply Status")
critical_threshold = {
    SupplyType.MUNITIONS.value: 5_000,
    SupplyType.AVGAS.value: 10_000,
    SupplyType.FOOD.value: 2_000,
    SupplyType.WATER.value: 2_000,
    SupplyType.PARTS.value: 1_000,
    SupplyType.RUNWAY_MATERIAL.value: 2_000,
    SupplyType.ELECTRICITY.value: 1_000,
    SupplyType.MOGAS.value: 1_000,
}

any_alert = False
for b in bases_data:
    for sup_type in SUPPLY_TYPES:
        val = b.get("supplies", {}).get(sup_type.value, 0)
        threshold = critical_threshold.get(sup_type.value, 1_000)
        if val < threshold:
            st.warning(f"{b['name']} — **{sup_type.value}** low: {val:,.0f} lbs (threshold {threshold:,})")
            any_alert = True

if not any_alert:
    st.success("All supply levels nominal.")

# ------------------------------------------------------------------ Asset recommendations

st.subheader("Asset Recommendations")
for a in assets_data:
    if a.get("is_destroyed"):
        st.markdown(f"~~**{a['name']}**~~ — DESTROYED")
        continue

    status = "In Transit" if a.get("in_transit") else "On Ground"
    fuel_pct = a.get("fuel_pct", 0)
    cargo_pct = a.get("cargo_pct", 0)
    recs = []

    if not a.get("in_transit"):
        if fuel_pct < 30:
            recs.append("Refuel immediately — fuel below 30%")
        if cargo_pct < 10:
            recs.append("Load supplies before next sortie — cargo near empty")
        if fuel_pct > 80 and cargo_pct < 10:
            recs.append("Consider loading supplies and departing for a target base")

    label = f"**{a['name']}** [{a['type']}] — {status} | Fuel {fuel_pct:.0f}% | Cargo {cargo_pct:.0f}%"
    if recs:
        with st.expander(label, expanded=True):
            for rec in recs:
                st.write(f"• {rec}")
    else:
        st.markdown(label + " — No action required.")

# ------------------------------------------------------------------ Supply comparison chart

st.subheader("Base Supply Comparison")
if bases_data:
    supply_keys = [sup.value for sup in SUPPLY_TYPES if sup != SupplyType.AVGAS]
    fig = go.Figure()
    for b in bases_data:
        vals = [b.get("supplies", {}).get(k, 0) for k in supply_keys]
        fig.add_trace(go.Bar(name=b["name"], x=supply_keys, y=vals))
    fig.update_layout(
        barmode="group",
        height=380,
        xaxis_title="Supply Type",
        yaxis_title="lbs",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ Runway health

st.subheader("Runway Status")
rw_cols = st.columns(len(bases_data)) if bases_data else []
for col, b in zip(rw_cols, bases_data):
    pct = b.get("runway_health_pct", 100)
    craters = b.get("num_craters", 0)
    col.metric(b["name"], f"{pct:.0f}%", delta=f"{craters} craters" if craters else "No damage",
               delta_color="inverse")
