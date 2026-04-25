"""Resource Planning — supply inventory analysis across all scenario bases."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eye.config import SUPPLY_TYPES, SupplyType
from eye.services.scenario import ScenarioService

st.set_page_config(page_title="Resource Planning — ACES", layout="wide")
st.title("Resource Planning")

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

# ------------------------------------------------------------------ Data source

active_sim = st.session_state.get("sim_service") or st.session_state.get("debug_sim")

if active_sim and active_sim.is_initialized():
    state = active_sim.get_state()
    bases_data = state.get("bases", [])
    st.info("Showing **live simulation** supply levels.")
else:
    installs = svc.list_installations(sid)
    if not installs:
        st.warning("No installations found. Add some in Scenario Editor.")
        st.stop()
    bases_data = [
        {
            "name": i["name"],
            "latitude": i["latitude"],
            "longitude": i["longitude"],
            "runway_health_pct": 100.0,
            "supplies": i.get("supply_inventory", {}),
            "is_target": i.get("is_target", False),
        }
        for i in installs
    ]
    st.info("Showing **initial scenario** supply levels — no active simulation running.")

# ------------------------------------------------------------------ Supply table

st.subheader("Current Supply Levels (lbs)")
supply_keys = [sup.value for sup in SUPPLY_TYPES]
rows = []
for b in bases_data:
    row: dict = {"Base": b["name"], "Target": "Yes" if b.get("is_target") else "No"}
    for k in supply_keys:
        row[k] = round(b.get("supplies", {}).get(k, 0.0), 0)
    rows.append(row)

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

# ------------------------------------------------------------------ Heatmap

st.subheader("Supply Heatmap")
z_vals = [[float(r.get(k, 0)) for k in supply_keys] for r in rows]
text_vals = [[f"{r.get(k, 0):,.0f}" for k in supply_keys] for r in rows]
base_names = [r["Base"] for r in rows]

fig_heat = go.Figure(
    data=go.Heatmap(
        z=z_vals,
        x=supply_keys,
        y=base_names,
        colorscale="RdYlGn",
        text=text_vals,
        texttemplate="%{text}",
        textfont={"size": 11},
    )
)
fig_heat.update_layout(
    title="Supply Levels — Green = High, Red = Low",
    height=max(300, len(bases_data) * 80 + 100),
    xaxis_title="Supply Type",
    yaxis_title="Base",
)
st.plotly_chart(fig_heat, use_container_width=True)

# ------------------------------------------------------------------ Consumption analysis

installs = svc.list_installations(sid)
consumption_rows = []
for inst in installs:
    rates = inst.get("consumption_rates", {})
    inv = inst.get("supply_inventory", {})
    for k, rate in rates.items():
        if rate > 0:
            stock = inv.get(k, 0.0)
            days = stock / (rate * 24.0)
            consumption_rows.append(
                {
                    "Base": inst["name"],
                    "Supply": k,
                    "Stock (lbs)": round(stock, 0),
                    "Rate (lbs/hr)": rate,
                    "Days Remaining": round(days, 1),
                }
            )

if consumption_rows:
    st.subheader("Consumption Rate Analysis")
    cdf = pd.DataFrame(consumption_rows).sort_values("Days Remaining")
    st.dataframe(cdf, use_container_width=True)

    critical = [r for r in consumption_rows if r["Days Remaining"] < 3.0]
    if critical:
        st.error("Critical: the following supplies will run out in under 3 days:")
        for item in critical:
            st.write(f"• **{item['Base']}** — {item['Supply']}: {item['Days Remaining']} days remaining")

# ------------------------------------------------------------------ Target bases

target_bases = [b for b in bases_data if b.get("is_target")]
if target_bases:
    st.subheader("Target Base Supply Status")
    st.caption("Target bases are the resupply priority. Focus munitions and fuel deliveries here.")
    priority_supplies = [SupplyType.MUNITIONS, SupplyType.AVGAS, SupplyType.FOOD, SupplyType.WATER]
    for tb in target_bases:
        with st.expander(f"**{tb['name']}** (Target)", expanded=True):
            cols = st.columns(len(priority_supplies))
            for col, sup in zip(cols, priority_supplies):
                val = tb.get("supplies", {}).get(sup.value, 0.0)
                col.metric(sup.value, f"{val:,.0f} lbs")

# ------------------------------------------------------------------ Total inventory

st.subheader("Total Inventory Across All Bases")
totals = {k: sum(b.get("supplies", {}).get(k, 0.0) for b in bases_data) for k in supply_keys}
total_cols = st.columns(len(supply_keys))
for col, k in zip(total_cols, supply_keys):
    col.metric(k, f"{totals[k]:,.0f} lbs")
