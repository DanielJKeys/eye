"""Debug Console — step-by-step manual mission control.

The user can:
  - Advance one step at a time
  - Review the agent's proposed action and override it before execution
  - Run until the next event (strike, destruction)
  - Inspect per-asset and per-base state at each step
"""
import numpy as np
import streamlit as st

from aces.config import FUEL_LEVELS, SUPPLY_LEVELS, SUPPLY_TYPES
from aces.services.scenario import ScenarioService
from aces.services.simulation import SimulationService
from aces.spaces.action import ActionBuilder

st.set_page_config(page_title="Debug Console — ACES", layout="wide")
st.title("Debug Console")

if "scenario_service" not in st.session_state:
    st.session_state.scenario_service = ScenarioService()
    st.session_state.scenario_service.ensure_seed_data()
if "debug_sim" not in st.session_state:
    st.session_state.debug_sim = SimulationService()

svc: ScenarioService = st.session_state.scenario_service
sim: SimulationService = st.session_state.debug_sim
sid = st.session_state.get("active_scenario_id")

if not sid:
    st.warning("Select a scenario on the home page.")
    st.stop()

# ------------------------------------------------------------------ Init controls
c1, c2, c3 = st.columns(3)
with c1:
    debug_seed = st.number_input("Seed", 0, 99999, 42, key="debug_seed")
with c2:
    agent_choice = st.selectbox("Suggestion Agent", ["Heuristic Greedy", "None (random)"])
with c3:
    pass

col_init, col_event, col_step = st.columns(3)
with col_init:
    if st.button("Initialize / Reset", type="primary"):
        scenario = svc.get_scenario_config(sid)
        mc = st.session_state.get("custom_mission_config")
        rc = st.session_state.get("custom_reward_config")
        if mc:
            scenario.mission_config = mc
        if rc:
            scenario.reward_config = rc
        sim.initialize(scenario, seed=int(debug_seed))
        if agent_choice == "Heuristic Greedy":
            sim.set_heuristic_agent()
        st.session_state.debug_override = {}
        st.rerun()

if not sim.is_initialized():
    st.info("Click **Initialize / Reset** to start a mission.")
    st.stop()

state = sim.get_state()
bases_data = state.get("bases", [])
assets_data = state.get("assets", [])
time_hr = state.get("time_hr", 0)
step_n = state.get("step", 0)
cum_r = state.get("cumulative_reward", 0)

# Header
m1, m2, m3, m4 = st.columns(4)
m1.metric("Time", f"{time_hr:.1f} hr")
m2.metric("Step", step_n)
m3.metric("Cumulative Reward", f"{cum_r:.1f}")
m4.metric("Last Reward", f"{state.get('last_reward', 0):.2f}")

if sim.is_terminated():
    st.success("Mission complete. Reset to run again.")

# ------------------------------------------------------------------ Action builder / override
st.divider()
st.subheader("Next Action")

env = sim.env
ab = ActionBuilder(env.scenario)
install_names = [b["name"] for b in bases_data]

# Get agent suggestion
suggested = sim.get_agent_suggestion() if not sim.is_terminated() else None
agent_actions = ab.decode(suggested) if suggested is not None else [{"destination_idx": 0, "fuel_pct": 0, "supply_pcts": {st.value: 0 for st in SUPPLY_TYPES}} for _ in assets_data]

override = st.session_state.get("debug_override", {})

for i, (asset, ag_act) in enumerate(zip(assets_data, agent_actions)):
    if asset.get("is_destroyed"):
        st.markdown(f"~~**{asset['name']}**~~ — DESTROYED")
        continue

    with st.expander(
        f"**{asset['name']}** [{asset['type']}] — {'In Transit' if asset['in_transit'] else 'On Ground'} | Fuel: {asset.get('fuel_pct',0):.0f}%",
        expanded=not asset["in_transit"]
    ):
        if asset["in_transit"]:
            st.info("Asset is in transit. Destination is locked.")
            dest_idx = ag_act["destination_idx"]
            st.write(f"Destination: **{install_names[dest_idx] if dest_idx < len(install_names) else '?'}**")
            override[i] = {"destination_idx": dest_idx, "fuel_pct": 0,
                           "supply_pcts": {st.value: 0 for st in SUPPLY_TYPES}}
            continue

        c1, c2 = st.columns(2)
        dest = c1.selectbox(
            "Destination", install_names,
            index=min(ag_act["destination_idx"], len(install_names)-1),
            key=f"dest_{i}",
        )
        dest_idx_chosen = install_names.index(dest)

        fuel_lvl = c2.select_slider(
            "Fuel Load", options=[f"{p}%" for p in FUEL_LEVELS],
            value=f"{ag_act['fuel_pct']}%",
            key=f"fuel_{i}",
        )
        fuel_pct = int(fuel_lvl.replace("%", ""))

        st.markdown("Supply Loads")
        supply_cols = st.columns(4)
        supply_pcts = {}
        for j, st_type in enumerate(SUPPLY_TYPES):
            ag_pct = ag_act["supply_pcts"].get(st_type.value, 0)
            chosen = supply_cols[j % 4].select_slider(
                st_type.value, options=[f"{p}%" for p in SUPPLY_LEVELS],
                value=f"{ag_pct}%", key=f"sup_{i}_{j}",
            )
            supply_pcts[st_type.value] = int(chosen.replace("%", ""))

        override[i] = {"destination_idx": dest_idx_chosen, "fuel_pct": fuel_pct, "supply_pcts": supply_pcts}

st.session_state.debug_override = override

# ------------------------------------------------------------------ Step controls
st.divider()
col_step2, col_event2, col_run = st.columns(3)

def _build_override_action() -> np.ndarray:
    action = np.zeros(ab.total_dims, dtype=np.int64)
    dim = 0
    for i in range(len(assets_data)):
        act = override.get(i, agent_actions[i] if i < len(agent_actions) else {})
        action[dim] = act.get("destination_idx", 0); dim += 1
        fuel_pct = act.get("fuel_pct", 0)
        fuel_idx = FUEL_LEVELS.index(fuel_pct) if fuel_pct in FUEL_LEVELS else 0
        action[dim] = fuel_idx; dim += 1
        for st_type in SUPPLY_TYPES:
            sp = act.get("supply_pcts", {}).get(st_type.value, 0)
            sp_idx = SUPPLY_LEVELS.index(sp) if sp in SUPPLY_LEVELS else 0
            action[dim] = sp_idx; dim += 1
    return action

with col_step2:
    if not sim.is_terminated() and st.button("Step (Override Action)", type="primary"):
        action_to_use = _build_override_action()
        sim.step(action_to_use)
        st.rerun()

with col_event2:
    if not sim.is_terminated() and st.button("Run to Next Event"):
        snapshots = sim.run_to_event()
        st.rerun()

with col_run:
    if not sim.is_terminated() and st.button("Run to Completion"):
        sim.run_to_completion()
        st.rerun()

# ------------------------------------------------------------------ State detail
st.divider()
tab_bases, tab_assets_detail, tab_log = st.tabs(["Base Status", "Asset Status", "Event Log"])

with tab_bases:
    for b in bases_data:
        alert = " ⚠ MISSILE ALERT" if b.get("missile_alert", 0) > 0 else ""
        pct = b.get("runway_health_pct", 100)
        bar_color = "green" if pct > 50 else ("orange" if pct > 0 else "red")
        st.markdown(f"**{b['name']}**{alert}")
        st.progress(pct / 100.0, text=f"Runway {pct:.0f}% ({b.get('num_craters',0)} craters)")
        sup = b.get("supplies", {})
        cols = st.columns(4)
        for idx, (k, v) in enumerate(sup.items()):
            cols[idx % 4].metric(k, f"{v:,.0f} lbs")
        st.markdown(f"Personnel: {b.get('personnel_count', 0)}")
        st.divider()

with tab_assets_detail:
    for a in assets_data:
        status = "DESTROYED" if a["is_destroyed"] else ("In Transit" if a["in_transit"] else "On Ground")
        st.markdown(f"**{a['name']}** ({a['type']}) — {status}")
        if not a["is_destroyed"]:
            c1, c2, c3 = st.columns(3)
            c1.metric("Fuel", f"{a.get('fuel_pct',0):.0f}%")
            c2.metric("Cargo", f"{a.get('cargo_pct',0):.0f}%")
            c3.metric("Broken Parts", f"{a.get('broken_parts_lbs',0):.0f} lbs")
            if a.get("supplies"):
                sup_cols = st.columns(4)
                for idx, (k, v) in enumerate(a["supplies"].items()):
                    if v > 0:
                        sup_cols[idx % 4].metric(k, f"{v:,.0f} lbs")
        st.divider()

with tab_log:
    history = sim.env._history if sim.env else []
    if history:
        st.markdown(f"**{len(history)} steps recorded**")
        for h in reversed(history[-20:]):
            destroyed = sum(1 for a in h["assets"] if a["is_destroyed"])
            craters = sum(len(b.get("num_craters", [])) if isinstance(b.get("num_craters"), list) else b.get("num_craters", 0) for b in h["bases"])
            st.text(f"T+{h['time_hr']:.1f}hr | Step {h['step']} | Assets destroyed: {destroyed} | Total craters: {craters}")
    else:
        st.info("No history yet.")
