"""ACES Platform — Streamlit home page.

Run with:  streamlit run aces/ui/app.py
Or:        aces-ui  (after pip install -e .)
"""
import streamlit as st

from aces.services.scenario import ScenarioService

st.set_page_config(
    page_title="ACES Platform",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ Bootstrap

if "scenario_service" not in st.session_state:
    st.session_state.scenario_service = ScenarioService()
    st.session_state.scenario_service.ensure_seed_data()

if "active_scenario_id" not in st.session_state:
    scenarios = st.session_state.scenario_service.list_scenarios()
    st.session_state.active_scenario_id = scenarios[0]["id"] if scenarios else None

# ------------------------------------------------------------------ Layout

st.title("ACES Platform")
st.caption("ACE Simulation Environment — Military Logistics Decision Support")

st.markdown("""
ACES Platform is a military logistics simulation and AI decision-support tool.
Select a page from the sidebar to begin.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Mission Planner")
    st.write("Configure scenarios, select an agent, and launch a full automated run.")
    st.page_link("pages/2_Mission_Planner.py", label="Open Mission Planner")

with col2:
    st.subheader("Debug Console")
    st.write("Step through missions manually. Review and override agent decisions at each timestep.")
    st.page_link("pages/4_Debug_Console.py", label="Open Debug Console")

with col3:
    st.subheader("Scenario Editor")
    st.write("Add, edit, or remove asset types, installations, threats, and attack schedules.")
    st.page_link("pages/1_Scenario_Editor.py", label="Open Scenario Editor")

st.divider()

# Active scenario summary
svc: ScenarioService = st.session_state.scenario_service
sid = st.session_state.active_scenario_id

if sid:
    scenarios = svc.list_scenarios()
    sc_names = {s["id"]: s["name"] for s in scenarios}
    chosen = st.selectbox(
        "Active Scenario",
        options=list(sc_names.keys()),
        format_func=lambda x: sc_names[x],
        index=list(sc_names.keys()).index(sid) if sid in sc_names else 0,
    )
    if chosen != sid:
        st.session_state.active_scenario_id = chosen
        st.rerun()

    installs = svc.list_installations(sid)
    assets = svc.list_mission_assets(sid)
    attacks = svc.list_attacks(sid)

    c1, c2, c3 = st.columns(3)
    c1.metric("Installations", len(installs))
    c2.metric("Mission Assets", len(assets))
    c3.metric("Scheduled Strikes", len(attacks))

    target_bases = [i["name"] for i in installs if i.get("is_target")]
    if target_bases:
        st.info(f"Target installation(s): {', '.join(target_bases)}")
