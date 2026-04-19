"""Scenario Editor — CRUD for all scenario data.

Replaces the Google Sheets dependency. All changes write to SQLite immediately.
"""
import streamlit as st

from aces.config import SUPPLY_TYPES, SupplyType
from aces.services.scenario import ScenarioService

st.set_page_config(page_title="Scenario Editor — ACES", layout="wide")
st.title("Scenario Editor")

if "scenario_service" not in st.session_state:
    st.session_state.scenario_service = ScenarioService()
    st.session_state.scenario_service.ensure_seed_data()

svc: ScenarioService = st.session_state.scenario_service

# ------------------------------------------------------------------ Scenario selector
scenarios = svc.list_scenarios()
sc_map = {s["id"]: s["name"] for s in scenarios}
sid = st.session_state.get("active_scenario_id")
if not sid or sid not in sc_map:
    sid = list(sc_map.keys())[0] if sc_map else None

if not sid:
    st.warning("No scenarios found. Create one below.")
    new_name = st.text_input("New scenario name")
    new_desc = st.text_area("Description")
    if st.button("Create Scenario") and new_name:
        new_id = svc.create_scenario(new_name, new_desc)
        st.session_state.active_scenario_id = new_id
        st.rerun()
    st.stop()

col_sel, col_new = st.columns([3, 1])
with col_sel:
    chosen = st.selectbox("Active Scenario", options=list(sc_map.keys()), format_func=lambda x: sc_map[x])
    if chosen != sid:
        st.session_state.active_scenario_id = chosen
        sid = chosen

with col_new:
    with st.popover("+ New Scenario"):
        new_name = st.text_input("Name", key="new_sc_name")
        new_desc = st.text_area("Description", key="new_sc_desc")
        if st.button("Create", key="create_sc") and new_name:
            new_id = svc.create_scenario(new_name, new_desc)
            st.session_state.active_scenario_id = new_id
            st.rerun()

tab_at, tab_inst, tab_assets, tab_threats, tab_missiles, tab_attacks = st.tabs([
    "Asset Types", "Installations", "Mission Assets", "Threat Zones", "Missile Types", "Attack Schedule"
])

# ================================================================== ASSET TYPES
with tab_at:
    st.subheader("Asset Types")
    ats = svc.list_asset_types(sid)
    if ats:
        st.dataframe(
            [{
                "Name": a["name"], "Speed (kts)": a["speed_kts"],
                "Fuel Cap (lbs)": a["fuel_capacity_lbs"],
                "Max Cargo (lbs)": a["max_cargo_weight_lbs"],
                "Runway Req (ft)": a["runway_required_ft"],
                "Seats": a["personnel_seats"],
            } for a in ats],
            use_container_width=True,
        )
        del_at = st.selectbox("Delete asset type", ["—"] + [a["name"] for a in ats], key="del_at")
        if del_at != "—" and st.button("Delete", key="del_at_btn"):
            rec = next(a for a in ats if a["name"] == del_at)
            svc.delete_asset_type(sid, rec["id"])
            st.rerun()

    with st.expander("Add Asset Type"):
        with st.form("add_at"):
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Name*")
            speed = c2.number_input("Speed (kts)", 0.0, 3000.0, 400.0)
            cost = c3.number_input("Cost ($)", 0.0, value=50_000_000.0, step=1_000_000.0)
            c4, c5, c6 = st.columns(3)
            fuel_cap = c4.number_input("Fuel Capacity (lbs)", 0.0, 500_000.0, 50_000.0)
            fuel_burn = c5.number_input("Fuel Burn (lbs/hr)", 0.0, 50_000.0, 8_000.0)
            sig = c6.number_input("Signature (%)", 0.0, 100.0, 20.0)
            c7, c8, c9 = st.columns(3)
            cargo_w = c7.number_input("Max Cargo Weight (lbs)", 0.0, 500_000.0, 40_000.0)
            cargo_v = c8.number_input("Max Cargo Volume (ft³)", 0.0, 50_000.0, 5_000.0)
            runway = c9.number_input("Runway Required (ft)", 0.0, 15_000.0, 3_000.0)
            c10, c11, c12 = st.columns(3)
            seats = c10.number_input("Personnel Seats", 0, 500, 0)
            break_chance = c11.number_input("Parts Break Chance (%)", 0.0, 100.0, 2.0)
            break_min = c12.number_input("Parts Break Min (lbs)", 0.0, 5_000.0, 100.0)
            break_max = st.number_input("Parts Break Max (lbs)", 0.0, 5_000.0, 500.0)
            cargo_types = st.multiselect(
                "Cargo Types", [st.value for st in SUPPLY_TYPES],
                default=[st.value for st in SUPPLY_TYPES]
            )
            submitted = st.form_submit_button("Add")
            if submitted and name:
                svc.add_asset_type(
                    sid, name=name, speed_kts=speed, cost=cost,
                    fuel_capacity_lbs=fuel_cap, fuel_burn_lbs_per_hr=fuel_burn,
                    signature_pct=sig, max_cargo_weight_lbs=cargo_w,
                    max_cargo_volume_ft3=cargo_v, runway_required_ft=runway,
                    personnel_seats=seats, part_break_chance_pct=break_chance,
                    part_break_min_lbs=break_min, part_break_max_lbs=break_max,
                    cargo_types=cargo_types,
                )
                st.rerun()

# ================================================================== INSTALLATIONS
with tab_inst:
    st.subheader("Installations (Bases)")
    installs = svc.list_installations(sid)
    if installs:
        st.dataframe(
            [{
                "Name": i["name"], "Lat": i["latitude"], "Lon": i["longitude"],
                "Runway Max (ft)": i["runway_length_max_ft"],
                "Repair Rate (ft/hr)": i["runway_repair_rate_ft_per_hr"],
                "Personnel": i["initial_personnel"],
                "Target": "Yes" if i["is_target"] else "No",
            } for i in installs],
            use_container_width=True,
        )
        del_inst = st.selectbox("Delete installation", ["—"] + [i["name"] for i in installs], key="del_inst")
        if del_inst != "—" and st.button("Delete Installation", key="del_inst_btn"):
            rec = next(i for i in installs if i["name"] == del_inst)
            svc.delete_installation(sid, rec["id"])
            st.rerun()

    with st.expander("Add Installation"):
        with st.form("add_inst"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Name*")
            access = c2.selectbox("Access Type", ["air", "sea", "land"])
            c3, c4 = st.columns(2)
            lat = c3.number_input("Latitude", -90.0, 90.0, 13.0)
            lon = c4.number_input("Longitude", -180.0, 180.0, 144.0)
            c5, c6, c7 = st.columns(3)
            runway_max = c5.number_input("Runway Max (ft)", 1000.0, 20_000.0, 10_000.0)
            repair_rate = c6.number_input("Repair Rate (ft/hr)", 0.0, 500.0, 50.0)
            personnel = c7.number_input("Personnel", 0, 10_000, 500)
            is_target = st.checkbox("Is Target Installation")
            st.markdown("**Initial Supplies (lbs)**")
            supply_cols = st.columns(4)
            supply_inv = {}
            for idx, st_type in enumerate(SUPPLY_TYPES):
                supply_inv[st_type.value] = supply_cols[idx % 4].number_input(
                    st_type.value, 0.0, value=10_000.0, key=f"inv_{st_type.value}"
                )
            submitted = st.form_submit_button("Add")
            if submitted and name:
                svc.add_installation(
                    sid, name=name, latitude=lat, longitude=lon,
                    runway_length_max_ft=runway_max,
                    runway_repair_rate_ft_per_hr=repair_rate,
                    is_target=is_target, access_type=access,
                    initial_personnel=personnel,
                    supply_inventory=supply_inv,
                    consumption_rates={st.value: 0.0 for st in SUPPLY_TYPES},
                    alert_thresholds={st.value: 25.0 for st in SUPPLY_TYPES},
                )
                st.rerun()

# ================================================================== MISSION ASSETS
with tab_assets:
    st.subheader("Mission Assets")
    assets = svc.list_mission_assets(sid)
    ats = svc.list_asset_types(sid)
    installs = svc.list_installations(sid)
    at_map = {a["id"]: a["name"] for a in ats}
    inst_map = {i["id"]: i["name"] for i in installs}

    if assets:
        st.dataframe(
            [{
                "Name": a["name"],
                "Type": at_map.get(a["asset_type_id"], "?"),
                "Starting Base": inst_map.get(a["starting_installation_id"], "?"),
                "Fuel (lbs)": a["starting_fuel_lbs"],
                "Personnel": a["starting_personnel"],
            } for a in assets],
            use_container_width=True,
        )
        del_asset = st.selectbox("Delete asset", ["—"] + [a["name"] for a in assets], key="del_ma")
        if del_asset != "—" and st.button("Delete Asset", key="del_ma_btn"):
            rec = next(a for a in assets if a["name"] == del_asset)
            svc.delete_mission_asset(sid, rec["id"])
            st.rerun()

    if ats and installs:
        with st.expander("Add Mission Asset"):
            with st.form("add_ma"):
                c1, c2, c3 = st.columns(3)
                name = c1.text_input("Asset Name*")
                at_choice = c2.selectbox("Asset Type", list(at_map.keys()), format_func=lambda x: at_map[x])
                inst_choice = c3.selectbox("Starting Base", list(inst_map.keys()), format_func=lambda x: inst_map[x])
                c4, c5 = st.columns(2)
                fuel = c4.number_input("Starting Fuel (lbs)", 0.0, 500_000.0, 20_000.0)
                personnel = c5.number_input("Starting Personnel", 0, 500, 0)
                submitted = st.form_submit_button("Add")
                if submitted and name:
                    svc.add_mission_asset(
                        sid, name=name, asset_type_id=at_choice,
                        starting_installation_id=inst_choice,
                        starting_fuel_lbs=fuel, starting_personnel=personnel,
                        starting_supplies={},
                    )
                    st.rerun()
    else:
        st.info("Add asset types and installations first.")

# ================================================================== THREAT ZONES
with tab_threats:
    st.subheader("Threat Zones")
    threats = svc.list_threat_zones(sid)
    if threats:
        st.dataframe(
            [{
                "Lat": t["latitude"], "Lon": t["longitude"],
                "Radius (nm)": t["radius_nm"],
                "Ground (%)": t["ground_threat_pct"],
                "Sea (%)": t["sea_threat_pct"],
                "Air (%)": t["air_threat_pct"],
            } for t in threats],
            use_container_width=True,
        )
        del_t = st.selectbox("Delete threat zone by index", ["—"] + [str(i) for i in range(len(threats))], key="del_tz")
        if del_t != "—" and st.button("Delete Zone", key="del_tz_btn"):
            svc.delete_threat_zone(sid, threats[int(del_t)]["id"])
            st.rerun()

    with st.expander("Add Threat Zone"):
        with st.form("add_tz"):
            c1, c2, c3 = st.columns(3)
            lat = c1.number_input("Center Latitude", -90.0, 90.0, 20.0)
            lon = c2.number_input("Center Longitude", -180.0, 180.0, 120.0)
            radius = c3.number_input("Radius (nautical miles)", 1.0, 2000.0, 200.0)
            c4, c5, c6 = st.columns(3)
            ground = c4.number_input("Ground Threat (%)", 0.0, 100.0, 10.0)
            sea = c5.number_input("Sea Threat (%)", 0.0, 100.0, 30.0)
            air = c6.number_input("Air Threat (%)", 0.0, 100.0, 40.0)
            if st.form_submit_button("Add"):
                svc.add_threat_zone(
                    sid, latitude=lat, longitude=lon, radius_nm=radius,
                    ground_threat_pct=ground, sea_threat_pct=sea, air_threat_pct=air,
                )
                st.rerun()

# ================================================================== MISSILE TYPES
with tab_missiles:
    st.subheader("Missile Types")
    missiles = svc.list_missile_types(sid)
    if missiles:
        st.dataframe(
            [{
                "Name": m["name"],
                "Impact Chance (%)": m["impact_chance_pct"],
                "Supply Hit (%)": m["supplies_hit_pct"],
                "Runway Hit (%)": m["runway_hit_pct"],
                "Asset Hit (%)": m["asset_hit_pct"],
                "Craters": m["craters_created"],
            } for m in missiles],
            use_container_width=True,
        )

    with st.expander("Add Missile Type"):
        with st.form("add_mt"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Name*")
            craters = c2.number_input("Craters Created", 0, 20, 3)
            c3, c4, c5, c6 = st.columns(4)
            impact = c3.number_input("Impact Chance (%)", 0.0, 100.0, 70.0)
            supply_hit = c4.number_input("Supply Hit (%)", 0.0, 100.0, 10.0)
            runway_hit = c5.number_input("Runway Hit (%)", 0.0, 100.0, 30.0)
            asset_hit = c6.number_input("Asset Hit (%)", 0.0, 100.0, 10.0)
            if st.form_submit_button("Add") and name:
                svc.add_missile_type(
                    sid, name=name, craters_created=int(craters),
                    impact_chance_pct=impact, supplies_hit_pct=supply_hit,
                    runway_hit_pct=runway_hit, asset_hit_pct=asset_hit,
                )
                st.rerun()

# ================================================================== ATTACK SCHEDULE
with tab_attacks:
    st.subheader("Attack Schedule")
    attacks = svc.list_attacks(sid)
    missiles = svc.list_missile_types(sid)
    installs = svc.list_installations(sid)
    inst_map = {i["id"]: i["name"] for i in installs}
    missile_map = {m["id"]: m["name"] for m in missiles}

    if attacks:
        st.dataframe(
            [{
                "Time (hr)": a["time_hr"],
                "Target": inst_map.get(a["target_installation_id"], "?"),
                "Missiles": a["num_missiles"],
                "Type": missile_map.get(a["missile_type_id"], "?"),
            } for a in attacks],
            use_container_width=True,
        )
        del_a = st.selectbox("Delete attack by index", ["—"] + [str(i) for i in range(len(attacks))], key="del_atk")
        if del_a != "—" and st.button("Delete Attack", key="del_atk_btn"):
            svc.delete_attack(sid, attacks[int(del_a)]["id"])
            st.rerun()

    if missiles and installs:
        with st.expander("Add Attack Event"):
            with st.form("add_atk"):
                c1, c2, c3 = st.columns(3)
                time_hr = c1.number_input("Mission Time (hr)", 0.0, 240.0, 24.0)
                target = c2.selectbox("Target Installation", list(inst_map.keys()), format_func=lambda x: inst_map[x])
                missile_type = c3.selectbox("Missile Type", list(missile_map.keys()), format_func=lambda x: missile_map[x])
                num_m = st.number_input("Number of Missiles", 1, 20, 2)
                if st.form_submit_button("Add"):
                    svc.add_attack(
                        sid, time_hr=time_hr,
                        target_installation_id=target,
                        num_missiles=int(num_m),
                        missile_type_id=missile_type,
                    )
                    st.rerun()
    else:
        st.info("Add missile types and installations first.")
