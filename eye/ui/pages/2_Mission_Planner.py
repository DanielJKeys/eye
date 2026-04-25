"""Mission Planner — configure and launch RL training runs or full automated simulations."""
import time

import streamlit as st

from eye.agents.registry import AVAILABLE_ALGORITHMS
from eye.config import TrainingConfig, MissionConfig, RewardConfig
from eye.services.scenario import ScenarioService
from eye.services.training import TrainingService

st.set_page_config(page_title="Mission Planner — ACES", layout="wide")
st.title("Mission Planner")

if "scenario_service" not in st.session_state:
    st.session_state.scenario_service = ScenarioService()
    st.session_state.scenario_service.ensure_seed_data()
if "training_service" not in st.session_state:
    st.session_state.training_service = TrainingService()

svc: ScenarioService = st.session_state.scenario_service
tsvc: TrainingService = st.session_state.training_service
sid = st.session_state.get("active_scenario_id")

if not sid:
    st.warning("No active scenario. Go to the home page and select one.")
    st.stop()

scenarios = svc.list_scenarios()
sc_map = {s["id"]: s["name"] for s in scenarios}
st.subheader(f"Scenario: {sc_map.get(sid, sid)}")

tab_train, tab_config = st.tabs(["Train Agent", "Mission Configuration"])

# ================================================================== TRAIN
with tab_train:
    st.markdown("Train a reinforcement learning agent on the active scenario.")

    c1, c2 = st.columns(2)
    algorithm = c1.selectbox(
        "Algorithm",
        options=list(AVAILABLE_ALGORITHMS.keys()),
        format_func=lambda x: AVAILABLE_ALGORITHMS[x],
    )
    n_envs = c2.number_input("Parallel Environments", 1, 16, 4)

    c3, c4, c5 = st.columns(3)
    total_steps = c3.number_input("Total Timesteps", 10_000, 10_000_000, 500_000, step=10_000)
    lr = c4.number_input("Learning Rate", 1e-5, 1e-2, 3e-4, format="%.5f")
    seed = c5.number_input("Random Seed (0 = random)", 0, 99_999, 42)

    c6, c7, c8 = st.columns(3)
    n_steps = c6.number_input("N Steps (PPO)", 128, 8192, 2048, step=128)
    batch_size = c7.number_input("Batch Size", 32, 1024, 64, step=32)
    n_epochs = c8.number_input("N Epochs (PPO)", 1, 20, 10)

    norm_obs = st.checkbox("Normalize Observations (VecNormalize)", value=True)
    norm_reward = st.checkbox("Normalize Rewards (VecNormalize)", value=True)

    model_name = st.text_input("Model save name", value=f"{algorithm}_{sc_map.get(sid,'scenario')}")

    progress_placeholder = st.empty()
    log_placeholder = st.empty()

    if st.button("Start Training", type="primary"):
        if "heuristic" in algorithm:
            st.info("Heuristic agents don't require training. Go to Debug Console to run.")
        else:
            cfg = TrainingConfig(
                algorithm=algorithm,
                n_envs=int(n_envs),
                total_timesteps=int(total_steps),
                learning_rate=lr,
                n_steps=int(n_steps),
                batch_size=int(batch_size),
                n_epochs=int(n_epochs),
                norm_obs=norm_obs,
                norm_reward=norm_reward,
                seed=int(seed) if seed > 0 else None,
            )
            scenario_config = svc.get_scenario_config(sid)

            def on_complete(model_path, vecnorm_path, steps, reward):
                svc.register_saved_model(
                    sid,
                    name=model_name,
                    algorithm=algorithm,
                    file_path=model_path,
                    vecnormalize_path=vecnorm_path,
                    training_steps=steps,
                    mean_reward=float(reward),
                )

            st.session_state.active_progress = tsvc.start_training(scenario_config, cfg, on_complete)
            st.rerun()

    # Show live progress if training is running
    if "active_progress" in st.session_state:
        prog = tsvc.get_progress()
        if prog:
            pct = prog["pct"]
            with progress_placeholder.container():
                st.progress(pct / 100.0, text=f"Training: {prog['timesteps']:,} / {prog['total_timesteps']:,} steps")
                c1, c2, c3 = st.columns(3)
                c1.metric("Steps", f"{prog['timesteps']:,}")
                c2.metric("Mean Reward", f"{prog['mean_reward']:.2f}")
                c3.metric("Status", "Running" if prog["is_running"] else ("Done" if prog["is_done"] else "Error"))

            if prog["is_done"]:
                st.success(f"Training complete. Final mean reward: {prog['mean_reward']:.2f}")
                del st.session_state.active_progress
            elif prog["error"]:
                st.error(f"Training failed: {prog['error']}")
                del st.session_state.active_progress
            else:
                time.sleep(2)
                st.rerun()

    # Saved models
    st.divider()
    st.subheader("Saved Models")
    models = svc.list_saved_models(sid)
    if models:
        st.dataframe(
            [{
                "Name": m["name"], "Algorithm": m["algorithm"],
                "Steps": m["training_steps"], "Mean Reward": f"{m['mean_reward']:.2f}",
                "Created": m["created_at"][:10],
            } for m in models],
            use_container_width=True,
        )
        del_m = st.selectbox("Delete model", ["—"] + [m["name"] for m in models], key="del_model")
        if del_m != "—" and st.button("Delete Model"):
            rec = next(m for m in models if m["name"] == del_m)
            svc.delete_saved_model(sid, rec["id"])
            st.rerun()
    else:
        st.info("No trained models yet for this scenario.")

# ================================================================== MISSION CONFIG
with tab_config:
    st.markdown("Adjust mission simulation parameters. Changes apply to new runs only.")

    mc = MissionConfig()
    rc = RewardConfig()

    st.subheader("Mission Parameters")
    c1, c2, c3 = st.columns(3)
    step_hr = c1.number_input("Step Duration (hours)", 0.1, 4.0, mc.step_to_hour)
    mission_len = c2.number_input("Mission Length (hours)", 12.0, 720.0, mc.mission_length_hr)
    alert_win = c3.number_input("Alert Window (hours)", 0.1, 6.0, mc.alert_window_hr)
    c4, c5 = st.columns(2)
    load_time = c4.number_input("Load/Unload Time (hours)", 0.1, 4.0, mc.load_time_hr)
    arrival_r = c5.number_input("Arrival Radius (nautical miles)", 1.0, 100.0, mc.arrival_radius_nm)

    st.subheader("Reward Coefficients")
    c1, c2, c3 = st.columns(3)
    mun_r = c1.number_input("Munitions Delivered (per lb)", 0.0, 100.0, rc.munitions_delivered_per_lb)
    sup_r = c2.number_input("Other Supply Delivered (per lb)", 0.0, 10.0, rc.supply_delivered_per_lb)
    asset_r = c3.number_input("Asset Destroyed (penalty)", -500.0, 0.0, rc.asset_destroyed)
    c4, c5, c6 = st.columns(3)
    runway_r = c4.number_input("Runway Destroyed (penalty)", -500.0, 0.0, rc.base_runway_destroyed)
    threat_r = c5.number_input("Threat Exposure (per step)", -10.0, 0.0, rc.threat_exposure_per_step)
    complete_r = c6.number_input("Mission Complete Bonus", 0.0, 1000.0, rc.mission_complete_bonus)

    if st.button("Save to Session"):
        st.session_state.custom_mission_config = MissionConfig(
            step_to_hour=step_hr, mission_length_hr=mission_len,
            alert_window_hr=alert_win, load_time_hr=load_time,
            arrival_radius_nm=arrival_r,
        )
        st.session_state.custom_reward_config = RewardConfig(
            munitions_delivered_per_lb=mun_r, supply_delivered_per_lb=sup_r,
            asset_destroyed=asset_r, base_runway_destroyed=runway_r,
            threat_exposure_per_step=threat_r, mission_complete_bonus=complete_r,
        )
        st.success("Configuration saved to session. Use Debug Console or a new training run to apply.")
