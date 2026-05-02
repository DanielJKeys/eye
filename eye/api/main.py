"""ACES FastAPI backend — wraps ScenarioService and TrainingService for the React UI."""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from eye.services.scenario import ScenarioService
from eye.services.training import TrainingService
from eye.config import TrainingConfig
from eye.agents.registry import AVAILABLE_ALGORITHMS

_svc: Optional[ScenarioService] = None
_tsvc: Optional[TrainingService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _svc, _tsvc
    _svc = ScenarioService()
    _svc.ensure_seed_data()
    _tsvc = TrainingService()
    yield


app = FastAPI(title="ACES API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ scenarios

@app.get("/api/scenarios")
def list_scenarios():
    return _svc.list_scenarios()


# ------------------------------------------------------------------ algorithms

@app.get("/api/algorithms")
def list_algorithms():
    return [{"key": k, "label": v} for k, v in AVAILABLE_ALGORITHMS.items()]


# ------------------------------------------------------------------ training

class TrainRequest(BaseModel):
    scenario_id: str
    algorithm: str = "maskable_ppo"
    n_envs: int = 4
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    norm_obs: bool = True
    norm_reward: bool = True
    seed: Optional[int] = 42
    model_name: str = "trained_model"


@app.post("/api/training/start")
def start_training(req: TrainRequest):
    prog = _tsvc.get_progress()
    if prog and prog["is_running"]:
        raise HTTPException(status_code=400, detail="Training already in progress")

    scenario_config = _svc.get_scenario_config(req.scenario_id)
    cfg = TrainingConfig(
        algorithm=req.algorithm,
        n_envs=req.n_envs,
        total_timesteps=req.total_timesteps,
        learning_rate=req.learning_rate,
        n_steps=req.n_steps,
        batch_size=req.batch_size,
        n_epochs=req.n_epochs,
        norm_obs=req.norm_obs,
        norm_reward=req.norm_reward,
        seed=req.seed if req.seed and req.seed > 0 else None,
    )

    def on_complete(model_path, vecnorm_path, steps, reward):
        _svc.register_saved_model(
            req.scenario_id,
            name=req.model_name,
            algorithm=req.algorithm,
            file_path=model_path,
            vecnormalize_path=vecnorm_path,
            training_steps=steps,
            mean_reward=float(reward),
        )

    _tsvc.start_training(scenario_config, cfg, on_complete)
    return {"status": "started"}


@app.get("/api/training/progress")
def get_progress():
    prog = _tsvc.get_progress()
    if prog is None:
        return {"status": "idle"}
    status = "running" if prog["is_running"] else ("done" if prog["is_done"] else "error")
    return {**prog, "status": status}


# ------------------------------------------------------------------ models

@app.get("/api/scenarios/{scenario_id}/models")
def list_models(scenario_id: str):
    return _svc.list_saved_models(scenario_id)


@app.delete("/api/scenarios/{scenario_id}/models/{model_id}")
def delete_model(scenario_id: str, model_id: str):
    _svc.delete_saved_model(scenario_id, model_id)
    return {"status": "deleted"}
