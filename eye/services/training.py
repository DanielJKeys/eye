"""TrainingService — manages RL training runs.

Wraps the environment in SubprocVecEnv + VecNormalize for parallel training.
Saves both model weights and normalization statistics so inference is correct.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Optional

import numpy as np

from eye.config import ScenarioConfig, TrainingConfig
from eye.domain.environment import LogisticsEnv
from eye.agents.registry import get_algorithm_class, is_heuristic, requires_masking


class TrainingProgress:
    """Thread-safe container for training progress updates."""
    def __init__(self) -> None:
        self.timesteps: int = 0
        self.total_timesteps: int = 0
        self.mean_reward: float = 0.0
        self.is_running: bool = False
        self.is_done: bool = False
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def update(self, timesteps: int, mean_reward: float) -> None:
        with self._lock:
            self.timesteps = timesteps
            self.mean_reward = mean_reward

    def finish(self, mean_reward: float) -> None:
        with self._lock:
            self.is_running = False
            self.is_done = True
            self.mean_reward = mean_reward

    def fail(self, error: str) -> None:
        with self._lock:
            self.is_running = False
            self.error = error

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "timesteps": self.timesteps,
                "total_timesteps": self.total_timesteps,
                "mean_reward": self.mean_reward,
                "is_running": self.is_running,
                "is_done": self.is_done,
                "error": self.error,
                "pct": (self.timesteps / self.total_timesteps * 100) if self.total_timesteps > 0 else 0,
            }


class TrainingService:
    def __init__(self) -> None:
        self._progress: Optional[TrainingProgress] = None
        self._thread: Optional[threading.Thread] = None
        self._last_model_path: Optional[str] = None
        self._last_vecnorm_path: Optional[str] = None

    def start_training(
        self,
        scenario: ScenarioConfig,
        training_config: TrainingConfig,
        on_complete: Optional[Callable] = None,
    ) -> TrainingProgress:
        """Launch training in a background thread. Returns a progress tracker."""
        progress = TrainingProgress()
        progress.total_timesteps = training_config.total_timesteps
        progress.is_running = True
        self._progress = progress

        thread = threading.Thread(
            target=self._train_worker,
            args=(scenario, training_config, progress, on_complete),
            daemon=True,
        )
        thread.start()
        self._thread = thread
        return progress

    def get_progress(self) -> Optional[dict]:
        if self._progress is None:
            return None
        return self._progress.snapshot()

    def get_last_model_paths(self) -> tuple[str, str]:
        return self._last_model_path or "", self._last_vecnorm_path or ""

    # ------------------------------------------------------------------ internal

    def _train_worker(
        self,
        scenario: ScenarioConfig,
        cfg: TrainingConfig,
        progress: TrainingProgress,
        on_complete: Optional[Callable],
    ) -> None:
        try:
            from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
            from stable_baselines3.common.callbacks import BaseCallback

            def make_env(seed_offset: int):
                def _init():
                    env = LogisticsEnv(scenario)
                    env.reset(seed=(cfg.seed or 0) + seed_offset)
                    return env
                return _init

            # Use SubprocVecEnv for n_envs > 1, DummyVecEnv otherwise
            if cfg.n_envs > 1:
                vec_env = SubprocVecEnv([make_env(i) for i in range(cfg.n_envs)])
            else:
                vec_env = DummyVecEnv([make_env(0)])

            vec_env = VecNormalize(
                vec_env,
                norm_obs=cfg.norm_obs,
                norm_reward=cfg.norm_reward,
                gamma=cfg.gamma,
            )

            agent_cls = get_algorithm_class(cfg.algorithm)

            class ProgressCallback(BaseCallback):
                def __init__(self, prog: TrainingProgress, report_freq: int = 1000) -> None:
                    super().__init__()
                    self.prog = prog
                    self.report_freq = report_freq
                    self._reward_buf: list[float] = []

                def _on_step(self) -> bool:
                    if self.n_calls % self.report_freq == 0:
                        mean_r = float(np.mean(self.locals.get("rewards", [0.0]) or [0.0]))
                        self.prog.update(self.num_timesteps, mean_r)
                    return True

            algo_kwargs = {
                "learning_rate": cfg.learning_rate,
                "gamma": cfg.gamma,
                "verbose": 0,
                "seed": cfg.seed,
            }
            if cfg.algorithm in ("ppo", "maskable_ppo"):
                algo_kwargs.update({
                    "n_steps": cfg.n_steps,
                    "batch_size": cfg.batch_size,
                    "n_epochs": cfg.n_epochs,
                })

            model = agent_cls("MlpPolicy", vec_env, **algo_kwargs)
            model.learn(
                total_timesteps=cfg.total_timesteps,
                callback=ProgressCallback(progress),
            )

            # Save model and normalization stats
            os.makedirs(cfg.model_save_dir, exist_ok=True)
            algo_tag = cfg.algorithm.replace("_", "-")
            model_path = os.path.join(cfg.model_save_dir, f"{algo_tag}_model")
            vecnorm_path = os.path.join(cfg.model_save_dir, f"{algo_tag}_vecnorm.pkl")
            model.save(model_path)
            vec_env.save(vecnorm_path)
            self._last_model_path = model_path + ".zip"
            self._last_vecnorm_path = vecnorm_path

            # Evaluate
            eval_env = LogisticsEnv(scenario)
            obs, _ = eval_env.reset(seed=42)
            total_r = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = eval_env.step(action)
                total_r += r
                done = terminated or truncated

            progress.finish(total_r)
            if on_complete:
                on_complete(model_path + ".zip", vecnorm_path, int(cfg.total_timesteps), total_r)

        except Exception as exc:
            progress.fail(str(exc))
            raise
