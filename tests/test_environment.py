"""Tests for LogisticsEnv."""
import numpy as np
import pytest

from aces.config import MissionConfig, RewardConfig, ScenarioConfig
from aces.domain.environment import LogisticsEnv
from aces.services.scenario import ScenarioService


@pytest.fixture
def scenario_config():
    """Get the default scenario config."""
    svc = ScenarioService()
    scenarios = svc.list_scenarios()
    return svc.get_scenario_config(scenarios[0]['id'])


@pytest.fixture
def env(scenario_config):
    """Create a LogisticsEnv instance."""
    return LogisticsEnv(scenario_config)


def test_env_reset(env):
    """Test environment reset."""
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (env._obs_builder.obs_dim,)
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() <= obs.max() <= 1.0
    assert isinstance(info, dict)
    assert 'time_hr' in info
    assert 'step' in info


def test_env_step(env):
    """Test environment step."""
    obs, info = env.reset(seed=42)
    action = env.action_space.sample()
    new_obs, reward, terminated, truncated, new_info = env.step(action)
    
    assert isinstance(new_obs, np.ndarray)
    assert new_obs.shape == obs.shape
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(new_info, dict)
    assert new_info['step'] == info['step'] + 1


def test_action_masks(env):
    """Test action masking."""
    env.reset(seed=42)
    masks = env.action_masks()
    assert isinstance(masks, np.ndarray)
    assert masks.dtype == bool
    assert masks.shape == (env._action_builder.mask_size,)
    # At least some actions should be valid
    assert masks.any()


def test_episode_termination(env):
    """Test that episodes terminate."""
    obs, info = env.reset(seed=42)
    terminated = False
    steps = 0
    while not terminated and steps < 1000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
    assert terminated or steps < 1000  # Should terminate within reasonable steps


def test_deterministic_reset(env):
    """Test deterministic reset with seed."""
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)