"""Tests for ObservationBuilder."""
import numpy as np
import pytest

from eye.services.scenario import ScenarioService
from eye.spaces.observation import ObservationBuilder


@pytest.fixture
def scenario_config():
    """Get the default scenario config."""
    svc = ScenarioService()
    scenarios = svc.list_scenarios()
    return svc.get_scenario_config(scenarios[0]['id'])


@pytest.fixture
def obs_builder(scenario_config):
    """Create an ObservationBuilder instance."""
    return ObservationBuilder(scenario_config)


@pytest.fixture
def mock_bases(scenario_config):
    """Create mock bases for testing."""
    from eye.domain.base import Base
    from eye.config import MissionConfig
    mc = MissionConfig()
    return [Base(inst, mc) for inst in scenario_config.installations]


@pytest.fixture
def mock_assets(scenario_config):
    """Create mock assets for testing."""
    from eye.domain.asset import Asset
    from eye.config import MissionConfig
    mc = MissionConfig()
    assets = []
    for ma in scenario_config.mission_assets:
        at = next(at for at in scenario_config.asset_types if at.id == ma.asset_type_id)
        inst = next(inst for inst in scenario_config.installations if inst.id == ma.starting_installation_id)
        assets.append(Asset(ma, at, inst.latitude, inst.longitude, mc))
    return assets


def test_obs_space(obs_builder):
    """Test observation space creation."""
    space = obs_builder.space
    assert space is not None
    assert hasattr(space, 'contains')


def test_build_observation(obs_builder, mock_bases, mock_assets):
    """Test observation building."""
    obs = obs_builder.build(mock_bases, mock_assets)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (obs_builder.obs_dim,)
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() <= obs.max() <= 1.0
