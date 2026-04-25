"""Tests for ActionBuilder."""
import numpy as np
import pytest

from eye.services.scenario import ScenarioService
from eye.spaces.action import ActionBuilder


@pytest.fixture
def scenario_config():
    """Get the default scenario config."""
    svc = ScenarioService()
    scenarios = svc.list_scenarios()
    return svc.get_scenario_config(scenarios[0]['id'])


@pytest.fixture
def action_builder(scenario_config):
    """Create an ActionBuilder instance."""
    return ActionBuilder(scenario_config)


def test_action_space(action_builder):
    """Test action space creation."""
    space = action_builder.space
    assert space is not None
    assert hasattr(space, 'sample')


def test_decode_action(action_builder):
    """Test action decoding."""
    action = action_builder.space.sample()
    decoded = action_builder.decode(action)
    
    assert isinstance(decoded, list)
    assert len(decoded) == action_builder.n_assets
    for asset_actions in decoded:
        assert isinstance(asset_actions, dict)
        assert 'destination_idx' in asset_actions
        assert 'fuel_pct' in asset_actions
        assert 'priority_pcts' in asset_actions


def test_mask_offsets(action_builder):
    """Test mask offset calculation."""
    offsets = action_builder.mask_offsets_for_asset(0)
    assert 'dest_start' in offsets
    assert 'fuel_start' in offsets
    assert 'priority_starts' in offsets
    
    # Check that offsets are within bounds
    assert 0 <= offsets['dest_start'] < action_builder.mask_size
    assert 0 <= offsets['fuel_start'] < action_builder.mask_size


def test_full_mask(action_builder):
    """Test full mask generation."""
    mask = action_builder.full_mask()
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (action_builder.mask_size,)
    assert mask.all()  # All actions valid in full mask
