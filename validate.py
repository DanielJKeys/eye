#!/usr/bin/env python3
"""Validation script to test EYE environment functionality."""

import sys
import numpy as np

def main():
    print("Validating EYE Platform...")
    
    try:
        # Test imports
        print("Testing imports...")
        from eye.services.scenario import ScenarioService
        from eye.domain.environment import LogisticsEnv
        from eye.config import ScenarioConfig
        print("✓ Imports successful")
        
        # Test scenario service
        print("Testing scenario service...")
        svc = ScenarioService()
        scenarios = svc.list_scenarios()
        if not scenarios:
            print("✗ No scenarios found")
            return 1
        print(f"✓ Found {len(scenarios)} scenarios")
        
        # Test scenario config loading
        print("Testing scenario config...")
        scenario_id = scenarios[0]['id']
        config = svc.get_scenario_config(scenario_id)
        if not isinstance(config, ScenarioConfig):
            print("✗ Invalid scenario config")
            return 1
        print(f"✓ Loaded scenario: {config.name}")
        
        # Test environment creation
        print("Testing environment creation...")
        env = LogisticsEnv(config)
        print("✓ Environment created")
        
        # Test reset
        print("Testing environment reset...")
        obs, info = env.reset(seed=42)
        expected_shape = (env._obs_builder.obs_dim,)
        if not isinstance(obs, np.ndarray) or obs.shape != expected_shape:
            print(f"✗ Invalid observation shape: {obs.shape} != {expected_shape}")
            return 1
        print(f"✓ Reset successful, obs shape: {obs.shape}")
        
        # Test step
        print("Testing environment step...")
        action = env.action_space.sample()
        new_obs, reward, terminated, truncated, new_info = env.step(action)
        if not isinstance(new_obs, np.ndarray) or new_obs.shape != obs.shape:
            print("✗ Invalid new observation")
            return 1
        print(f"✓ Step successful, reward: {reward:.2f}")
        
        # Test action masks
        print("Testing action masks...")
        masks = env.action_masks()
        expected_mask_shape = (env._action_builder.mask_size,)
        if not isinstance(masks, np.ndarray) or masks.shape != expected_mask_shape:
            print(f"✗ Invalid mask shape: {masks.shape} != {expected_mask_shape}")
            return 1
        print(f"✓ Masks generated, shape: {masks.shape}")
        
        # Test render
        print("Testing render...")
        env.render()
        print("✓ Render successful")
        
        print("\n🎉 All validations passed! EYE is operational.")
        return 0
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
