"""
Scenario Variant Generator for ACES

Generates scenario variants through parameter sweeps, Monte Carlo simulations,
and what-if analysis for comprehensive military logistics planning and decision support.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import itertools
from scipy import stats
import random

from aces.domain.environment import LogisticsEnvironment
from aces.domain.asset import Asset
from aces.domain.threat import Threat
from aces.services.scenario import ScenarioService


@dataclass
class ParameterRange:
    """Defines a parameter range for variation."""
    name: str
    min_value: float
    max_value: float
    distribution: str = 'uniform'  # 'uniform', 'normal', 'lognormal'
    mean: Optional[float] = None
    std: Optional[float] = None
    steps: Optional[int] = None  # For discrete sweeps


@dataclass
class ScenarioVariant:
    """Represents a single scenario variant."""
    id: str
    base_scenario: str
    parameters: Dict[str, Any]
    description: str
    expected_difficulty: str
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    variant_id: str
    runs: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    percentile_5: float
    percentile_95: float
    success_rate: float  # Percentage of runs above threshold
    confidence_interval: Tuple[float, float]
    distribution_stats: Dict[str, float]


class ScenarioVariantGenerator:
    """
    Generates scenario variants for comprehensive analysis and decision support.

    Supports parameter sweeps, Monte Carlo simulations, and what-if analysis
    to explore different military logistics scenarios and their outcomes.
    """

    def __init__(self, scenario_service: Optional[ScenarioService] = None):
        """
        Initialize the variant generator.

        Args:
            scenario_service: Service for creating and managing scenarios
        """
        self.scenario_service = scenario_service
        self.variants: Dict[str, ScenarioVariant] = {}
        self.parameter_ranges: Dict[str, ParameterRange] = {}
        self.monte_carlo_results: Dict[str, MonteCarloResult] = {}

        # Default parameter ranges for common logistics factors
        self._initialize_default_ranges()

    def _initialize_default_ranges(self):
        """Initialize default parameter ranges for common scenario factors."""
        self.parameter_ranges.update({
            'num_assets': ParameterRange('num_assets', 5, 20, 'uniform', steps=16),
            'num_threats': ParameterRange('num_threats', 2, 15, 'uniform', steps=14),
            'asset_speed': ParameterRange('asset_speed', 0.5, 2.0, 'normal', mean=1.0, std=0.2),
            'threat_severity': ParameterRange('threat_severity', 0.1, 1.0, 'uniform'),
            'resource_availability': ParameterRange('resource_availability', 0.3, 1.0, 'uniform'),
            'time_pressure': ParameterRange('time_pressure', 0.5, 2.0, 'lognormal', mean=1.0, std=0.3),
            'communication_reliability': ParameterRange('communication_reliability', 0.7, 1.0, 'uniform'),
            'weather_impact': ParameterRange('weather_impact', 0.0, 0.8, 'uniform')
        })

    def add_parameter_range(self, param_range: ParameterRange):
        """
        Add a custom parameter range for variation.

        Args:
            param_range: Parameter range definition
        """
        self.parameter_ranges[param_range.name] = param_range

    def generate_parameter_sweep(
        self,
        base_scenario: str,
        parameters_to_vary: List[str],
        steps_per_parameter: Optional[Dict[str, int]] = None
    ) -> List[ScenarioVariant]:
        """
        Generate scenario variants through parameter sweep.

        Args:
            base_scenario: Name of the base scenario
            parameters_to_vary: List of parameter names to vary
            steps_per_parameter: Optional custom steps per parameter

        Returns:
            List of generated scenario variants
        """
        variants = []

        # Get parameter values for sweep
        parameter_values = {}
        for param_name in parameters_to_vary:
            if param_name not in self.parameter_ranges:
                raise ValueError(f"Parameter {param_name} not defined")

            param_range = self.parameter_ranges[param_name]
            steps = steps_per_parameter.get(param_name, param_range.steps or 5)

            if param_range.distribution == 'uniform':
                values = np.linspace(param_range.min_value, param_range.max_value, steps)
            elif param_range.distribution == 'normal':
                # Generate values around mean with given std
                values = np.linspace(
                    param_range.mean - 2*param_range.std,
                    param_range.mean + 2*param_range.std,
                    steps
                )
            else:
                values = np.linspace(param_range.min_value, param_range.max_value, steps)

            parameter_values[param_name] = values

        # Generate all combinations
        param_names = list(parameter_values.keys())
        param_combinations = list(itertools.product(*[parameter_values[name] for name in param_names]))

        for i, combination in enumerate(param_combinations):
            variant_params = dict(zip(param_names, combination))

            # Create variant description
            param_str = ", ".join([f"{k}={v:.2f}" for k, v in variant_params.items()])
            description = f"Parameter sweep variant: {param_str}"

            # Estimate difficulty based on parameters
            difficulty = self._estimate_difficulty(variant_params)

            variant = ScenarioVariant(
                id=f"{base_scenario}_sweep_{i:03d}",
                base_scenario=base_scenario,
                parameters=variant_params,
                description=description,
                expected_difficulty=difficulty,
                tags=['parameter_sweep', base_scenario]
            )

            variants.append(variant)
            self.variants[variant.id] = variant

        return variants

    def generate_monte_carlo_variants(
        self,
        base_scenario: str,
        num_variants: int,
        parameters_to_vary: List[str],
        seed: Optional[int] = None
    ) -> List[ScenarioVariant]:
        """
        Generate scenario variants using Monte Carlo sampling.

        Args:
            base_scenario: Name of the base scenario
            num_variants: Number of variants to generate
            parameters_to_vary: List of parameter names to vary
            seed: Random seed for reproducibility

        Returns:
            List of generated scenario variants
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        variants = []

        for i in range(num_variants):
            variant_params = {}

            for param_name in parameters_to_vary:
                if param_name not in self.parameter_ranges:
                    raise ValueError(f"Parameter {param_name} not defined")

                param_range = self.parameter_ranges[param_name]

                if param_range.distribution == 'uniform':
                    value = np.random.uniform(param_range.min_value, param_range.max_value)
                elif param_range.distribution == 'normal':
                    value = np.random.normal(param_range.mean, param_range.std)
                    # Clamp to reasonable bounds
                    value = np.clip(value, param_range.min_value, param_range.max_value)
                elif param_range.distribution == 'lognormal':
                    # Generate lognormal from normal parameters
                    normal_value = np.random.normal(param_range.mean, param_range.std)
                    value = np.exp(normal_value)
                    value = np.clip(value, param_range.min_value, param_range.max_value)
                else:
                    value = np.random.uniform(param_range.min_value, param_range.max_value)

                variant_params[param_name] = float(value)

            # Create variant description
            param_str = ", ".join([f"{k}={v:.2f}" for k, v in variant_params.items()])
            description = f"Monte Carlo variant: {param_str}"

            difficulty = self._estimate_difficulty(variant_params)

            variant = ScenarioVariant(
                id=f"{base_scenario}_mc_{i:03d}",
                base_scenario=base_scenario,
                parameters=variant_params,
                description=description,
                expected_difficulty=difficulty,
                tags=['monte_carlo', base_scenario]
            )

            variants.append(variant)
            self.variants[variant.id] = variant

        return variants

    def _estimate_difficulty(self, parameters: Dict[str, Any]) -> str:
        """Estimate scenario difficulty based on parameters."""
        difficulty_score = 0

        # Factors that increase difficulty
        if 'num_threats' in parameters:
            difficulty_score += parameters['num_threats'] * 0.1
        if 'threat_severity' in parameters:
            difficulty_score += parameters['threat_severity'] * 2
        if 'time_pressure' in parameters:
            difficulty_score += (parameters['time_pressure'] - 1) * 1.5
        if 'communication_reliability' in parameters:
            difficulty_score += (1 - parameters['communication_reliability']) * 2
        if 'weather_impact' in parameters:
            difficulty_score += parameters['weather_impact'] * 1.5
        if 'resource_availability' in parameters:
            difficulty_score += (1 - parameters['resource_availability']) * 1.5

        # Factors that decrease difficulty
        if 'num_assets' in parameters:
            difficulty_score -= parameters['num_assets'] * 0.05
        if 'asset_speed' in parameters:
            difficulty_score -= (parameters['asset_speed'] - 1) * 0.5

        if difficulty_score < 1:
            return "Easy"
        elif difficulty_score < 2.5:
            return "Medium"
        elif difficulty_score < 4:
            return "Hard"
        else:
            return "Very Hard"

    def run_monte_carlo_simulation(
        self,
        variant: ScenarioVariant,
        evaluation_function: Callable[[Dict[str, Any]], float],
        num_runs: int = 100,
        success_threshold: float = 0.0,
        seed: Optional[int] = None
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for a scenario variant.

        Args:
            variant: Scenario variant to simulate
            evaluation_function: Function that evaluates the scenario and returns a reward
            num_runs: Number of simulation runs
            success_threshold: Threshold for considering a run successful
            seed: Random seed for reproducibility

        Returns:
            Monte Carlo simulation results
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        rewards = []

        for _ in range(num_runs):
            # Add some stochasticity to parameters for each run
            run_params = self._add_stochasticity(variant.parameters)
            reward = evaluation_function(run_params)
            rewards.append(reward)

        rewards = np.array(rewards)

        # Calculate statistics
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        min_reward = float(np.min(rewards))
        max_reward = float(np.max(rewards))
        percentile_5 = float(np.percentile(rewards, 5))
        percentile_95 = float(np.percentile(rewards, 95))
        success_rate = float(np.mean(rewards > success_threshold))

        # 95% confidence interval for the mean
        confidence_interval = stats.t.interval(
            0.95, len(rewards)-1, loc=mean_reward, scale=stats.sem(rewards)
        )

        # Distribution statistics
        distribution_stats = {
            'skewness': float(stats.skew(rewards)),
            'kurtosis': float(stats.kurtosis(rewards)),
            'shapiro_p_value': float(stats.shapiro(rewards).pvalue)
        }

        result = MonteCarloResult(
            variant_id=variant.id,
            runs=num_runs,
            mean_reward=mean_reward,
            std_reward=std_reward,
            min_reward=min_reward,
            max_reward=max_reward,
            percentile_5=percentile_5,
            percentile_95=percentile_95,
            success_rate=success_rate,
            confidence_interval=(float(confidence_interval[0]), float(confidence_interval[1])),
            distribution_stats=distribution_stats
        )

        self.monte_carlo_results[variant.id] = result
        return result

    def _add_stochasticity(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add random variation to parameters for Monte Carlo runs."""
        stochastic_params = parameters.copy()

        for param_name, value in parameters.items():
            if param_name in self.parameter_ranges:
                param_range = self.parameter_ranges[param_name]
                # Add 5-10% random variation
                variation = np.random.normal(0, 0.05 * abs(value))
                new_value = value + variation

                # Clamp to parameter bounds
                new_value = np.clip(new_value, param_range.min_value, param_range.max_value)
                stochastic_params[param_name] = float(new_value)

        return stochastic_params

    def generate_what_if_analysis(
        self,
        base_scenario: str,
        what_if_questions: List[Dict[str, Any]]
    ) -> List[ScenarioVariant]:
        """
        Generate variants for what-if analysis questions.

        Args:
            base_scenario: Name of the base scenario
            what_if_questions: List of what-if scenarios with parameter changes

        Returns:
            List of what-if scenario variants
        """
        variants = []

        for i, question in enumerate(what_if_questions):
            variant_params = question.get('parameter_changes', {})
            description = question.get('description', f'What-if scenario {i+1}')
            tags = question.get('tags', ['what_if', base_scenario])

            difficulty = self._estimate_difficulty(variant_params)

            variant = ScenarioVariant(
                id=f"{base_scenario}_whatif_{i:03d}",
                base_scenario=base_scenario,
                parameters=variant_params,
                description=description,
                expected_difficulty=difficulty,
                tags=tags
            )

            variants.append(variant)
            self.variants[variant.id] = variant

        return variants

    def export_variants(self, output_path: Path, format: str = 'json'):
        """
        Export generated variants to file.

        Args:
            output_path: Path to save the export
            format: Export format ('json', 'csv')
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            data = {
                'variants': [vars(v) for v in self.variants.values()],
                'monte_carlo_results': [vars(r) for r in self.monte_carlo_results.values()],
                'parameter_ranges': [vars(p) for p in self.parameter_ranges.values()],
                'exported_at': datetime.now().isoformat()
            }
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        elif format == 'csv':
            # Export variants as CSV
            variant_data = []
            for variant in self.variants.values():
                row = {
                    'id': variant.id,
                    'base_scenario': variant.base_scenario,
                    'description': variant.description,
                    'expected_difficulty': variant.expected_difficulty,
                    'tags': ','.join(variant.tags),
                    'created_at': variant.created_at.isoformat()
                }
                row.update(variant.parameters)
                variant_data.append(row)

            df = pd.DataFrame(variant_data)
            df.to_csv(output_path, index=False)

    def get_variant_summary(self) -> Dict[str, Any]:
        """Get summary statistics of generated variants."""
        if not self.variants:
            return {'total_variants': 0}

        difficulties = [v.expected_difficulty for v in self.variants.values()]
        tags = [tag for v in self.variants.values() for tag in v.tags]

        difficulty_counts = pd.Series(difficulties).value_counts().to_dict()
        tag_counts = pd.Series(tags).value_counts().to_dict()

        base_scenarios = list(set(v.base_scenario for v in self.variants.values()))

        return {
            'total_variants': len(self.variants),
            'base_scenarios': base_scenarios,
            'difficulty_distribution': difficulty_counts,
            'tag_distribution': tag_counts,
            'monte_carlo_simulations': len(self.monte_carlo_results)
        }

    def find_similar_variants(
        self,
        target_parameters: Dict[str, Any],
        max_distance: float = 0.1
    ) -> List[Tuple[ScenarioVariant, float]]:
        """
        Find variants similar to target parameters.

        Args:
            target_parameters: Target parameter values
            max_distance: Maximum normalized distance to consider similar

        Returns:
            List of (variant, distance) tuples for similar variants
        """
        similar = []

        for variant in self.variants.values():
            distance = self._calculate_parameter_distance(variant.parameters, target_parameters)
            if distance <= max_distance:
                similar.append((variant, distance))

        # Sort by distance
        similar.sort(key=lambda x: x[1])
        return similar

    def _calculate_parameter_distance(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any]
    ) -> float:
        """Calculate normalized Euclidean distance between parameter sets."""
        all_keys = set(params1.keys()) | set(params2.keys())
        distance = 0
        count = 0

        for key in all_keys:
            if key in params1 and key in params2 and key in self.parameter_ranges:
                param_range = self.parameter_ranges[key]
                range_size = param_range.max_value - param_range.min_value

                if range_size > 0:
                    normalized_diff = abs(params1[key] - params2[key]) / range_size
                    distance += normalized_diff ** 2
                    count += 1

        return np.sqrt(distance / count) if count > 0 else 1.0