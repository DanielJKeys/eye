"""
Strategy Comparator Module for ACES

This module provides comparative analysis capabilities for different reinforcement learning
algorithms and strategies in military logistics scenarios. It enables head-to-head
comparisons, statistical significance testing, and performance benchmarking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

from aces.domain.environment import LogisticsEnv
from aces.services.training import TrainingService
from aces.analytics.audit_logger import AuditLogger


@dataclass
class ComparisonResult:
    """Results from comparing two strategies."""
    strategy_a: str
    strategy_b: str
    metric: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    t_statistic: float
    p_value: float
    significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    sample_size_a: int
    sample_size_b: int
    timestamp: datetime


@dataclass
class BenchmarkResult:
    """Benchmark results for multiple strategies."""
    strategies: List[str]
    metrics: Dict[str, List[float]]
    rankings: Dict[str, int]
    best_strategy: str
    confidence_levels: Dict[str, float]
    timestamp: datetime


class StrategyComparator:
    """
    Compares different RL algorithms and strategies for military logistics scenarios.

    Provides statistical analysis, performance benchmarking, and comparative
    visualization capabilities for decision support in war games.
    """

    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        """
        Initialize the Strategy Comparator.

        Args:
            audit_logger: Optional audit logger for tracking comparisons
        """
        self.audit_logger = audit_logger
        self.comparison_history: List[ComparisonResult] = []
        self.benchmark_history: List[BenchmarkResult] = []

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def compare_strategies(
        self,
        strategy_a: str,
        strategy_b: str,
        results_a: List[float],
        results_b: List[float],
        metric_name: str = "reward",
        alpha: float = 0.05
    ) -> ComparisonResult:
        """
        Perform statistical comparison between two strategies.

        Args:
            strategy_a: Name of first strategy
            strategy_b: Name of second strategy
            results_a: Performance results for strategy A
            results_b: Performance results for strategy B
            metric_name: Name of the metric being compared
            alpha: Significance level for statistical tests

        Returns:
            ComparisonResult with statistical analysis
        """
        # Calculate basic statistics
        mean_a, std_a = np.mean(results_a), np.std(results_a, ddof=1)
        mean_b, std_b = np.mean(results_b), np.std(results_b, ddof=1)

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(results_a, results_b, equal_var=False)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        effect_size = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

        # Calculate confidence interval for difference
        diff_mean = mean_a - mean_b
        se_diff = np.sqrt(std_a**2/len(results_a) + std_b**2/len(results_b))
        t_critical = stats.t.ppf(1 - alpha/2, len(results_a) + len(results_b) - 2)
        ci_lower = diff_mean - t_critical * se_diff
        ci_upper = diff_mean + t_critical * se_diff

        # Determine significance
        significant = p_value < alpha

        result = ComparisonResult(
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            metric=metric_name,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            t_statistic=t_stat,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            sample_size_a=len(results_a),
            sample_size_b=len(results_b),
            timestamp=datetime.now()
        )

        self.comparison_history.append(result)

        # Log to audit if available
        if self.audit_logger:
            self.audit_logger.log_decision(
                decision_type="strategy_comparison",
                context={"comparison": result.__dict__},
                confidence=1.0 - p_value,
                alternatives=[strategy_a, strategy_b],
                reasoning=f"Statistical comparison of {strategy_a} vs {strategy_b} on {metric_name}"
            )

        return result

    def benchmark_strategies(
        self,
        strategy_results: Dict[str, List[float]],
        metric_name: str = "reward"
    ) -> BenchmarkResult:
        """
        Benchmark multiple strategies against each other.

        Args:
            strategy_results: Dictionary mapping strategy names to result lists
            metric_name: Name of the metric being benchmarked

        Returns:
            BenchmarkResult with rankings and statistics
        """
        strategies = list(strategy_results.keys())
        metrics = strategy_results.copy()

        # Calculate rankings based on mean performance
        means = {strat: np.mean(results) for strat, results in strategy_results.items()}
        rankings = {strat: rank for rank, (strat, _) in
                   enumerate(sorted(means.items(), key=lambda x: x[1], reverse=True), 1)}

        best_strategy = max(means, key=means.get)

        # Calculate confidence levels (simplified approach)
        confidence_levels = {}
        for strat in strategies:
            # Use coefficient of variation as inverse confidence measure
            cv = np.std(strategy_results[strat]) / abs(np.mean(strategy_results[strat]))
            confidence_levels[strat] = 1.0 / (1.0 + cv)  # Higher CV = lower confidence

        result = BenchmarkResult(
            strategies=strategies,
            metrics=metrics,
            rankings=rankings,
            best_strategy=best_strategy,
            confidence_levels=confidence_levels,
            timestamp=datetime.now()
        )

        self.benchmark_history.append(result)
        return result

    def run_algorithm_comparison(
        self,
        algorithms: List[str],
        environment: LogisticsEnv,
        training_service: TrainingService,
        episodes_per_algorithm: int = 100,
        seeds: Optional[List[int]] = None
    ) -> Dict[str, List[float]]:
        """
        Run multiple algorithms on the same environment and collect results.

        Args:
            algorithms: List of algorithm names to compare
            environment: The logistics environment to train on
            training_service: Service for training agents
            episodes_per_algorithm: Number of episodes per algorithm
            seeds: Random seeds for reproducibility

        Returns:
            Dictionary mapping algorithm names to reward lists
        """
        if seeds is None:
            seeds = [42 + i for i in range(len(algorithms))]

        results = {}

        for i, algorithm in enumerate(algorithms):
            print(f"Training {algorithm}...")

            # Configure training with specific algorithm
            config = {
                'algorithm': algorithm,
                'total_timesteps': episodes_per_algorithm * 1000,  # Rough estimate
                'seed': seeds[i]
            }

            # Train the agent
            rewards = training_service.train_agent(
                environment=environment,
                config=config,
                episodes=episodes_per_algorithm
            )

            results[algorithm] = rewards

        return results

    def generate_comparison_report(
        self,
        comparison_results: List[ComparisonResult],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a detailed comparison report.

        Args:
            comparison_results: List of comparison results
            output_path: Optional path to save the report

        Returns:
            Formatted report as string
        """
        report_lines = []
        report_lines.append("# Strategy Comparison Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        for result in comparison_results:
            report_lines.append(f"## {result.strategy_a} vs {result.strategy_b}")
            report_lines.append(f"**Metric:** {result.metric}")
            report_lines.append(f"**Sample Sizes:** {result.strategy_a}={result.sample_size_a}, "
                              f"{result.strategy_b}={result.sample_size_b}")
            report_lines.append(f"**Means:** {result.strategy_a}={result.mean_a:.3f}, "
                              f"{result.strategy_b}={result.mean_b:.3f}")
            report_lines.append(f"**Standard Deviations:** {result.strategy_a}={result.std_a:.3f}, "
                              f"{result.strategy_b}={result.std_b:.3f}")
            report_lines.append(f"**T-statistic:** {result.t_statistic:.3f}")
            report_lines.append(f"**P-value:** {result.p_value:.4f}")
            report_lines.append(f"**Significant:** {'Yes' if result.significant else 'No'}")
            report_lines.append(f"**Effect Size:** {result.effect_size:.3f}")
            report_lines.append(f"**95% CI for Difference:** ({result.confidence_interval[0]:.3f}, "
                              f"{result.confidence_interval[1]:.3f})")
            report_lines.append("")

        report = "\n".join(report_lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)

        return report

    def plot_comparison(
        self,
        comparison_result: ComparisonResult,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create a visualization of the comparison results.

        Args:
            comparison_result: Result to visualize
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Bar plot of means with error bars
        strategies = [comparison_result.strategy_a, comparison_result.strategy_b]
        means = [comparison_result.mean_a, comparison_result.mean_b]
        stds = [comparison_result.std_a, comparison_result.std_b]

        bars = ax1.bar(strategies, means, yerr=stds, capsize=5,
                      color=['skyblue', 'lightcoral'], alpha=0.7)
        ax1.set_ylabel(comparison_result.metric.capitalize())
        ax1.set_title('Mean Performance Comparison')
        ax1.grid(True, alpha=0.3)

        # Add significance annotation
        if comparison_result.significant:
            max_y = max(means) + max(stds)
            ax1.annotate('*', xy=(0.5, max_y * 1.05), ha='center',
                        fontsize=20, color='red')

        # Effect size plot
        ax2.bar(['Effect Size'], [comparison_result.effect_size],
               color='lightgreen', alpha=0.7)
        ax2.set_title('Effect Size (Cohen\'s d)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def export_comparison_history(self, output_path: Path) -> None:
        """
        Export all comparison history to JSON.

        Args:
            output_path: Path to save the export
        """
        data = {
            'comparisons': [vars(result) for result in self.comparison_history],
            'benchmarks': [vars(result) for result in self.benchmark_history]
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def get_strategy_recommendations(
        self,
        benchmark_result: BenchmarkResult,
        scenario_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate strategic recommendations based on benchmark results.

        Args:
            benchmark_result: Results from benchmarking
            scenario_context: Optional context about the scenario

        Returns:
            Dictionary with recommendations and reasoning
        """
        recommendations = {
            'recommended_strategy': benchmark_result.best_strategy,
            'ranking': benchmark_result.rankings,
            'confidence_assessment': benchmark_result.confidence_levels,
            'reasoning': [],
            'risks': [],
            'alternatives': []
        }

        # Generate reasoning based on rankings
        best_strategy = benchmark_result.best_strategy
        best_confidence = benchmark_result.confidence_levels[best_strategy]

        recommendations['reasoning'].append(
            f"{best_strategy} shows the highest mean performance across all tested strategies."
        )

        if best_confidence > 0.8:
            recommendations['reasoning'].append(
                f"High confidence ({best_confidence:.2f}) in {best_strategy}'s superiority."
            )
        elif best_confidence > 0.6:
            recommendations['reasoning'].append(
                f"Moderate confidence ({best_confidence:.2f}) in {best_strategy}'s performance."
            )
        else:
            recommendations['reasoning'].append(
                f"Low confidence ({best_confidence:.2f}) - consider additional testing."
            )
            recommendations['risks'].append(
                "Performance variability suggests inconsistent results."
            )

        # Identify close alternatives
        sorted_strategies = sorted(
            benchmark_result.rankings.items(),
            key=lambda x: x[1]
        )
        if len(sorted_strategies) > 1:
            second_best = sorted_strategies[1][0]
            recommendations['alternatives'].append(second_best)
            recommendations['reasoning'].append(
                f"{second_best} is the next best alternative if {best_strategy} encounters issues."
            )

        return recommendations