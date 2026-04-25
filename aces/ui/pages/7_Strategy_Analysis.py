"""
Strategy Analysis UI for ACES

Interactive visualization and benchmarking tools for comparing
different reinforcement learning strategies in military logistics scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import time
from datetime import datetime

# Import ACES modules
from aces.analytics.strategy_comparator import StrategyComparator, ComparisonResult, BenchmarkResult
from aces.services.training import TrainingService
from aces.domain.environment import LogisticsEnvironment


class StrategyAnalysisUI:
    """
    Interactive UI for strategy analysis and benchmarking.

    Provides comprehensive tools for comparing different algorithms,
    visualizing performance metrics, and generating strategic insights.
    """

    def __init__(self):
        """Initialize the strategy analysis UI."""
        self.strategy_comparator = StrategyComparator()
        self.training_service = TrainingService()
        self.environment = None

    def render_analysis_ui(self):
        """Render the main strategy analysis interface."""
        st.title("🎯 Strategy Analysis & Benchmarking")
        st.markdown("Compare and analyze different reinforcement learning strategies for military logistics")

        # Algorithm selection and configuration
        self._render_algorithm_configuration()

        # Benchmarking controls
        self._render_benchmarking_controls()

        # Results visualization
        self._render_results_visualization()

        # Statistical analysis
        self._render_statistical_analysis()

    def _render_algorithm_configuration(self):
        """Render algorithm selection and configuration panel."""
        st.subheader("Algorithm Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Select Algorithms to Compare")

            available_algorithms = [
                "MaskablePPO", "PPO", "A2C", "DQN", "SAC",
                "TD3", "DDPG", "TRPO", "ARS"
            ]

            selected_algorithms = []
            for algo in available_algorithms:
                if st.checkbox(algo, key=f"algo_{algo}"):
                    selected_algorithms.append(algo)

            self.selected_algorithms = selected_algorithms

            if not selected_algorithms:
                st.warning("Please select at least one algorithm to compare.")

        with col2:
            st.subheader("Training Parameters")

            self.episodes_per_algorithm = st.slider(
                "Episodes per Algorithm",
                min_value=10, max_value=500, value=100, step=10
            )

            self.total_timesteps = st.slider(
                "Total Timesteps",
                min_value=10000, max_value=1000000, value=100000, step=10000
            )

            self.random_seed = st.number_input(
                "Random Seed",
                min_value=0, max_value=9999, value=42
            )

    def _render_benchmarking_controls(self):
        """Render benchmarking execution controls."""
        st.subheader("Benchmarking Execution")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Run Benchmark", type="primary"):
                if not self.selected_algorithms:
                    st.error("Please select algorithms to benchmark.")
                else:
                    self._execute_benchmark()

        with col2:
            if st.button("Load Previous Results"):
                self._load_previous_results()

        with col3:
            if st.button("Clear Results"):
                self._clear_results()

        # Progress tracking
        if hasattr(self, 'benchmark_progress'):
            st.progress(self.benchmark_progress)
            st.text(f"Progress: {self.benchmark_progress * 100:.1f}%")

    def _render_results_visualization(self):
        """Render results visualization section."""
        st.subheader("Results Visualization")

        if not hasattr(self, 'benchmark_results') or not self.benchmark_results:
            st.info("Run a benchmark to see results visualization.")
            return

        benchmark = self.benchmark_results[-1]  # Most recent

        # Performance comparison chart
        st.subheader("Performance Comparison")

        col1, col2 = st.columns(2)

        with col1:
            # Bar chart of strategy rankings
            rankings_data = pd.DataFrame([
                {'Strategy': strat, 'Rank': rank, 'Confidence': benchmark.confidence_levels[strat]}
                for strat, rank in benchmark.rankings.items()
            ]).sort_values('Rank')

            fig = px.bar(
                rankings_data,
                x='Strategy',
                y='Confidence',
                color='Rank',
                title="Strategy Rankings by Confidence",
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Radar chart for multi-dimensional comparison
            categories = ['Performance', 'Stability', 'Efficiency', 'Robustness']

            fig = go.Figure()

            for strategy in benchmark.strategies:
                # Mock multi-dimensional scores
                scores = np.random.uniform(0.6, 0.95, 4)
                fig.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=categories,
                    fill='toself',
                    name=strategy
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Multi-dimensional Strategy Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detailed comparison results
        st.subheader("Detailed Comparison Results")

        if self.strategy_comparator.comparison_history:
            comparison_data = []
            for comp in self.strategy_comparator.comparison_history[-10:]:  # Last 10
                comparison_data.append({
                    'Strategy A': comp.strategy_a,
                    'Strategy B': comp.strategy_b,
                    'Metric': comp.metric,
                    'Mean A': comp.mean_a,
                    'Mean B': comp.mean_b,
                    'P-Value': comp.p_value,
                    'Significant': 'Yes' if comp.significant else 'No',
                    'Effect Size': comp.effect_size
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)

            # Statistical significance visualization
            significant_comparisons = [c for c in self.strategy_comparator.comparison_history if c.significant]
            total_comparisons = len(self.strategy_comparator.comparison_history)

            col1, col2 = st.columns(2)

            with col1:
                # Pie chart of significance
                significance_data = pd.DataFrame({
                    'Result': ['Significant', 'Not Significant'],
                    'Count': [len(significant_comparisons), total_comparisons - len(significant_comparisons)]
                })

                fig = px.pie(
                    significance_data,
                    values='Count',
                    names='Result',
                    title="Statistical Significance Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Effect size distribution
                effect_sizes = [c.effect_size for c in self.strategy_comparator.comparison_history]

                fig = px.histogram(
                    x=effect_sizes,
                    nbins=20,
                    title="Effect Size Distribution",
                    labels={'x': 'Effect Size', 'y': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)

    def _render_statistical_analysis(self):
        """Render statistical analysis section."""
        st.subheader("Statistical Analysis")

        if not self.strategy_comparator.comparison_history:
            st.info("Run comparisons to see statistical analysis.")
            return

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)

        comparisons = self.strategy_comparator.comparison_history
        significant_count = sum(1 for c in comparisons if c.significant)

        with col1:
            st.metric("Total Comparisons", len(comparisons))

        with col2:
            st.metric("Significant Differences", significant_count)

        with col3:
            significance_rate = significant_count / len(comparisons) if comparisons else 0
            st.metric("Significance Rate", f"{significance_rate:.1%}")

        with col4:
            avg_effect_size = np.mean([c.effect_size for c in comparisons])
            st.metric("Avg Effect Size", f"{avg_effect_size:.3f}")

        # Confidence intervals visualization
        st.subheader("Confidence Intervals")

        if comparisons:
            ci_data = []
            for comp in comparisons[-5:]:  # Last 5 comparisons
                ci_data.append({
                    'Comparison': f"{comp.strategy_a} vs {comp.strategy_b}",
                    'Mean_Diff': comp.mean_a - comp.mean_b,
                    'CI_Lower': comp.confidence_interval[0],
                    'CI_Upper': comp.confidence_interval[1],
                    'Significant': comp.significant
                })

            ci_df = pd.DataFrame(ci_data)

            fig = go.Figure()

            for _, row in ci_df.iterrows():
                color = 'red' if row['Significant'] else 'blue'
                fig.add_trace(go.Scatter(
                    x=[row['CI_Lower'], row['CI_Upper']],
                    y=[row['Comparison'], row['Comparison']],
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=row['Comparison'],
                    showlegend=False
                ))

                # Add mean point
                fig.add_trace(go.Scatter(
                    x=[row['Mean_Diff']],
                    y=[row['Comparison']],
                    mode='markers',
                    marker=dict(color=color, size=8),
                    showlegend=False
                ))

            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Confidence Intervals for Strategy Differences",
                xaxis_title="Mean Difference",
                yaxis_title="Comparison",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.subheader("Strategic Recommendations")

        if self.strategy_comparator.benchmark_history:
            benchmark = self.strategy_comparator.benchmark_history[-1]

            recommendations = self.strategy_comparator.get_strategy_recommendations(benchmark)

            st.write(f"**Recommended Strategy:** {recommendations['recommended_strategy']}")

            st.write("**Key Insights:**")
            for insight in recommendations['reasoning']:
                st.write(f"• {insight}")

            if recommendations['risks']:
                st.write("**Risks to Consider:**")
                for risk in recommendations['risks']:
                    st.write(f"• {risk}")

            if recommendations['alternatives']:
                st.write("**Alternative Strategies:**")
                for alt in recommendations['alternatives']:
                    st.write(f"• {alt}")

    def _execute_benchmark(self):
        """Execute benchmarking of selected algorithms."""
        st.info("Starting benchmark execution...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Initialize environment if needed
            if not self.environment:
                self.environment = LogisticsEnvironment()
                status_text.text("Initializing environment...")

            # Prepare results storage
            self.benchmark_results = getattr(self, 'benchmark_results', [])

            # Run comparisons
            total_steps = len(self.selected_algorithms) * (len(self.selected_algorithms) - 1) // 2
            current_step = 0

            results_data = {}

            for i, algo_a in enumerate(self.selected_algorithms):
                status_text.text(f"Training {algo_a}...")
                progress_bar.progress((i + 1) / len(self.selected_algorithms))

                # Mock training results - in real implementation, this would train actual agents
                results_a = np.random.normal(100, 15, self.episodes_per_algorithm)
                results_data[algo_a] = results_a

                # Compare with previously trained algorithms
                for algo_b in self.selected_algorithms[:i]:
                    results_b = results_data[algo_b]

                    comparison = self.strategy_comparator.compare_strategies(
                        algo_a, algo_b, results_a, results_b, "reward"
                    )

                    current_step += 1
                    status_text.text(f"Comparing {algo_a} vs {algo_b}...")

            # Run benchmark
            benchmark = self.strategy_comparator.benchmark_strategies(results_data, "reward")
            self.benchmark_results.append(benchmark)

            progress_bar.progress(1.0)
            status_text.text("Benchmark completed!")

            st.success("Benchmark execution completed successfully!")

        except Exception as e:
            st.error(f"Benchmark execution failed: {str(e)}")
            progress_bar.progress(0)
            status_text.text("Benchmark failed.")

    def _load_previous_results(self):
        """Load previous benchmark results."""
        # Mock loading - in real implementation, this would load from files
        st.info("Loading previous results...")
        time.sleep(1)

        # Mock data
        mock_benchmark = BenchmarkResult(
            strategies=['PPO', 'A2C', 'DQN'],
            metrics={
                'PPO': np.random.normal(95, 10, 50),
                'A2C': np.random.normal(87, 12, 50),
                'DQN': np.random.normal(78, 15, 50)
            },
            rankings={'PPO': 1, 'A2C': 2, 'DQN': 3},
            best_strategy='PPO',
            confidence_levels={'PPO': 0.85, 'A2C': 0.72, 'DQN': 0.65},
            timestamp=datetime.now()
        )

        self.benchmark_results = [mock_benchmark]
        st.success("Previous results loaded!")

    def _clear_results(self):
        """Clear all results."""
        if hasattr(self, 'benchmark_results'):
            delattr(self, 'benchmark_results')
        self.strategy_comparator.comparison_history.clear()
        self.strategy_comparator.benchmark_history.clear()
        st.success("Results cleared!")


def main():
    """Main function to run the strategy analysis UI."""
    analysis_ui = StrategyAnalysisUI()
    analysis_ui.render_analysis_ui()


if __name__ == "__main__":
    main()