"""
Decision Support Dashboard for ACES

Real-time dashboard providing decision support with recommendations,
risk alerts, and comprehensive analytics for military logistics scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

# Import EYE modules
from eye.analytics.decision_analyzer import DecisionAnalyzer
from eye.analytics.audit_logger import AuditLogger
from eye.analytics.strategy_comparator import StrategyComparator, ComparisonResult, BenchmarkResult
from eye.reporting.report_generator import ReportGenerator
from eye.optimization.resource_optimizer import ResourceOptimizer
from eye.threat_modeling.threat_model import ThreatModel, IntelligenceReport
from eye.multiagent.multiagent_scenario import MultiAgentScenario
from eye.domain.environment import LogisticsEnvironment


class DecisionSupportDashboard:
    """
    Real-time decision support dashboard for military logistics operations.

    Provides comprehensive analytics, risk assessment, and strategic recommendations
    through an interactive Streamlit interface.
    """

    def __init__(self):
        """Initialize the dashboard."""
        self.decision_analyzer = DecisionAnalyzer()
        self.audit_logger = AuditLogger()
        self.strategy_comparator = StrategyComparator(self.audit_logger)
        self.report_generator = ReportGenerator()
        self.resource_optimizer = ResourceOptimizer()
        self.threat_model = ThreatModel()
        self.multiagent_scenario = None

        # Dashboard state
        self.current_scenario = None
        self.last_update = datetime.now()

    def render_dashboard(self):
        """Render the main dashboard interface."""
        st.title("🛡️ ACES Decision Support Dashboard")
        st.markdown("Real-time analytics and strategic recommendations for military logistics operations")

        # Sidebar controls
        self._render_sidebar()

        # Main dashboard content
        self._render_main_content()

    def _render_sidebar(self):
        """Render sidebar with controls and settings."""
        st.sidebar.title("Dashboard Controls")

        # Scenario selection
        st.sidebar.subheader("Scenario Management")
        scenario_options = ["Load Scenario", "Create New Scenario", "Import from File"]
        selected_scenario_action = st.sidebar.selectbox("Scenario Action", scenario_options)

        if selected_scenario_action == "Load Scenario":
            scenario_id = st.sidebar.text_input("Scenario ID")
            if st.sidebar.button("Load"):
                self._load_scenario(scenario_id)

        # Real-time updates toggle
        st.sidebar.subheader("Real-time Updates")
        self.auto_refresh = st.sidebar.checkbox("Auto-refresh every 30 seconds", value=True)

        # Risk threshold settings
        st.sidebar.subheader("Risk Settings")
        self.risk_threshold = st.sidebar.slider("Risk Alert Threshold", 0.0, 1.0, 0.7, 0.1)

        # Export options
        st.sidebar.subheader("Export & Reports")
        if st.sidebar.button("Generate Executive Summary"):
            self._generate_executive_summary()

        if st.sidebar.button("Export Analytics Data"):
            self._export_analytics_data()

    def _render_main_content(self):
        """Render the main dashboard content."""
        # Key metrics overview
        self._render_key_metrics()

        # Real-time alerts and recommendations
        self._render_alerts_recommendations()

        # Analytics tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Strategic Analysis", "Resource Optimization",
            "Threat Intelligence", "Multi-Agent Dynamics", "Performance Monitoring"
        ])

        with tab1:
            self._render_strategic_analysis()

        with tab2:
            self._render_resource_optimization()

        with tab3:
            self._render_threat_intelligence()

        with tab4:
            self._render_multiagent_dynamics()

        with tab5:
            self._render_performance_monitoring()

    def _render_key_metrics(self):
        """Render key performance metrics."""
        st.subheader("Key Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        # Mock metrics - in real implementation, these would come from actual data
        with col1:
            st.metric(
                label="Mission Success Rate",
                value="87%",
                delta="+5%",
                help="Percentage of objectives completed successfully"
            )

        with col2:
            st.metric(
                label="Resource Efficiency",
                value="92%",
                delta="-2%",
                help="Efficiency of resource utilization"
            )

        with col3:
            st.metric(
                label="Risk Level",
                value="Medium",
                delta="↔️",
                help="Current operational risk assessment"
            )

        with col4:
            st.metric(
                label="Communication Reliability",
                value="94%",
                delta="+3%",
                help="Success rate of inter-agent communications"
            )

    def _render_alerts_recommendations(self):
        """Render real-time alerts and strategic recommendations."""
        st.subheader("Real-time Alerts & Recommendations")

        # Risk alerts
        risk_alerts = self._get_risk_alerts()
        if risk_alerts:
            for alert in risk_alerts:
                if alert['severity'] == 'high':
                    st.error(f"🚨 {alert['message']}")
                elif alert['severity'] == 'medium':
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")

        # Strategic recommendations
        recommendations = self._get_strategic_recommendations()
        if recommendations:
            st.success("💡 Strategic Recommendations:")
            for rec in recommendations:
                st.write(f"• {rec}")

    def _render_strategic_analysis(self):
        """Render strategic analysis tab."""
        st.subheader("Strategic Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Strategy Comparison")
            if self.strategy_comparator.comparison_history:
                # Display recent comparisons
                comparisons_df = pd.DataFrame([
                    {
                        'Strategy A': comp.strategy_a,
                        'Strategy B': comp.strategy_b,
                        'Metric': comp.metric,
                        'P-Value': comp.p_value,
                        'Significant': comp.significant,
                        'Effect Size': comp.effect_size
                    }
                    for comp in self.strategy_comparator.comparison_history[-5:]
                ])
                st.dataframe(comparisons_df)

                # Visualization
                fig = self._create_comparison_visualization()
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strategy comparisons available yet.")

        with col2:
            st.subheader("Benchmark Results")
            if self.strategy_comparator.benchmark_history:
                benchmark = self.strategy_comparator.benchmark_history[-1]
                st.write(f"**Best Strategy:** {benchmark.best_strategy}")

                # Rankings
                rankings_df = pd.DataFrame([
                    {'Strategy': strat, 'Rank': rank, 'Confidence': benchmark.confidence_levels[strat]}
                    for strat, rank in benchmark.rankings.items()
                ]).sort_values('Rank')
                st.dataframe(rankings_df)
            else:
                st.info("No benchmark results available yet.")

    def _render_resource_optimization(self):
        """Render resource optimization tab."""
        st.subheader("Resource Optimization")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Supply Chain Optimization")
            # Mock supply chain data
            supply_data = pd.DataFrame({
                'Node': ['Base A', 'Base B', 'Forward Operating Base', 'Supply Depot'],
                'Supply': [1000, 800, 200, 1500],
                'Demand': [300, 500, 800, 200]
            })

            fig = px.bar(supply_data, x='Node', y=['Supply', 'Demand'],
                        title="Supply vs Demand Analysis",
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Asset Deployment Plan")
            # Mock deployment visualization
            deployment_data = pd.DataFrame({
                'Asset': ['Truck 1', 'Truck 2', 'Helicopter 1', 'Drone 1'],
                'Status': ['Active', 'Active', 'Maintenance', 'Active'],
                'Efficiency': [95, 87, 0, 92]
            })

            fig = px.bar(deployment_data, x='Asset', y='Efficiency',
                        color='Status', title="Asset Efficiency")
            st.plotly_chart(fig, use_container_width=True)

        # Optimization controls
        st.subheader("Optimization Controls")
        if st.button("Run Supply Optimization"):
            with st.spinner("Optimizing supply allocation..."):
                time.sleep(2)  # Mock computation
                st.success("Optimization complete! Results displayed above.")

    def _render_threat_intelligence(self):
        """Render threat intelligence tab."""
        st.subheader("Threat Intelligence")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Active Threat Zones")
            # Mock threat zone data
            threat_data = pd.DataFrame({
                'Zone ID': ['TZ-001', 'TZ-002', 'TZ-003'],
                'Threat Level': [0.8, 0.6, 0.3],
                'Type': ['Enemy Force', 'Minefield', 'Air Defense'],
                'Detection Prob': [0.9, 0.7, 0.5]
            })

            fig = px.scatter(threat_data, x='Zone ID', y='Threat Level',
                           size='Detection Prob', color='Type',
                           title="Threat Zone Analysis")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Intelligence Reports")
            # Mock intelligence data
            intel_data = pd.DataFrame({
                'Report ID': ['INT-001', 'INT-002'],
                'Confidence': [0.85, 0.72],
                'Threats Identified': [3, 2],
                'Recommendations': ['Increase patrols', 'Deploy countermeasures']
            })

            st.dataframe(intel_data)

            # Counter-intelligence measures
            st.subheader("Counter-Intelligence Measures")
            measures = ['Signal Jamming', 'Deception Ops', 'Stealth Routing']
            for measure in measures:
                if st.checkbox(measure, key=f"ci_{measure}"):
                    st.success(f"{measure} activated!")

    def _render_multiagent_dynamics(self):
        """Render multi-agent dynamics tab."""
        st.subheader("Multi-Agent Dynamics")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Team Performance")
            # Mock team performance data
            team_data = pd.DataFrame({
                'Team': ['Friendly Forces', 'Adversary Forces'],
                'Score': [1250, 980],
                'Objectives Completed': [8, 3],
                'Communication Success': [94, 76]
            })

            fig = px.bar(team_data, x='Team', y='Score',
                        color='Objectives Completed',
                        title="Team Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Communication Network")
            # Mock communication network visualization
            comm_data = pd.DataFrame({
                'Agent': ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4'],
                'Connections': [3, 2, 4, 1],
                'Message Success Rate': [96, 89, 98, 78]
            })

            fig = px.scatter(comm_data, x='Connections', y='Message Success Rate',
                           text='Agent', title="Communication Network Analysis")
            st.plotly_chart(fig, use_container_width=True)

        # Scenario controls
        st.subheader("Scenario Controls")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Start Competitive Scenario"):
                with st.spinner("Initializing multi-agent scenario..."):
                    time.sleep(2)
                    st.success("Competitive scenario started!")

        with col2:
            if st.button("Activate Jamming"):
                st.warning("Communication jamming activated!")

        with col3:
            if st.button("Generate After-Action Report"):
                st.info("After-action report generated and saved.")

    def _render_performance_monitoring(self):
        """Render performance monitoring tab."""
        st.subheader("Performance Monitoring")

        # Real-time metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("System Performance")
            # Mock performance metrics
            perf_data = pd.DataFrame({
                'Metric': ['CPU Usage', 'Memory Usage', 'Network Latency', 'Simulation Speed'],
                'Value': [45, 67, 23, 98],
                'Unit': ['%', '%', 'ms', 'fps']
            })

            fig = go.Figure(data=[
                go.Bar(x=perf_data['Metric'], y=perf_data['Value'],
                      marker_color=['red', 'blue', 'green', 'orange'])
            ])
            fig.update_layout(title="System Performance Metrics")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Decision Quality Trends")
            # Mock decision quality over time
            time_points = pd.date_range(start='2024-01-01', periods=10, freq='H')
            decision_quality = np.random.normal(0.85, 0.05, 10)

            fig = px.line(x=time_points, y=decision_quality,
                         title="Decision Quality Over Time")
            fig.update_layout(xaxis_title="Time", yaxis_title="Quality Score")
            st.plotly_chart(fig, use_container_width=True)

        # Audit log preview
        st.subheader("Recent Audit Log")
        audit_data = pd.DataFrame({
            'Timestamp': pd.date_range(start='now', periods=5, freq='-1H'),
            'Decision Type': ['Resource Allocation', 'Route Planning', 'Threat Response', 'Communication', 'Objective Update'],
            'Confidence': [0.92, 0.87, 0.95, 0.78, 0.89],
            'Risk Level': ['Low', 'Medium', 'Low', 'High', 'Low']
        })

        st.dataframe(audit_data)

    def _create_comparison_visualization(self) -> go.Figure:
        """Create visualization for strategy comparisons."""
        if not self.strategy_comparator.comparison_history:
            return go.Figure()

        latest_comparison = self.strategy_comparator.comparison_history[-1]

        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Performance Comparison', 'Statistical Significance'))

        # Performance comparison
        fig.add_trace(
            go.Bar(x=[latest_comparison.strategy_a, latest_comparison.strategy_b],
                  y=[latest_comparison.mean_a, latest_comparison.mean_b],
                  name='Mean Performance',
                  marker_color=['lightblue', 'lightcoral']),
            row=1, col=1
        )

        # Statistical significance
        fig.add_trace(
            go.Bar(x=['T-Statistic', 'P-Value'],
                  y=[latest_comparison.t_statistic, latest_comparison.p_value],
                  name='Statistics',
                  marker_color=['orange', 'purple']),
            row=1, col=2
        )

        fig.update_layout(title="Strategy Comparison Analysis")
        return fig

    def _get_risk_alerts(self) -> List[Dict[str, Any]]:
        """Get current risk alerts."""
        # Mock alerts - in real implementation, these would be based on actual analysis
        alerts = [
            {
                'severity': 'high',
                'message': 'High threat zone detected in sector Alpha - recommend route change'
            },
            {
                'severity': 'medium',
                'message': 'Communication reliability dropping in northern region'
            },
            {
                'severity': 'low',
                'message': 'Resource utilization approaching capacity limits'
            }
        ]
        return alerts

    def _get_strategic_recommendations(self) -> List[str]:
        """Get strategic recommendations."""
        # Mock recommendations
        recommendations = [
            "Deploy additional reconnaissance assets to threat zone TZ-001",
            "Optimize supply routes to reduce transportation costs by 15%",
            "Increase communication redundancy for critical operations",
            "Consider strategy B for high-risk scenarios based on recent benchmarks"
        ]
        return recommendations

    def _load_scenario(self, scenario_id: str):
        """Load a scenario by ID."""
        # Mock scenario loading
        st.sidebar.success(f"Scenario {scenario_id} loaded successfully!")

    def _generate_executive_summary(self):
        """Generate and display executive summary."""
        with st.spinner("Generating executive summary..."):
            time.sleep(2)

            # Mock executive summary
            summary = """
            # Executive Summary

            ## Current Situation
            - Mission success rate: 87%
            - 3 active threat zones identified
            - Resource utilization: 92% efficiency

            ## Key Recommendations
            1. Deploy countermeasures for threat zone TZ-001
            2. Optimize supply chain logistics
            3. Enhance communication protocols

            ## Risk Assessment
            - Overall risk level: Medium
            - Highest risk: Communication disruption
            """

            st.sidebar.markdown(summary)

    def _export_analytics_data(self):
        """Export analytics data."""
        # Mock export
        st.sidebar.success("Analytics data exported to reports/ directory!")


def main():
    """Main function to run the dashboard."""
    dashboard = DecisionSupportDashboard()
    dashboard.render_dashboard()

    # Auto-refresh functionality
    if dashboard.auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
