"""
Resource Planning UI for ACES

Interactive optimization interfaces for supply chain management,
asset deployment planning, and resource allocation in military logistics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime
from io import BytesIO

# Import EYE modules
from eye.optimization.resource_optimizer import ResourceOptimizer, AllocationResult, DeploymentPlan
from eye.domain.asset import Asset
from eye.domain.environment import LogisticsEnvironment
from eye.utils.geometry import Point2D


class ResourcePlanningUI:
    """
    Interactive UI for resource planning and optimization.

    Provides tools for supply chain optimization, asset deployment planning,
    and resource allocation decision support.
    """

    def __init__(self):
        """Initialize the resource planning UI."""
        self.resource_optimizer = ResourceOptimizer()
        self.environment = LogisticsEnvironment()
        self.assets: List[Asset] = []
        self.optimization_results: List[AllocationResult] = []

    def render_planning_ui(self):
        """Render the main resource planning interface."""
        st.title("📦 Resource Planning & Optimization")
        st.markdown("Optimize supply chains, deploy assets, and allocate resources for military logistics operations")

        # Scenario setup
        self._render_scenario_setup()

        # Optimization tabs
        tab1, tab2, tab3 = st.tabs([
            "Supply Chain Optimization",
            "Asset Deployment Planning",
            "Resource Allocation"
        ])

        with tab1:
            self._render_supply_chain_optimization()

        with tab2:
            self._render_asset_deployment()

        with tab3:
            self._render_resource_allocation()

        # Results and export
        self._render_results_summary()

    def _render_scenario_setup(self):
        """Render scenario setup controls."""
        st.subheader("Scenario Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Supply Nodes")
            num_supply_nodes = st.slider("Number of Supply Nodes", 2, 10, 4)

            supply_data = []
            for i in range(num_supply_nodes):
                col_a, col_b = st.columns(2)
                with col_a:
                    supply = st.number_input(f"Supply Node {i+1} Capacity",
                                           min_value=0, max_value=10000, value=1000,
                                           key=f"supply_{i}")
                with col_b:
                    cost = st.number_input(f"Supply Node {i+1} Cost",
                                         min_value=0.0, max_value=100.0, value=10.0,
                                         key=f"cost_{i}")
                supply_data.append({'id': f"S{i+1}", 'supply': supply, 'cost': cost})

            self.supply_nodes = supply_data

        with col2:
            st.subheader("Demand Nodes")
            num_demand_nodes = st.slider("Number of Demand Nodes", 2, 10, 3)

            demand_data = []
            for i in range(num_demand_nodes):
                col_a, col_b = st.columns(2)
                with col_a:
                    demand = st.number_input(f"Demand Node {i+1} Required",
                                           min_value=0, max_value=5000, value=500,
                                           key=f"demand_{i}")
                with col_b:
                    priority = st.slider(f"Demand Node {i+1} Priority",
                                       min_value=1, max_value=5, value=3,
                                       key=f"priority_{i}")
                demand_data.append({'id': f"D{i+1}", 'demand': demand, 'priority': priority})

            self.demand_nodes = demand_data

        with col3:
            st.subheader("Transportation Costs")
            st.write("Define costs between supply and demand nodes:")

            # Create cost matrix
            cost_matrix = np.zeros((len(self.supply_nodes), len(self.demand_nodes)))

            for i, supply in enumerate(self.supply_nodes):
                for j, demand in enumerate(self.demand_nodes):
                    cost = st.number_input(
                        f"{supply['id']} → {demand['id']}",
                        min_value=0.0, max_value=100.0, value=5.0,
                        key=f"cost_{i}_{j}"
                    )
                    cost_matrix[i, j] = cost

            self.transportation_costs = {}
            for i, supply in enumerate(self.supply_nodes):
                for j, demand in enumerate(self.demand_nodes):
                    self.transportation_costs[(supply['id'], demand['id'])] = cost_matrix[i, j]

    def _render_supply_chain_optimization(self):
        """Render supply chain optimization interface."""
        st.subheader("Supply Chain Optimization")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Network Visualization")

            # Create supply chain network graph
            G = nx.DiGraph()

            # Add nodes
            for node in self.supply_nodes:
                G.add_node(node['id'], type='supply', capacity=node['supply'])
            for node in self.demand_nodes:
                G.add_node(node['id'], type='demand', requirement=node['demand'])

            # Add edges with costs
            for (supply_id, demand_id), cost in self.transportation_costs.items():
                G.add_edge(supply_id, demand_id, cost=cost)

            # Visualize network
            pos = nx.spring_layout(G)

            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x = []
            node_y = []
            node_text = []
            node_color = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                node_color.append('lightblue' if G.nodes[node]['type'] == 'supply' else 'lightcoral')

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="bottom center",
                hoverinfo='text',
                marker=dict(
                    color=node_color,
                    size=20,
                    line_width=2))

            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                title="Supply Chain Network",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False))
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Optimization Results")

            if st.button("Optimize Supply Chain", key="optimize_supply"):
                with st.spinner("Running optimization..."):
                    self._run_supply_optimization()

            if hasattr(self, 'supply_optimization_result'):
                result = self.supply_optimization_result

                st.metric("Total Cost", f"${result.total_cost:.2f}")
                st.metric("Total Utility", f"{result.total_utility:.2f}")

                # Display allocations
                st.subheader("Optimal Allocations")
                allocation_data = []
                for from_node, allocations in result.asset_allocations.items():
                    for to_node, amount in allocations.items():
                        if amount > 0:
                            allocation_data.append({
                                'From': from_node,
                                'To': to_node,
                                'Amount': amount
                            })

                if allocation_data:
                    allocation_df = pd.DataFrame(allocation_data)
                    st.dataframe(allocation_df)

                    # Visualization
                    fig = px.bar(
                        allocation_df,
                        x='From',
                        y='Amount',
                        color='To',
                        title="Supply Allocations by Source"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    def _render_asset_deployment(self):
        """Render asset deployment planning interface."""
        st.subheader("Asset Deployment Planning")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Asset Configuration")

            num_assets = st.slider("Number of Assets", 1, 10, 4)

            asset_types = ['truck', 'helicopter', 'drone', 'ship']
            self.assets = []

            for i in range(num_assets):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    asset_type = st.selectbox(f"Asset {i+1} Type",
                                            asset_types,
                                            key=f"asset_type_{i}")
                with col_b:
                    speed = st.slider(f"Asset {i+1} Speed",
                                    min_value=10, max_value=200, value=60,
                                    key=f"asset_speed_{i}")
                with col_c:
                    capacity = st.slider(f"Asset {i+1} Capacity",
                                       min_value=100, max_value=5000, value=1000,
                                       key=f"asset_capacity_{i}")

                asset = Asset(
                    id=f"Asset_{i+1}",
                    position=self.environment.get_random_position(),
                    asset_type=asset_type,
                    status='active'
                )
                # Add custom attributes
                asset.speed = speed
                asset.capacity = capacity
                self.assets.append(asset)

        with col2:
            st.subheader("Mission Objectives")

            num_objectives = st.slider("Number of Objectives", 1, 5, 3)

            self.mission_objectives = []
            for i in range(num_objectives):
                col_a, col_b = st.columns(2)
                with col_a:
                    priority = st.slider(f"Objective {i+1} Priority",
                                       min_value=1, max_value=5, value=3,
                                       key=f"obj_priority_{i}")
                with col_b:
                    urgency = st.selectbox(f"Objective {i+1} Urgency",
                                         ['Low', 'Medium', 'High', 'Critical'],
                                         key=f"obj_urgency_{i}")

                objective = {
                    'id': f"OBJ_{i+1}",
                    'position': self.environment.get_random_position(),
                    'priority': priority,
                    'urgency': urgency
                }
                self.mission_objectives.append(objective)

        # Deployment optimization
        st.subheader("Deployment Optimization")

        col1, col2 = st.columns(2)

        with col1:
            time_horizon = st.slider("Time Horizon (hours)", 1, 72, 24)

            if st.button("Optimize Deployment", key="optimize_deployment"):
                with st.spinner("Optimizing asset deployment..."):
                    self._run_deployment_optimization(time_horizon)

        with col2:
            if st.button("Visualize Deployment", key="visualize_deployment"):
                self._visualize_deployment_plan()

        # Display deployment results
        if hasattr(self, 'deployment_plan'):
            self._display_deployment_results()

    def _render_resource_allocation(self):
        """Render resource allocation interface."""
        st.subheader("Resource Allocation")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Resource Constraints")

            # Define resource types
            resource_types = ['Fuel', 'Ammunition', 'Personnel', 'Maintenance']

            self.resource_constraints = {}
            for resource in resource_types:
                total_available = st.number_input(
                    f"Total {resource} Available",
                    min_value=0, max_value=10000, value=1000,
                    key=f"resource_{resource}"
                )
                unit_cost = st.number_input(
                    f"{resource} Unit Cost",
                    min_value=0.0, max_value=100.0, value=10.0,
                    key=f"cost_{resource}"
                )
                priority = st.slider(
                    f"{resource} Priority Weight",
                    min_value=0.1, max_value=2.0, value=1.0,
                    key=f"priority_{resource}"
                )

                constraint = ResourceConstraint(
                    resource_type=resource.lower(),
                    total_available=total_available,
                    unit_cost=unit_cost,
                    priority_weight=priority
                )
                self.resource_constraints[resource.lower()] = constraint

        with col2:
            st.subheader("Allocation Objectives")

            # Define objectives
            objectives = [
                ('minimize_cost', 'Minimize Total Cost'),
                ('maximize_efficiency', 'Maximize Resource Efficiency'),
                ('balance_distribution', 'Balance Resource Distribution'),
                ('prioritize_critical', 'Prioritize Critical Operations')
            ]

            self.selected_objectives = []
            for obj_key, obj_name in objectives:
                if st.checkbox(obj_name, key=f"obj_{obj_key}"):
                    self.selected_objectives.append(obj_key)

        # Optimization execution
        st.subheader("Multi-Objective Optimization")

        if st.button("Run Resource Allocation", key="allocate_resources"):
            with st.spinner("Running multi-objective optimization..."):
                self._run_resource_allocation()

        # Display allocation results
        if hasattr(self, 'allocation_result'):
            self._display_allocation_results()

    def _render_results_summary(self):
        """Render results summary and export options."""
        st.subheader("Results Summary & Export")

        if not self.optimization_results:
            st.info("Run optimizations to see results summary.")
            return

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        total_optimizations = len(self.optimization_results)
        avg_cost = np.mean([r.total_cost for r in self.optimization_results])
        avg_utility = np.mean([r.total_utility for r in self.optimization_results])
        success_rate = sum(1 for r in self.optimization_results if r.solver_status == 'optimal') / total_optimizations

        with col1:
            st.metric("Total Optimizations", total_optimizations)
        with col2:
            st.metric("Average Cost", f"${avg_cost:.2f}")
        with col3:
            st.metric("Average Utility", f"{avg_utility:.2f}")
        with col4:
            st.metric("Success Rate", f"{success_rate:.1%}")

        # Export options
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Export to CSV"):
                self._export_results_csv()

        with col2:
            if st.button("Export to JSON"):
                self._export_results_json()

        with col3:
            if st.button("Generate Report"):
                self._generate_optimization_report()

    def _run_supply_optimization(self):
        """Run supply chain optimization."""
        try:
            # Prepare data for optimization
            supply_limits = {node['id']: node['supply'] for node in self.supply_nodes}
            demand_requirements = {node['id']: node['demand'] for node in self.demand_nodes}

            result = self.resource_optimizer.optimize_supply_allocation(
                self.supply_nodes,
                self.demand_nodes,
                self.transportation_costs,
                supply_limits,
                demand_requirements
            )

            self.supply_optimization_result = result
            self.optimization_results.append(result)

            st.success("Supply optimization completed!")

        except Exception as e:
            st.error(f"Supply optimization failed: {str(e)}")

    def _run_deployment_optimization(self, time_horizon: int):
        """Run asset deployment optimization."""
        try:
            # Mock threat zones
            threat_zones = [
                {'position': self.environment.get_random_position(), 'range': 20.0, 'severity': 0.7},
                {'position': self.environment.get_random_position(), 'range': 15.0, 'severity': 0.5}
            ]

            deployment_plan = self.resource_optimizer.optimize_asset_deployment(
                self.assets,
                self.mission_objectives,
                threat_zones,
                time_horizon
            )

            self.deployment_plan = deployment_plan
            st.success("Deployment optimization completed!")

        except Exception as e:
            st.error(f"Deployment optimization failed: {str(e)}")

    def _run_resource_allocation(self):
        """Run resource allocation optimization."""
        try:
            # Mock decision variables and objectives
            n_vars = len(self.resource_constraints)
            decision_vars = np.ones(n_vars) * 100  # Initial allocation

            # Define bounds
            bounds = [(0, constraint.total_available) for constraint in self.resource_constraints.values()]

            # Mock objective functions
            def cost_objective(x):
                return sum(x[i] * list(self.resource_constraints.values())[i].unit_cost for i in range(n_vars))

            def efficiency_objective(x):
                return -sum(x[i] * list(self.resource_constraints.values())[i].priority_weight for i in range(n_vars))

            objective_functions = [cost_objective, efficiency_objective]
            constraints = []  # Add constraints as needed

            result = self.resource_optimizer.optimize_multi_objective(
                decision_vars,
                objective_functions,
                constraints,
                bounds
            )

            # Create mock allocation result
            allocation_result = AllocationResult(
                asset_allocations={'optimal': {k: v for k, v in zip(self.resource_constraints.keys(), result['optimal_solution'])}},
                total_cost=cost_objective(result['optimal_solution']),
                total_utility=-efficiency_objective(result['optimal_solution']),
                constraint_violations={},
                objective_values={'cost': cost_objective(result['optimal_solution']),
                                'efficiency': -efficiency_objective(result['optimal_solution'])},
                solver_status='optimal' if result['success'] else 'failed',
                computation_time=0.1
            )

            self.allocation_result = allocation_result
            self.optimization_results.append(allocation_result)

            st.success("Resource allocation completed!")

        except Exception as e:
            st.error(f"Resource allocation failed: {str(e)}")

    def _visualize_deployment_plan(self):
        """Visualize the deployment plan."""
        if not hasattr(self, 'deployment_plan'):
            st.warning("No deployment plan available. Run optimization first.")
            return

        plan = self.deployment_plan

        # Create deployment visualization
        fig = go.Figure()

        # Plot asset routes
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, (asset_id, route) in enumerate(plan.asset_routes.items()):
            if route:
                route_points = [p.to_array() for p in route]
                route_x, route_y = zip(*route_points)

                fig.add_trace(go.Scatter(
                    x=route_x, y=route_y,
                    mode='lines+markers',
                    name=f"{asset_id} Route",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=8)
                ))

        # Plot objectives
        obj_x = [obj['position'].x for obj in self.mission_objectives]
        obj_y = [obj['position'].y for obj in self.mission_objectives]

        fig.add_trace(go.Scatter(
            x=obj_x, y=obj_y,
            mode='markers',
            name='Objectives',
            marker=dict(color='gold', size=12, symbol='star')
        ))

        fig.update_layout(
            title="Asset Deployment Plan",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    def _display_deployment_results(self):
        """Display deployment optimization results."""
        plan = self.deployment_plan

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Efficiency Metrics")
            metrics_df = pd.DataFrame([
                {'Metric': k, 'Value': v} for k, v in plan.efficiency_metrics.items()
            ])
            st.dataframe(metrics_df)

        with col2:
            st.subheader("Risk Assessment")
            risk_df = pd.DataFrame([
                {'Asset': asset, 'Risk': risk} for asset, risk in plan.risk_assessment.items()
            ])
            st.dataframe(risk_df)

    def _display_allocation_results(self):
        """Display resource allocation results."""
        result = self.allocation_result

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Cost", f"${result.total_cost:.2f}")
            st.metric("Total Utility", f"{result.total_utility:.2f}")

        with col2:
            st.subheader("Resource Allocation")
            allocation_data = []
            for resource, amount in result.asset_allocations['optimal'].items():
                allocation_data.append({
                    'Resource': resource.capitalize(),
                    'Allocated': amount,
                    'Available': self.resource_constraints[resource].total_available,
                    'Utilization': amount / self.resource_constraints[resource].total_available
                })

            allocation_df = pd.DataFrame(allocation_data)
            st.dataframe(allocation_df)

            # Utilization chart
            fig = px.bar(
                allocation_df,
                x='Resource',
                y='Utilization',
                title="Resource Utilization"
            )
            st.plotly_chart(fig, use_container_width=True)

    def _export_results_csv(self):
        """Export optimization results to CSV."""
        if not self.optimization_results:
            st.error("No results to export.")
            return

        results_data = []
        for result in self.optimization_results:
            results_data.append({
                'timestamp': result.timestamp.isoformat(),
                'total_cost': result.total_cost,
                'total_utility': result.total_utility,
                'solver_status': result.solver_status,
                'computation_time': result.computation_time
            })

        df = pd.DataFrame(results_data)
        csv = df.to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="optimization_results.csv",
            mime="text/csv"
        )

    def _export_results_json(self):
        """Export optimization results to JSON."""
        if not self.optimization_results:
            st.error("No results to export.")
            return

        results_dict = {
            'optimizations': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'total_cost': r.total_cost,
                    'total_utility': r.total_utility,
                    'solver_status': r.solver_status,
                    'computation_time': r.computation_time
                }
                for r in self.optimization_results
            ]
        }

        json_str = pd.io.json.dumps(results_dict, indent=2)

        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="optimization_results.json",
            mime="application/json"
        )

    def _generate_optimization_report(self):
        """Generate a comprehensive optimization report."""
        if not self.optimization_results:
            st.error("No results to report.")
            return

        # Generate report content
        report_content = "# Resource Optimization Report\n\n"
        report_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        report_content += "## Summary\n\n"
        report_content += f"- Total optimizations run: {len(self.optimization_results)}\n"
        report_content += f"- Average cost: ${np.mean([r.total_cost for r in self.optimization_results]):.2f}\n"
        report_content += f"- Success rate: {sum(1 for r in self.optimization_results if r.solver_status == 'optimal') / len(self.optimization_results):.1%}\n\n"

        report_content += "## Detailed Results\n\n"

        for i, result in enumerate(self.optimization_results, 1):
            report_content += f"### Optimization {i}\n\n"
            report_content += f"- Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_content += f"- Total Cost: ${result.total_cost:.2f}\n"
            report_content += f"- Total Utility: {result.total_utility:.2f}\n"
            report_content += f"- Solver Status: {result.solver_status}\n"
            report_content += f"- Computation Time: {result.computation_time:.3f} seconds\n\n"

        # Display report
        st.text_area("Optimization Report", report_content, height=400)

        # Download button
        st.download_button(
            label="Download Report",
            data=report_content,
            file_name="optimization_report.md",
            mime="text/markdown"
        )


def main():
    """Main function to run the resource planning UI."""
    planning_ui = ResourcePlanningUI()
    planning_ui.render_planning_ui()


if __name__ == "__main__":
    main()
