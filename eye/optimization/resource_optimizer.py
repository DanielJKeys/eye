"""
Resource Optimizer for ACES

Provides optimization algorithms for military logistics resource allocation,
asset deployment, and supply chain management using linear programming,
heuristic methods, and reinforcement learning approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
from scipy.optimize import linprog, minimize_scalar
from sklearn.cluster import KMeans
import networkx as nx
from collections import defaultdict
import heapq

from eye.domain.asset import Asset
from eye.domain.environment import LogisticsEnv
from eye.utils.geometry import Point2D


@dataclass
class ResourceConstraint:
    """Defines constraints for resource optimization."""
    resource_type: str
    total_available: float
    unit_cost: float
    priority_weight: float = 1.0


@dataclass
class OptimizationObjective:
    """Defines optimization objectives."""
    name: str
    weight: float
    direction: str  # 'maximize' or 'minimize'
    function: Callable


@dataclass
class AllocationResult:
    """Results from resource allocation optimization."""
    asset_allocations: Dict[str, Dict[str, float]]
    total_cost: float
    total_utility: float
    constraint_violations: Dict[str, float]
    objective_values: Dict[str, float]
    solver_status: str
    computation_time: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DeploymentPlan:
    """Asset deployment plan with routes and schedules."""
    asset_routes: Dict[str, List[Point2D]]
    deployment_schedule: Dict[str, List[Tuple[datetime, str]]]
    resource_requirements: Dict[str, Dict[str, float]]
    risk_assessment: Dict[str, float]
    efficiency_metrics: Dict[str, float]


class ResourceOptimizer:
    """
    Optimizes resource allocation and asset deployment for military logistics.

    Uses multiple optimization approaches including linear programming, heuristics,
    and machine learning methods to provide decision support for complex logistics scenarios.
    """

    def __init__(self):
        """Initialize the resource optimizer."""
        self.constraints: Dict[str, ResourceConstraint] = {}
        self.objectives: List[OptimizationObjective] = []
        self.optimization_history: List[AllocationResult] = []

    def add_constraint(self, constraint: ResourceConstraint):
        """
        Add a resource constraint to the optimization problem.

        Args:
            constraint: Resource constraint definition
        """
        self.constraints[constraint.resource_type] = constraint

    def add_objective(self, objective: OptimizationObjective):
        """
        Add an optimization objective.

        Args:
            objective: Objective definition
        """
        self.objectives.append(objective)

    def optimize_supply_allocation(
        self,
        supply_nodes: List[Dict[str, Any]],
        demand_nodes: List[Dict[str, Any]],
        transportation_costs: Dict[Tuple[str, str], float],
        supply_limits: Dict[str, float],
        demand_requirements: Dict[str, float]
    ) -> AllocationResult:
        """
        Optimize supply allocation using transportation problem formulation.

        Args:
            supply_nodes: List of supply node dictionaries with 'id', 'supply' keys
            demand_nodes: List of demand node dictionaries with 'id', 'demand' keys
            transportation_costs: Dictionary mapping (supply_id, demand_id) to cost
            supply_limits: Maximum supply per supply node
            demand_requirements: Required demand per demand node

        Returns:
            Optimal allocation results
        """
        import time
        start_time = time.time()

        # Create supply and demand arrays
        supply_ids = [node['id'] for node in supply_nodes]
        demand_ids = [node['id'] for node in demand_nodes]

        num_supply = len(supply_ids)
        num_demand = len(demand_ids)

        # Cost matrix
        cost_matrix = np.zeros((num_supply, num_demand))
        for i, s_id in enumerate(supply_ids):
            for j, d_id in enumerate(demand_ids):
                cost_matrix[i, j] = transportation_costs.get((s_id, d_id), float('inf'))

        # Supply and demand vectors
        supply_vector = np.array([supply_limits.get(s_id, 0) for s_id in supply_ids])
        demand_vector = np.array([demand_requirements.get(d_id, 0) for d_id in demand_ids])

        # Set up linear programming problem
        # Minimize total transportation cost
        c = cost_matrix.flatten()

        # Equality constraints: supply = demand for balanced problem
        total_supply = np.sum(supply_vector)
        total_demand = np.sum(demand_vector)

        if total_supply != total_demand:
            # Add dummy supply/demand to balance
            if total_supply > total_demand:
                demand_vector = np.append(demand_vector, total_supply - total_demand)
                cost_matrix = np.column_stack([cost_matrix, np.zeros(num_supply)])
            else:
                supply_vector = np.append(supply_vector, total_demand - total_supply)
                cost_matrix = np.row_stack([cost_matrix, np.full(num_demand, 0)])

        # Update cost vector
        c = cost_matrix.flatten()

        # Supply constraints (each supply node capacity)
        A_eq = []
        b_eq = []

        # Demand constraints
        for j in range(len(demand_vector)):
            row = np.zeros(num_supply * len(demand_vector))
            for i in range(num_supply):
                row[i * len(demand_vector) + j] = 1
            A_eq.append(row)
            b_eq.append(demand_vector[j])

        # Supply constraints
        for i in range(num_supply):
            row = np.zeros(num_supply * len(demand_vector))
            for j in range(len(demand_vector)):
                row[i * len(demand_vector) + j] = 1
            A_eq.append(row)
            b_eq.append(supply_vector[i])

        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)

        # Bounds: non-negative allocations
        bounds = [(0, None) for _ in range(len(c))]

        # Solve
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        computation_time = time.time() - start_time

        if result.success:
            # Extract allocations
            allocations = result.x.reshape((num_supply, len(demand_vector)))

            asset_allocations = {}
            for i, s_id in enumerate(supply_ids):
                asset_allocations[s_id] = {}
                for j, d_id in enumerate(demand_ids):
                    if j < len(allocations[i]):
                        asset_allocations[s_id][d_id] = allocations[i][j]

            total_cost = result.fun
            total_utility = -total_cost  # For maximization problems

            # Calculate constraint violations
            constraint_violations = {}
            for s_id in supply_ids:
                allocated = sum(asset_allocations[s_id].values())
                limit = supply_limits.get(s_id, 0)
                constraint_violations[f"supply_{s_id}"] = max(0, allocated - limit)

            for d_id in demand_ids:
                received = sum(asset_allocations[s_id].get(d_id, 0) for s_id in supply_ids)
                required = demand_requirements.get(d_id, 0)
                constraint_violations[f"demand_{d_id}"] = max(0, required - received)

            objective_values = {'total_cost': total_cost}

            allocation_result = AllocationResult(
                asset_allocations=asset_allocations,
                total_cost=total_cost,
                total_utility=total_utility,
                constraint_violations=constraint_violations,
                objective_values=objective_values,
                solver_status='optimal',
                computation_time=computation_time
            )

        else:
            # Create empty result for failed optimization
            allocation_result = AllocationResult(
                asset_allocations={},
                total_cost=0,
                total_utility=0,
                constraint_violations={},
                objective_values={},
                solver_status=result.message,
                computation_time=computation_time
            )

        self.optimization_history.append(allocation_result)
        return allocation_result

    def optimize_asset_deployment(
        self,
        assets: List[Asset],
        mission_objectives: List[Dict[str, Any]],
        threat_zones: List[Dict[str, Any]],
        time_horizon: int = 24
    ) -> DeploymentPlan:
        """
        Optimize asset deployment using heuristic methods.

        Args:
            assets: List of available assets
            mission_objectives: List of mission objective dictionaries
            threat_zones: List of threat zone dictionaries
            time_horizon: Time horizon in hours

        Returns:
            Optimized deployment plan
        """
        # Create graph for path planning
        G = nx.Graph()

        # Add asset positions as nodes
        for asset in assets:
            G.add_node(f"asset_{asset.id}", pos=asset.position, type='asset')

        # Add objective positions
        for i, obj in enumerate(mission_objectives):
            G.add_node(f"objective_{i}", pos=obj['position'], type='objective',
                      priority=obj.get('priority', 1.0))

        # Add threat zones (avoid these areas)
        for i, threat in enumerate(threat_zones):
            G.add_node(f"threat_{i}", pos=threat['position'], type='threat',
                      severity=threat.get('severity', 1.0))

        # Calculate heuristic deployment using clustering
        asset_positions = np.array([asset.position.to_array() for asset in assets])
        objective_positions = np.array([obj['position'].to_array() for obj in mission_objectives])

        if len(assets) > 0 and len(mission_objectives) > 0:
            # Use K-means to cluster objectives and assign assets
            n_clusters = min(len(assets), len(mission_objectives))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            objective_clusters = kmeans.fit_predict(objective_positions)

            # Assign assets to clusters
            asset_routes = {}
            deployment_schedule = {}
            resource_requirements = {}

            for i, asset in enumerate(assets):
                cluster_objectives = [j for j, cluster in enumerate(objective_clusters) if cluster == i % n_clusters]

                if cluster_objectives:
                    # Plan route visiting objectives in this cluster
                    route = self._plan_asset_route(asset, [mission_objectives[j] for j in cluster_objectives])
                    asset_routes[asset.id] = route

                    # Create schedule
                    schedule = self._create_deployment_schedule(asset, route, time_horizon)
                    deployment_schedule[asset.id] = schedule

                    # Estimate resource requirements
                    resource_requirements[asset.id] = self._estimate_resource_needs(asset, route, schedule)
                else:
                    asset_routes[asset.id] = [asset.position]
                    deployment_schedule[asset.id] = []
                    resource_requirements[asset.id] = {}

            # Risk assessment
            risk_assessment = self._assess_deployment_risks(assets, asset_routes, threat_zones)

            # Efficiency metrics
            efficiency_metrics = self._calculate_efficiency_metrics(asset_routes, deployment_schedule)

        else:
            asset_routes = {asset.id: [asset.position] for asset in assets}
            deployment_schedule = {asset.id: [] for asset in assets}
            resource_requirements = {asset.id: {} for asset in assets}
            risk_assessment = {}
            efficiency_metrics = {}

        return DeploymentPlan(
            asset_routes=asset_routes,
            deployment_schedule=deployment_schedule,
            resource_requirements=resource_requirements,
            risk_assessment=risk_assessment,
            efficiency_metrics=efficiency_metrics
        )

    def _plan_asset_route(self, asset: Asset, objectives: List[Dict[str, Any]]) -> List[Point2D]:
        """Plan an efficient route for an asset visiting multiple objectives."""
        if not objectives:
            return [asset.position]

        # Simple nearest neighbor heuristic
        route = [asset.position]
        remaining_objectives = objectives.copy()

        while remaining_objectives:
            current_pos = route[-1]
            # Find nearest objective
            nearest_idx = min(range(len(remaining_objectives)),
                             key=lambda i: current_pos.distance_to(remaining_objectives[i]['position']))

            nearest_obj = remaining_objectives.pop(nearest_idx)
            route.append(nearest_obj['position'])

        return route

    def _create_deployment_schedule(
        self,
        asset: Asset,
        route: List[Point2D],
        time_horizon: int
    ) -> List[Tuple[datetime, str]]:
        """Create a time-based deployment schedule."""
        schedule = []
        current_time = datetime.now()

        for i, position in enumerate(route):
            action = "moving_to_objective" if i > 0 else "at_start_position"
            schedule.append((current_time, action))

            # Estimate travel time (simplified)
            if i < len(route) - 1:
                distance = position.distance_to(route[i + 1])
                travel_time_hours = distance / asset.speed if hasattr(asset, 'speed') else 1.0
                current_time = current_time.replace(hour=(current_time.hour + int(travel_time_hours)) % 24)

        return schedule

    def _estimate_resource_needs(
        self,
        asset: Asset,
        route: List[Point2D],
        schedule: List[Tuple[datetime, str]]
    ) -> Dict[str, float]:
        """Estimate resource requirements for deployment."""
        total_distance = sum(route[i].distance_to(route[i+1]) for i in range(len(route)-1))

        # Simplified resource estimation
        fuel_needed = total_distance * 0.1  # Arbitrary units
        time_deployed = len(schedule) * 0.5  # Hours

        return {
            'fuel': fuel_needed,
            'time': time_deployed,
            'maintenance': total_distance * 0.01
        }

    def _assess_deployment_risks(
        self,
        assets: List[Asset],
        asset_routes: Dict[str, List[Point2D]],
        threat_zones: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Assess risks for the deployment plan."""
        risk_assessment = {}

        for asset in assets:
            route = asset_routes.get(asset.id, [])
            total_risk = 0

            for position in route:
                for threat in threat_zones:
                    distance = position.distance_to(threat['position'])
                    threat_range = threat.get('range', 10.0)
                    severity = threat.get('severity', 1.0)

                    if distance < threat_range:
                        # Risk decreases with distance
                        risk_contribution = severity * (1 - distance / threat_range)
                        total_risk += risk_contribution

            risk_assessment[asset.id] = min(total_risk, 1.0)  # Cap at 1.0

        return risk_assessment

    def _calculate_efficiency_metrics(
        self,
        asset_routes: Dict[str, List[Point2D]],
        deployment_schedule: Dict[str, List[Tuple[datetime, str]]]
    ) -> Dict[str, float]:
        """Calculate efficiency metrics for the deployment."""
        total_distance = sum(
            sum(route[i].distance_to(route[i+1]) for i in range(len(route)-1))
            for route in asset_routes.values()
        )

        total_objectives = sum(len(route) - 1 for route in asset_routes.values())  # Subtract starting positions
        total_assets = len(asset_routes)

        avg_distance_per_objective = total_distance / max(total_objectives, 1)
        asset_utilization = total_objectives / max(total_assets, 1)

        return {
            'total_distance': total_distance,
            'total_objectives_covered': total_objectives,
            'avg_distance_per_objective': avg_distance_per_objective,
            'asset_utilization': asset_utilization
        }

    def optimize_multi_objective(
        self,
        decision_variables: np.ndarray,
        objective_functions: List[Callable],
        constraints: List[Callable],
        bounds: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Optimize multi-objective problems using weighted sum approach.

        Args:
            decision_variables: Initial guess for decision variables
            objective_functions: List of objective functions to minimize
            constraints: List of constraint functions
            bounds: Variable bounds

        Returns:
            Optimization results
        """
        def weighted_sum_objective(x):
            weights = np.ones(len(objective_functions)) / len(objective_functions)
            return sum(w * obj(x) for w, obj in zip(weights, objective_functions))

        # Apply constraints
        cons = [{'type': 'ineq', 'fun': c} for c in constraints]

        result = minimize_scalar(
            weighted_sum_objective,
            bounds=bounds,
            constraints=cons,
            method='SLSQP'
        )

        return {
            'optimal_solution': result.x,
            'optimal_value': result.fun,
            'success': result.success,
            'message': result.message
        }

    def generate_resource_report(self, allocation_result: AllocationResult) -> str:
        """
        Generate a detailed report of resource allocation results.

        Args:
            allocation_result: Results from optimization

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("# Resource Allocation Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        report_lines.append("## Summary")
        report_lines.append(f"- **Total Cost:** ${allocation_result.total_cost:.2f}")
        report_lines.append(f"- **Total Utility:** {allocation_result.total_utility:.2f}")
        report_lines.append(f"- **Solver Status:** {allocation_result.solver_status}")
        report_lines.append(f"- **Computation Time:** {allocation_result.computation_time:.3f} seconds")
        report_lines.append("")

        report_lines.append("## Allocations")
        for from_node, allocations in allocation_result.asset_allocations.items():
            report_lines.append(f"### From {from_node}")
            for to_node, amount in allocations.items():
                if amount > 0:
                    report_lines.append(f"- {to_node}: {amount:.2f} units")
            report_lines.append("")

        if allocation_result.constraint_violations:
            report_lines.append("## Constraint Violations")
            for constraint, violation in allocation_result.constraint_violations.items():
                if violation > 0:
                    report_lines.append(f"- {constraint}: {violation:.2f} units")
            report_lines.append("")

        report_lines.append("## Objective Values")
        for obj_name, value in allocation_result.objective_values.items():
            report_lines.append(f"- {obj_name}: {value:.2f}")
        report_lines.append("")

        return "\n".join(report_lines)

    def export_optimization_results(self, output_path: Path):
        """
        Export optimization history to JSON.

        Args:
            output_path: Path to save the export
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'optimization_history': [
                {
                    'asset_allocations': result.asset_allocations,
                    'total_cost': result.total_cost,
                    'total_utility': result.total_utility,
                    'constraint_violations': result.constraint_violations,
                    'objective_values': result.objective_values,
                    'solver_status': result.solver_status,
                    'computation_time': result.computation_time,
                    'timestamp': result.timestamp.isoformat()
                }
                for result in self.optimization_history
            ],
            'constraints': [vars(c) for c in self.constraints.values()],
            'objectives': [
                {
                    'name': obj.name,
                    'weight': obj.weight,
                    'direction': obj.direction
                }
                for obj in self.objectives
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
