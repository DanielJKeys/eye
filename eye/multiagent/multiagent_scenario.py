"""
Multi-Agent Scenario for ACES

Implements competitive and cooperative multi-agent logistics scenarios
with communication constraints, team coordination, and adversarial dynamics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import defaultdict
import random
import networkx as nx
from scipy.spatial.distance import pdist, squareform

from eye.domain.asset import Asset
from eye.utils.geometry import Point2D
from eye.domain.environment import LogisticsEnv
from eye.services.training import TrainingService
from eye.spaces.action import ActionBuilder
from eye.spaces.observation import ObservationBuilder


@dataclass
class AgentTeam:
    """Represents a team of agents in multi-agent scenarios."""
    team_id: str
    agent_ids: List[str]
    team_type: str  # 'friendly', 'adversary', 'neutral'
    communication_range: float
    cooperation_level: float  # 0-1 scale
    shared_objectives: List[str] = field(default_factory=list)


@dataclass
class CommunicationMessage:
    """Represents a message between agents."""
    sender_id: str
    receiver_id: str
    message_type: str  # 'position_update', 'threat_warning', 'resource_request', etc.
    content: Dict[str, Any]
    timestamp: datetime
    reliability: float  # 0-1 scale (affected by jamming, distance, etc.)
    received: bool = False


@dataclass
class MultiAgentState:
    """State representation for multi-agent scenarios."""
    agent_states: Dict[str, Dict[str, Any]]
    team_states: Dict[str, Dict[str, Any]]
    communication_network: Dict[str, List[str]]
    shared_knowledge: Dict[str, Any]
    global_objectives: List[Dict[str, Any]]
    timestamp: datetime


@dataclass
class ScenarioResult:
    """Results from a multi-agent scenario run."""
    scenario_id: str
    duration: float
    team_scores: Dict[str, float]
    agent_performance: Dict[str, Dict[str, float]]
    communication_stats: Dict[str, int]
    objective_completion: Dict[str, bool]
    conflicts_encountered: int
    timestamp: datetime


class MultiAgentScenario:
    """
    Multi-agent logistics simulation with competitive and cooperative dynamics.

    Supports team-based scenarios with communication constraints, adversarial
    interactions, and complex coordination requirements.
    """

    def __init__(self, environment: LogisticsEnv):
        """
        Initialize multi-agent scenario.

        Args:
            environment: The logistics environment
        """
        self.environment = environment
        self.teams: Dict[str, AgentTeam] = {}
        self.agents: Dict[str, Asset] = {}
        self.communication_log: List[CommunicationMessage] = []
        self.scenario_history: List[MultiAgentState] = []
        self.results_history: List[ScenarioResult] = []

        # Communication constraints
        self.jamming_active = False
        self.jamming_strength = 0.0
        self.base_communication_range = 50.0  # km

    def add_team(self, team: AgentTeam):
        """
        Add a team to the scenario.

        Args:
            team: Team definition
        """
        self.teams[team.team_id] = team

        # Initialize agents for the team
        for agent_id in team.agent_ids:
            if agent_id not in self.agents:
                # Create a basic agent (would be more sophisticated in real implementation)
                # For testing, create a simple dict-based agent
                agent = {
                    'id': agent_id,
                    'position': Point2D(0, 0),
                    'asset_type': 'transport' if team.team_type == 'friendly' else 'threat',
                    'status': 'active'
                }
                self.agents[agent_id] = agent

    def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: str,
        content: Dict[str, Any],
        current_time: datetime
    ) -> bool:
        """
        Send a message between agents with communication constraints.

        Args:
            sender_id: Sending agent ID
            receiver_id: Receiving agent ID
            message_type: Type of message
            content: Message content
            current_time: Current simulation time

        Returns:
            Whether message was successfully delivered
        """
        if sender_id not in self.agents or receiver_id not in self.agents:
            return False

        sender = self.agents[sender_id]
        receiver = self.agents[receiver_id]

        # Calculate distance
        if hasattr(sender, 'position') and hasattr(receiver, 'position'):
            distance = sender.position.distance_to(receiver.position)
        elif isinstance(sender, dict) and isinstance(receiver, dict):
            distance = sender['position'].distance_to(receiver['position'])
        else:
            distance = 0.0  # Default if positions not available

        # Check if within communication range
        sender_team = self._get_agent_team(sender_id)
        comm_range = sender_team.communication_range if sender_team else self.base_communication_range

        # Apply jamming effects
        effective_range = comm_range * (1 - self.jamming_strength)

        if distance > effective_range:
            return False

        # Calculate reliability based on distance and jamming
        distance_factor = 1 - (distance / effective_range)
        reliability = distance_factor * (1 - self.jamming_strength)

        # Add some randomness
        reliability *= random.uniform(0.8, 1.2)
        reliability = min(reliability, 1.0)

        # Determine if message is received
        received = random.random() < reliability

        message = CommunicationMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=current_time,
            reliability=reliability,
            received=received
        )

        self.communication_log.append(message)
        return received

    def _get_agent_team(self, agent_id: str) -> Optional[AgentTeam]:
        """Get the team an agent belongs to."""
        for team in self.teams.values():
            if agent_id in team.agent_ids:
                return team
        return None

    def update_communication_network(self) -> Dict[str, List[str]]:
        """
        Update the communication network based on current positions and constraints.

        Returns:
            Current communication network
        """
        network = defaultdict(list)

        agent_ids = list(self.agents.keys())

        for i, agent_a in enumerate(agent_ids):
            for j, agent_b in enumerate(agent_ids):
                if i != j:
                    team_a = self._get_agent_team(agent_a)
                    team_b = self._get_agent_team(agent_b)

                    # Same team or allied teams can communicate
                    if team_a and team_b and team_a.team_id == team_b.team_id:
                        comm_range = team_a.communication_range
                    else:
                        comm_range = self.base_communication_range * 0.5  # Limited inter-team comm

                    effective_range = comm_range * (1 - self.jamming_strength)

                    # Calculate distance
                    agent_a_obj = self.agents[agent_a]
                    agent_b_obj = self.agents[agent_b]
                    if isinstance(agent_a_obj, dict) and isinstance(agent_b_obj, dict):
                        distance = agent_a_obj['position'].distance_to(agent_b_obj['position'])
                    elif hasattr(agent_a_obj, 'position') and hasattr(agent_b_obj, 'position'):
                        distance = agent_a_obj.position.distance_to(agent_b_obj.position)
                    else:
                        distance = 0.0

                    if distance <= effective_range:
                        network[agent_a].append(agent_b)

        return dict(network)

    def simulate_competitive_scenario(
        self,
        duration_hours: int,
        objectives: List[Dict[str, Any]],
        random_seed: Optional[int] = None
    ) -> ScenarioResult:
        """
        Simulate a competitive multi-agent scenario.

        Args:
            duration_hours: Scenario duration in hours
            objectives: Scenario objectives
            random_seed: Random seed for reproducibility

        Returns:
            Scenario results
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        start_time = datetime.now()
        current_time = start_time

        # Initialize scenario state
        initial_state = MultiAgentState(
            agent_states={aid: self._get_agent_state(aid) for aid in self.agents.keys()},
            team_states={tid: self._get_team_state(tid) for tid in self.teams.keys()},
            communication_network=self.update_communication_network(),
            shared_knowledge={},
            global_objectives=objectives,
            timestamp=current_time
        )

        self.scenario_history.append(initial_state)

        # Simulation loop
        team_scores = defaultdict(float)
        agent_performance = defaultdict(dict)
        communication_stats = defaultdict(int)
        conflicts = 0

        for hour in range(duration_hours):
            current_time = start_time + timedelta(hours=hour)

            # Update agent positions and actions
            for agent_id, agent in self.agents.items():
                team = self._get_agent_team(agent_id)
                if not team:
                    continue

                # Simulate agent decision making
                action = self._simulate_agent_action(agent_id, team, objectives, current_time)

                # Execute action
                self._execute_agent_action(agent_id, action)

                # Update performance metrics
                agent_performance[agent_id] = self._calculate_agent_performance(agent_id)

            # Update communication network
            comm_network = self.update_communication_network()

            # Simulate inter-agent communication
            messages_sent = self._simulate_team_communication(current_time)
            communication_stats['messages_sent'] += messages_sent

            # Check for conflicts between teams
            conflicts += self._detect_team_conflicts()

            # Update team scores
            for team_id, team in self.teams.items():
                team_scores[team_id] = self._calculate_team_score(team_id, objectives)

            # Record state
            state = MultiAgentState(
                agent_states={aid: self._get_agent_state(aid) for aid in self.agents.keys()},
                team_states={tid: self._get_team_state(tid) for tid in self.teams.keys()},
                communication_network=comm_network,
                shared_knowledge=self._get_shared_knowledge(),
                global_objectives=objectives,
                timestamp=current_time
            )
            self.scenario_history.append(state)

        # Calculate final objective completion
        objective_completion = {}
        for obj in objectives:
            obj_id = obj.get('id', f"obj_{objectives.index(obj)}")
            objective_completion[obj_id] = self._check_objective_completion(obj)

        result = ScenarioResult(
            scenario_id=f"competitive_{start_time.strftime('%Y%m%d_%H%M%S')}",
            duration=duration_hours,
            team_scores=dict(team_scores),
            agent_performance=dict(agent_performance),
            communication_stats=dict(communication_stats),
            objective_completion=objective_completion,
            conflicts_encountered=conflicts,
            timestamp=start_time
        )

        self.results_history.append(result)
        return result

    def _get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get current state of an agent."""
        agent = self.agents[agent_id]
        if isinstance(agent, dict):
            return {
                'position': {'x': agent['position'].x, 'y': agent['position'].y},
                'status': agent['status'],
                'fuel': agent.get('fuel', 100),
                'cargo': agent.get('cargo', 0),
                'last_action': agent.get('last_action', None)
            }
        else:
            return {
                'position': {'x': agent.position.x, 'y': agent.position.y},
                'status': agent.status,
                'fuel': getattr(agent, 'fuel', 100),
                'cargo': getattr(agent, 'cargo', 0),
                'last_action': getattr(agent, 'last_action', None)
            }

    def _get_team_state(self, team_id: str) -> Dict[str, Any]:
        """Get current state of a team."""
        team = self.teams[team_id]
        active_count = 0
        for aid in team.agent_ids:
            agent = self.agents[aid]
            if isinstance(agent, dict):
                if agent.get('status') == 'active':
                    active_count += 1
            else:
                if getattr(agent, 'status', 'active') == 'active':
                    active_count += 1

        return {
            'active_agents': active_count,
            'communication_network_size': len(self.update_communication_network().get(team.team_id, [])),
            'shared_objectives_completed': len([obj for obj in team.shared_objectives if self._check_objective_completion({'id': obj})])
        }

    def _simulate_agent_action(
        self,
        agent_id: str,
        team: AgentTeam,
        objectives: List[Dict[str, Any]],
        current_time: datetime
    ) -> Dict[str, Any]:
        """Simulate an agent's action decision."""
        agent = self.agents[agent_id]

        # Simple decision logic based on team type and objectives
        if team.team_type == 'friendly':
            # Friendly agents try to achieve objectives
            agent_pos = agent['position'] if isinstance(agent, dict) else agent.position
            nearest_objective = min(
                objectives,
                key=lambda obj: agent_pos.distance_to(obj['position'])
            )
            action = {
                'type': 'move_to_objective',
                'target': nearest_objective['position'],
                'reason': 'achieving_objective'
            }
        elif team.team_type == 'adversary':
            # Adversary agents try to interfere
            friendly_agents = [
                aid for aid in self.agents.keys()
                if self._get_agent_team(aid) and self._get_agent_team(aid).team_type == 'friendly'
            ]
            if friendly_agents:
                nearest_friendly = min(
                    friendly_agents,
                    key=lambda aid: agent.position.distance_to(self.agents[aid].position)
                )
                action = {
                    'type': 'interfere',
                    'target': nearest_friendly,
                    'reason': 'disrupting_enemy'
                }
            else:
                action = {'type': 'patrol', 'reason': 'waiting_for_targets'}
        else:
            action = {'type': 'idle', 'reason': 'neutral_behavior'}

        return action

    def _execute_agent_action(self, agent_id: str, action: Dict[str, Any]):
        """Execute an agent's action."""
        agent = self.agents[agent_id]

        if action['type'] == 'move_to_objective':
            # Move toward target (simplified)
            target_pos = action['target']
            direction = (target_pos - agent.position).normalize()
            move_distance = min(10.0, agent.position.distance_to(target_pos))  # Max 10km per hour
            agent.position = agent.position + direction * move_distance

        elif action['type'] == 'interfere':
            # Adversary interference (simplified)
            target_id = action['target']
            if target_id in self.agents:
                # Reduce target efficiency
                if hasattr(self.agents[target_id], 'efficiency'):
                    self.agents[target_id].efficiency *= 0.9

        # Update agent attributes
        agent.last_action = action

    def _simulate_team_communication(self, current_time: datetime) -> int:
        """Simulate communication within and between teams."""
        messages_sent = 0

        for team in self.teams.values():
            # Intra-team communication
            for sender_id in team.agent_ids:
                for receiver_id in team.agent_ids:
                    if sender_id != receiver_id:
                        # Send position update
                        success = self.send_message(
                            sender_id, receiver_id, 'position_update',
                            {'position': self.agents[sender_id].position.to_dict()},
                            current_time
                        )
                        if success:
                            messages_sent += 1

        return messages_sent

    def _detect_team_conflicts(self) -> int:
        """Detect conflicts between teams."""
        conflicts = 0

        for team_a in self.teams.values():
            for team_b in self.teams.values():
                if team_a.team_id != team_b.team_id and team_a.team_type != team_b.team_type:
                    # Check for proximity conflicts
                    for agent_a in team_a.agent_ids:
                        for agent_b in team_b.agent_ids:
                            distance = self.agents[agent_a].position.distance_to(self.agents[agent_b].position)
                            if distance < 5.0:  # Conflict threshold
                                conflicts += 1

        return conflicts

    def _calculate_agent_performance(self, agent_id: str) -> Dict[str, float]:
        """Calculate performance metrics for an agent."""
        agent = self.agents[agent_id]
        team = self._get_agent_team(agent_id)

        performance = {
            'distance_traveled': getattr(agent, 'distance_traveled', 0),
            'objectives_completed': getattr(agent, 'objectives_completed', 0),
            'communication_success_rate': getattr(agent, 'comm_success_rate', 0.8),
            'survival_time': getattr(agent, 'survival_time', 100)
        }

        return performance

    def _calculate_team_score(self, team_id: str, objectives: List[Dict[str, Any]]) -> float:
        """Calculate score for a team."""
        team = self.teams[team_id]
        score = 0

        # Base score from agent performance
        for agent_id in team.agent_ids:
            perf = self._calculate_agent_performance(agent_id)
            score += perf.get('objectives_completed', 0) * 10
            score += perf.get('survival_time', 0) * 0.1

        # Bonus for cooperation
        score *= team.cooperation_level

        return score

    def _check_objective_completion(self, objective: Dict[str, Any]) -> bool:
        """Check if an objective has been completed."""
        obj_type = objective.get('type', 'deliver')
        obj_pos = objective['position']

        if obj_type == 'deliver':
            # Check if any friendly agent is near the objective
            friendly_agents = [
                aid for aid in self.agents.keys()
                if self._get_agent_team(aid) and self._get_agent_team(aid).team_type == 'friendly'
            ]

            for agent_id in friendly_agents:
                if self.agents[agent_id].position.distance_to(obj_pos) < 2.0:  # Delivery threshold
                    return True

        return False

    def _get_shared_knowledge(self) -> Dict[str, Any]:
        """Get knowledge shared between team members."""
        shared = {}

        for team in self.teams.values():
            team_knowledge = []
            for agent_id in team.agent_ids:
                # Collect recent messages received by this agent
                agent_messages = [
                    msg for msg in self.communication_log
                    if msg.receiver_id == agent_id and msg.received
                ][-5:]  # Last 5 messages

                team_knowledge.extend([msg.content for msg in agent_messages])

            shared[team.team_id] = team_knowledge

        return shared

    def activate_jamming(self, strength: float = 0.5):
        """
        Activate communication jamming.

        Args:
            strength: Jamming strength (0-1)
        """
        self.jamming_active = True
        self.jamming_strength = min(strength, 1.0)

    def deactivate_jamming(self):
        """Deactivate communication jamming."""
        self.jamming_active = False
        self.jamming_strength = 0.0

    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get summary statistics of multi-agent scenarios."""
        if not self.results_history:
            return {'total_scenarios': 0}

        total_scenarios = len(self.results_history)
        avg_duration = np.mean([r.duration for r in self.results_history])
        total_conflicts = sum([r.conflicts_encountered for r in self.results_history])
        total_messages = sum([sum(r.communication_stats.values()) for r in self.results_history])

        team_types = set()
        for result in self.results_history:
            team_types.update(result.team_scores.keys())

        return {
            'total_scenarios': total_scenarios,
            'average_duration': avg_duration,
            'total_conflicts': total_conflicts,
            'total_messages': total_messages,
            'team_types': list(team_types),
            'communication_efficiency': total_messages / max(total_scenarios, 1)
        }

    def export_scenario_data(self, output_path: Path):
        """
        Export multi-agent scenario data to JSON.

        Args:
            output_path: Path to save the export
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'teams': [
                {
                    'team_id': t.team_id,
                    'agent_ids': t.agent_ids,
                    'team_type': t.team_type,
                    'communication_range': t.communication_range,
                    'cooperation_level': t.cooperation_level,
                    'shared_objectives': t.shared_objectives
                }
                for t in self.teams.values()
            ],
            'agents': [
                {
                    'id': a.id,
                    'position': a.position.to_dict(),
                    'asset_type': a.asset_type,
                    'status': a.status
                }
                for a in self.agents.values()
            ],
            'communication_log': [
                {
                    'sender_id': msg.sender_id,
                    'receiver_id': msg.receiver_id,
                    'message_type': msg.message_type,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat(),
                    'reliability': msg.reliability,
                    'received': msg.received
                }
                for msg in self.communication_log
            ],
            'scenario_history': [
                {
                    'agent_states': state.agent_states,
                    'team_states': state.team_states,
                    'communication_network': state.communication_network,
                    'timestamp': state.timestamp.isoformat()
                }
                for state in self.scenario_history
            ],
            'results_history': [
                {
                    'scenario_id': r.scenario_id,
                    'duration': r.duration,
                    'team_scores': r.team_scores,
                    'communication_stats': r.communication_stats,
                    'objective_completion': r.objective_completion,
                    'conflicts_encountered': r.conflicts_encountered,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.results_history
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
