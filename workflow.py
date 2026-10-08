from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Workflow:
    workflow_id: str
    agents: list[str]
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        normalized = {}
        for agent in self.agents:
            normalized[agent] = list(self.dependencies.get(agent, []))
        self.dependencies = normalized

    def validate(self) -> None:
        for agent in self.agents:
            for dependency in self.dependencies.get(agent, []):
                if dependency not in self.agents:
                    raise ValueError(f"Dependency {dependency!r} for agent {agent!r} is not declared in the workflow.")
        self.resolve_order()

    def resolve_order(self) -> list[str]:
        in_degree = {agent: 0 for agent in self.agents}
        dependents: dict[str, list[str]] = defaultdict(list)

        for agent in self.agents:
            for dependency in self.dependencies.get(agent, []):
                in_degree[agent] += 1
                dependents[dependency].append(agent)

        queue = deque(sorted(agent for agent, degree in in_degree.items() if degree == 0))
        order: list[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for dependent in sorted(dependents.get(current, [])):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(self.agents):
            raise ValueError(f"Workflow {self.workflow_id!r} contains a cycle or invalid dependency graph.")
        return order

    def get_upstream(self, agent_id: str) -> list[str]:
        return list(self.dependencies.get(agent_id, []))

    def get_downstream(self, agent_id: str) -> list[str]:
        dependents: list[str] = []
        for candidate, upstream in self.dependencies.items():
            if agent_id in upstream:
                dependents.append(candidate)
        return dependents

    def __len__(self) -> int:
        return len(self.agents)
