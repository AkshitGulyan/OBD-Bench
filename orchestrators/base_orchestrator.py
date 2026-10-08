from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseRunResult:
    architecture: str
    run_id: str
    task_id: str
    workflow_size: int
    success: bool
    total_latency_ms: int
    total_llm_calls: int
    total_agent_calls: int
    total_orchestrator_calls: int
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    output: Any = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "architecture": self.architecture,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "workflow_size": self.workflow_size,
            "success": self.success,
            "total_latency_ms": self.total_latency_ms,
            "total_llm_calls": self.total_llm_calls,
            "total_agent_calls": self.total_agent_calls,
            "total_orchestrator_calls": self.total_orchestrator_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "output": self.output,
            "events": self.events,
        }
        return payload
