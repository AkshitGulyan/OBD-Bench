from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class LLMEvent:
    timestamp: str = field(default_factory=utc_timestamp)
    run_id: str = ""
    architecture: str = ""
    agent_id: str = ""
    call_type: str = "agent"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    status: str = "success"
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Instrumentation:
    def __init__(self, run_id: str, architecture: str, task_id: str = "", workflow_size: int = 0):
        self.run_id = run_id
        self.architecture = architecture
        self.task_id = task_id
        self.workflow_size = workflow_size
        self.events: list[LLMEvent] = []

    def add_llm_event(self, event: LLMEvent) -> None:
        if event.run_id == "":
            event.run_id = self.run_id
        if event.architecture == "":
            event.architecture = self.architecture
        self.events.append(event)

    def aggregate_metrics(self, input_price_per_1m: float = 0.0, output_price_per_1m: float = 0.0) -> dict[str, Any]:
        total_input = sum(event.input_tokens for event in self.events)
        total_output = sum(event.output_tokens for event in self.events)
        total_tokens = total_input + total_output
        estimated_cost = (
            (total_input / 1_000_000) * input_price_per_1m
            + (total_output / 1_000_000) * output_price_per_1m
        )

        return {
            "run_id": self.run_id,
            "architecture": self.architecture,
            "task_id": self.task_id,
            "workflow_size": self.workflow_size,
            "total_llm_calls": len(self.events),
            "total_agent_calls": sum(1 for event in self.events if event.call_type in {"agent", "compiler"}),
            "total_orchestrator_calls": sum(1 for event in self.events if event.call_type == "orchestrator"),
            "total_compiler_calls": sum(1 for event in self.events if event.call_type == "compiler"),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "estimated_cost": estimated_cost,
        }

    def finalize_run(self, success: bool, total_latency_ms: int, **extra: Any) -> dict[str, Any]:
        metrics = self.aggregate_metrics(**extra.get("pricing", {}))
        metrics.update(
            {
                "success": success,
                "total_latency_ms": total_latency_ms,
                "run_id": self.run_id,
                "task_id": self.task_id,
                "architecture": self.architecture,
                "workflow_size": self.workflow_size,
            }
        )
        metrics.update({key: value for key, value in extra.items() if key not in {"pricing"}})
        return metrics
