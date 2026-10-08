from __future__ import annotations

import time
import uuid
from typing import Any

from instrumentation import Instrumentation, LLMEvent
from orchestrators.base_orchestrator import BaseRunResult


class SequentialOrchestrator:
    def __init__(self, agents: list[Any], model: str = "gpt-4o-mini", task_id: str = "task_001"):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.model = model
        self.task_id = task_id

    async def run(self, task: str, workflow: Any) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        instrumentation = Instrumentation(run_id=run_id, architecture="sequential", task_id=self.task_id, workflow_size=len(workflow.agents))
        workflow_order = workflow.resolve_order()
        previous_results: dict[str, Any] = {}
        final_summary: dict[str, Any] = {"task": task, "results": {}}
        start_time = time.perf_counter()

        for agent_id in workflow_order:
            agent = self.agents[agent_id]
            context = {
                "task": task,
                "workflow_id": workflow.workflow_id,
                "dependencies": workflow.dependencies,
                "previous_results": previous_results,
            }
            agent_result = await agent.execute(task, context)
            if isinstance(agent_result, dict):
                payload = agent_result
            else:
                payload = {"value": agent_result}
            payload["agent_id"] = agent_id
            previous_results[agent_id] = payload
            final_summary["results"][agent_id] = payload

            agent_event = self._make_event(
                run_id=run_id,
                architecture="sequential",
                agent_id=agent_id,
                call_type="agent",
                input_tokens=self._estimate_tokens(str({"task": task, "context": context})),
                output_tokens=self._estimate_tokens(str(payload)),
                latency_ms=12,
                status="success",
            )
            instrumentation.add_llm_event(agent_event)

        final_answer = {
            "task": task,
            "final_summary": final_summary["results"],
            "status": "complete",
        }
        total_latency_ms = int((time.perf_counter() - start_time) * 1000)
        metrics = instrumentation.aggregate_metrics(input_price_per_1m=0.0, output_price_per_1m=0.0)
        result = BaseRunResult(
            architecture="sequential",
            run_id=run_id,
            task_id=self.task_id,
            workflow_size=len(workflow.agents),
            success=True,
            total_latency_ms=total_latency_ms,
            total_llm_calls=metrics["total_llm_calls"],
            total_agent_calls=metrics["total_agent_calls"],
            total_orchestrator_calls=metrics["total_orchestrator_calls"],
            input_tokens=metrics["input_tokens"],
            output_tokens=metrics["output_tokens"],
            total_tokens=metrics["total_tokens"],
            estimated_cost=metrics["estimated_cost"],
            events=[event.__dict__ for event in instrumentation.events],
            output=final_answer,
        )
        return result.as_dict()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text.split()))

    @staticmethod
    def _make_event(run_id: str, architecture: str, agent_id: str, call_type: str, input_tokens: int, output_tokens: int, latency_ms: int, status: str) -> LLMEvent:
        return LLMEvent(
            timestamp="",
            run_id=run_id,
            architecture=architecture,
            agent_id=agent_id,
            call_type=call_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            status=status,
            model="gpt-4o-mini",
        )
