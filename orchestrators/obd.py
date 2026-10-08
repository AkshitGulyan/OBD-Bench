from __future__ import annotations

import time
import uuid
from typing import Any

from instrumentation import Instrumentation, LLMEvent
from orchestrators.base_orchestrator import BaseRunResult


class OBDOrchestrator:
    def __init__(self, agents: list[Any], final_compiler: Any, model: str = "gpt-4o-mini", task_id: str = "task_001"):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.final_compiler = final_compiler
        self.model = model
        self.task_id = task_id

    async def run(self, task: str, workflow: Any) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        instrumentation = Instrumentation(run_id=run_id, architecture="obd", task_id=self.task_id, workflow_size=len(workflow.agents))
        workflow_order = workflow.resolve_order()
        prior_results: dict[str, Any] = {}
        start_time = time.perf_counter()

        plan = {
            "plan_id": f"plan_{run_id[:8]}",
            "intent": f"Execute the {len(workflow_order)}-stage workflow for the provided task.",
            "execution_order": workflow_order,
            "tasks": [
                {
                    "agent_id": agent_id,
                    "objective": f"Execute step {idx} for {agent_id}",
                    "context": {"task": task, "workflow_id": workflow.workflow_id, "depends_on": workflow.get_upstream(agent_id)},
                    "output_schema": {"type": "object"},
                    "depends_on": workflow.get_upstream(agent_id),
                    "is_final": agent_id == workflow_order[-1],
                }
                for idx, agent_id in enumerate(workflow_order)
            ],
            "compilation_strategy": "final_compiler",
        }

        planning_event = self._make_event(
            run_id=run_id,
            architecture="obd",
            agent_id="orchestrator",
            call_type="orchestrator",
            input_tokens=self._estimate_tokens(str({"task": task, "plan": plan})),
            output_tokens=self._estimate_tokens(str(plan)),
            latency_ms=15,
            status="success",
        )
        instrumentation.add_llm_event(planning_event)

        result_envelopes: dict[str, Any] = {}
        for index, agent_id in enumerate(workflow_order):
            agent = self.agents[agent_id]
            context = {
                "task": task,
                "workflow_id": workflow.workflow_id,
                "plan_id": plan["plan_id"],
                "prior_results": prior_results,
                "depends_on": workflow.get_upstream(agent_id),
                "hop_count": index + 1,
                "max_hops": len(workflow_order),
            }
            output = await agent.execute(task, context)
            envelope = {
                "agent_id": agent_id,
                "plan_id": plan["plan_id"],
                "output": output if isinstance(output, dict) else {"value": output},
                "status": "success",
                "next_agent": workflow_order[index + 1] if index + 1 < len(workflow_order) else self.final_compiler.agent_id,
                "compiled_result_yet": False,
                "hop_count": index + 1,
                "max_hops": len(workflow_order),
                "prior_results": prior_results.copy(),
            }
            prior_results[agent_id] = envelope["output"]
            result_envelopes[agent_id] = envelope

            agent_event = self._make_event(
                run_id=run_id,
                architecture="obd",
                agent_id=agent_id,
                call_type="agent",
                input_tokens=self._estimate_tokens(str({"task": task, "context": context})),
                output_tokens=self._estimate_tokens(str(envelope)),
                latency_ms=12,
                status="success",
            )
            instrumentation.add_llm_event(agent_event)

        compiler_context = {
            "task": task,
            "workflow_id": workflow.workflow_id,
            "plan_id": plan["plan_id"],
            "result_envelopes": result_envelopes,
            "prior_results": prior_results,
            "final_compilation": True,
        }
        compiler_output = await self.final_compiler.execute(task, compiler_context)
        compiler_envelope = {
            "agent_id": self.final_compiler.agent_id,
            "plan_id": plan["plan_id"],
            "output": compiler_output if isinstance(compiler_output, dict) else {"value": compiler_output},
            "status": "success",
            "next_agent": None,
            "compiled_result_yet": True,
            "hop_count": len(workflow_order) + 1,
            "max_hops": len(workflow_order) + 1,
            "prior_results": prior_results.copy(),
        }
        compiler_event = self._make_event(
            run_id=run_id,
            architecture="obd",
            agent_id=self.final_compiler.agent_id,
            call_type="compiler",
            input_tokens=self._estimate_tokens(str(compiler_context)),
            output_tokens=self._estimate_tokens(str(compiler_envelope)),
            latency_ms=14,
            status="success",
        )
        instrumentation.add_llm_event(compiler_event)

        final_compilation = {
            "plan_id": plan["plan_id"],
            "final_output": compiler_envelope["output"],
            "summarized_results": result_envelopes,
            "status": "complete",
        }
        final_orchestrator_event = self._make_event(
            run_id=run_id,
            architecture="obd",
            agent_id="orchestrator",
            call_type="orchestrator",
            input_tokens=self._estimate_tokens(str({"results": result_envelopes, "compiler": compiler_envelope})),
            output_tokens=self._estimate_tokens(str(final_compilation)),
            latency_ms=18,
            status="success",
        )
        instrumentation.add_llm_event(final_orchestrator_event)

        total_latency_ms = int((time.perf_counter() - start_time) * 1000)
        metrics = instrumentation.aggregate_metrics(input_price_per_1m=0.0, output_price_per_1m=0.0)
        result = BaseRunResult(
            architecture="obd",
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
            output={"task": task, "status": "complete", "plan_id": plan["plan_id"], "results": result_envelopes, "final_output": compiler_envelope["output"]},
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
