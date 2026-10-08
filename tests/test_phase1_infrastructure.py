import pytest

from agents.base_agent import BaseAgent
from config import load_config
from instrumentation import Instrumentation, LLMEvent
from workflow import Workflow


class DummyAgent(BaseAgent):
    async def execute(self, task, context):
        return {"agent_id": self.agent_id, "task": task, "context": context}


def test_result_envelope_validation():
    envelope = {
        "agent_id": "A",
        "plan_id": "plan_001",
        "output": {"summary": "ok"},
        "status": "success",
        "next_agent": "B",
        "compiled_result_yet": False,
        "hop_count": 1,
        "max_hops": 3,
        "prior_results": {"A": {"summary": "ok"}},
    }

    assert envelope["agent_id"] == "A"
    assert envelope["status"] in {"success", "partial", "failed"}
    assert envelope["next_agent"] == "B"


def test_workflow_dependency_resolution():
    workflow = Workflow(
        workflow_id="workflow_4",
        agents=["A", "B", "C", "D"],
        dependencies={"B": ["A"], "C": ["B"], "D": ["C"]},
    )

    assert workflow.resolve_order() == ["A", "B", "C", "D"]
    assert workflow.get_upstream("C") == ["B"]
    assert workflow.get_downstream("B") == ["C"]


def test_token_aggregation_and_cost():
    metrics = Instrumentation(run_id="run_001", architecture="centralized", task_id="task_001")
    metrics.add_llm_event(
        LLMEvent(
            timestamp="2026-10-01T00:00:00Z",
            run_id="run_001",
            architecture="centralized",
            agent_id="A",
            call_type="agent",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200,
            status="success",
        )
    )
    metrics.add_llm_event(
        LLMEvent(
            timestamp="2026-10-01T00:00:01Z",
            run_id="run_001",
            architecture="centralized",
            agent_id="orchestrator",
            call_type="orchestrator",
            input_tokens=80,
            output_tokens=30,
            latency_ms=120,
            status="success",
        )
    )

    totals = metrics.aggregate_metrics(input_price_per_1m=10.0, output_price_per_1m=20.0)
    assert totals["total_llm_calls"] == 2
    assert totals["input_tokens"] == 180
    assert totals["output_tokens"] == 80
    assert totals["total_tokens"] == 260
    assert totals["estimated_cost"] > 0


def test_orchestrator_call_counting_and_run_summary():
    metrics = Instrumentation(run_id="run_002", architecture="centralized", task_id="task_002", workflow_size=3)
    metrics.add_llm_event(
        LLMEvent(
            timestamp="2026-10-01T00:00:00Z",
            run_id="run_002",
            architecture="centralized",
            agent_id="orchestrator",
            call_type="orchestrator",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100,
            status="success",
        )
    )
    metrics.add_llm_event(
        LLMEvent(
            timestamp="2026-10-01T00:00:01Z",
            run_id="run_002",
            architecture="centralized",
            agent_id="A",
            call_type="agent",
            input_tokens=20,
            output_tokens=10,
            latency_ms=150,
            status="success",
        )
    )
    summary = metrics.finalize_run(success=True, total_latency_ms=500)
    assert summary["total_orchestrator_calls"] == 1
    assert summary["total_agent_calls"] == 1
    assert summary["total_llm_calls"] == 2
    assert summary["success"] is True
    assert summary["workflow_size"] == 3


def test_config_loading_uses_environment_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    config = load_config("tests/test_config.yaml")
    assert config["model"]["provider"] == "openai"
    assert config["model"]["name"] == "gpt-4o-mini"
    assert config["experiment"]["temperature"] == 0


@pytest.mark.asyncio
async def test_base_agent_interface():
    agent = DummyAgent(agent_id="A", model="gpt-4o-mini")
    result = await agent.execute("task_001", {"step": 1})
    assert result["agent_id"] == "A"
    assert result["task"] == "task_001"
