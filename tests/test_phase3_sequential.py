import pytest

from agents.analysis_agent import AnalysisAgent
from agents.critic_agent import CriticAgent
from agents.research_agent import ResearchAgent
from orchestrators.sequential import SequentialOrchestrator
from workflow import Workflow


@pytest.mark.asyncio
async def test_sequential_orchestration_has_no_intermediate_orchestrator_calls():
    workflow = Workflow(
        workflow_id="workflow_3",
        agents=["A", "B", "C"],
        dependencies={"B": ["A"], "C": ["B"]},
    )

    orchestrator = SequentialOrchestrator(
        agents=[
            ResearchAgent(agent_id="A", model="gpt-4o-mini"),
            AnalysisAgent(agent_id="B", model="gpt-4o-mini"),
            CriticAgent(agent_id="C", model="gpt-4o-mini"),
        ],
        model="gpt-4o-mini",
    )

    result = await orchestrator.run(task="Compute the final answer for this task.", workflow=workflow)

    assert result["architecture"] == "sequential"
    assert result["total_orchestrator_calls"] == 0
    assert result["total_agent_calls"] == 3
    assert result["total_llm_calls"] == 3
    assert result["success"] is True
    assert result["workflow_size"] == 3
