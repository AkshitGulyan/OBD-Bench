import pytest

from agents.analysis_agent import AnalysisAgent
from agents.critic_agent import CriticAgent
from agents.research_agent import ResearchAgent
from orchestrators.centralized import CentralizedOrchestrator
from workflow import Workflow


@pytest.mark.asyncio
async def test_centralized_orchestration_counts_calls_and_success():
    workflow = Workflow(
        workflow_id="workflow_3",
        agents=["A", "B", "C"],
        dependencies={"B": ["A"], "C": ["B"]},
    )

    orchestrator = CentralizedOrchestrator(
        agents=[
            ResearchAgent(agent_id="A", model="gpt-4o-mini"),
            AnalysisAgent(agent_id="B", model="gpt-4o-mini"),
            CriticAgent(agent_id="C", model="gpt-4o-mini"),
        ],
        model="gpt-4o-mini",
    )

    result = await orchestrator.run(task="Compute the final answer for this task.", workflow=workflow)

    assert result["architecture"] == "centralized"
    assert result["total_orchestrator_calls"] == 4
    assert result["total_agent_calls"] == 3
    assert result["total_llm_calls"] == 7
    assert result["success"] is True
    assert result["workflow_size"] == 3
