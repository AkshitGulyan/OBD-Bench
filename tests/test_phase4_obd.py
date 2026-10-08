import pytest

from agents.analysis_agent import AnalysisAgent
from agents.critic_agent import CriticAgent
from agents.final_compiler import FinalCompiler
from agents.research_agent import ResearchAgent
from orchestrators.obd import OBDOrchestrator
from workflow import Workflow


@pytest.mark.asyncio
async def test_obd_has_single_initial_and_final_orchestrator_calls():
    workflow = Workflow(
        workflow_id="workflow_3",
        agents=["A", "B", "C"],
        dependencies={"B": ["A"], "C": ["B"]},
    )

    orchestrator = OBDOrchestrator(
        agents=[
            ResearchAgent(agent_id="A", model="gpt-4o-mini"),
            AnalysisAgent(agent_id="B", model="gpt-4o-mini"),
            CriticAgent(agent_id="C", model="gpt-4o-mini"),
        ],
        final_compiler=FinalCompiler(agent_id="compiler", model="gpt-4o-mini"),
        model="gpt-4o-mini",
    )

    result = await orchestrator.run(task="Compute the final answer for this task.", workflow=workflow)

    assert result["architecture"] == "obd"
    assert result["total_orchestrator_calls"] == 2
    assert result["total_agent_calls"] == 4
    assert result["total_llm_calls"] == 6
    assert result["success"] is True
    assert result["workflow_size"] == 3
    assert result["output"]["status"] == "complete"
