import json

import pytest

from experiments.run_experiment import build_workflow, run_single_experiment
from config import load_config


@pytest.mark.asyncio
async def test_run_single_experiment_returns_serializable_result():
    config = load_config()
    task = {
        "task_id": "task_test",
        "category": "reasoning_chain",
        "input": "Compute the total for 2, 4, 6.",
        "expected_properties": ["total"]
    }
    result = await run_single_experiment(task, workflow_size=2, architecture="centralized", repetition=0, config=config)

    assert result["architecture"] == "centralized"
    assert result["workflow_size"] == 2
    assert result["repetition"] == 0
    assert json.loads(json.dumps(result)) == result


def test_build_workflow_resolves_unique_dependencies():
    workflow = build_workflow(4, "workflow_4")
    assert workflow.resolve_order() == ["A", "B", "C", "D"]
    assert workflow.dependencies["C"] == ["B"]
