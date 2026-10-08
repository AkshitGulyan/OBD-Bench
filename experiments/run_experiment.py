from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents.analysis_agent import AnalysisAgent
from agents.critic_agent import CriticAgent
from agents.final_compiler import FinalCompiler
from agents.research_agent import ResearchAgent
from config import load_config
from orchestrators.centralized import CentralizedOrchestrator
from orchestrators.obd import OBDOrchestrator
from orchestrators.sequential import SequentialOrchestrator
from workflow import Workflow

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TASKS_PATH = ROOT_DIR / "benchmarks" / "tasks.json"
DEFAULT_RESULTS_PATH = ROOT_DIR / "results" / "raw"


def parse_failure_probabilities(raw: str | None) -> list[float]:
    if raw is None or not str(raw).strip():
        return [0.0]
    candidates = []
    for token in re.split(r"[,\s]+", str(raw).strip()):
        if not token:
            continue
        value = token.strip().replace("%", "")
        if not value:
            continue
        try:
            numeric = float(value)
        except ValueError:
            continue
        if "%" in token:
            numeric = numeric / 100.0
        candidates.append(max(0.0, min(1.0, numeric)))
    if not candidates:
        return [0.0]
    return sorted(set(round(value, 4) for value in candidates))


async def run_single_experiment(
    task: dict[str, Any],
    workflow_size: int,
    architecture: str,
    repetition: int,
    config: dict[str, Any],
    failure_probability: float = 0.0,
) -> dict[str, Any]:
    workflow = build_workflow(workflow_size=workflow_size, workflow_id=f"workflow_{workflow_size}")
    agent_ids = [chr(65 + idx) for idx in range(workflow_size)]
    agents = [
        ResearchAgent(agent_id=agent_ids[0], model=config["model"]["name"]),
        AnalysisAgent(agent_id=agent_ids[1] if workflow_size > 1 else agent_ids[0], model=config["model"]["name"]),
    ]

    for idx in range(2, workflow_size):
        agents.append(CriticAgent(agent_id=agent_ids[idx], model=config["model"]["name"]))

    if architecture == "centralized":
        orchestrator = CentralizedOrchestrator(agents=agents, model=config["model"]["name"], task_id=task["task_id"])
    elif architecture == "sequential":
        orchestrator = SequentialOrchestrator(agents=agents, model=config["model"]["name"], task_id=task["task_id"])
    elif architecture == "obd":
        final_compiler = FinalCompiler(agent_id="compiler", model=config["model"]["name"])
        orchestrator = OBDOrchestrator(agents=agents, final_compiler=final_compiler, model=config["model"]["name"], task_id=task["task_id"])
    else:
        raise ValueError(f"Unsupported architecture: {architecture!r}")

    result = await orchestrator.run(task=task["input"], workflow=workflow)
    result["failure_probability"] = float(failure_probability)
    result["failure_injected"] = False

    if config.get("failure", {}).get("enabled", False) or failure_probability > 0:
        seed_base = int(config.get("experiment", {}).get("seed", 42))
        task_seed = ord(task["task_id"][0]) if task.get("task_id") else 0
        failure_rng = random.Random(
            seed_base + workflow_size * 1000 + repetition * 100 + int(failure_probability * 1000) + task_seed
        )
        if failure_rng.random() < float(failure_probability):
            result["failure_injected"] = True
            result["success"] = False
            result["status"] = "failed"
            result["failure"] = {
                "triggered": True,
                "probability": float(failure_probability),
                "phase": "simulation",
                "seed": seed_base,
            }
            result["output"] = result.get("output", {})
            if isinstance(result["output"], dict):
                result["output"]["failure_note"] = "Simulated agent failure injected at benchmark runtime."
            else:
                result["output"] = {"failure_note": "Simulated agent failure injected at benchmark runtime."}

            recovery_latency = int(200 + (failure_probability * 4000))
            recovery_tokens = int(100 + (failure_probability * 2500))
            result["total_latency_ms"] = int(result.get("total_latency_ms", 0)) + recovery_latency
            result["total_tokens"] = int(result.get("total_tokens", 0)) + recovery_tokens
            result["input_tokens"] = int(result.get("input_tokens", 0)) + max(10, int(recovery_tokens * 0.6))
            result["output_tokens"] = int(result.get("output_tokens", 0)) + max(5, int(recovery_tokens * 0.4))
            result["total_llm_calls"] = int(result.get("total_llm_calls", 0)) + 1
            result["total_orchestrator_calls"] = int(result.get("total_orchestrator_calls", 0)) + 1
            result["estimated_cost"] = float(result.get("estimated_cost", 0.0)) + (recovery_tokens / 1_000_000) * 0.0

    result["run_id"] = uuid4().hex
    result["task_id"] = task["task_id"]
    result["architecture"] = architecture
    result["workflow_size"] = workflow_size
    result["repetition"] = repetition
    result["model_name"] = config["model"]["name"]
    return result


def build_workflow(workflow_size: int, workflow_id: str) -> Workflow:
    agent_ids = [chr(65 + idx) for idx in range(workflow_size)]
    dependencies: dict[str, list[str]] = {}
    for idx, agent_id in enumerate(agent_ids):
        if idx == 0:
            dependencies[agent_id] = []
        else:
            dependencies[agent_id] = [agent_ids[idx - 1]]
    return Workflow(workflow_id=workflow_id, agents=agent_ids, dependencies=dependencies)


def load_tasks(path: Path | str = DEFAULT_TASKS_PATH) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


async def run_matrix(
    config: dict[str, Any],
    workflow_sizes: list[int],
    repetitions: int,
    task_file: Path | str = DEFAULT_TASKS_PATH,
    failure_probabilities: list[float] | None = None,
) -> Path:
    tasks = load_tasks(task_file)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = DEFAULT_RESULTS_PATH / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    if failure_probabilities is None:
        failure_probabilities = [float(config.get("failure", {}).get("probability", 0.0))]

    run_file = output_dir / "runs.jsonl"
    with run_file.open("w", encoding="utf-8") as handle:
        for failure_probability in failure_probabilities:
            for task in tasks:
                for workflow_size in workflow_sizes:
                    for architecture in config["workflows"]["architectures"]:
                        for repetition in range(repetitions):
                            result = await run_single_experiment(
                                task,
                                workflow_size,
                                architecture,
                                repetition,
                                config,
                                failure_probability=failure_probability,
                            )
                            result["failure_probability"] = float(failure_probability)
                            handle.write(json.dumps(result, sort_keys=True) + "\n")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OBD-Bench experiment matrix.")
    parser.add_argument("--tasks", type=str, default=str(DEFAULT_TASKS_PATH), help="Path to the benchmark tasks JSON file.")
    parser.add_argument("--sizes", type=str, default="2,4,6", help="Comma-separated workflow sizes to evaluate.")
    parser.add_argument("--repetitions", type=int, default=5, help="How many repetitions to run per task/workflow/architecture.")
    parser.add_argument(
        "--failures",
        type=str,
        default="0,0.05,0.10,0.20,0.30",
        help="Comma- or whitespace-separated failure probabilities to test, e.g. 0,0.05,0.1,0.2,0.3",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    config = load_config()
    workflow_sizes = [int(value.strip()) for value in args.sizes.split(",") if value.strip()]
    failure_probabilities = parse_failure_probabilities(args.failures)
    output_dir = await run_matrix(
        config=config,
        workflow_sizes=workflow_sizes,
        repetitions=args.repetitions,
        task_file=args.tasks,
        failure_probabilities=failure_probabilities,
    )
    print(f"Experiment output written to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
