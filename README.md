# OBD-Bench

OBD-Bench is a research-oriented benchmark for measuring the cost and behavior of Orchestrator Bypass Delegation (OBD) against standard centralized and sequential orchestration baselines.

## Research motivation

The benchmark is designed to compare orchestration strategies under controlled experimental conditions. The key research question is whether OBD can reduce orchestration overhead while preserving task quality and reliability.

## OBD concept

OBD begins with one initial orchestrator planning phase, then delegates execution across agents with result envelopes rather than repeated orchestrator checkpoints. The architecture is designed to reduce orchestrator call volume while keeping the task and model configuration consistent across baselines.

## Architectures

The benchmark includes three orchestration patterns:

1. Centralized orchestration
2. Sequential pipeline
3. Orchestrator Bypass Delegation (OBD)

All architectures receive the same workflow definition, same task, same model, same prompts, and same experimental conditions.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
```

## Configuration

Set the model and experiment settings in the environment or in `experiments/config.yaml`.

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_MODEL="gpt-4o-mini"
```

## Running an experiment

```bash
python -m experiments.run_experiment
```

The experiment runner writes raw results to `results/raw/` as timestamped directories.

## Reproducing figures

```bash
python -m experiments.generate_figures
```

Figures are written to `results/figures/`.

## Metrics

The benchmark records actual LLM events and computes totals from recorded values, including:

- orchestrator invocations
- total LLM invocations
- agent invocations
- input tokens
- output tokens
- total tokens
- estimated cost
- latency
- task success
- failure counts

## Benchmark tasks

The initial benchmark uses deterministic task structures designed to isolate orchestration overhead from subjective judgments.

## Limitations

This is a research benchmark designed for reproducibility, not product marketing. It does not assume that OBD is superior before measurement, and it keeps evaluation grounded in recorded experiment data.

## Phase 1 status

This repository now contains the shared infrastructure for configuration, workflow modeling, agent interfaces, and instrumentation needed for the first development phase.
