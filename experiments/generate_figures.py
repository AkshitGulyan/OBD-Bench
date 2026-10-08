from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median, pstdev

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "results" / "raw"
FIGURES_DIR = ROOT_DIR / "results" / "figures"


def load_latest_results() -> list[dict]:
    directories = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not directories:
        raise FileNotFoundError(f"No raw experiment directories found under {RAW_DIR}")
    latest_dir = directories[-1]
    path = latest_dir / "runs.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No runs.jsonl found in {latest_dir}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize_by_architecture(rows: list[dict]) -> dict[str, dict[int, list[dict]]]:
    grouped: dict[str, dict[int, list[dict]]] = {}
    for row in rows:
        architecture = row["architecture"]
        workflow_size = row["workflow_size"]
        grouped.setdefault(architecture, {}).setdefault(workflow_size, []).append(row)
    return grouped


def summarize_results(rows: list[dict]) -> dict[str, dict[str, object]]:
    grouped = summarize_by_architecture(rows)
    summary: dict[str, dict[str, object]] = {}
    for architecture, by_size in grouped.items():
        all_runs = [run for size_runs in by_size.values() for run in size_runs]
        if not all_runs:
            continue
        by_size_summary = {}
        for size in sorted(by_size):
            size_runs = by_size[size]
            by_size_summary[str(size)] = {
                "n_runs": len(size_runs),
                "mean_total_orchestrator_calls": mean(float(run["total_orchestrator_calls"]) for run in size_runs),
                "mean_total_llm_calls": mean(float(run["total_llm_calls"]) for run in size_runs),
                "mean_total_tokens": mean(float(run["total_tokens"]) for run in size_runs),
                "success_rate": sum(1 for run in size_runs if run.get("success")) / len(size_runs),
            }
        summary[architecture] = {
            "n_runs": len(all_runs),
            "sizes": sorted(by_size),
            "mean_total_orchestrator_calls": mean(float(run["total_orchestrator_calls"]) for run in all_runs),
            "mean_total_llm_calls": mean(float(run["total_llm_calls"]) for run in all_runs),
            "mean_total_tokens": mean(float(run["total_tokens"]) for run in all_runs),
            "success_rate": sum(1 for run in all_runs if run.get("success")) / len(all_runs),
            "by_size": by_size_summary,
        }
    return summary


def summarize_failure_results(rows: list[dict]) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[float, list[dict]]] = {}
    for row in rows:
        architecture = row["architecture"]
        failure_probability = float(row.get("failure_probability", 0.0))
        grouped.setdefault(architecture, {}).setdefault(failure_probability, []).append(row)

    summary: dict[str, dict[str, object]] = {}
    for architecture, by_failure in grouped.items():
        summary[architecture] = {}
        for failure_probability in sorted(by_failure):
            runs = by_failure[failure_probability]
            summary[architecture][str(failure_probability)] = {
                "n_runs": len(runs),
                "success_rate": sum(1 for run in runs if run.get("success")) / len(runs),
                "mean_total_latency_ms": mean(float(run["total_latency_ms"]) for run in runs),
                "mean_total_tokens": mean(float(run["total_tokens"]) for run in runs),
                "mean_total_orchestrator_calls": mean(float(run["total_orchestrator_calls"]) for run in runs),
                "mean_total_llm_calls": mean(float(run["total_llm_calls"]) for run in runs),
            }
    return summary


def metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def generate_plot(rows: list[dict], metric: str, title: str, output_name: str) -> None:
    if plt is None:
        return
    grouped = summarize_by_architecture(rows)
    architectures = ["centralized", "sequential", "obd"]
    sizes = sorted({row["workflow_size"] for row in rows})
    plt.figure(figsize=(8, 5))
    for architecture in architectures:
        if architecture not in grouped:
            continue
        series = []
        for size in sizes:
            records = grouped[architecture].get(size, [])
            if not records:
                continue
            values = [float(r.get(metric, 0.0)) for r in records]
            series.append(sum(values) / len(values))
        plt.plot(sizes, series, marker="o", label=architecture)
    plt.title(title)
    plt.xlabel("Workflow size")
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=200)
    plt.close()


def generate_failure_plot(rows: list[dict], output_name: str) -> None:
    if plt is None:
        return
    grouped = summarize_failure_results(rows)
    plt.figure(figsize=(8, 5))
    for architecture in ["centralized", "sequential", "obd"]:
        if architecture not in grouped:
            continue
        failure_levels = sorted(float(prob) for prob in grouped[architecture])
        success_rates = [grouped[architecture][str(level)]["success_rate"] for level in failure_levels]
        plt.plot(failure_levels, success_rates, marker="o", label=architecture)
    plt.title("Success rate vs failure probability")
    plt.xlabel("Failure probability")
    plt.ylabel("Success rate")
    plt.xticks(failure_levels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=200)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_latest_results()
    summary = summarize_results(rows)
    summary_path = FIGURES_DIR / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failure_summary = summarize_failure_results(rows)
    failure_summary_path = FIGURES_DIR / "failure_summary.json"
    failure_summary_path.write_text(json.dumps(failure_summary, indent=2), encoding="utf-8")

    metrics = [
        ("total_orchestrator_calls", "Orchestrator calls vs number of agents", "orchestrator_calls_vs_agents.png"),
        ("total_llm_calls", "Total LLM calls vs number of agents", "llm_calls_vs_agents.png"),
        ("total_tokens", "Total token usage vs number of agents", "tokens_vs_agents.png"),
        ("total_latency_ms", "Latency vs number of agents", "latency_vs_agents.png"),
        ("success", "Task success rate vs number of agents", "success_vs_agents.png"),
        ("estimated_cost", "Estimated cost vs number of agents", "cost_vs_agents.png"),
    ]
    for metric, title, output_name in metrics:
        generate_plot(rows, metric, title, output_name)

    generate_failure_plot(rows, "failure_success_vs_probability.png")
    print(f"Figures saved to {FIGURES_DIR}")
    print(f"Summary saved to {summary_path}")
    print(f"Failure summary saved to {failure_summary_path}")


if __name__ == "__main__":
    main()
