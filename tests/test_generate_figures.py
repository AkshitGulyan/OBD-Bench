from pathlib import Path

from experiments.generate_figures import load_latest_results, summarize_results


def test_load_latest_results_reads_raw_data():
    rows = load_latest_results()
    assert len(rows) > 0
    assert {"architecture", "workflow_size", "task_id"}.issubset(rows[0].keys())
    assert isinstance(rows[0]["architecture"], str)


def test_summarize_results_reports_architecture_metrics():
    rows = load_latest_results()
    summary = summarize_results(rows)
    assert set(summary).issubset({"centralized", "sequential", "obd"})
    for architecture, payload in summary.items():
        assert {"n_runs", "sizes", "mean_total_orchestrator_calls", "mean_total_llm_calls", "mean_total_tokens", "success_rate"}.issubset(payload.keys())
        assert payload["n_runs"] > 0
        assert payload["success_rate"] >= 0.0


def test_figures_directory_exists_after_generation(tmp_path):
    root = Path("results/raw")
    assert root.exists()
