from __future__ import annotations

from typing import Any

from instrumentation import Instrumentation


def calculate_cost(input_tokens: int, output_tokens: int, input_price_per_1m: float, output_price_per_1m: float) -> float:
    return ((input_tokens / 1_000_000) * input_price_per_1m) + ((output_tokens / 1_000_000) * output_price_per_1m)


def aggregate_events(events: list[Any], input_price_per_1m: float = 0.0, output_price_per_1m: float = 0.0) -> dict[str, Any]:
    collector = Instrumentation(run_id="aggregate", architecture="aggregate")
    for event in events:
        collector.add_llm_event(event)
    return collector.aggregate_metrics(input_price_per_1m, output_price_per_1m)
