import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "provider": "openai",
        "name": "gpt-4o-mini",
        "base_url": "",
        "api_key": "",
    },
    "experiment": {
        "repetitions": 20,
        "temperature": 0,
        "seed": 42,
    },
    "pricing": {
        "input_cost_per_1m_tokens": 0.0,
        "output_cost_per_1m_tokens": 0.0,
    },
    "workflows": {
        "agent_counts": [2, 4, 6, 8, 10],
        "architectures": ["centralized", "sequential", "obd"],
    },
    "failure": {
        "enabled": False,
        "probability": 0.0,
    },
}


def _resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")
        return pattern.sub(lambda match: os.getenv(match.group(1), match.group(0)), value)
    return value


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    target_path = Path(config_path) if config_path else ROOT_DIR / "experiments" / "config.yaml"
    config_data: dict[str, Any] = deepcopy(DEFAULT_CONFIG)

    if target_path.exists():
        with target_path.open("r", encoding="utf-8") as handle:
            file_data = yaml.safe_load(handle) or {}
        config_data = _merge_dicts(config_data, _resolve_env_placeholders(file_data))

    env_overrides: dict[str, Any] = {"model": {}, "experiment": {}, "failure": {}}

    model_provider = os.getenv("MODEL_PROVIDER") or os.getenv("LLM_PROVIDER")
    model_name = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_NAME")
    model_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("MODEL_BASE_URL")
    model_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MODEL_API_KEY")
    if model_provider:
        env_overrides["model"]["provider"] = model_provider
    if model_name:
        env_overrides["model"]["name"] = model_name
    if model_base_url:
        env_overrides["model"]["base_url"] = model_base_url
    if model_api_key:
        env_overrides["model"]["api_key"] = model_api_key

    experiment_temperature = _coerce_int_or_float(os.getenv("EXPERIMENT_TEMPERATURE"))
    experiment_repetitions = _coerce_int(os.getenv("EXPERIMENT_REPETITIONS"))
    if experiment_temperature is not None:
        env_overrides["experiment"]["temperature"] = experiment_temperature
    if experiment_repetitions is not None:
        env_overrides["experiment"]["repetitions"] = experiment_repetitions

    failure_enabled = _coerce_bool(os.getenv("FAILURE_ENABLED"))
    failure_probability = _coerce_float(os.getenv("FAILURE_PROBABILITY"))
    if failure_enabled is not None:
        env_overrides["failure"]["enabled"] = failure_enabled
    if failure_probability is not None:
        env_overrides["failure"]["probability"] = failure_probability

    config_data = _merge_dicts(config_data, _resolve_env_placeholders(env_overrides))

    if not config_data["model"].get("api_key"):
        config_data["model"]["api_key"] = os.getenv("OPENAI_API_KEY") or ""

    return config_data


def _coerce_int_or_float(value: str | None):
    if value is None:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: str | None):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: str | None):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: str | None):
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
