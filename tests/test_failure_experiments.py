import pytest

from experiments.run_experiment import parse_failure_probabilities


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0,0.05,0.1,0.2,0.3", [0.0, 0.05, 0.1, 0.2, 0.3]),
        ("0% 5% 10% 20% 30%", [0.0, 0.05, 0.1, 0.2, 0.3]),
    ],
)
def test_parse_failure_probabilities(raw, expected):
    assert parse_failure_probabilities(raw) == expected
