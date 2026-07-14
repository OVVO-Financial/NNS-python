from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, found {count}\nOLD:\n{old}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


bounded_helper = '''def _r2(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    """Racine-Hastie within-sample goodness-of-fit, bounded in [0, 1]."""
    actual_centered = actual - float(np.mean(actual))
    predicted_centered = predicted - float(np.mean(predicted))
    denominator = float(np.sum(actual_centered**2) * np.sum(predicted_centered**2))
    if denominator == 0.0:
        return 1.0 if np.array_equal(actual, predicted) else 0.0
    numerator = float(np.sum(actual_centered * predicted_centered) ** 2)
    return float(np.clip(numerator / denominator, 0.0, 1.0))
'''

replace_once(
    "src/nns/_reg_engine.py",
    '''def _r2(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    sse = float(np.sum((actual - predicted) ** 2))
    sst = float(np.sum((actual - np.mean(actual)) ** 2))
    if sst == 0.0:
        return 1.0 if sse == 0.0 else 0.0
    return 1.0 - sse / sst
''',
    bounded_helper,
)

replace_once(
    "src/nns/regression.py",
    '''def _r2(y: NDArray[np.float64], y_hat: NDArray[np.float64]) -> float:
    y_mean = float(np.mean(y))
    numerator = float(np.sum((y - y_mean) * (y_hat - y_mean)) ** 2)
    denominator = float(np.sum((y - y_mean) ** 2) * np.sum((y_hat - y_mean) ** 2))
    return numerator / denominator if denominator > 0.0 else float("nan")
''',
    bounded_helper.replace("actual", "y").replace("predicted", "y_hat"),
)

replace_once(
    "src/nns/regression.py",
    '''    """Repaired NNS.reg port: one consistent prediction rule, real distance
    dispatch, training-fitted encodings, predictive R2. Plotting arguments and
    the legacy class_levels/factor_levels emulation parameters are accepted for
    API compatibility and ignored."""
''',
    '''    """Repaired NNS.reg port with the original bounded Racine-Hastie R2.

    Plotting arguments and the legacy class_levels/factor_levels emulation
    parameters are accepted for API compatibility and ignored.
    """
''',
)

for path in (
    "tests/property/test_regression.py",
    "tests/property/test_multivariate_regression.py",
):
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    old = 'assert np.isnan(result["R2"]) or result["R2"] <= 1.0 + 1e-12'
    count = text.count(old)
    if count < 1:
        raise RuntimeError(f"{path}: no stale R2 property assertion found")
    text = text.replace(old, 'assert 0.0 <= result["R2"] <= 1.0')
    file_path.write_text(text, encoding="utf-8")

replace_once(
    "tests/_r.py",
    '_NNS_VERSION = "13.1-21be6d9"',
    '_NNS_VERSION = "13.1-racine-hastie-r2"',
)


def squared_correlation(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("R2 cache vectors must be nonempty and have equal length")
    actual_mean = math.fsum(actual) / len(actual)
    predicted_mean = math.fsum(predicted) / len(predicted)
    actual_centered = [value - actual_mean for value in actual]
    predicted_centered = [value - predicted_mean for value in predicted]
    denominator = math.fsum(value * value for value in actual_centered) * math.fsum(
        value * value for value in predicted_centered
    )
    if denominator == 0.0:
        return 1.0 if actual == predicted else 0.0
    numerator = math.fsum(
        left * right for left, right in zip(actual_centered, predicted_centered, strict=True)
    ) ** 2
    return min(1.0, max(0.0, numerator / denominator))


def numeric_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def migrate(node: Any) -> int:
    updated = 0
    if isinstance(node, dict):
        fitted = node.get("Fitted.xy")
        class_levels = node.get("class.levels")
        prediction_accuracy = node.get("Prediction.Accuracy")
        is_continuous = prediction_accuracy is None and class_levels in (None, [], "")
        if is_continuous and "R2" in node and isinstance(fitted, dict):
            actual = numeric_vector(fitted.get("y"))
            predicted = numeric_vector(fitted.get("y.hat"))
            if actual is not None and predicted is not None and len(actual) == len(predicted) and actual:
                node["R2"] = squared_correlation(actual, predicted)
                updated += 1
        for value in node.values():
            updated += migrate(value)
    elif isinstance(node, list):
        for value in node:
            updated += migrate(value)
    return updated


cache_path = Path("tests/_r_cache.json")
payload = json.loads(cache_path.read_text(encoding="utf-8"))
if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
    raise RuntimeError("Unexpected R cache structure")
updated = migrate(payload["entries"])
if updated < 1:
    raise RuntimeError("No continuous regression R2 cache entries were migrated")
payload["nns_version"] = "13.1-racine-hastie-r2"
cache_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(f"Migrated {updated} continuous regression R2 cache values.")
