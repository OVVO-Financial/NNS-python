from __future__ import annotations

import numpy as np
import pytest

from nns.stack import nns_stack


def test_nns_stack_method1_uses_pooled_oof_selection_not_fold_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Method 1 must score conceptual candidates after pooling all folds."""

    import nns.stack as stack_mod

    x = np.arange(12, dtype=np.float64)
    variable = np.column_stack((x, x**2))
    y = np.zeros(12, dtype=np.float64)

    folds = [
        (np.arange(4, 12, dtype=np.int64), np.arange(0, 4, dtype=np.int64)),
        (np.r_[0:4, 8:12].astype(np.int64), np.arange(4, 8, dtype=np.int64)),
        (np.arange(0, 8, dtype=np.int64), np.arange(8, 12, dtype=np.int64)),
    ]
    split_iter = iter(folds)
    monkeypatch.setattr(stack_mod, "_cv_split", lambda *_args, **_kwargs: next(split_iter))

    class Model:
        rpm = np.zeros((3, 3), dtype=np.float64)

    monkeypatch.setattr(stack_mod, "_mreg_prepare_model", lambda *_args, **_kwargs: Model())

    paths = iter(
        [
            {1: np.zeros(4), 2: np.full(4, 10.0), 3: np.full(4, 20.0)},
            {1: np.full(4, 2.0), 2: np.zeros(4), 3: np.full(4, 20.0)},
            {1: np.full(4, 2.0), 2: np.ones(4), 3: np.full(4, 20.0)},
        ]
    )
    monkeypatch.setattr(stack_mod, "_mreg_predict_path", lambda *_args, **_kwargs: next(paths))

    selected: list[int | str | None] = []

    def fake_nns_reg(*_args: object, **kwargs: object) -> dict[str, object]:
        selected.append(kwargs.get("n_best"))
        n = np.asarray(kwargs["point_est"]).shape[0]
        return {
            "Point.est": np.zeros(n),
            "Fitted.xy": {"y.hat": np.zeros(12), "y": np.zeros(12)},
            "pred.int": None,
        }

    monkeypatch.setattr(stack_mod, "nns_reg", fake_nns_reg)
    result = nns_stack(variable, y, variable[:2], folds=3, method=1)

    # Fold-local winners are 1, 2, 2, but pooled SSE chooses candidate 1.
    assert result["NNS.reg.n.best"] == 1.0
    assert selected[-1] == 1


def test_nns_stack_method1_all_is_not_encoded_as_training_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ALL remains conceptual until it resolves against the prepared RPM."""

    import nns.stack as stack_mod

    x = np.linspace(-1.0, 1.0, 16)
    variable = np.column_stack((x, x**2))
    y = x
    requested_paths: list[tuple[int, ...]] = []

    class Model:
        rpm = np.zeros((5, 3), dtype=np.float64)

    monkeypatch.setattr(stack_mod, "_mreg_prepare_model", lambda *_args, **_kwargs: Model())

    def fake_path(*args: object, **kwargs: object) -> dict[int, np.ndarray]:
        ks = tuple(int(k) for k in kwargs["k_values"])
        requested_paths.append(ks)
        n = np.asarray(args[1]).shape[0]
        return {k: np.zeros(n) for k in ks}

    monkeypatch.setattr(stack_mod, "_mreg_predict_path", fake_path)
    nns_stack(variable, y, variable[:2], folds=1, method=1)

    assert requested_paths
    assert 16 not in requested_paths[0]
    assert max(requested_paths[0]) <= int(np.floor(np.sqrt(16)))
