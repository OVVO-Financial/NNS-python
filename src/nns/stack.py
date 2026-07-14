"""Repaired NNS.stack port.

Python port of the audited R/Stack.R (NNS 13.1 Beta, merged): out-of-fold
selection of ``n.best``, dimension counts, classification thresholds, and
ensemble weights; rolling chronological ``ts.test`` folds; per-fold
balancing; training-only X* scaling; strict argument validation; and the
historical classification contract (numeric codes 1..K with
``class.levels`` carrying the label map).

Deviation from R, by design: random fold assignment and balancing use a
local ``numpy.random.Generator`` seeded from ``seed`` rather than R's
Mersenne ``sample()`` stream, so randomized-fold outputs match R
distributionally, not bit-for-bit. The chronological ``ts_test`` mode is
fully deterministic and matches R exactly. The caller's global NumPy RNG
state is never touched.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from nns._reg_engine import (
    _mreg_predict_path,
    _mreg_prepare,
    _validate_order,
    nns_reg_engine,
)
from nns._rrng import RRNG
from nns.central_tendencies import nns_gravity
from nns.dependence import nns_dep

Objective = Literal["min", "max"]

_RESPONSE_OFFSET = 0.123456789

StackResult = dict[str, Any]
"""R-style NNS.stack result dict (OBJfn.reg, NNS.reg.n.best,
probability.threshold, OBJfn.dim.red, NNS.dim.red.threshold, reg,
reg.pred.int, dim.red, dim.red.pred.int, stack, pred.int, weights,
class.levels)."""


# --------------------------------------------------------------------------
# Small shared helpers (kept for nns.var imports)
# --------------------------------------------------------------------------

def _rank_average(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _pearson(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(a @ a) * float(b @ b))
    if denom == 0.0:
        return float("nan")
    return float(a @ b) / denom


def _spearman_scores(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    y_rank = _rank_average(y)
    scores = np.empty(x.shape[1], dtype=np.float64)
    for col in range(x.shape[1]):
        scores[col] = abs(_pearson(_rank_average(x[:, col]), y_rank))
    scores[~np.isfinite(scores)] = 0.0
    return scores


def _spearman_signed(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Signed Spearman correlations, R cor(method='spearman') semantics."""
    y_rank = _rank_average(y)
    out = np.empty(x.shape[1], dtype=np.float64)
    for col in range(x.shape[1]):
        out[col] = _pearson(_rank_average(x[:, col]), y_rank)
    out[~np.isfinite(out)] = 0.0
    return out


def _scalar_logical(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"[{name}] must be TRUE or FALSE.")
    return bool(value)


def _scalar_integer(
    value: Any, name: str, minimum: int = 0, allow_null: bool = False
) -> int | None:
    if allow_null and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer, float, np.floating)):
        raise ValueError(f"[{name}] must be an integer >= {minimum}.")
    v = float(value)
    if not math.isfinite(v) or v < minimum or v != math.floor(v):
        raise ValueError(f"[{name}] must be an integer >= {minimum}.")
    return int(v)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def nns_stack(
    ivs_train: NDArray[Any],
    dv_train: NDArray[Any],
    ivs_test: NDArray[Any] | None = None,
    *,
    type: str | None = None,
    obj_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float] | None = None,
    objective: Objective = "min",
    optimize_threshold: bool = True,
    dist: str = "L2",
    cv_size: float | None = None,
    balance: bool = False,
    ts_test: int | None = None,
    folds: int = 5,
    order: Any = None,
    method: Any = (1, 2),
    stack: bool = True,
    dim_red_method: Any = "cor",
    pred_int: float | None = None,
    status: bool = False,
    ncores: int | None = None,
    seed: int | None = 123,
    # Legacy compatibility (ignored):
    class_levels: Any = None,
    factor_levels: Any = None,
    random_seed: int | None = None,
) -> StackResult:
    del class_levels, factor_levels, ncores
    if random_seed is not None and seed == 123:
        seed = random_seed

    optimize_threshold = _scalar_logical(optimize_threshold, "optimize.threshold")
    balance = _scalar_logical(balance, "balance")
    stack = _scalar_logical(stack, "stack")
    status = _scalar_logical(status, "status")
    folds_value = cast(int, _scalar_integer(folds, "folds", minimum=1))
    ts_test_value = _scalar_integer(ts_test, "ts.test", minimum=1, allow_null=True)

    # R sets its Mersenne-Twister stream once via set.seed(seed) and draws every
    # split from it. Reproduce that stream exactly for deterministic parity; when
    # seed is NULL R leaves the global stream untouched (non-deterministic), so
    # the port seeds from system entropy.
    seed_value = (
        _scalar_integer(seed, "seed", minimum=0)
        if seed is not None
        else int(np.random.SeedSequence().generate_state(1)[0])
    )
    rng = RRNG(seed_value)

    if not isinstance(objective, str) or objective.lower() not in {"min", "max"}:
        raise ValueError("[objective] must be exactly 'min' or 'max'.")
    objective_value: Objective = cast(Objective, objective.lower())

    if type is not None:
        if not isinstance(type, str) or type.lower() != "class":
            raise ValueError("[type] must be NULL or 'CLASS'.")
        type = "class"

    order_value = _validate_order(order)

    method_arr = np.atleast_1d(np.asarray(method))
    if method_arr.size == 0 or not np.issubdtype(method_arr.dtype, np.number) or np.any(
        method_arr != np.floor(method_arr.astype(np.float64))
    ):
        raise ValueError("[method] must contain only 1 and/or 2.")
    methods = sorted(set(int(v) for v in method_arr.tolist()))
    if not all(m in (1, 2) for m in methods):
        raise ValueError("[method] must contain only 1 and/or 2.")

    if cv_size is not None:
        cv_size = float(cv_size)
        if not math.isfinite(cv_size) or cv_size <= 0 or cv_size >= 1:
            raise ValueError("[CV.size] must be a finite scalar strictly between 0 and 1.")

    if pred_int is not None:
        pred_int = float(pred_int)
        if not math.isfinite(pred_int) or pred_int <= 0 or pred_int >= 1:
            raise ValueError("[pred.int] must be a finite scalar strictly between 0 and 1.")

    if not isinstance(dist, str) or dist.lower() not in {"l2", "l1", "dtw", "factor"}:
        raise ValueError("[dist] must be one character value among 'L2', 'L1', 'DTW', 'FACTOR'.")
    if dist.lower() != "l2":
        raise ValueError(
            "The corrected NNS.stack currently supports dist = 'L2' only. "
            "The production multivariate NNS.reg path does not yet implement "
            "distinct L1, DTW, or FACTOR estimators, so those values are "
            "rejected rather than silently treated as L2."
        )
    dist_value = "L2"

    # ----------------------------------------------------------------------
    # Input data and response coding
    # ----------------------------------------------------------------------
    x = np.asarray(ivs_train)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("[IVs.train] must contain at least one predictor.")

    dv = np.asarray(dv_train)
    if dv.ndim == 2 and dv.shape[1] == 1:
        dv = dv[:, 0]
    if dv.ndim != 1:
        raise ValueError("[DV.train] must contain exactly one response column.")
    if dv.size != x.shape[0]:
        raise ValueError("nrow(IVs.train) must equal length(DV.train).")
    if dv.size < 4:
        raise ValueError("NNS.stack requires at least four training observations.")

    response_categorical = dv.dtype.kind in {"U", "S", "O", "b"}
    if not response_categorical:
        dv_num = dv.astype(np.float64)
        if np.any(np.isnan(dv_num)):
            raise ValueError("[DV.train] contains missing values.")
        if np.any(~np.isfinite(dv_num)):
            raise ValueError("[DV.train] contains non-finite values.")

    auto_class = response_categorical or (
        not response_categorical and np.unique(dv).size == 2
    )
    if auto_class and type is None:
        type = "class"
    if balance and type is None:
        warnings.warn("type = 'CLASS' selected because balance = TRUE.", stacklevel=2)
        type = "class"
    is_class = type == "class"

    class_values: list[Any] | None = None
    if is_class:
        if dv.dtype.kind == "b":
            class_values = sorted(set(dv.tolist()))
        elif response_categorical:
            seen: list[Any] = []
            for v in dv.tolist():
                if v not in seen:
                    seen.append(v)
            class_values = seen
        else:
            class_values = sorted(set(dv.astype(np.float64).tolist()))
        y = np.asarray(
            [
                class_values.index(v) + 1
                for v in (
                    dv.tolist()
                    if response_categorical
                    else dv.astype(np.float64).tolist()
                )
            ],
            dtype=np.float64,
        )
        if np.unique(y).size < 2:
            raise ValueError("Classification requires at least two response classes.")
        _, counts = np.unique(y, return_counts=True)
        if counts.min() < 2:
            raise ValueError(
                "Each response class requires at least two observations for cross-validation."
            )
        if obj_fn is None:
            obj_fn = lambda predicted, actual: float(np.mean(predicted == actual))  # noqa: E731
            objective_value = "max"
    else:
        if response_categorical:
            raise ValueError("A nonnumeric response requires type = 'CLASS'.")
        y = dv.astype(np.float64)
        if obj_fn is None:
            obj_fn = lambda predicted, actual: float(np.sum((predicted - actual) ** 2))  # noqa: E731

    if balance and not is_class:
        raise ValueError("[balance = TRUE] requires classification.")

    if ivs_test is None:
        z = x.copy()
    else:
        z = np.asarray(ivs_test)
        if z.ndim == 0:
            z = z.reshape(1, 1)
        if z.ndim == 1:
            if x.shape[1] == 1:
                z = z.reshape(-1, 1)
            elif z.size == x.shape[1]:
                z = z.reshape(1, -1)
            else:
                raise ValueError(
                    "A vector [IVs.test] must contain one complete test row, "
                    "unless [IVs.train] has one predictor."
                )
        if z.shape[1] != x.shape[1]:
            raise ValueError(
                "[IVs.test] must have the same number of predictors as [IVs.train]."
            )

    n_obs = x.shape[0]
    original_p = x.shape[1]

    if ts_test_value is not None and ts_test_value >= n_obs:
        raise ValueError(
            "[ts.test] must be smaller than the number of training observations."
        )

    if original_p == 1 and 2 in methods:
        warnings.warn(
            "Method 2 was removed because dimension reduction requires more than "
            "one original predictor.",
            stacklevel=2,
        )
        methods = [m for m in methods if m != 2] or [1]

    dim_red_value: Any = dim_red_method
    if 2 in methods:
        if isinstance(dim_red_method, (list, tuple, np.ndarray)):
            dim_red_value = np.asarray(dim_red_method, dtype=np.float64).reshape(-1)
            if dim_red_value.size == 0 or np.any(~np.isfinite(dim_red_value)):
                raise ValueError("Numeric [dim.red.method] coefficients must be finite.")
        else:
            if not isinstance(dim_red_method, str):
                raise ValueError(
                    "[dim.red.method] must be one supported character value or a numeric vector."
                )
            dim_red_value = dim_red_method.lower()
            if dim_red_value not in {"cor", "nns.dep", "nns.caus", "equal", "all"}:
                raise ValueError("Unsupported [dim.red.method].")

    # Constant non-integer translation keeps NNS.reg from auto-classifying
    # integer-valued regression or class-code targets.
    y_fit_all = y + _RESPONSE_OFFSET

    # ----------------------------------------------------------------------
    # Shared closures
    # ----------------------------------------------------------------------
    n_classes = 0 if class_values is None else len(class_values)

    def score(predicted: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
        if predicted.size != actual.size:
            raise ValueError(
                "The objective received predicted and actual vectors of different lengths."
            )
        value = float(cast(Callable[..., float], obj_fn)(predicted, actual))
        return value if math.isfinite(value) else float("nan")

    def sanitize_raw(
        predicted: NDArray[np.float64], fallback_y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        predicted = np.asarray(predicted, dtype=np.float64).copy()
        bad = ~np.isfinite(predicted)
        if np.any(bad):
            good = predicted[~bad]
            replacement = float(nns_gravity(good)) if good.size else float(nns_gravity(fallback_y))
            if not math.isfinite(replacement):
                replacement = float(np.mean(fallback_y))
            if not math.isfinite(replacement):
                raise ValueError("NNS.reg returned no finite predictions.")
            predicted[bad] = replacement
        return predicted

    def round_codes(raw: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
        raw = np.clip(np.asarray(raw, dtype=np.float64), 1, n_classes)
        lo = np.floor(raw)
        hi = np.ceil(raw)
        frac = raw - lo
        out = np.where(frac < threshold, lo, hi)
        return np.clip(out, 1, n_classes)

    def best_threshold(raw: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
        if not is_class or not optimize_threshold:
            return 0.5
        grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
        scores = np.asarray([score(round_codes(raw, th), actual) for th in grid])
        valid = np.flatnonzero(np.isfinite(scores))
        if valid.size == 0:
            return 0.5
        best = scores[valid].min() if objective_value == "min" else scores[valid].max()
        tied = valid[scores[valid] == best]
        return float(grid[tied[math.ceil(tied.size / 2) - 1]])

    def evaluate_raw(
        raw: NDArray[np.float64], actual: NDArray[np.float64]
    ) -> tuple[float, float]:
        threshold = best_threshold(raw, actual) if is_class else 0.5
        predicted = round_codes(raw, threshold) if is_class else raw
        return score(predicted, actual), threshold

    def decode_codes(code: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            np.clip(np.asarray(code, dtype=np.float64), 1, n_classes)
            .astype(np.int64)
            .astype(np.float64)
        )

    def decode_interval(
        interval: dict[str, NDArray[np.float64]] | None, threshold: float
    ) -> dict[str, NDArray[np.float64]] | None:
        if interval is None:
            return None
        return {
            k: decode_codes(round_codes(v, threshold)) for k, v in interval.items()
        }

    def translate_interval(
        interval: dict[str, NDArray[np.float64]] | None,
    ) -> dict[str, NDArray[np.float64]] | None:
        if interval is None:
            return None
        return {k: np.asarray(v, dtype=np.float64) - _RESPONSE_OFFSET for k, v in interval.items()}

    def has_all_classes(train_y: NDArray[np.float64]) -> bool:
        return not is_class or set(np.unique(train_y).tolist()) == set(np.unique(y).tolist())

    def balance_indices(train_y: NDArray[np.float64]) -> NDArray[np.int64]:
        if not balance:
            return np.arange(train_y.size, dtype=np.int64)
        values = np.unique(train_y)
        groups = [np.flatnonzero(train_y == v) for v in values]
        if len(groups) < 2 or any(g.size == 0 for g in groups):
            raise ValueError(
                "Balancing requires at least two non-empty classes in the fitting split."
            )
        smallest = min(g.size for g in groups)
        largest = max(g.size for g in groups)
        down = np.concatenate([rng.sample(g, smallest, replace=False) for g in groups])
        up = np.concatenate([rng.sample(g, largest, replace=True) for g in groups])
        combined = np.concatenate([down, up])
        return rng.sample(combined).astype(np.int64)

    def numeric_design(
        train: NDArray[Any], test: NDArray[Any]
    ) -> dict[str, NDArray[np.float64]]:
        train_blocks: list[NDArray[np.float64]] = []
        test_blocks: list[NDArray[np.float64]] = []
        for j in range(train.shape[1]):
            tr = train[:, j]
            te = test[:, j]
            if tr.dtype.kind in {"U", "S", "O", "b"} and not _numeric_column(tr):
                tr_chr = np.asarray([str(v) for v in tr.tolist()])
                te_chr = np.asarray([str(v) for v in te.tolist()])
                levels: list[str] = []
                for v in tr_chr.tolist():
                    if v not in levels:
                        levels.append(v)
                unseen = sorted(set(te_chr.tolist()) - set(levels))
                if unseen:
                    raise ValueError(
                        f"Test predictor [{j + 1}] contains unseen level(s): "
                        + ", ".join(unseen)
                    )
                train_blocks.append(
                    np.column_stack([(tr_chr == lv).astype(np.float64) for lv in levels])
                )
                test_blocks.append(
                    np.column_stack([(te_chr == lv).astype(np.float64) for lv in levels])
                )
            else:
                trn = np.asarray(tr, dtype=np.float64)
                ten = np.asarray(te, dtype=np.float64)
                if np.any(np.isnan(trn)):
                    raise ValueError("[IVs.train] contains missing values.")
                if np.any(~np.isfinite(trn)):
                    raise ValueError(
                        f"[IVs.train] predictor [{j + 1}] contains non-finite values."
                    )
                if np.any(np.isnan(ten)):
                    raise ValueError("[IVs.test] contains missing values.")
                if np.any(~np.isfinite(ten)):
                    raise ValueError(
                        f"[IVs.test] predictor [{j + 1}] contains non-finite values."
                    )
                train_blocks.append(trn.reshape(-1, 1))
                test_blocks.append(ten.reshape(-1, 1))
        train_matrix = np.hstack(train_blocks)
        test_matrix = np.hstack(test_blocks)
        tmin = train_matrix.min(axis=0)
        tmax = train_matrix.max(axis=0)
        trange = tmax - tmin
        trange = np.where(~np.isfinite(trange) | (trange == 0), 1.0, trange)
        return {
            "train": (train_matrix - tmin) / trange,
            "test": (test_matrix - tmin) / trange,
            "minimum": tmin,
            "range": trange,
        }

    def make_splits() -> list[dict[str, NDArray[np.int64]]]:
        all_index = np.arange(n_obs, dtype=np.int64)

        if ts_test_value is not None:
            possible = (n_obs - 1) // ts_test_value
            if possible < 1:
                raise ValueError("Not enough observations for the requested [ts.test].")
            use_folds = min(folds_value, possible)
            if use_folds < folds_value:
                warnings.warn(
                    f"Only {use_folds} non-overlapping chronological fold(s) are available.",
                    stacklevel=2,
                )
            starts = n_obs - np.arange(use_folds, 0, -1) * ts_test_value
            out = []
            for start in starts.tolist():
                validation = np.arange(start, start + ts_test_value, dtype=np.int64)
                training = np.arange(start, dtype=np.int64)
                out.append({"train": training, "validation": validation})
            out = [
                s
                for s in out
                if s["train"].size >= 3 and has_all_classes(y[s["train"]])
            ]
            if not out:
                raise ValueError(
                    "No chronological fold retained enough training observations "
                    "and all response classes."
                )
            return out

        holdout_size = cv_size
        if holdout_size is None and folds_value == 1:
            holdout_size = round(float(rng.runif(0.20, 1.0 / 3.0)), 3)

        if holdout_size is not None:
            out = []
            for _ in range(folds_value):
                if is_class:
                    validation_parts = []
                    for v in np.unique(y):
                        g = np.flatnonzero(y == v)
                        size = min(g.size - 1, max(1, round(holdout_size * g.size)))
                        if size > 0:
                            validation_parts.append(rng.sample(g, size, replace=False))
                    validation = np.unique(np.concatenate(validation_parts)).astype(np.int64)
                else:
                    size = max(1, min(n_obs - 1, round(holdout_size * n_obs)))
                    validation = np.sort(
                        rng.sample_int(n_obs, size, replace=False) - 1
                    ).astype(np.int64)
                training = np.setdiff1d(all_index, validation)
                if training.size < 3 or not has_all_classes(y[training]):
                    raise ValueError(
                        "Unable to create a repeated holdout retaining enough "
                        "fitting observations and every response class."
                    )
                out.append({"train": training, "validation": validation})
            return out

        use_folds = min(folds_value, n_obs)
        if is_class:
            _, counts = np.unique(y, return_counts=True)
            if counts.min() < 2:
                raise ValueError(
                    "Each response class requires at least two observations for cross-validation."
                )
            use_folds = min(use_folds, int(counts.max()))
        if use_folds < 2:
            raise ValueError(
                "At least two folds are required when [CV.size] and [ts.test] are NULL."
            )
        if use_folds < folds_value:
            warnings.warn(
                f"Cross-validation folds reduced to {use_folds} for the available data.",
                stacklevel=2,
            )
        fold_id = np.zeros(n_obs, dtype=np.int64)
        if is_class:
            for v in np.unique(y):
                g = np.flatnonzero(y == v)
                shuffled = rng.sample(g)
                fold_id[shuffled] = np.resize(np.arange(1, use_folds + 1), g.size)
        else:
            shuffled = rng.sample(all_index)
            fold_id[shuffled] = np.resize(np.arange(1, use_folds + 1), n_obs)
        out = []
        for b in range(1, use_folds + 1):
            validation = np.flatnonzero(fold_id == b).astype(np.int64)
            training = np.setdiff1d(all_index, validation)
            if validation.size > 0 and training.size >= 3 and has_all_classes(y[training]):
                out.append({"train": training, "validation": validation})
        if len(out) < 2:
            raise ValueError("Unable to create at least two valid cross-validation folds.")
        return out

    splits = make_splits()

    def coefficient_vector(
        design: NDArray[np.float64], response: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        p = design.shape[1]
        if isinstance(dim_red_value, np.ndarray):
            coef = dim_red_value.astype(np.float64).copy()
            if coef.size != p:
                raise ValueError(
                    f"Numeric [dim.red.method] must contain {p} encoded coefficients."
                )
            coef[~np.isfinite(coef)] = 0.0
            return coef

        def cor_coef() -> NDArray[np.float64]:
            return _spearman_signed(design, response)

        def dep_coef() -> NDArray[np.float64]:
            out = np.zeros(p, dtype=np.float64)
            for j in range(p):
                try:
                    v = float(nns_dep(design[:, j], response, asym=True)["Dependence"])
                except Exception:
                    v = 0.0
                out[j] = v if math.isfinite(v) else 0.0
            return out

        def caus_coef() -> NDArray[np.float64]:
            from nns.causation import _tau_value, _ts_tau_values, _uni_caus

            out = np.zeros(p, dtype=np.float64)
            for j in range(p):
                try:
                    if ts_test_value is not None:
                        _, y_tau = _ts_tau_values(response, design[:, j])
                        v = float(_uni_caus(response, design[:, j], y_tau))
                    else:
                        v = float(_uni_caus(response, design[:, j], _tau_value("cs")))
                except Exception:
                    v = 0.0
                out[j] = v if math.isfinite(v) else 0.0
            return out

        if dim_red_value == "cor":
            coef = cor_coef()
        elif dim_red_value == "nns.dep":
            coef = dep_coef()
        elif dim_red_value == "nns.caus":
            coef = caus_coef()
        elif dim_red_value == "equal":
            coef = np.ones(p, dtype=np.float64)
        elif dim_red_value == "all":
            coef = np.mean(
                np.column_stack((caus_coef(), dep_coef(), cor_coef())), axis=1
            )
        else:
            raise ValueError("Unsupported [dim.red.method].")
        coef[~np.isfinite(coef)] = 0.0
        if not np.any(np.abs(coef) > 0):
            coef = np.ones(p, dtype=np.float64)
        return coef

    def active_coefficients(coef: NDArray[np.float64], count: int) -> NDArray[np.float64]:
        coef = np.asarray(coef, dtype=np.float64)
        count = max(1, min(int(count), coef.size))
        order_idx = np.argsort(-np.abs(coef), kind="stable")[:count]
        out = np.zeros(coef.size, dtype=np.float64)
        out[order_idx] = coef[order_idx]
        if not np.any(np.abs(out) > 0):
            out[order_idx] = 1.0
        return out

    def project_xstar(
        train_design: NDArray[np.float64],
        test_design: NDArray[np.float64],
        coef: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        denom = int(np.sum(np.abs(coef) > 0))
        if denom < 1:
            raise ValueError("No active dimension-reduction coefficients.")
        return (
            np.asarray(train_design @ coef / denom, dtype=np.float64),
            np.asarray(test_design @ coef / denom, dtype=np.float64),
        )

    def fit_univariate_raw(
        train_x: NDArray[np.float64],
        train_y_fit: NDArray[np.float64],
        test_x: NDArray[np.float64],
        confidence_interval: float | None = None,
        point_only: bool = True,
        allow_failure: bool = False,
    ) -> dict[str, Any]:
        try:
            fit = nns_reg_engine(
                train_x,
                train_y_fit,
                point_est=test_x,
                order=order_value,
                type=None,
                factor_2_dummy=False,
                dist=dist_value,
                point_only=point_only,
                confidence_interval=confidence_interval,
            )
        except Exception:
            if allow_failure:
                return {"raw": np.full(test_x.size, np.nan), "fit": None}
            raise
        raw = sanitize_raw(
            np.asarray(fit["Point.est"], dtype=np.float64).reshape(-1) - _RESPONSE_OFFSET,
            train_y_fit - _RESPONSE_OFFSET,
        )
        return {"raw": raw, "fit": fit}

    def production_multivariate_path(
        rpm: dict[str, NDArray[np.float64]],
        xtest: NDArray[np.float64],
        train_design: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        feature_names = [k for k in rpm if k != "y.hat"]
        rpm_x = np.column_stack([np.asarray(rpm[k], dtype=np.float64) for k in feature_names])
        rpm_yhat = np.asarray(rpm["y.hat"], dtype=np.float64)
        mins = train_design.min(axis=0)
        maxs = train_design.max(axis=0)
        path = _mreg_predict_path(
            xtest, rpm_x, rpm_yhat, rpm_x.shape[0], dist_value, mins, maxs
        )
        if path.shape[0] != xtest.shape[0]:
            raise ValueError("The production distance path returned an invalid row count.")
        return path

    def candidate_from_oof(
        sum_matrix: NDArray[np.float64],
        count_matrix: NDArray[np.int64],
        candidate: int,
    ) -> dict[str, Any]:
        count = count_matrix[:, candidate]
        raw = np.full(n_obs, np.nan)
        has = count > 0
        raw[has] = sum_matrix[has, candidate] / count[has]
        valid = has & np.isfinite(raw)
        if not np.any(valid):
            return {"score": float("nan"), "threshold": 0.5, "raw": raw}
        s, t = evaluate_raw(raw[valid], y[valid])
        return {"score": s, "threshold": t, "raw": raw}

    def select_best(scores: NDArray[np.float64]) -> int:
        valid = np.flatnonzero(np.isfinite(scores))
        if valid.size == 0:
            raise ValueError("No candidate produced a finite out-of-fold objective.")
        best = scores[valid].min() if objective_value == "min" else scores[valid].max()
        tied = valid[scores[valid] == best]
        return int(tied[math.ceil(tied.size / 2) - 1])

    full_design = numeric_design(x, z)
    encoded_p = full_design["train"].shape[1]

    if (
        2 in methods
        and isinstance(dim_red_value, np.ndarray)
        and dim_red_value.size != encoded_p
    ):
        raise ValueError(
            f"Numeric [dim.red.method] must contain {encoded_p} encoded coefficients."
        )

    # ----------------------------------------------------------------------
    # Method 2: dimension count from OOF predictions
    # ----------------------------------------------------------------------
    dim_best_count: int | None = None
    dim_best_score = float("nan")
    dim_component_threshold = 0.5
    dim_oof_raw = np.full(n_obs, np.nan)
    dim_threshold_report = float("nan")
    dim_full_xstar_train: NDArray[np.float64] | None = None
    dim_full_xstar_test: NDArray[np.float64] | None = None

    if 2 in methods:
        dim_sum = np.zeros((n_obs, encoded_p))
        dim_count = np.zeros((n_obs, encoded_p), dtype=np.int64)

        for b, split in enumerate(splits):
            train_idx = split["train"]
            valid_idx = split["validation"]
            fold_design = numeric_design(x[train_idx], x[valid_idx])
            coef_fold = coefficient_vector(fold_design["train"], y[train_idx])
            balanced_idx = balance_indices(y[train_idx])

            for m in range(1, encoded_p + 1):
                active = active_coefficients(coef_fold, m)
                xs_train, xs_test = project_xstar(
                    fold_design["train"], fold_design["test"], active
                )
                fit = fit_univariate_raw(
                    xs_train[balanced_idx],
                    y_fit_all[train_idx][balanced_idx],
                    xs_test,
                    point_only=True,
                    allow_failure=True,
                )
                good = np.isfinite(fit["raw"])
                if np.any(good):
                    rows = valid_idx[good]
                    dim_sum[rows, m - 1] += fit["raw"][good]
                    dim_count[rows, m - 1] += 1
            if status:
                print(f"Dimension-reduction folds remaining = {len(splits) - b - 1}")

        dim_scores = np.full(encoded_p, np.nan)
        dim_thresholds = np.full(encoded_p, 0.5)
        dim_raws: list[NDArray[np.float64]] = []
        for m in range(encoded_p):
            cand = candidate_from_oof(dim_sum, dim_count, m)
            dim_scores[m] = cand["score"]
            dim_thresholds[m] = cand["threshold"]
            dim_raws.append(cand["raw"])
            if status:
                print(
                    f"Current dimension count = {m + 1} | OOF eval(obj.fn) = "
                    f"{cand['score']:.6g} | Iterations remaining = {encoded_p - m - 1}"
                )

        best_idx = select_best(dim_scores)
        dim_best_count = best_idx + 1
        dim_best_score = float(dim_scores[best_idx])
        dim_component_threshold = float(dim_thresholds[best_idx])
        dim_oof_raw = dim_raws[best_idx]

        dim_full_coef = active_coefficients(
            coefficient_vector(full_design["train"], y), dim_best_count
        )
        dim_full_xstar_train, dim_full_xstar_test = project_xstar(
            full_design["train"], full_design["test"], dim_full_coef
        )
        active_magnitudes = np.abs(dim_full_coef[np.abs(dim_full_coef) > 0])
        if dim_best_count >= dim_full_coef.size:
            dim_threshold_report = 0.0
        elif active_magnitudes.size:
            dim_threshold_report = float(active_magnitudes.min())
        else:
            dim_threshold_report = 0.0
        if not math.isfinite(dim_threshold_report):
            dim_threshold_report = 0.0

    # ----------------------------------------------------------------------
    # Method 1: production RPM + production distance path for every k
    # ----------------------------------------------------------------------
    reg_best_k: int | None = None
    reg_best_score = float("nan")
    reg_component_threshold = 0.5
    reg_oof_raw = np.full(n_obs, np.nan)

    def method1_design_for_split(
        train_idx: NDArray[np.int64], valid_idx: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        fold_design = numeric_design(x[train_idx], x[valid_idx])
        if stack and 2 in methods:
            coef_fold = coefficient_vector(fold_design["train"], y[train_idx])
            active = active_coefficients(coef_fold, cast(int, dim_best_count))
            xs_train, xs_test = project_xstar(
                fold_design["train"], fold_design["test"], active
            )
            return (
                np.column_stack((xs_train, xs_train)),
                np.column_stack((xs_test, xs_test)),
            )
        return fold_design["train"], fold_design["test"]

    reg_best_is_all = False
    if 1 in methods:
        l_small = max(1, math.floor(math.sqrt(n_obs)))
        candidate_ids = [str(k) for k in range(1, l_small + 1)] + ["all"]
        sum_list = {cid: np.zeros(n_obs) for cid in candidate_ids}
        count_list = {cid: np.zeros(n_obs, dtype=np.int64) for cid in candidate_ids}
        fold_small_kmax: list[int] = []

        for b, split in enumerate(splits):
            train_idx = split["train"]
            valid_idx = split["validation"]
            train_design, valid_design = method1_design_for_split(train_idx, valid_idx)
            balanced_idx = balance_indices(y[train_idx])
            train_design = train_design[balanced_idx]
            train_y_fit = y_fit_all[train_idx][balanced_idx]

            if train_design.shape[1] == 1:
                # Univariate: uniform k-NN cumulative means on |x - test|.
                train_x = train_design[:, 0]
                test_x = valid_design[:, 0]
                small_kmax = min(l_small, train_x.size)
                d = np.abs(train_x[None, :] - test_x[:, None])
                ord_idx = np.argsort(d, axis=1, kind="stable")
                cums = np.cumsum(train_y_fit[ord_idx], axis=1)[:, :small_kmax]
                pred_mat = cums / np.arange(1, small_kmax + 1)
                small_raw = [pred_mat[:, k] - _RESPONSE_OFFSET for k in range(small_kmax)]
            else:
                rpm_x_f, rpm_yhat_f = _mreg_prepare(
                    train_design, train_y_fit, order_value, "off", False
                )
                if rpm_x_f.shape[0] < 1:
                    raise ValueError(
                        "NNS.reg did not return a usable regression-point matrix."
                    )
                small_kmax = min(l_small, rpm_x_f.shape[0])
                if small_kmax >= 1:
                    small_path = _mreg_predict_path(
                        valid_design,
                        rpm_x_f,
                        rpm_yhat_f,
                        small_kmax,
                        dist_value,
                        train_design.min(axis=0),
                        train_design.max(axis=0),
                    ) - _RESPONSE_OFFSET
                    small_raw = [
                        sanitize_raw(small_path[:, k], y[train_idx])
                        for k in range(small_kmax)
                    ]
                else:
                    small_raw = []

            fold_small_kmax.append(len(small_raw))

            # Aggregate every available local candidate for this fold. No
            # fold-specific early stopping is allowed because it would create
            # incomparable OOF coverage across candidates (NNS#44).
            for k in range(1, len(small_raw) + 1):
                raw = small_raw[k - 1]
                good = np.isfinite(raw)
                if np.any(good):
                    rows = valid_idx[good]
                    sum_list[str(k)][rows] += raw[good]
                    count_list[str(k)][rows] += 1

            # Mandatory all-points candidate: constant mean response.
            all_raw = np.full(valid_idx.size, float(np.mean(train_y_fit)) - _RESPONSE_OFFSET)
            good = np.isfinite(all_raw)
            rows = valid_idx[good]
            sum_list["all"][rows] += all_raw[good]
            count_list["all"][rows] += 1
            if status:
                print(f"Method 1 folds remaining = {len(splits) - b - 1}")

        common_small_kmax = min(fold_small_kmax) if fold_small_kmax else 0
        if common_small_kmax < 1:
            raise ValueError("No Method 1 local candidate was available in every fold.")

        candidate_scores = {cid: float("nan") for cid in candidate_ids}
        candidate_thresholds = {cid: 0.5 for cid in candidate_ids}
        candidate_raws: dict[str, NDArray[np.float64]] = {}

        # Candidate k=1 defines the required complete OOF coverage pattern; a
        # candidate competes only when it covers exactly the same observations.
        reference_count = count_list["1"]

        def score_method1_candidate(cid: str) -> bool:
            count_vec = count_list[cid]
            covered = count_vec > 0
            raw = np.full(n_obs, np.nan)
            raw[covered] = sum_list[cid][covered] / count_vec[covered]
            candidate_raws[cid] = raw
            complete_coverage = np.array_equal(count_vec, reference_count)
            valid = covered & np.isfinite(raw)
            if not complete_coverage or not np.any(valid):
                candidate_scores[cid] = float("nan")
                candidate_thresholds[cid] = 0.5
                return False
            s, t = evaluate_raw(raw[valid], y[valid])
            candidate_scores[cid] = s
            candidate_thresholds[cid] = t
            return math.isfinite(s)

        evaluated_small_ids: list[str] = []
        for k in range(1, common_small_kmax + 1):
            cid = str(k)
            if not score_method1_candidate(cid):
                break
            evaluated_small_ids.append(cid)
            if len(evaluated_small_ids) >= 4:
                recent = [candidate_scores[i] for i in evaluated_small_ids[-3:]]
                stop_local = (
                    recent[2] >= recent[1] and recent[2] >= recent[0]
                    if objective_value == "min"
                    else recent[2] <= recent[1] and recent[2] <= recent[0]
                )
                if stop_local:
                    break

        # ALL is a separate limit condition, always scored after the local
        # sequence and never subject to the diminishing-returns stop.
        all_usable = score_method1_candidate("all")

        eligible_ids = list(evaluated_small_ids)
        if all_usable:
            eligible_ids.append("all")
        valid_ids = [cid for cid in eligible_ids if math.isfinite(candidate_scores[cid])]
        if not valid_ids:
            raise ValueError(
                "No Method 1 candidate produced a finite complete-coverage OOF objective."
            )
        best_val = (
            min(candidate_scores[cid] for cid in valid_ids)
            if objective_value == "min"
            else max(candidate_scores[cid] for cid in valid_ids)
        )
        best_id = next(cid for cid in valid_ids if candidate_scores[cid] == best_val)

        if best_id == "all":
            reg_best_is_all = True
            reg_best_k = None
        else:
            reg_best_k = int(best_id)
        reg_best_score = float(candidate_scores[best_id])
        reg_component_threshold = float(candidate_thresholds[best_id])
        reg_oof_raw = candidate_raws[best_id]
        if status:
            label = "all (k = all RPM rows)" if best_id == "all" else f"k = {best_id}"
            print(f"Best Method 1 candidate: {label}, score = {reg_best_score:.6g}")

    # ----------------------------------------------------------------------
    # OOF blend and final threshold
    # ----------------------------------------------------------------------
    component_weights = {"reg": 0.0, "dim.red": 0.0}
    probability_threshold = 0.5

    if methods == [1]:
        component_weights["reg"] = 1.0
        probability_threshold = reg_component_threshold
    elif methods == [2]:
        component_weights["dim.red"] = 1.0
        probability_threshold = dim_component_threshold
    else:
        valid = np.isfinite(reg_oof_raw) & np.isfinite(dim_oof_raw)
        if not np.any(valid):
            raise ValueError(
                "The two component models have no common finite OOF predictions."
            )
        weight_grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)
        blend_scores = np.full(weight_grid.size, np.nan)
        blend_thresholds = np.full(weight_grid.size, 0.5)
        for i, w in enumerate(weight_grid.tolist()):
            raw = w * reg_oof_raw[valid] + (1 - w) * dim_oof_raw[valid]
            s, t = evaluate_raw(raw, y[valid])
            blend_scores[i] = s
            blend_thresholds[i] = t
        valid_grid = np.flatnonzero(np.isfinite(blend_scores))
        if valid_grid.size == 0:
            raise ValueError("No ensemble weight produced a finite OOF objective.")
        best = (
            blend_scores[valid_grid].min()
            if objective_value == "min"
            else blend_scores[valid_grid].max()
        )
        tied = valid_grid[blend_scores[valid_grid] == best]
        selected = int(tied[math.ceil(tied.size / 2) - 1])
        component_weights = {
            "reg": float(weight_grid[selected]),
            "dim.red": float(1 - weight_grid[selected]),
        }
        probability_threshold = float(blend_thresholds[selected])

    # ----------------------------------------------------------------------
    # Final production fits
    # ----------------------------------------------------------------------
    if status:
        print("Generating final estimates")

    reg_raw_final: NDArray[np.float64] | None = None
    reg_pred_int_raw: dict[str, NDArray[np.float64]] | None = None
    dim_raw_final: NDArray[np.float64] | None = None
    dim_pred_int_raw: dict[str, NDArray[np.float64]] | None = None

    if 2 in methods:
        balanced_idx = balance_indices(y)
        dim_final = fit_univariate_raw(
            cast(NDArray[np.float64], dim_full_xstar_train)[balanced_idx],
            y_fit_all[balanced_idx],
            cast(NDArray[np.float64], dim_full_xstar_test),
            confidence_interval=pred_int,
            point_only=False,
        )
        dim_raw_final = dim_final["raw"]
        dim_pred_int_raw = translate_interval(dim_final["fit"]["pred.int"])

    if 1 in methods:
        if stack and 2 in methods:
            final_train: NDArray[np.float64] = np.column_stack(
                (dim_full_xstar_train, dim_full_xstar_train)
            )
            final_test: NDArray[np.float64] = np.column_stack(
                (dim_full_xstar_test, dim_full_xstar_test)
            )
        else:
            final_train = full_design["train"]
            final_test = full_design["test"]
        balanced_idx = balance_indices(y)
        train_design = final_train[balanced_idx]
        train_y_fit = y_fit_all[balanced_idx]

        final_n_best: Any = "all" if reg_best_is_all else reg_best_k
        if train_design.shape[1] == 1:
            reg_fit = nns_reg_engine(
                train_design[:, 0],
                train_y_fit,
                point_est=final_test[:, 0],
                n_best=final_n_best,
                order=order_value,
                type=None,
                factor_2_dummy=False,
                dist=dist_value,
                point_only=False,
                confidence_interval=pred_int,
            )
            reg_raw_final = sanitize_raw(
                np.asarray(reg_fit["Point.est"], dtype=np.float64).reshape(-1)
                - _RESPONSE_OFFSET,
                y,
            )
            reg_pred_int_raw = translate_interval(reg_fit["pred.int"])
            if reg_best_is_all:
                reg_best_k = train_design.shape[0]
        else:
            reg_fit = nns_reg_engine(
                train_design,
                train_y_fit,
                point_est=final_test,
                n_best=final_n_best,
                order=order_value,
                type=None,
                factor_2_dummy=False,
                dist=dist_value,
                point_only=False,
                confidence_interval=pred_int,
            )
            reg_raw_final = sanitize_raw(
                np.asarray(reg_fit["Point.est"], dtype=np.float64).reshape(-1)
                - _RESPONSE_OFFSET,
                y,
            )
            reg_pred_int_raw = translate_interval(reg_fit["pred.int"])
            if reg_best_is_all:
                rpm_full_x, _ = _mreg_prepare(
                    train_design, train_y_fit, order_value, "off", False
                )
                reg_best_k = rpm_full_x.shape[0] if rpm_full_x.size else train_design.shape[0]

    # Component outputs with their own OOF thresholds.
    if is_class:
        reg_output = (
            decode_codes(round_codes(reg_raw_final, reg_component_threshold))
            if reg_raw_final is not None
            else None
        )
        dim_output = (
            decode_codes(round_codes(dim_raw_final, dim_component_threshold))
            if dim_raw_final is not None
            else None
        )
    else:
        reg_output = reg_raw_final
        dim_output = dim_raw_final

    if methods == [1]:
        stacked_raw = cast(NDArray[np.float64], reg_raw_final)
    elif methods == [2]:
        stacked_raw = cast(NDArray[np.float64], dim_raw_final)
    else:
        reg_use = cast(NDArray[np.float64], reg_raw_final).copy()
        dim_use = cast(NDArray[np.float64], dim_raw_final).copy()
        reg_bad = ~np.isfinite(reg_use)
        dim_bad = ~np.isfinite(dim_use)
        if np.any(reg_bad & dim_bad):
            raise ValueError(
                "Both final component models failed for at least one test observation."
            )
        reg_use[reg_bad] = dim_use[reg_bad]
        dim_use[dim_bad] = reg_use[dim_bad]
        stacked_raw = (
            component_weights["reg"] * reg_use + component_weights["dim.red"] * dim_use
        )

    stacked_output = (
        decode_codes(round_codes(stacked_raw, probability_threshold))
        if is_class
        else stacked_raw
    )

    reg_pred_int = (
        decode_interval(reg_pred_int_raw, reg_component_threshold)
        if is_class
        else reg_pred_int_raw
    )
    dim_pred_int = (
        decode_interval(dim_pred_int_raw, dim_component_threshold)
        if is_class
        else dim_pred_int_raw
    )

    if pred_int is None:
        stacked_pred_int_raw = None
    elif methods == [1]:
        stacked_pred_int_raw = reg_pred_int_raw
    elif methods == [2]:
        stacked_pred_int_raw = dim_pred_int_raw
    elif reg_pred_int_raw is None:
        stacked_pred_int_raw = dim_pred_int_raw
    elif dim_pred_int_raw is None:
        stacked_pred_int_raw = reg_pred_int_raw
    else:
        stacked_pred_int_raw = {
            k: component_weights["reg"] * reg_pred_int_raw[k]
            + component_weights["dim.red"] * dim_pred_int_raw[k]
            for k in reg_pred_int_raw
        }

    stacked_pred_int = (
        decode_interval(stacked_pred_int_raw, probability_threshold)
        if is_class
        else stacked_pred_int_raw
    )

    return {
        "OBJfn.reg": reg_best_score,
        "NNS.reg.n.best": float(reg_best_k) if reg_best_k is not None else float("nan"),
        "probability.threshold": probability_threshold,
        "OBJfn.dim.red": dim_best_score,
        "NNS.dim.red.threshold": dim_threshold_report,
        "reg": reg_output,
        "reg.pred.int": reg_pred_int,
        "dim.red": dim_output,
        "dim.red.pred.int": dim_pred_int,
        "stack": stacked_output,
        "pred.int": stacked_pred_int,
        "weights": component_weights,
        "class.levels": class_values if is_class else None,
    }


def _numeric_column(z: NDArray[Any]) -> bool:
    try:
        np.asarray(z, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return True
