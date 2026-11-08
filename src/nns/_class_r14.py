"""NNS 14.0 empirical-frequency nominal classification.

Reference: OVVO-Financial/NNS@7b07a837c130ba7220dbe16c584221508757d5ae,
R/Classification.R. Class codes identify indicator columns only; numeric
interpolation is never performed on ordinalized class labels.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nns._rblas import r_crossprod_vec
from nns._rmean import r_approx, r_tie_means

F = NDArray[np.float64]


def _normalize(probabilities: F, priors: F) -> F:
    p = np.asarray(probabilities, dtype=np.float64).copy()
    p[~np.isfinite(p) | (p < 0.0)] = 0.0
    p[p > 1.0] = 1.0
    totals = np.sum(p, axis=1)
    valid = totals > 0.0
    if np.any(valid):
        p[valid] /= totals[valid, None]
    if np.any(~valid):
        p[~valid] = priors
    return p


def _select(probabilities: F | None, values: F, order: NDArray[np.intp]) -> F | None:
    if probabilities is None:
        return None
    selected = np.argmax(probabilities[:, order], axis=1)
    return np.asarray(values[order[selected]], dtype=np.float64)


def _class_base(
    design_x: F,
    indicators: F,
    points: F | None,
    order: Any,
    n_best: Any,
    dist: str,
    categorical: bool,
    point_only: bool,
    labels: list[str],
) -> dict[str, Any]:
    """R .nns_class_base: empirical class shares in deterministic NNS cells."""
    from nns._reg_engine import (
        _build_points,
        _dependence,
        _mreg_distances,
        _mreg_ensemble_weights,
    )

    n, p = design_x.shape
    mins = np.min(design_x, axis=0)
    maxs = np.max(design_x, axis=0)
    strength = np.array(
        [
            max(
                0.0,
                max(
                    _dependence(design_x[:, j], indicators[:, c])
                    for c in range(indicators.shape[1])
                ),
            )
            for j in range(p)
        ],
        dtype=np.float64,
    )
    boundaries: list[F] = []
    exact: list[bool] = []
    memberships: list[NDArray[np.int64]] = []
    for j in range(p):
        coord = design_x[:, j]
        coord_exact = categorical or order == "max" or np.unique(coord).size <= 2
        exact.append(coord_exact)
        if coord_exact:
            b = np.unique(coord)
        else:
            if order is None:
                depth = max(1, math.floor(float(strength[j]) * 10.0 + 0.5))
                if n < 100:
                    depth = max(1, depth // 2)
            else:
                depth = int(order)
            b = np.asarray(_build_points(coord, coord, depth, "mean", False)["x"], dtype=np.float64)
        boundaries.append(b)
        if coord_exact:
            memberships.append(np.searchsorted(b, coord, side="left").astype(np.int64) + 1)
        else:
            positions = np.searchsorted(b, coord, side="right").astype(np.int64)
            # R findInterval(..., rightmost.closed=TRUE) maps the last
            # breakpoint to the final interval, not to one past it.
            memberships.append(np.minimum(positions, b.size - 1))

    row_ids = [".".join(str(int(membership[i])) for membership in memberships) for i in range(n)]
    groups: dict[str, list[int]] = {}
    for i, ident in enumerate(row_ids):
        groups.setdefault(ident, []).append(i)

        # split(seq_len(n), ids) groups names in lexicographic order in R.
    keys = sorted(groups)
    centres = np.array([np.mean(design_x[groups[key]], axis=0) for key in keys], dtype=np.float64)
    shares = np.array([np.mean(indicators[groups[key]], axis=0) for key in keys], dtype=np.float64)
    prior = np.mean(indicators, axis=0)

    if n_best == "all":
        k = centres.shape[0]
    elif n_best is not None:
        k = int(n_best)
    elif order == "max" or categorical:
        k = 1
    else:
        k = max(1, math.floor((1.0 - float(np.mean(strength))) * math.sqrt(n)))
    k = max(1, min(k, centres.shape[0]))

    def predict(query: F | None) -> F | None:
        if query is None:
            return None
        q = np.asarray(query, dtype=np.float64).reshape(-1, p)
        if p == 1 and not categorical:
            sorted_idx = np.argsort(centres[:, 0], kind="stable")
            xx = centres[sorted_idx, 0]
            if xx.size == 1:
                out = np.tile(shares[sorted_idx[0]], (q.shape[0], 1))
            else:
                # R: approx(centres[o, 1], shares[o, j], xout, rule = 2,
                # ties = mean)$y per class, with R's approx1 arithmetic (the
                # fraction first, exact knot ordinates), so an exact
                # probability tie is not turned into a one-ULP preference.
                yy = shares[sorted_idx]
                columns = []
                for col in range(yy.shape[1]):
                    kx, ky = r_tie_means(xx, yy[:, col])
                    columns.append(r_approx(kx, ky, q[:, 0], ky[0], ky[-1]))
                out = np.column_stack(columns)
            return _normalize(out, prior)
        ds = _mreg_distances(centres, q, dist, mins, maxs)
        out = np.empty((q.shape[0], shares.shape[1]), dtype=np.float64)
        for i, row in enumerate(ds):
            matches = np.flatnonzero(row == 0.0)
            if matches.size:
                out[i] = np.mean(shares[matches], axis=0)
                continue
            ordered = np.argsort(row, kind="stable")
            distance_sorted = row[ordered]
            limit = distance_sorted[k - 1]
            chosen = ordered[distance_sorted <= limit]
            chosen_d = row[chosen]
            if chosen.size == 1:
                out[i] = shares[chosen[0]]
                continue
            w = _mreg_ensemble_weights(chosen_d[None, :])[0]
            for value in np.unique(chosen_d):
                same = chosen_d == value
                if np.count_nonzero(same) > 1:
                    w[same] = np.mean(w[same])
            # R: as.numeric(crossprod(weights, shares[o[seq_len(m)], ])).
            out[i] = r_crossprod_vec(w, shares[chosen])
        return _normalize(out, prior)

    return {
        "fitted": None if point_only else predict(design_x),
        "predicted": predict(points),
        "centres": centres,
        "shares": shares,
        "ids": row_ids,
        "boundaries": boundaries,
        "k": k,
        "strength": strength,
    }


def classify(
    encoded: dict[str, Any],
    task: dict[str, Any],
    *,
    order: Any,
    n_best: Any,
    dist: str,
    dim_red_method: Any,
    tau: Any,
    threshold: Any,
    point_only: bool,
    confidence_interval: float | None,
    dim_count: int | None = None,
) -> dict[str, Any]:
    """R .nns_reg_classify: return nominal posterior probabilities and sets."""
    from nns._reg_engine import _dimred_coefficients, _dimreduce, _validate_threshold

    design_x = np.asarray(encoded["x"], dtype=np.float64)
    query_x = encoded["point_est"]
    y = np.asarray(task["y"], dtype=np.float64)
    values = np.asarray(task["class_values"], dtype=np.float64)
    levels = task["class_levels"]
    labels = (
        list(levels)
        if levels is not None and len(levels) == values.size
        else [f"{v:.15g}" for v in values]
    )
    indicators = np.asarray(y[:, None] == values[None, :], dtype=np.float64)
    priors = np.mean(indicators, axis=0)
    _, first = np.unique(y, return_index=True)
    first_values = y[np.sort(first)]
    tie_order = np.array([np.flatnonzero(values == z)[0] for z in first_values], dtype=np.intp)
    metadata = encoded.get("metadata")
    categorical = (
        metadata is not None and bool(metadata) and all(bool(m["categorical"]) for m in metadata)
    )

    equations = None
    if dim_red_method is not None and design_x.shape[1] > 1:
        # R: one cutoff for every class index, or one per class (named by
        # class, or in class order), as NNS.stack reports for classification.
        cutoffs = _validate_threshold(threshold)
        per_class = "[threshold] must be one number, or one cutoff per class."
        if isinstance(cutoffs, dict):
            if len(cutoffs) <= 1:
                threshold_values = np.full(values.size, next(iter(cutoffs.values()), 0.0))
            elif all(str(label) in cutoffs for label in labels):
                threshold_values = np.array([cutoffs[str(label)] for label in labels])
            elif len(cutoffs) == values.size:
                threshold_values = np.array(list(cutoffs.values()), dtype=np.float64)
            else:
                raise ValueError(per_class)
        elif isinstance(cutoffs, float):
            threshold_values = np.full(values.size, cutoffs)
        else:
            threshold_values = np.asarray(cutoffs, dtype=np.float64)
            if threshold_values.size != values.size:
                raise ValueError(per_class)
        fitted_cols, point_cols, equations = [], [], {}
        for c in range(values.size):
            coefficients = _dimred_coefficients(
                design_x, indicators[:, c], dim_red_method, tau, threshold_values[c]
            )
            if dim_count is not None and dim_count < coefficients.size:
                # R/Classification.R: retain the first dim_count columns
                # in decreasing |coefficient| order (stable original order).
                keep = np.argsort(-np.abs(coefficients), kind="stable")[:dim_count]
                narrowed = np.zeros_like(coefficients)
                narrowed[keep] = coefficients[keep]
                coefficients = narrowed
            if not np.any(np.abs(coefficients) > 0):
                coefficients[:] = 1.0
            reduced = _dimreduce(
                design_x,
                query_x,
                indicators[:, c],
                coefficients,
                tau,
                0,
                variable_names=encoded.get("encoded_names"),
            )
            y_indicator = np.column_stack((1.0 - indicators[:, c], indicators[:, c]))
            reduced_x = np.asarray(reduced["x_star"], dtype=np.float64).reshape(-1, 1)
            reduced_q = (
                None
                if query_x is None
                else np.asarray(reduced["point_star"], dtype=np.float64).reshape(-1, 1)
            )
            fit = _class_base(
                reduced_x,
                y_indicator,
                reduced_q,
                order,
                n_best,
                dist,
                False,
                point_only,
                ["not " + labels[c], labels[c]],
            )
            if not point_only:
                fitted_cols.append(np.asarray(fit["fitted"][:, 1], dtype=np.float64))
            if query_x is not None:
                point_cols.append(np.asarray(fit["predicted"][:, 1], dtype=np.float64))
            equations[labels[c]] = reduced["equation"]
        fit_prob = None if point_only else _normalize(np.column_stack(fitted_cols), priors)
        point_prob = None if query_x is None else _normalize(np.column_stack(point_cols), priors)
        centres, shares, boundaries, ids, k = (
            None,
            None,
            None,
            np.arange(1, design_x.shape[0] + 1, dtype=np.int64),  # R: seq_len(nrow(X))
            float("nan"),  # R: k <- NA_integer_
        )
    else:
        fit = _class_base(
            design_x,
            indicators,
            query_x,
            order,
            n_best,
            dist,
            categorical,
            point_only,
            labels,
        )
        fit_prob = fit["fitted"]
        point_prob = fit["predicted"]
        centres, shares = fit["centres"], fit["shares"]
        boundaries, ids, k = fit["boundaries"], fit["ids"], fit["k"]

    point_pred = _select(point_prob, values, tie_order)
    fitted_pred = _select(fit_prob, values, tie_order)
    accuracy = None if point_only else float(np.mean(fitted_pred == y))
    fitted: dict[str, Any] | None = None
    if not point_only and fitted_pred is not None:
        names = encoded.get("encoded_names") or [f"X{i + 1}" for i in range(design_x.shape[1])]
        fitted = {str(name): design_x[:, j] for j, name in enumerate(names)}
        fitted.update(
            {
                "y": y,
                "y.hat": fitted_pred,
                "NNS.ID": ids,
                "error": np.asarray(fitted_pred != y, dtype=np.float64),
            }
        )
    point_sets = None
    if point_prob is not None and confidence_interval is not None:
        point_sets = []
        for row in point_prob:
            order_idx = tie_order[np.argsort(-row[tie_order], kind="stable")]
            cumulative = np.cumsum(row[order_idx])
            index = min(
                int(np.searchsorted(cumulative, confidence_interval - 1e-12)) + 1, order_idx.size
            )
            point_sets.append(values[order_idx[:index]].tolist())
    rp = None
    if centres is not None:
        names = encoded.get("encoded_names") or [f"X{i + 1}" for i in range(design_x.shape[1])]
        rp = {str(name): centres[:, j] for j, name in enumerate(names)}
    rpm_probs = None if shares is None else {labels[c]: shares[:, c] for c in range(values.size)}
    point_probs = (
        None if point_prob is None else {labels[c]: point_prob[:, c] for c in range(values.size)}
    )
    fitted_probs = (
        None if fit_prob is None else {labels[c]: fit_prob[:, c] for c in range(values.size)}
    )
    return {
        "R2": accuracy,
        "SE": None,
        "Prediction.Accuracy": accuracy,
        "equation": equations,
        "x.star": None,
        "derivative": None,
        "Point.est": point_pred,
        "pred.int": None,
        "Prediction.set": point_sets,
        "regression.points": rp,
        "RPM": rp,
        "RPM.probabilities": rpm_probs,
        "rhs.partitions": boundaries,
        "Fitted.xy": fitted,
        "Fitted": fitted,
        "Fitted.probabilities": fitted_probs,
        "Point.probabilities": point_probs,
        "n.best": k,
        "class.levels": task["class_levels"],
        "class.values": values,
    }
