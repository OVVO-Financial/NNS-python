from __future__ import annotations

from typing import Any, Literal, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from nns._helpers import _warn_unsupported
from nns._native import honours_ncores
from nns.part import NoiseReduction
from nns.regression import Order, _normalize_type

NBest = int | Literal["all"] | None

MRegFitted = dict[str, NDArray[np.float64] | NDArray[np.str_]]
"""Fit table keyed by encoded predictor name (frame column names, factor dummies
``<name>_<level>``, ``V1..Vn`` for an unnamed matrix) plus ``y``, ``y.hat``,
``NNS.ID``, ``residuals``, and (with ``confidence_interval``)
``conf.int.pos``/``conf.int.neg``. Column count depends on the input, so keys
stay dynamic."""

MRegPredInt = TypedDict(
    "MRegPredInt",
    {
        "pred.int.neg": NDArray[np.float64],
        "pred.int.pos": NDArray[np.float64],
    },
)
"""Prediction interval bounds for the requested ``point_est`` rows."""

MRegResult = TypedDict(
    "MRegResult",
    {
        "Point.est": "NDArray[np.float64] | None",
        "RPM": dict[str, NDArray[np.float64]],
        "R2": "float | None",
        "rhs.partitions": dict[str, NDArray[np.float64]],
        "pred.int": "MRegPredInt | None",
        "Fitted.xy": "MRegFitted | None",
        "n.best": int,
        "dist": str,
        "class.levels": "list[str] | None",
    },
)
"""``nns_m_reg`` result.

Every key is always present: ``R2`` and ``Fitted.xy`` are ``None`` with
``point_only=True``, ``Point.est`` is ``None`` without ``point_est``,
``pred.int`` is ``None`` unless ``confidence_interval`` is set, and
``class.levels`` is ``None`` for a numeric response.
"""


@honours_ncores
def nns_m_reg(
    x: NDArray[Any],
    y: NDArray[Any],
    *,
    factor_2_dummy: bool = True,
    order: Order = None,
    n_best: NBest = None,
    type: str | None = None,
    point_est: NDArray[np.float64] | None = None,
    point_only: bool = False,
    plot: bool = False,
    residual_plot: bool = False,
    location: object | None = None,
    noise_reduction: NoiseReduction = "off",
    dist: str | None = None,
    return_values: bool = False,
    plot_regions: bool = False,
    ncores: int | None = None,
    confidence_interval: float | None = None,
    class_levels: list[object] | None = None,
    factor_levels: Any = None,
    categorical: bool = False,
) -> MRegResult:
    """Multivariate regression matching R's ``NNS.M.reg``.

    Delegates to :func:`nns._reg_engine.nns_reg_engine`, whose multivariate
    branch is the port of R's ``NNS.M.reg``. Predictors are encoded exactly as
    in :func:`nns.nns_reg`: string and categorical columns become one-hot
    dummies (``factor_2_dummy=True``, R's default) or integer codes, pandas
    column names are kept as the ``RPM`` / ``Fitted.xy`` keys, and
    [point_est] columns are matched by name. A string, bool or categorical
    response (or one with ``class_levels``) is a classification target, as in
    R; ``class_levels`` fixes the code order. At least two encoded predictor
    columns are required. ``categorical=True`` (R's ``categorical``) fits exact
    joint cells: maximum order and, unless ``n_best`` is given, one neighbour.
    As in R, it is not inferred from the predictors here; ``nns_reg`` sets it
    when every predictor is categorical.
    """
    _warn_unsupported(
        location=location is not None,
        dist=False,
        return_values=return_values is not False,
        plot_regions=plot_regions,
    )
    type_value = _normalize_type(type)
    from nns._reg_engine import nns_reg_engine

    result = cast(
        "MRegResult",
        nns_reg_engine(
            x,
            y,
            factor_2_dummy=factor_2_dummy,
            order=order,
            n_best=cast(Any, n_best),
            type=type_value,
            point_est=point_est,
            point_only=point_only,
            noise_reduction=noise_reduction,
            dist=dist,
            confidence_interval=confidence_interval,
            class_levels=class_levels,
            factor_levels=factor_levels,
            multivariate_only=True,
            categorical=bool(categorical),
        ),
    )
    if (plot or residual_plot) and isinstance(result.get("Fitted.xy"), dict):
        _render_m_reg(cast(Any, result["Fitted.xy"]))
    return result


def _render_m_reg(fitted: dict[str, NDArray[np.float64] | NDArray[np.str_]]) -> None:
    """Render R's NNS.M.reg residual plot (Multivariate_Regression.R:367-377).

    The multivariate plot output is the residual plot: actual ``y`` over the
    observation index as ``steelblue`` open circles, fitted ``y.hat`` as a
    ``red`` line, and a pink (alpha 0.375) confidence band when present.
    """
    from nns.plotting import palette
    from nns.plotting._mpl import resolve_ax

    y = np.asarray(fitted["y"], dtype=np.float64)
    y_hat = np.asarray(fitted["y.hat"], dtype=np.float64)
    if y.size == 0 or y.size != y_hat.size:
        return
    ax = resolve_ax(None)
    index = np.arange(1, y.size + 1)
    ax.scatter(index, y, facecolors="none", edgecolors="steelblue", marker="o")
    ax.plot(index, y_hat, color="red", linewidth=2)
    if "conf.int.pos" in fitted and "conf.int.neg" in fitted:
        pos = np.asarray(fitted["conf.int.pos"], dtype=np.float64)
        neg = np.asarray(fitted["conf.int.neg"], dtype=np.float64)
        mask = np.isfinite(pos) & np.isfinite(neg)
        if mask.any():
            ax.fill_between(
                index[mask], neg[mask], pos[mask],
                color=palette.PINK, alpha=palette.CI_ALPHA_REG, linewidth=0.0,
            )
    ax.set_xlabel("Index")
    ax.set_ylabel("y (blue)   y.hat (red)")
    ax.set_title("NNS.M.reg Residual Plot")


