"""Public NNS.reg wrapper that preserves computation/plot separation."""

from __future__ import annotations

from typing import Any

from nns.regression import _maybe_render_reg
from nns.regression import nns_reg as _nns_reg


def nns_reg(
    x: Any,
    y: Any,
    *,
    factor_2_dummy: bool = True,
    order: Any = None,
    dim_red_method: Any = None,
    tau: Any = None,
    type: str | None = None,
    point_est: Any = None,
    return_values: bool = True,
    plot: bool = False,
    plot_regions: bool = False,
    residual_plot: bool = False,
    confidence_interval: float | None = None,
    threshold: float = 0.0,
    n_best: Any = None,
    smooth: bool = False,
    noise_reduction: Any = "off",
    dist: str | None = None,
    ncores: int | None = None,
    point_only: bool = False,
    multivariate_call: bool = False,
    class_levels: list[object] | None = None,
    factor_levels: Any = None,
) -> Any:
    """Nonlinear nonparametric regression and classification (R ``NNS.reg``).

    Parameters
    ----------
    x
        Predictor(s): a vector, a matrix, or a pandas frame. Frame column
        names are kept. String and categorical columns are factors (declared
        category order, else sorted levels); ``bool`` and numeric columns are
        numeric. More than one encoded predictor runs the multivariate
        ``NNS.M.reg`` branch unless ``dim_red_method`` is given.
    y
        Response. Strings, ``bool`` and pandas categoricals are classification
        targets (as is an exact 0/1 numeric response); everything else,
        including an object array of numbers, is regression.
    factor_2_dummy
        One-hot encode factor predictors (default, as R) instead of using
        integer level codes.
    order, n_best, noise_reduction, dist, threshold, tau
        Partition order (``None``, an integer or ``"max"``), number of nearest
        regression points in the multivariate branch (``None``, an integer or
        ``"all"``), within-partition reducer (``"off"``, ``"mean"``,
        ``"median"``, ``"mode"``), distance (``None``/``"NNS"``, ``"L1"``,
        ``"L2"``, ``"FACTOR"``), dimension-reduction coefficient threshold and
        causation lag type (``"cs"`` / ``"ts"``).
    dim_red_method
        ``None``, ``"cor"``, ``"NNS.dep"``, ``"NNS.caus"``, ``"all"``,
        ``"equal"`` or a numeric coefficient vector: collapse the predictors
        to one synthetic regressor ``x.star``.
    type
        ``None``, ``"CLASS"`` (force classification) or ``"XONLY"``.
    point_est
        Points to predict. Frame (or named-row) columns are matched to the
        training columns by name: the same names are reordered, names
        disjoint from the training names are positional, a partial overlap
        raises ``ValueError``.
    confidence_interval
        Residual-quantile band level in (0, 1) for ``Fitted.xy`` and
        ``pred.int``.
    smooth
        Smoothing-spline fit through the regression points (univariate).
    point_only
        Skip the fit statistics (``R2``, ``SE``, ``Fitted.xy`` are ``None``).
    multivariate_call
        Return only the regression points ``{"x", "y"}``.
    class_levels
        Response levels in code order (codes ``1..K``); implies
        classification.
    factor_levels
        Predictor levels: a mapping from column name to levels, one entry per
        column (``None`` for numeric columns), or a flat list for a single
        predictor. Forces those columns to be factors.
    plot, plot_regions, residual_plot
        Render plots as a side effect; the returned values never change.
    return_values, ncores
        Accepted for R compatibility and ignored.

    Returns
    -------
    dict
        R's list names: ``R2``, ``SE``, ``Prediction.Accuracy``, ``equation``,
        ``x.star``, ``derivative``, ``Point.est``, ``pred.int``,
        ``regression.points``, ``Fitted.xy``, ``class.levels`` (univariate),
        or ``R2``, ``rhs.partitions``, ``RPM``, ``Point.est``, ``pred.int``,
        ``Fitted.xy``, ``n.best``, ``dist``, ``class.levels`` (multivariate).
        Classification returns numeric class codes; ``class.levels`` maps
        them to labels.
    """
    result = _nns_reg(  # type: ignore[misc]
        x,
        y,
        factor_2_dummy=factor_2_dummy,
        order=order,
        dim_red_method=dim_red_method,
        tau=tau,
        type=type,
        point_est=point_est,
        return_values=return_values,
        plot=False,
        plot_regions=False,
        residual_plot=False,
        confidence_interval=confidence_interval,
        threshold=threshold,
        n_best=n_best,
        smooth=smooth,
        noise_reduction=noise_reduction,
        dist=dist,
        ncores=ncores,
        point_only=point_only,
        multivariate_call=multivariate_call,
        class_levels=class_levels,
        factor_levels=factor_levels,
    )
    _maybe_render_reg(
        result,
        plot=plot,
        plot_regions=plot_regions,
        residual_plot=residual_plot,
        point_est=point_est,
    )
    return result
