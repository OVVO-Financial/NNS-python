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
    dist: str = "L2",
    ncores: int | None = None,
    point_only: bool = False,
    multivariate_call: bool = False,
    class_levels: list[object] | None = None,
    factor_levels: Any = None,
) -> Any:
    """Run the repaired regression engine and render only when requested.

    Plot flags are handled strictly as side effects and therefore cannot alter
    the statistical return value used by the parity suite.
    """
    result = _nns_reg(
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
