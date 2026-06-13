"""Lazy matplotlib loading for the plotting API.

matplotlib is a regular dependency of this package, but it is still imported
lazily (never at package import time) so ``import nns`` stays light. Every plot
function calls :func:`require_mpl` to import it on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

_INSTALL_HINT = (
    "matplotlib is required for nns.plotting but could not be imported; "
    "reinstall ovvo-nns to restore it (`pip install --force-reinstall ovvo-nns`)."
)


def require_mpl() -> Any:
    """Import and return the ``matplotlib.pyplot`` module, or raise ImportError."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via test monkeypatch
        raise ImportError(_INSTALL_HINT) from exc
    return plt


def horizontal_boxplot(ax: Axes, data: Any, **kwargs: Any) -> Any:
    """``ax.boxplot`` rendered horizontally, compatible across matplotlib versions.

    ``vert=`` was deprecated for ``orientation=`` in matplotlib 3.11; prefer the
    new keyword when present and fall back to the old one for >= 3.7.
    """
    import matplotlib

    version = tuple(int(p) for p in matplotlib.__version__.split(".")[:2])
    if version >= (3, 11):
        return ax.boxplot(data, orientation="horizontal", **kwargs)
    return ax.boxplot(data, vert=False, **kwargs)


def resolve_ax(ax: Axes | None) -> Axes:
    """Return ``ax`` if given, otherwise create a fresh Axes.

    Plot functions never call ``plt.show()``; they return the Axes/Figure so the
    caller controls display and saving.
    """
    if ax is not None:
        return ax
    plt = require_mpl()
    _, new_ax = plt.subplots()
    return cast("Axes", new_ax)


__all__ = ["require_mpl", "resolve_ax"]
