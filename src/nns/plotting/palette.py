"""R ``grDevices`` colors, pinned to exact hex for plot parity with R NNS.

R and matplotlib agree on ``steelblue`` and ``red`` but **disagree** on
``green`` and ``grey``/``gray``. To stay faithful to the ``col=`` usage in
``tools/NNS/R/*.R`` we pin every non-trivial color to its exact R hex value.

Fidelity traps (do **not** trust matplotlib's same-named color):

* R ``green`` is pure green ``#00FF00`` (matplotlib ``green`` is ``#008000``).
* R ``grey``/``gray`` is ``#BEBEBE`` (matplotlib ``gray`` is ``#808080``).

The literal names ``"steelblue"`` and ``"red"`` are identical in both stacks
and may be used as-is; everything else should reference the constants here.
"""

from __future__ import annotations

import colorsys

# -- Colors that are identical in R and matplotlib (safe to use by name) ----
STEELBLUE = "#4682B4"  # R "steelblue" == mpl "steelblue"
RED = "#FF0000"  # R "red"       == mpl "red"
BLUE = "#0000FF"  # R "blue"      == mpl "blue"
PINK = "#FFC0CB"  # R rgb(1, 192/255, 203/255) CI band == mpl "pink"

# -- Colors where the same name means something DIFFERENT in matplotlib -----
GREEN = "#00FF00"  # R "green" is PURE green; mpl "green" is #008000 -> use this
GREY = "#BEBEBE"  # R "grey"/"gray"; mpl "gray" is #808080 -> DIFFERENT
AZURE4 = "#838B8B"  # R "azure4" (no matplotlib name)

# -- Confidence/prediction-interval fill alphas R uses ----------------------
CI_ALPHA_REG = 0.375  # NNS.reg pink band: rgb(1, 192/255, 203/255, alpha = 0.375)
CI_ALPHA_ARMA = 0.5  # NNS.ARMA pink band & NNS.ARMA.optim steelblue band: alpha = 0.5


def rainbow(n: int) -> list[tuple[float, float, float]]:
    """Emulate R's ``grDevices::rainbow(n)`` (HSV with ``s = v = 1``).

    R sweeps hues ``0, 1/n, ..., (n-1)/n`` at full saturation and value. This
    is *not* identical to ``plt.cm.rainbow``; use this for color parity on the
    multi-series plots (``NNS.norm``, ``NNS.ANOVA``).
    """
    if n <= 0:
        return []
    return [colorsys.hsv_to_rgb(i / n, 1.0, 1.0) for i in range(n)]


__all__ = [
    "AZURE4",
    "BLUE",
    "CI_ALPHA_ARMA",
    "CI_ALPHA_REG",
    "GREEN",
    "GREY",
    "PINK",
    "RED",
    "STEELBLUE",
    "rainbow",
]
