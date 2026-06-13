"""The palette pins exact R grDevices hex -- especially the fidelity traps."""

from __future__ import annotations

import matplotlib.colors as mcolors

from nns.plotting import palette


def test_safe_names_match_matplotlib() -> None:
    # steelblue and red are identical in R and matplotlib.
    assert mcolors.to_hex(palette.STEELBLUE) == "#4682b4"
    assert mcolors.to_hex("steelblue") == "#4682b4"
    assert mcolors.to_hex(palette.RED) == "#ff0000"
    assert mcolors.to_hex("red") == "#ff0000"


def test_green_trap() -> None:
    # R "green" is PURE green; matplotlib "green" is #008000 -- they DIFFER.
    assert palette.GREEN == "#00FF00"
    assert mcolors.to_hex("green") == "#008000"
    assert mcolors.to_hex(palette.GREEN) != mcolors.to_hex("green")


def test_grey_trap() -> None:
    # R "grey"/"gray" is #BEBEBE; matplotlib "gray" is #808080 -- they DIFFER.
    assert palette.GREY == "#BEBEBE"
    assert mcolors.to_hex("gray") == "#808080"
    assert mcolors.to_hex(palette.GREY) != mcolors.to_hex("gray")


def test_azure4_pinned() -> None:
    assert mcolors.to_hex(palette.AZURE4) == "#838b8b"


def test_pink_band_matches_r_rgb() -> None:
    # R uses rgb(1, 192/255, 203/255) for the CI band == matplotlib "pink".
    assert mcolors.to_hex(palette.PINK) == "#ffc0cb"


def test_ci_alphas() -> None:
    assert palette.CI_ALPHA_REG == 0.375
    assert palette.CI_ALPHA_ARMA == 0.5


def test_rainbow_matches_r_hsv() -> None:
    # R rainbow(n): HSV with s=v=1 and hues 0, 1/n, ... -> for n=3, RGB primaries.
    rgb = palette.rainbow(3)
    assert rgb[0] == (1.0, 0.0, 0.0)
    assert rgb[1] == (0.0, 1.0, 0.0)
    assert rgb[2] == (0.0, 0.0, 1.0)
    assert palette.rainbow(0) == []
