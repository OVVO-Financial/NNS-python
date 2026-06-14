"""Force the non-interactive Agg backend for all plotting tests (no display)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

# Plotting tests open many short-lived figures (closed per-test); don't warn
# about the open-figure count if this session shares a process with others.
matplotlib.rcParams["figure.max_open_warning"] = 0
