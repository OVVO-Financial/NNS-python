"""Force the non-interactive Agg backend for all plotting tests (no display)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
