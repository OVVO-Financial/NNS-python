"""Allow numbered vignette scripts to import shared support from the repository.

Python imports ``sitecustomize`` from the script directory during startup.  By
adding the repository root here, each vignette remains directly executable as
``python examples/vignettes/NN_topic.py`` without packaging the examples.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
