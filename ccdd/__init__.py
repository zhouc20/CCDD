"""Public package entry point for CCDD.

The code was historically imported as ``ccdd`` and, in some scripts, as
``ccdd``.  Keep those names available so existing checkpoints and downstream
scripts continue to work after the public package is renamed to ``CCDD``.
"""

from __future__ import annotations

import sys

sys.modules.setdefault("ccdd", sys.modules[__name__])
sys.modules.setdefault("ccdd", sys.modules[__name__])

__all__ = ["GiddPipeline"]


def __getattr__(name: str):
    if name == "GiddPipeline":
        from .pipeline import GiddPipeline

        return GiddPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
