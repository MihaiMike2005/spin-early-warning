"""
Smoke test: every spinwarn module imports, and so does driftscore.

Catches a broken install, a syntax error or a circular import before any
other test runs. The driftscore check proves the git dependency installed,
which only works while DriftScoreUI is public.
"""

import importlib

import pytest


MODULES = [
    "config", "data", "labeling", "features", "evaluate", "baselines",
    "models", "cv", "run", "spotcheck", "report",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"spinwarn.{name}")


def test_driftscore_installed():
    from driftscore.signals import slip_angle_deg

    # Going straight along +z with heading 0 is zero slip.
    assert slip_angle_deg((0.0, 0.0, 20.0), 0.0) == 0.0
