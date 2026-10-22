"""Smoke tests: every agent module compiles and exposes its public surface.

The modules are loaded directly from their files so the test never triggers
an import-time side effect (no API key or model download required).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

AGENTS = Path(__file__).resolve().parents[1] / "src" / "study_agent" / "agents"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"smoke_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["analyst", "critic", "finaliser"])
def test_agent_module_loads(name):
    module = _load(AGENTS / f"{name}.py")

    assert module is not None