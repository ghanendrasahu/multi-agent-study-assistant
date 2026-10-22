"""Smoke tests: the package and its agents import cleanly.

`src` is put on the path so the package resolves exactly as it does in
production, and every agent module is imported to prove the surface exists.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def test_package_imports():
    module = importlib.import_module("study_agent")

    assert module is not None


@pytest.mark.parametrize("name", ["analyst", "critic", "finaliser"])
def test_agent_modules_are_importable(name):
    module = importlib.import_module(f"study_agent.agents.{name}")

    assert module is not None