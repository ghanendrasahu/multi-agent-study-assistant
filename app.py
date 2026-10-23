"""
Streamlit entry point for local dev and Render deployment.

Render starts this file (`streamlit run app.py`). It adds src/ to the
Python path, then executes the real app module.

`runpy` is required here: a plain `from study_agent.app import *` only
builds the UI on the first import. Streamlit reruns this file on every
widget interaction, but the imported module stays cached, so the page
would go blank after the first click.
"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

runpy.run_module("study_agent.app", run_name="__main__")
