"""Streamlit application entrypoint.

Runs main.py to support 'streamlit run app.py' as described in the documentation.
"""
import runpy
import sys
from pathlib import Path

# Prevent crash from broken torchaudio dynamic library on macOS
try:
    import torchaudio  # noqa: F401
except (OSError, ImportError, Exception):
    sys.modules["torchaudio"] = None

main_file = Path(__file__).parent / "main.py"
runpy.run_path(str(main_file), run_name="__main__")
