import os
import sys
import runpy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.join(CURRENT_DIR, "notebook")

if NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOK_DIR)

runpy.run_path(os.path.join(NOTEBOOK_DIR, "app.py"), run_name="__main__")
