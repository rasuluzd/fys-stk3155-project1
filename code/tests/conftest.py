import sys
from pathlib import Path

# make the modules in code/ importable when running `pytest` from the project root or code/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
