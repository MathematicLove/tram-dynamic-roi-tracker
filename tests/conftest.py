import sys
from pathlib import Path

# Make `algorithm` importable when pytest is run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
