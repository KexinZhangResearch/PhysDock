import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PHYSDOCK_ROOT", str(Path(__file__).resolve().parents[2]))          # data/v1 of PhysDock (the tests on real proteins skip without it)
