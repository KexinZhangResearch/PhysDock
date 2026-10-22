import os
import sys
from pathlib import Path

POMELO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POMELO))
# PhysDock: its diffuser, loss and data are Pomelo's (PHYSDOCK_ROOT, else the parent of Pomelo, else the shared checkout)
for root in (os.environ.get("PHYSDOCK_ROOT"), str(POMELO.parent), "/2024133006/zkx_temp/PhysDock"):
    if root and (Path(root) / "PhysDock" / "models").is_dir():
        os.environ["PHYSDOCK_ROOT"] = root
        sys.path.insert(1, root)
        break
