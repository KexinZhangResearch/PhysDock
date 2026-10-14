#!/usr/bin/env python3
"""Build a system pickle from a receptor PDB and a ligand SDF.

    python scripts/prepare/prepare_system.py --receptor_pdb receptor.pdb --ligand_sdf ligand.sdf --ligand_ccd_id EJQ \\
        --systems_dir systems

Writes <systems_dir>/<name>.pkl.gz and <systems_dir>/fastas/ (input of scripts/prepare/homo_search.py).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PhysDock.data.tools.generate_system import generate_system  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receptor_pdb", required=True, help="receptor PDB file")
    ap.add_argument("--ligand_sdf", required=True, help="ligand SDF file")
    ap.add_argument("--ligand_ccd_id", required=True, help="CCD id of the ligand (atom names must match the CCD library)")
    ap.add_argument("--systems_dir", required=True, help="output folder of the system pickles")
    ap.add_argument("--ccd", default=str(ROOT / "data" / "ccd_id_meta_data.pkl.gz"), help="CCD library")
    a = ap.parse_args()
    generate_system(receptor_pdb_path=a.receptor_pdb, ligand_sdf_path=a.ligand_sdf, ligand_ccd_id=a.ligand_ccd_id,
                    systems_dir=a.systems_dir, ccd_id_meta_data=a.ccd)
