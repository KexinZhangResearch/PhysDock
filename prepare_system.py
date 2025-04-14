import argparse
from PhysDock.data.generate_system import generate_system

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PhysDock Redocking Data Preparation')
    parser.add_argument('--receptor_pdb_path', type=str, required=True, help='Receptor PDB file path')
    parser.add_argument('--ligand_sdf_path', type=str, required=True, help='Input ligand SDF file path')
    parser.add_argument('--ligand_ccd_id', type=str, required=True, help='Ligand CCD ID')
    parser.add_argument('--systems_dir', type=str, required=True, help='Output directory for system pickle files')
    args = parser.parse_args()
    generate_system(
        receptor_pdb_path=args.receptor_pdb_path,
        ligand_sdf_path=args.ligand_sdf_path,
        ligand_ccd_id=args.ligand_ccd_id,
        systems_dir=args.systems_dir,
    )
