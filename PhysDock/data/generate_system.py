import sys
import os
import argparse
import warnings

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem

from PhysDock.utils.io_utils import dump_pkl, load_pkl, convert_md5_string
from PhysDock.data.constants.PDBData import protein_letters_3to1_extended

warnings.filterwarnings("ignore")


def generate_system(
        receptor_pdb_path,
        ligand_sdf_path,
        ligand_ccd_id,
        systems_dir,
        ccd_id_meta_data=None,

        # bfd_database_path,
        # uniclust30_database_path,
        # uniref90_database_path,
        # mgnify_database_path,
        # uniprot_database_path,
        # jackhmmer_binary_path,
        # hhblits_binary_path,
        #
        # input_dir,
        # out_dir,
        #
        # n_cpus=16,
        # n_workers=1,

):
    """
    Parse PDB and SDF files to generate protein-ligand complex features.

    Args:
        input_pdb (str): Path to the input PDB file.
        input_sdf (str): Path to the input ligand SDF file.
        systems_dir (str): Directory to save system feature pickle files.
        feature_dir (str): Directory to save feature files (e.g., input FASTA).
        ligand_id (str): CCD ID of the ligand.
    """
    # Create output directories
    if ccd_id_meta_data is None:
        print("Loading CCD meta data ...")
        ccd_id_meta_data = load_pkl(os.path.join(os.path.split(__file__)[0], "../../params/ccd_id_meta_data.pkl.gz"))
    os.makedirs(systems_dir, exist_ok=True)


    # Initialize parser and data containers
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure("", receptor_pdb_path)
    model = structure[0]

    all_chain_features = {}
    used_chain_ids = []

    # Extract protein chains from PDB
    for chain in model:
        chain_id = chain.id
        used_chain_ids.append(chain_id)
        all_chain_features[chain_id] = {
            "all_atom_positions": [],
            "all_atom_mask": [],
            "ccds": []
        }

        offset = None
        for residue in chain:
            if offset is None:
                offset = int(residue.id[1])

            resname = residue.get_resname().strip().ljust(3)
            res_idx = int(residue.id[1]) - offset
            num_atoms = len(ccd_id_meta_data[resname]["ref_atom_name_chars"])

            # Fill missing residues
            while len(all_chain_features[chain_id]["ccds"]) < res_idx:
                all_chain_features[chain_id]["ccds"].append("UNK")
                all_chain_features[chain_id]["all_atom_positions"].append(np.zeros([1, 3], dtype=np.float32))
                all_chain_features[chain_id]["all_atom_mask"].append(np.zeros([1], dtype=np.int8))

            # Initialize residue data
            all_chain_features[chain_id]["ccds"].append(resname)
            all_chain_features[chain_id]["all_atom_positions"].append(np.zeros([num_atoms, 3], dtype=np.float32))
            all_chain_features[chain_id]["all_atom_mask"].append(np.zeros([num_atoms], dtype=np.int8))

            ref_atom_names = ccd_id_meta_data[resname]["ref_atom_name_chars"]
            for atom in residue:
                if atom.name in ref_atom_names:
                    atom_idx = ref_atom_names.index(atom.name)
                    all_chain_features[chain_id]["all_atom_positions"][res_idx][atom_idx] = atom.coord
                    all_chain_features[chain_id]["all_atom_mask"][res_idx][atom_idx] = 1

        # Add interaction features # TODO PLIP
        interaction_keys = ['salt bridges', 'pi-cation interactions', 'hydrophobic interactions',
                            'pi-stacking', 'hydrogen bonds', 'metal complexes']
        for key in interaction_keys:
            all_chain_features[chain_id][key] = np.zeros(len(all_chain_features[chain_id]["ccds"]), dtype=np.int8)

    # Extract ligand from SDF
    supplier = Chem.SDMolSupplier(ligand_sdf_path, removeHs=True, sanitize=False)
    mol = supplier[0]
    mol = Chem.RemoveAllHs(mol)
    conf = mol.GetConformer()
    ligand_chain_id = "1"
    used_chain_ids.append(ligand_chain_id)

    ligand_atom_count = mol.GetNumAtoms()
    ligand_positions = np.zeros([ligand_atom_count, 3], dtype=np.float32)
    ligand_masks = np.ones([ligand_atom_count], dtype=np.int8)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        ligand_positions[idx] = [pos.x, pos.y, pos.z]

    all_chain_features[ligand_chain_id] = {
        "all_atom_positions": [ligand_positions],
        "all_atom_mask": [ligand_masks],
        "ccds": [ligand_ccd_id.upper()]
    }

    for key in interaction_keys:
        all_chain_features[ligand_chain_id][key] = np.zeros(1, dtype=np.int8)

    # Generate system pickle file
    save_name = os.path.basename(receptor_pdb_path).replace('.pdb', '')
    for cid in used_chain_ids:
        save_name += f"_{cid}"

    dump_pkl(all_chain_features, os.path.join(systems_dir, f"{save_name}.pkl.gz"))

    # Generate FASTA files to run homo search
    for cid, features in all_chain_features.items():
        if cid == ligand_chain_id:
            continue
        sequence = ''.join(protein_letters_3to1_extended.get(ccd, "X") for ccd in features["ccds"])
        md5_hash = convert_md5_string(f"protein:{sequence}")
        os.makedirs(os.path.join(systems_dir, "fastas"), exist_ok=True)
        with open(os.path.join(systems_dir, "fastas", f"{md5_hash}.fasta"), "w") as f:
            f.write(f">{md5_hash}\n{sequence}\n")
    print("Make system successfully!")