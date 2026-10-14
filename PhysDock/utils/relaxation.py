import copy
import sys
import torch
import os
import tqdm
import pandas as pd
import argparse

sys.path.append("../")
from PhysDock.utils.io_utils import run_pool_tasks, load_txt, dump_txt
from pathlib import Path
import openmm.app as mm_app
import openmm.unit as mm_unit
import openmm as mm
import os.path
import sys
import mdtraj
from openmm.app import PDBFile, Modeller
import pdbfixer
from openmmforcefields.generators import SystemGenerator
from openff.toolkit import Molecule
from openff.toolkit.utils.exceptions import UndefinedStereochemistryError, RadicalsNotSupportedError
from openmm import CustomExternalForce
from posebusters import PoseBusters
from posebusters.posebusters import _dataframe_from_output
from posebusters.cli import _select_mode, _format_results


def get_bust_results(  # noqa: PLR0913
        mol_pred,
        mol_true,
        mol_cond,
        top_n: int | None = None,
):
    mol_pred = [Path(mol_pred)]
    mol_true = Path(mol_true)
    mol_cond = Path(mol_cond)  # Each bust running has different receptor

    # run on single input
    d = {k for k, v in dict(mol_pred=mol_pred, mol_true=mol_true, mol_cond=mol_cond).items() if v}
    mode = _select_mode(None, d)
    posebusters = PoseBusters(mode, top_n=top_n)
    cols = ["mol_pred", "mol_true", "mol_cond"]
    posebusters.file_paths = pd.DataFrame([[mol_pred, mol_true, mol_cond] for mol_pred in mol_pred], columns=cols)
    posebusters_results = posebusters._run()
    results = None
    for i, results_dict in enumerate(posebusters_results):
        results = _dataframe_from_output(results_dict, posebusters.config, full_report=True)
        break
    return results


def fix_pdb(pdbname, outdir, file_name):
    """add"""
    fixer = pdbfixer.PDBFixer(pdbname)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    # 根据文件名判断是否写入指定目录
    # if "relaxed_complex" in file_name:
    #     target_path = f'{outdir}/{file_name}_hydrogen_added.pdb'
    # else:
    #     target_path = f'{file_name}_hydrogen_added.pdb'
    # mm_app.PDBFile.writeFile(fixer.topology, fixer.positions, open(target_path, 'w'))
    return fixer.topology, fixer.positions


def set_system(topology):
    """
    Set the system using the topology from the pdb file
    """
    # Put it in a force field to skip adding all particles manually
    forcefield = mm_app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    system = forcefield.createSystem(topology,
                                     removeCMMotion=False,
                                     nonbondedMethod=mm_app.NoCutoff,
                                     rigidWater=True  # Use implicit solvent
                                     )
    return system


def minimize_energy(
        topology,
        system,
        positions,
        outdir,
        out_title
):
    '''Function that minimizes energy, given topology, OpenMM system, and positions '''
    # Use a Brownian Integrator
    integrator = mm.BrownianIntegrator(
        100 * mm.unit.kelvin,
        100. / mm.unit.picoseconds,
        2.0 * mm.unit.femtoseconds
    )
    # platform = Platform.getPlatformByName('CUDA')
    # properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
    simulation = mm.app.Simulation(topology, system, integrator)

    # Initialize the DCDReporter
    reportInterval = 100  # Adjust this value as needed
    reporter = mdtraj.reporters.DCDReporter('positions.dcd', reportInterval)

    # Add the reporter to the simulation
    simulation.reporters.append(reporter)

    simulation.context.setPositions(positions)

    simulation.minimizeEnergy(1, 100)
    # Save positions
    minpositions = simulation.context.getState(getPositions=True).getPositions()

    # 根据out_title决定是否写入指定目录
    if "relaxed_complex" in out_title:
        target_path = outdir + f'/{out_title}.pdb'
    else:
        target_path = f'{out_title}.pdb'
    mm_app.PDBFile.writeFile(topology, minpositions, open(target_path, 'w'))

    # Get and return the minimized energy
    minimized_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()

    reporter.close()

    return topology, minpositions, minimized_energy


def add_restraints(
        system,
        topology,
        positions,
        restraint_type
):
    '''Function to add restraints to specified group of atoms

    Code adapted from https://gist.github.com/peastman/ad8cda653242d731d75e18c836b2a3a5

    '''
    restraint = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
    system.addForce(restraint)
    restraint.addGlobalParameter('k', 100000000.0 * mm_unit.kilojoules_per_mole / mm_unit.nanometer ** 2)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    for atom in topology.atoms():
        if restraint_type == 'protein':
            if 'x' not in atom.name:
                restraint.addParticle(atom.index, positions[atom.index])
        elif restraint_type == 'CA+ligand':
            if ('x' in atom.name) or (atom.name == "CA"):
                restraint.addParticle(atom.index, positions[atom.index])

    return system


def run(
        # i
        input_pdb,
        outdir,
        mol_in,
        file_name,
        restraint_type="ca+ligand",
        relax_protein_first=False,
        steps=100,
):
    try:
        ligand_mol = Molecule.from_file(mol_in)
    # Check for undefined stereochemistry, allow undefined stereochemistry to be loaded
    except UndefinedStereochemistryError:
        print('Undefined Stereochemistry Error found! Trying with undefined stereo flag True')
        ligand_mol = Molecule.from_file(mol_in, allow_undefined_stereo=True)
    # Check for radicals -- break out of script if radical is encountered
    except RadicalsNotSupportedError:
        print('OpenFF does not currently support radicals -- use unrelaxed structure')
        sys.exit()
    # Assigning partial charges first because the default method (am1bcc) does not work
    ligand_mol.assign_partial_charges(partial_charge_method='gasteiger')

    ## Read protein PDB and add hydrogens
    protein_topology, protein_positions = fix_pdb(input_pdb, outdir, file_name)
    # print('Added all atoms...')

    # Minimize energy for the protein
    system = set_system(protein_topology)
    # print('Creating system...')
    # Relax
    if relax_protein_first:
        print('Relaxing ONLY protein structure...')
        protein_topology, protein_positions = minimize_energy(
            protein_topology,
            system,
            protein_positions,
            outdir,
            f'{file_name}_relaxed_protein'
        )

    # print('Preparing complex')
    ## Add protein first
    modeller = Modeller(protein_topology, protein_positions)
    # print('System has %d atoms' % modeller.topology.getNumAtoms())

    ## Then add ligand
    # print('Adding ligand...')
    lig_top = ligand_mol.to_topology()
    modeller.add(lig_top.to_openmm(), lig_top.get_positions().to_openmm())
    # print('System has %d atoms' % modeller.topology.getNumAtoms())

    # print('Preparing system')
    # Initialize a SystemGenerator using the GAFF for the ligand and implicit water.
    # forcefield_kwargs = {'constraints': mm_app.HBonds, 'rigidWater': True, 'removeCMMotion': False, 'hydrogenMass': 4*mm_unit.amu }
    system_generator = SystemGenerator(
        forcefields=['amber14-all.xml', 'implicit/gbn2.xml'],
        small_molecule_forcefield='gaff-2.11',
        molecules=[ligand_mol],
        # forcefield_kwargs=forcefield_kwargs
    )

    ## Create system
    system = system_generator.create_system(modeller.topology, molecules=ligand_mol)

    # if restraint_type == 'protein':
    #     print('Adding restraints on entire protein')
    # elif restraint_type == 'CA+ligand':
    #     print('Adding restraints on protein CAs and ligand atoms')

    system = add_restraints(system, modeller.topology, modeller.positions, restraint_type=restraint_type)

    ## Minimize energy for the complex and print the minimized energy
    _, _, minimized_energy = minimize_energy(
        modeller.topology,
        system,
        modeller.positions,
        outdir,
        f'{file_name}_relaxed_complex'
    )


def relax(receptor_pdb, ligand_mol_sdf):
    output_dir = os.path.split(receptor_pdb)[0]
    file_name = os.path.split(receptor_pdb)[1].split(".")[0]
    system_file_name = "system" + file_name.split("receptor")[1]
    try:
        run(
            input_pdb=receptor_pdb,
            outdir=output_dir,
            mol_in=ligand_mol_sdf,
            file_name=system_file_name
        )
        lines = load_txt(
            os.path.join(output_dir, f"{system_file_name}_relaxed_complex.pdb")).split("\n")
        receptor = "\n".join([i for i in lines if "HETATM" not in i])
        dump_txt(receptor, os.path.join(output_dir, f"{file_name}_relaxed_complex.pdb"))
    except Exception as e:
        print(dir, "can't relax,", e)