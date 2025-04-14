# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Feature processing logic for multimer data pipeline."""
from typing import Iterable, MutableMapping, List, Mapping, Dict, Any, Union
from scipy.sparse import coo_matrix
import numpy as np

from . import msa_pairing

FeatureDict = Dict[str, Union[np.ndarray, coo_matrix, None, Any]]
# TODO: Move this into the config
REQUIRED_FEATURES = frozenset({
    'aatype', 'all_atom_mask', 'all_atom_positions', 'all_chains_entity_ids',
    'all_crops_all_chains_mask', 'all_crops_all_chains_positions',
    'all_crops_all_chains_residue_ids', 'assembly_num_chains', 'asym_id',
    'bert_mask', 'cluster_bias_mask', 'deletion_matrix', 'deletion_mean',
    'entity_id', 'entity_mask', 'mem_peak', 'msa', 'msa_mask', 'num_alignments',
    'num_templates', 'queue_size', 'residue_index', 'resolution',
    'seq_length', 'seq_mask', 'sym_id', 'template_aatype',
    'template_all_atom_mask', 'template_all_atom_positions'
})

MAX_TEMPLATES = 4
MSA_CROP_SIZE = 16384


def _is_homomer_or_monomer(chains: Iterable[Mapping[str, np.ndarray]]) -> bool:
    """Checks if a list of chains represents a homomer/monomer example."""
    # Note that an entity_id of 0 indicates padding.
    # num_unique_chains = len(np.unique(np.concatenate(
    #     [np.unique(chain['entity_id'][chain['entity_id'] > 0]) for
    #      chain in chains])))

    # return num_unique_chains == 1
    num_chains = len(chains)
    return num_chains == 1


def pair_and_merge(
        all_chain_features: MutableMapping[str, Mapping[str, np.ndarray]],
        is_homomer_or_monomer,
) -> FeatureDict:
    """Runs processing on features to augment, pair and merge.

  Args:
    all_chain_features: A MutableMap of dictionaries of features for each chain.

  Returns:
    A dictionary of features.
  """

    process_unmerged_features(all_chain_features)

    np_chains_list = list(all_chain_features.values())
    np_chains_list_prot = [chain for chain in np_chains_list if
                           chain['chain_class'] in ['protein']]
    np_chains_list_dna = [chain for chain in np_chains_list if
                          chain['chain_class'] in ['dna']]
    np_chains_list_rna = [chain for chain in np_chains_list if
                          chain['chain_class'] in ['rna', ]]
    # TODO: ligand?
    np_chains_list_ligand = [chain for chain in np_chains_list if chain['chain_class'] in ['ligand']]
    # np_chains_list_ligand = []

    # pair_msa_sequences_prot = False
    # if np_chains_list_prot:
    #     pair_msa_sequences_prot = not _is_homomer_or_monomer(np_chains_list_prot)
    pair_msa_sequences = not _is_homomer_or_monomer(np_chains_list)
    pair_msa_sequences_prot = not is_homomer_or_monomer
    if pair_msa_sequences_prot and np_chains_list_prot:
        # uniprot : all_seq pairs
        np_chains_list_prot = msa_pairing.create_paired_features(
            chains=np_chains_list_prot
        )
        # deduplicate msa
        # np_chains_list_prot = msa_pairing.deduplicate_unpaired_sequences(np_chains_list_prot)
    else:
        if np_chains_list_prot:
            for prot in np_chains_list_prot:
                prot["num_alignments"] = np.ones([], dtype=np.int32)

    for chain in np_chains_list_prot:
        chain.pop("msa_species_identifiers", None)
        chain.pop("msa_species_identifiers_all_seq", None)

    np_chains_list_prot.extend(np_chains_list_rna)
    np_chains_list_prot.extend(np_chains_list_dna)
    np_chains_list_prot.extend(np_chains_list_ligand)

    np_chains_list = np_chains_list_prot

    np_chains_list = crop_chains(
        np_chains_list,
        msa_crop_size=MSA_CROP_SIZE,
        pair_msa_sequences=pair_msa_sequences_prot,
        max_templates=MAX_TEMPLATES
    )

    np_example = msa_pairing.merge_chain_features(
        np_chains_list=np_chains_list, pair_msa_sequences=pair_msa_sequences,
        max_templates=MAX_TEMPLATES
    )

    # np_example = print_final(np_example)
    return np_example


def crop_chains(
        chains_list: List[Mapping[str, np.ndarray]],
        msa_crop_size: int,
        pair_msa_sequences: bool,
        max_templates: int
) -> List[Mapping[str, np.ndarray]]:
    """Crops the MSAs for a set of chains.

  Args:
    chains_list: A list of chains to be cropped.
    msa_crop_size: The total number of sequences to crop from the MSA.
    pair_msa_sequences: Whether we are operating in sequence-pairing mode.
    max_templates: The maximum templates to use per chain.

  Returns:
    The chains cropped.
  """

    # Apply the cropping.
    cropped_chains = []
    for chain in chains_list:
        if chain['chain_class'] in ['protein']:
            # print(chain['chain_class'])
            cropped_chain = _crop_single_chain(
                chain,
                msa_crop_size=msa_crop_size,
                pair_msa_sequences=pair_msa_sequences,
                max_templates=max_templates)
        else:

            msa_size = chain['msa'].shape[0]
            msa_size_array = np.arange(msa_size)
            target_size = MSA_CROP_SIZE
            if msa_size < target_size:
                sample_msa_id = np.random.choice(msa_size_array, target_size - msa_size, replace=True)
                sample_msa = chain['msa'][sample_msa_id, :]
                chain['msa'] = np.concatenate([chain['msa'], sample_msa], axis=0)
                sample_msa_del = chain['deletion_matrix'][sample_msa_id, :]
                chain['deletion_matrix'] = np.concatenate([chain['deletion_matrix'], sample_msa_del], axis=0)

            else:
                chain['msa'] = chain['msa'][:target_size, :]
                msa_size = chain['msa'].shape[0]
                msa_size_array = np.arange(msa_size)
                chain['deletion_matrix'] = chain['deletion_matrix'][:target_size, :]

            cropped_chain = chain
        cropped_chains.append(cropped_chain)

    return cropped_chains


def _crop_single_chain(chain: Mapping[str, np.ndarray],
                       msa_crop_size: int,
                       pair_msa_sequences: bool,
                       max_templates: int) -> Mapping[str, np.ndarray]:
    """Crops msa sequences to `msa_crop_size`."""
    msa_size = len(chain['msa'])

    if pair_msa_sequences:
        # print(chain.keys())
        msa_size_all_seq = chain['msa_all_seq'].shape[0]
        msa_crop_size_all_seq = np.minimum(msa_size_all_seq, msa_crop_size // 2)



    else:

        msa_crop_size_all_seq = 0

    include_templates = 'template_aatype' in chain and max_templates
    if include_templates:
        num_templates = chain['template_aatype'].shape[0]
        templates_crop_size = np.minimum(num_templates, max_templates)

    target_size = MSA_CROP_SIZE - msa_crop_size_all_seq
    if msa_size < target_size:
        sample_msa_id = np.random.choice(np.arange(msa_size), target_size - msa_size, replace=True)
    for k in chain:
        k_split = k.split('_all_seq')[0]
        if k_split in msa_pairing.TEMPLATE_FEATURES:
            chain[k] = chain[k][:templates_crop_size, :]
        elif k_split in msa_pairing.MSA_FEATURES:
            if '_all_seq' in k:
                chain[k] = chain[k][:msa_crop_size_all_seq, :]
            else:

                if msa_size < target_size:

                    sample_msa = chain[k][sample_msa_id, :]
                    chain[k] = np.concatenate([chain[k], sample_msa], axis=0)


                else:
                    chain[k] = chain[k][:target_size, :]

    chain['num_alignments'] = np.asarray(len(chain['msa']), dtype=np.int32)
    if include_templates:
        chain['num_templates'] = np.asarray(templates_crop_size, dtype=np.int32)
    if pair_msa_sequences:
        chain['num_alignments_all_seq'] = np.asarray(
            len(chain['msa_all_seq']), dtype=np.int32)

    return chain


def print_final(
        np_example: Mapping[str, np.ndarray]
) -> Mapping[str, np.ndarray]:
    return np_example


def _filter_features(
        np_example: Mapping[str, np.ndarray]
) -> Mapping[str, np.ndarray]:
    """Filters features of example to only those requested."""
    return {k: v for (k, v) in np_example.items() if k in REQUIRED_FEATURES}


def process_unmerged_features(
        all_chain_features: MutableMapping[str, Mapping[str, np.ndarray]]
):
    """Postprocessing stage for per-chain features before merging."""
    num_chains = len(all_chain_features)
    for chain_features in all_chain_features.values():
        # if chain_features['chain_class'] in ['protein']:
        chain_features['deletion_mean'] = np.mean(
            chain_features['deletion_matrix'], axis=0
        )

        # Add assembly_num_chains.
        chain_features['assembly_num_chains'] = np.asarray(num_chains)

    # Add entity_mask.
    for chain_features in all_chain_features.values():
        chain_features['entity_mask'] = (
                chain_features['entity_id'] != 0).astype(np.int32)
