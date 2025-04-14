import sys


sys.path.append("../")
import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

import scipy
import random
import logging
import collections
from functools import partial
from typing import Union, Tuple, Dict
import itertools
from PhysDock.utils.tensor_utils import tensor_tree_map
from PhysDock.utils.io_utils import load_json, load_txt,dump_json,dump_txt,dump_pkl, load_pkl



def _calculate_bin_centers(breaks: np.ndarray):
  """Gets the bin centers from the bin edges.

  Args:
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    bin_centers: [num_bins] the error bin centers.
  """
  step = (breaks[1] - breaks[0])

  # Add half-step to get the center
  bin_centers = breaks + step / 2
  # Add a catch-all bin at the end.
  bin_centers = np.concatenate([bin_centers, [bin_centers[-1] + step]],
                               axis=0)
  return bin_centers

def _calculate_expected_aligned_error(
    alignment_confidence_breaks: np.ndarray,
    aligned_distance_error_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Calculates expected aligned distance errors for every pair of residues.

  Args:
    alignment_confidence_breaks: [num_bins - 1] the error bin edges.
    aligned_distance_error_probs: [num_res, num_res, num_bins] the predicted
      probs for each error bin, for each pair of residues.

  Returns:
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  bin_centers = _calculate_bin_centers(alignment_confidence_breaks)

  # Tuple of expected aligned distance error and max possible error.
  return (np.sum(aligned_distance_error_probs * bin_centers, axis=-1),
          np.asarray(bin_centers[-1]))


def compute_plddt(logits: np.ndarray) -> np.ndarray:
  """Computes per-residue pLDDT from logits.

  Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.

  Returns:
    plddt: [num_res] per-residue pLDDT.
  """
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
  probs = scipy.special.softmax(logits, axis=-1)
  predicted_lddt_ca = np.sum(probs * bin_centers[None, :], axis=-1)
  return predicted_lddt_ca * 100

def predicted_tm_score(
    logits: np.ndarray,
    breaks: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    asym_id: Optional[np.ndarray] = None,
    interface: bool = False) -> np.ndarray:
  """Computes predicted TM alignment or predicted interface TM alignment score.

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    asym_id: [num_res] the asymmetric unit ID - the chain ID. Only needed for
      ipTM calculation, i.e. when interface=True.
    interface: If True, interface predicted TM score is computed.

  Returns:
    ptm_score: The predicted TM alignment or the predicted iTM score.
  """

  # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
  # exp. resolved head's probability.
  if residue_weights is None:
    residue_weights = np.ones(logits.shape[0])

  bin_centers = _calculate_bin_centers(breaks)

  num_res = int(np.sum(residue_weights))
  # Clip num_res to avoid negative/undefined d0.
  clipped_num_res = max(num_res, 19)

  # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
  # "Scoring function for automated assessment of protein structure template
  # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
  d0 = 1.24 * (clipped_num_res - 15) ** (1./3) - 1.8

  # Convert logits to probs.
  probs = scipy.special.softmax(logits, axis=-1)

  # TM-Score term for every bin.
  tm_per_bin = 1. / (1 + np.square(bin_centers) / np.square(d0))
  # E_distances tm(distance).
  predicted_tm_term = np.sum(probs * tm_per_bin, axis=-1)

  pair_mask = np.ones_like(predicted_tm_term, dtype=bool)
  if interface:
    pair_mask *= asym_id[:, None] != asym_id[None, :]

  predicted_tm_term *= pair_mask

  pair_residue_weights = pair_mask * (
      residue_weights[None, :] * residue_weights[:, None])
  normed_residue_mask = pair_residue_weights / (1e-8 + np.sum(
      pair_residue_weights, axis=-1, keepdims=True))
  per_alignment = np.sum(predicted_tm_term * normed_residue_mask, axis=-1)
  return np.asarray(per_alignment[(per_alignment * residue_weights).argmax()])


def compute_predicted_aligned_error(
    logits: np.ndarray,
    breaks: np.ndarray) -> Dict[str, np.ndarray]:
  """Computes aligned confidence metrics from logits.

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    aligned_confidence_probs: [num_res, num_res, num_bins] the predicted
      aligned error probabilities over bins for each residue pair.
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  aligned_confidence_probs = scipy.special.softmax(
      logits,
      axis=-1)
  predicted_aligned_error, max_predicted_aligned_error = (
      _calculate_expected_aligned_error(
          alignment_confidence_breaks=breaks,
          aligned_distance_error_probs=aligned_confidence_probs))
  return {
      'aligned_confidence_probs': aligned_confidence_probs,
      'predicted_aligned_error': predicted_aligned_error,
      'max_predicted_aligned_error': max_predicted_aligned_error,
  }

def get_has_clash(atom_pos, atom_mask, asym_id, is_polymer_chain):
    """
    A structure is marked as having a clash (has_clash) if for any two
    polymer chains A,B in the prediction clashes(A,B) > 100 or
    clashes(A,B) / min(NA,NB) > 0.5 where NA is the number of atoms in
    chain A.
    Args:
        atom_pos: [N_atom, 3]
        atom_mask: [N_atom]
        asym_id: [N_atom]
        is_polymer_chain: [N_atom]
    """
    flag = np.logical_and(atom_mask == 1, is_polymer_chain == 1)
    atom_pos = atom_pos[flag]
    asym_id = asym_id[flag]
    uniq_asym_ids = np.unique(asym_id)
    n = len(uniq_asym_ids)
    if n == 1:
        return 0
    for aid1 in uniq_asym_ids[:-1]:
        for aid2 in uniq_asym_ids[1:]:
            pos1 = atom_pos[asym_id == aid1]
            pos2 = atom_pos[asym_id == aid2]
            dist = np.sqrt(np.sum((pos1[None] - pos2[:, None]) ** 2, -1))
            n_clash = np.sum(dist < 1.1).astype('float32')
            if n_clash > 100 or n_clash / min(len(pos1), len(pos2)) > 0.5:
                return 1
    return 0




def get_metrics(output, batch):
    """
    Args:
        logits_plddt: (B, N_atom, b_plddt)
        logits_pae: (B, N_token, N_token, b_pae)

    Returns:
        atom_plddts: (B, N_atom)
        mean_plddt: (B,)
        pae: (B, N_token, N_token)
        ptm: (B,)
        iptm: (B,)
        has_clash: (B,)
        ranking_confidence: (B,)
    """
    logit_value = output

    # B = logit_value['p_pae'].shape[0]
    breaks_pae = torch.linspace(0.,
                                 0.5 * 64,
                                 64 - 1)
    inputs = {
        's_mask': batch['s_mask'],
        'asym_id': batch['asym_id'],
        'breaks_pae': torch.tile(breaks_pae, [ 1]),
        # 'perm_asym_id': batch['perm_asym_id'],
        'is_polymer_chain': ((batch['is_protein'] +
                              batch['is_dna'] + batch['is_rna']) > 0),
        **logit_value,
        **batch

    }

    ret_list = []
    # for i in range(B):
    cur_input = tensor_tree_map(lambda x: x.numpy(), inputs)
    # cur_input = inputs
    ret = get_all_atom_confidence_metrics(cur_input,0)
    ret_list.append(ret)

    metrics = {}
    for k, v in ret_list[0].items():
        metrics[k] = torch.from_numpy(np.stack([r[k] for r in ret_list]))
    return metrics



def get_all_atom_confidence_metrics(
        prediction_result,b):
    """get_all_atom_confidence_metrics."""
    metrics = {}
    metrics['atom_plddts'] = compute_plddt(
            prediction_result['p_plddt'])
    metrics['mean_plddt'] = metrics['atom_plddts'].mean()
    metrics['pae'] = compute_predicted_aligned_error(
            logits=prediction_result['p_pae'],
            breaks=prediction_result['breaks_pae'])['predicted_aligned_error']
    metrics['ptm'] = predicted_tm_score(
            logits=prediction_result['p_pae'],
            breaks=prediction_result['breaks_pae'],
            residue_weights=prediction_result['s_mask'],
            asym_id=None)
    metrics['iptm'] = predicted_tm_score(
            logits=prediction_result['p_pae'],
            breaks=prediction_result['breaks_pae'],
            residue_weights=prediction_result['s_mask'],
            asym_id=prediction_result['asym_id'],
            interface=True)
    metrics['has_clash'] = get_has_clash(
            prediction_result['x_pred'][b],
            prediction_result['a_mask'],
            prediction_result['asym_id'][prediction_result["atom_id_to_token_id"]],
            ~prediction_result['is_ligand'][prediction_result["atom_id_to_token_id"]])
    metrics['ranking_confidence'] = (
            0.8 * metrics['iptm'] + 0.2 * metrics['ptm']
            - 1.0 * metrics['has_clash'])
    return metrics


# output = load_pkl("../output.pkl.gz")
# feats = load_pkl("../feats.pkl.gz")
# for k,v in output.items():
#     print(k,v.shape)
#     # output[k] = torch.from_numpy(v)
# for k,v in feats.items():
#     print(k,v.shape)
#     # feats[k] = torch.from_numpy(v)
# # dump_pkl(output,"../output.pkl.gz")
# # dump_pkl(feats,"../feats.pkl.gz")
#
# metrics = get_metrics(output,feats)
#
# for k,v in metrics.items():
#     print(k,v)