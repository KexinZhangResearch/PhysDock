# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

from functools import partial
import logging
from typing import Tuple, List, Callable, Any, Dict, Sequence, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def add(m1, m2, inplace):
    # The first operation in a checkpoint can't be in-place, but it's
    # nice to have in-place addition during inference. Thus...
    if (not inplace):
        m1 = m1 + m2
    else:
        m1 += m2

    return m1


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def reshape_final_dim(t: torch.Tensor, shape: Tuple):
    return t.reshape(t.shape[:-1] + shape)


def masked_mean(mask, value, dim, eps=1e-9):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def pts_to_distogram(pts, min_bin=2.3125, max_bin=21.6875, no_bins=64):
    boundaries = torch.linspace(
        min_bin, max_bin, no_bins - 1, device=pts.device
    )
    dists = torch.sqrt(
        torch.sum((pts.unsqueeze(-2) - pts.unsqueeze(-3)) ** 2, dim=-1)
    )
    return torch.bucketize(dists, boundaries)


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def one_hot(x, v_bins):
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()


def batched_gather(data: torch.Tensor, indices: torch.Tensor, dims: List[int]):
    """
    Batch gather operation.
    Args:
        data: Tensor of shape [*, ...] to gather from.
        indices: Tensor of shape [*, num_gathered], 1-D indices to gather for each batch at given dimension.
        dims: List of dimensions to gather along, can include negative values.
    Returns:
        Tensor where elements are gathered at specified dimensions using provided indices.
    """

    # Convert negative dimensions to positive
    total_dims = data.dim()
    dims = [(dim + total_dims if dim < 0 else dim) for dim in dims]
    print(len(dims), len(indices.shape), len(data.shape))
    # Check dimensions length
    if len(dims) != len(indices.shape) - len(data.shape) + 1:
        raise ValueError("Length of dims must match the dimensions of indices minus batch dimensions.")

    # Validate indices and dimensions alignment
    idx_dims = list(range(len(data.shape) - len(indices.shape) + 1, len(data.shape)))
    if dims != idx_dims:
        raise ValueError(f"indices should be aligned with dimensions {idx_dims}, but got {dims}")

    # Prepare indices for each dimension specified in dims
    expanded_indices = []
    for dim in dims:
        # Broadcast indices to match the size of the data tensor along the current dimension
        expand_shape = list(data.shape)
        expand_shape[dim] = -1
        broad_indices = indices.expand(*expand_shape)
        expanded_indices.append(broad_indices)

    # Perform the advanced indexing
    return data[tuple(expanded_indices)]


# With tree_map, a poor man's JAX tree_map
def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        print(type(tree))
        # raise ValueError("Not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def _fetch_dims(tree):
    shapes = []
    tree_type = type(tree)
    if tree_type is dict:
        for v in tree.values():
            shapes.extend(_fetch_dims(v))
    elif tree_type is list or tree_type is tuple:
        for t in tree:
            shapes.extend(_fetch_dims(t))
    elif tree_type is torch.Tensor:
        shapes.append(tree.shape)
    else:
        raise ValueError("Not supported")

    return shapes


def chunk_layer(
        layer: Callable,
        inputs: Dict[str, Any],
        chunk_size: int,
        no_batch_dims: int,
        low_mem: bool = False,
) -> Any:
    """
    Implements the "chunking" procedure described in section 1.11.8.

    Layer outputs and input_json are assumed to be simple "pytrees,"
    consisting only of (arbitrarily nested) lists, tuples, and dicts with
    torch.Tensor leaves.

    Args:
        layer:
            The layer to be applied chunk-wise
        inputs:
            A (non-nested) dictionary of keyworded input_json. All leaves must
            be tensors and must share the same batch dimensions.
        chunk_size:
            The number of sub-batches per chunk. If multiple batch
            dimensions are specified, a "sub-batch" is defined as a single
            indexing of all batch dimensions simultaneously (s.t. the
            number of sub-batches is the product of the batch dimensions).
        no_batch_dims:
            How many of the initial dimensions of each input tensor can
            be considered batch dimensions.
        low_mem:
            Avoids flattening potentially large input tensors. Unnecessary
            in most cases, and is ever so slightly slower than the default
            setting.
    Returns:
        The reassembled output of the layer on the input_json.
    """
    if not (len(inputs) > 0):
        raise ValueError("Must provide at least one input")

    initial_dims = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
    orig_batch_dims = tuple([max(s) for s in zip(*initial_dims)])

    def _prep_inputs(t):
        # TODO: make this more memory efficient. This sucks
        if (not low_mem):
            if not sum(t.shape[:no_batch_dims]) == no_batch_dims:
                t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
            t = t.reshape(-1, *t.shape[no_batch_dims:])
        else:
            t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
        return t

    prepped_inputs = tensor_tree_map(_prep_inputs, inputs)

    flat_batch_dim = 1
    for d in orig_batch_dims:
        flat_batch_dim *= d

    no_chunks = flat_batch_dim // chunk_size + (
            flat_batch_dim % chunk_size != 0
    )

    i = 0
    out = None
    for _ in range(no_chunks):
        # Chunk the input
        if (not low_mem):
            select_chunk = (
                lambda t: t[i: i + chunk_size] if t.shape[0] != 1 else t
            )
        else:
            select_chunk = (
                partial(
                    _chunk_slice,
                    flat_start=i,
                    flat_end=min(flat_batch_dim, i + chunk_size),
                    no_batch_dims=len(orig_batch_dims)
                )
            )

        chunks = tensor_tree_map(select_chunk, prepped_inputs)

        # Run the layer on the chunk
        output_chunk = layer(**chunks)

        # Allocate space for the output
        if out is None:
            allocate = lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:])
            out = tensor_tree_map(allocate, output_chunk)

        # Put the chunk in its pre-allocated space
        out_type = type(output_chunk)
        if out_type is dict:
            def assign(d1, d2):
                for k, v in d1.items():
                    if type(v) is dict:
                        assign(v, d2[k])
                    else:
                        v[i: i + chunk_size] = d2[k]

            assign(out, output_chunk)
        elif out_type is tuple:
            for x1, x2 in zip(out, output_chunk):
                x1[i: i + chunk_size] = x2
        elif out_type is torch.Tensor:
            out[i: i + chunk_size] = output_chunk
        else:
            raise ValueError("Not supported")

        i += chunk_size

    reshape = lambda t: t.view(orig_batch_dims + t.shape[1:])
    out = tensor_tree_map(reshape, out)

    return out


@torch.jit.ignore
def _flat_idx_to_idx(
        flat_idx: int,
        dims: Tuple[int],
) -> Tuple[int]:
    idx = []
    for d in reversed(dims):
        idx.append(flat_idx % d)
        flat_idx = flat_idx // d

    return tuple(reversed(idx))


def _get_minimal_slice_set(
        start: Sequence[int],
        end: Sequence[int],
        dims: int,
        start_edges: Optional[Sequence[bool]] = None,
        end_edges: Optional[Sequence[bool]] = None,
) -> Sequence[Tuple[int]]:
    """
        Produces an ordered sequence of tensor slices that, when used in
        sequence on a tensor with shape dims, yields tensors that contain every
        leaf in the contiguous range [start, end]. Care is taken to yield a
        short sequence of slices, and perhaps even the shortest possible (I'm
        pretty sure it's the latter).

        end is INCLUSIVE.
    """

    # start_edges and end_edges both indicate whether, starting from any given
    # dimension, the start/end index is at the top/bottom edge of the
    # corresponding tensor, modeled as a tree
    def reduce_edge_list(l):
        tally = 1
        for i in range(len(l)):
            reversed_idx = -1 * (i + 1)
            l[reversed_idx] *= tally
            tally = l[reversed_idx]

    if (start_edges is None):
        start_edges = [s == 0 for s in start]
        reduce_edge_list(start_edges)
    if (end_edges is None):
        end_edges = [e == (d - 1) for e, d in zip(end, dims)]
        reduce_edge_list(end_edges)

        # Base cases. Either start/end are empty and we're done, or the final,
    # one-dimensional tensor can be simply sliced
    if (len(start) == 0):
        return [tuple()]
    elif (len(start) == 1):
        return [(slice(start[0], end[0] + 1),)]

    slices = []
    path = []

    # Dimensions common to start and end can be selected directly
    for s, e in zip(start, end):
        if (s == e):
            path.append(slice(s, s + 1))
        else:
            break

    path = tuple(path)
    divergence_idx = len(path)

    # start == end, and we're done
    if (divergence_idx == len(dims)):
        return [tuple(path)]

    def upper():
        sdi = start[divergence_idx]
        return [
            path + (slice(sdi, sdi + 1),) + s for s in
            _get_minimal_slice_set(
                start[divergence_idx + 1:],
                [d - 1 for d in dims[divergence_idx + 1:]],
                dims[divergence_idx + 1:],
                start_edges=start_edges[divergence_idx + 1:],
                end_edges=[1 for _ in end_edges[divergence_idx + 1:]]
            )
        ]

    def lower():
        edi = end[divergence_idx]
        return [
            path + (slice(edi, edi + 1),) + s for s in
            _get_minimal_slice_set(
                [0 for _ in start[divergence_idx + 1:]],
                end[divergence_idx + 1:],
                dims[divergence_idx + 1:],
                start_edges=[1 for _ in start_edges[divergence_idx + 1:]],
                end_edges=end_edges[divergence_idx + 1:],
            )
        ]

    # If both start and end are at the edges of the subtree rooted at
    # divergence_idx, we can just select the whole subtree at once
    if (start_edges[divergence_idx] and end_edges[divergence_idx]):
        slices.append(
            path + (slice(start[divergence_idx], end[divergence_idx] + 1),)
        )
    # If just start is at the edge, we can grab almost all of the subtree,
    # treating only the ragged bottom edge as an edge case
    elif (start_edges[divergence_idx]):
        slices.append(
            path + (slice(start[divergence_idx], end[divergence_idx]),)
        )
        slices.extend(lower())
    # Analogous to the previous case, but the top is ragged this time
    elif (end_edges[divergence_idx]):
        slices.extend(upper())
        slices.append(
            path + (slice(start[divergence_idx] + 1, end[divergence_idx] + 1),)
        )
    # If both sides of the range are ragged, we need to handle both sides
    # separately. If there's contiguous meat in between them, we can index it
    # in one big chunk
    else:
        slices.extend(upper())
        middle_ground = end[divergence_idx] - start[divergence_idx]
        if (middle_ground > 1):
            slices.append(
                path + (slice(start[divergence_idx] + 1, end[divergence_idx]),)
            )
        slices.extend(lower())

    return [tuple(s) for s in slices]


@torch.jit.ignore
def _chunk_slice(
        t: torch.Tensor,
        flat_start: int,
        flat_end: int,
        no_batch_dims: int,
) -> torch.Tensor:
    """
        Equivalent to

            t.reshape((-1,) + t.shape[no_batch_dims:])[flat_start:flat_end]

        but without the need for the initial reshape call, which can be
        memory-intensive in certain situations. The only reshape operations
        in this function are performed on sub-tensors that scale with
        (flat_end - flat_start), the chunk size.
    """

    batch_dims = t.shape[:no_batch_dims]
    start_idx = list(_flat_idx_to_idx(flat_start, batch_dims))
    # _get_minimal_slice_set is inclusive
    end_idx = list(_flat_idx_to_idx(flat_end - 1, batch_dims))

    # Get an ordered list of slices to perform
    slices = _get_minimal_slice_set(
        start_idx,
        end_idx,
        batch_dims,
    )

    sliced_tensors = [t[s] for s in slices]

    return torch.cat(
        [s.view((-1,) + t.shape[no_batch_dims:]) for s in sliced_tensors]
    )


def map_fn(fun, x):
    ensembles = [fun(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = torch.stack(
            [dict_i[feat] for dict_i in ensembles], dim=-1
        )
    return ensembled_dict


def softmax_cross_entropy(logits, labels, num_classes):
    loss = -1 * torch.sum(
        F.one_hot(labels, num_classes=num_classes) * F.log_softmax(logits, dim=-1),
        dim=-1
    )
    return loss


def sigmoid_cross_entropy(logits, labels):
    logits_dtype = logits.dtype
    logits = logits.double()
    labels = labels.double()
    log_p = torch.nn.functional.logsigmoid(logits)
    # log_p = torch.log(torch.sigmoid(logits))
    log_not_p = torch.nn.functional.logsigmoid(-1 * logits)
    # log_not_p = torch.log(torch.sigmoid(-logits))
    loss = (-1. * labels) * log_p - (1. - labels) * log_not_p
    loss = loss.to(dtype=logits_dtype)
    return loss


def _uniform_sphere_point_np(size):
    phi = np.random.random(*size) * 2 * np.pi
    theta = np.arccos(np.random.random(*size) * 2 - 1)
    point = np.stack([
        np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    ], axis=-1)
    return point


def uniform_random_rotation_np(size):
    uniform_e0 = _uniform_sphere_point_np(size)
    uniform_e1 = _uniform_sphere_point_np(size)

    e1 = uniform_e1 - uniform_e0 * np.sum(uniform_e1 * uniform_e0, axis=-1, keepdims=True)

    e1 = e1 / np.linalg.norm(e1, axis=-1, keepdims=True)
    e0 = uniform_e0
    e2 = np.cross(e0, e1)
    R = np.stack([e0, e1, e2], axis=-2)
    return R


def centre_random_augmentation_np(x: np.ndarray, s: int = 1):
    """
    Implements Algorithm 19.
    """
    # """
    # Algorithm 19: CentreRandomAugmentation
    # Args:
    #     x: torch.Tensor, [..., num_samples, num_atoms, 3]
    #     s: int, default=1 Angstrom
    # """
    x_aug = x - x.mean(axis=-2, keepdims=True)

    R = uniform_random_rotation_np(x_aug.shape[:-2])
    # print("R",R.shape,R)
    x_aug = np.einsum("...ij,...kj->...ki", R, x_aug)
    t = s * np.random.normal(size=x_aug.shape[:-2] + (3,))[..., None, :]
    x_aug = x_aug + t
    return x_aug


def centre_random_augmentation_np_apply(ref_pos, atom_id_to_token_id):
    # num_tokens = len(set(atom_id_to_token_id))
    num_tokens = max(atom_id_to_token_id) + 1
    R = uniform_random_rotation_np([num_tokens])[atom_id_to_token_id]
    T = np.random.normal(size=[num_tokens, 3])[atom_id_to_token_id]
    ref_pos_aug = np.einsum("xij,xi->xj", R, ref_pos) + T
    return ref_pos_aug


def centre_random_augmentation_np_batch(batch_ref_pos):
    # num_tokens = len(set(atom_id_to_token_id))
    batch_ref_pos = batch_ref_pos - np.mean(batch_ref_pos, axis=1, keepdims=True)
    num_tokens = len(batch_ref_pos)
    R = uniform_random_rotation_np([num_tokens])
    T = np.random.normal(size=[num_tokens, 3])
    batch_ref_pos_aug = np.einsum("bij,bxi->bxj", R, batch_ref_pos) + T[:, None]
    return batch_ref_pos_aug


def _uniform_sphere_point(size, device, seed=None):
    if seed is not None:
        G = torch.Generator(device=device)
        G.manual_seed(seed)
        phi = torch.rand([*size], device=device, dtype=torch.float32,
                         generator=G) * 2 * torch.pi  # random.uniform(0,2*math.pi)
        theta = torch.acos(
            torch.rand([*size], device=device, dtype=torch.float32,
                       generator=G) * 2 - 1, )  # np.arccos(random.uniform(-1,1))
    else:
        phi = torch.rand([*size], device=device, dtype=torch.float32) * 2 * torch.pi  # random.uniform(0,2*math.pi)
        theta = torch.acos(
            torch.rand([*size], device=device, dtype=torch.float32) * 2 - 1, )  # np.arccos(random.uniform(-1,1))

    point = torch.stack([
        torch.cos(phi) * torch.sin(theta), torch.sin(phi) * torch.sin(theta), torch.cos(theta)
    ], dim=-1)
    return point


def uniform_random_rotation(size, device, seed=None):
    uniform_e0 = _uniform_sphere_point(size, device, seed=seed)
    uniform_e1 = _uniform_sphere_point(size, device, seed=None if seed is None else seed + 1)
    e1 = uniform_e1 - uniform_e0 * (uniform_e1 * uniform_e0).sum(dim=-1, keepdim=True)
    e1 = e1 / torch.norm(e1, dim=-1, keepdim=True)
    e0 = uniform_e0
    e2 = torch.cross(e0, e1, dim=-1)
    R = torch.stack([e0, e1, e2], dim=-2)
    return R


def centre_random_augmentation(x: torch.Tensor, x_exists, x_centre=None, s: float = 1.0, seed=None):
    mean = torch.sum(x * x_exists[None, :, None], dim=-2, keepdim=True) / torch.sum(x_exists)
    x_aug = x - mean
    R = uniform_random_rotation(x_aug.shape[:-2], device=x.device)
    # print("R",R.shape,R)
    x_aug = torch.einsum("...ij,...kj->...ki", R, x_aug)
    t = s * torch.normal(mean=0, std=1, size=x_aug.shape[:-2] + (3,),
                         device=x.device, dtype=x_aug.dtype, )[..., None, :]
    x_aug = x_aug + t

    return x_aug


def pocket_centre_random_augmentation(x_ligand, x_pocket, s: float = 1.0):
    mean = x_pocket.mean(dim=-2, keepdim=True)
    x_ligand = x_ligand - mean
    x_pocket = x_pocket - mean
    R = uniform_random_rotation(x_ligand.shape[:-2], device=x_ligand.device)
    t = s * torch.normal(mean=0, std=1, size=x_ligand.shape[:-2] + (3,),
                         device=x_ligand.device, dtype=x_ligand.dtype, )[..., None, :]

    x_ligand = torch.einsum("...ij,...kj->...ki", R, x_ligand) + t
    x_pocket = torch.einsum("...ij,...kj->...ki", R, x_pocket) + t
    return x_ligand, x_pocket


def batch_centre_random_augmentation(x: torch.Tensor, s: torch.Tensor, x_centre=None, seed=None):
    """
    Implements Algorithm 19.
    """
    # """
    # Algorithm 19: CentreRandomAugmentation
    # Args:
    #     x: torch.Tensor, [..., num_samples, num_atoms, 3]
    #     s: int, default=1 Angstrom
    # """
    dtype = x.dtype
    x = x.to(torch.float32)
    if x_centre is None:
        x_aug = x - x.mean(dim=-2, keepdim=True)
    else:
        x_aug = x - x_centre[None, None]
    if seed is not None:
        R = uniform_random_rotation(x_aug.shape[:-2], device=x.device, seed=seed)
        # print("R",R.shape,R)
        x_aug = torch.einsum("...ij,...kj->...ki", R, x_aug)
        G = torch.Generator(device=x.device)
        if seed is not None:
            G.manual_seed(seed)
        t = s * torch.normal(mean=0, std=1, size=x_aug.shape[:-2] + (3,), device=x.device, dtype=torch.float32,
                             generator=G)[..., None,
                :]
        x_aug = x_aug + t
    else:
        R = uniform_random_rotation(x_aug.shape[:-2], device=x.device)
        # print("R",R.shape,R)
        x_aug = torch.einsum("...ij,...kj->...ki", R, x_aug)
        t = s[:, None, None] * torch.normal(mean=0, std=1, size=x_aug.shape[:-2] + (3,), device=x.device,
                                            dtype=torch.float32, )[...,
                               None,
                               :]
        x_aug = x_aug + t

    return x_aug.type(dtype)


def gen_attn_mask(mask, neg_inf):
    assert neg_inf < -1e4
    attn_mask = torch.zeros_like(mask)
    attn_mask[mask == 0] = neg_inf
    return attn_mask


def GreedyMatching(feats, x_pred):
    with torch.no_grad():
        x_gt_decoy = feats["x_gt_decoy"]
        x_gt_decoy_exists = feats["x_gt_decoy_exists"]
        token_id_to_centre_atom_id = feats["token_id_to_centre_atom_id"]
        z_mask = feats["z_mask"]
        # x_pred: [?, num_atom,3]
        # x_gt_decoy: [128, num_atom, 3]
        # x_gt_decoy_exists: [128]

        x_pred_token = x_pred[0:1, token_id_to_centre_atom_id]
        x_pred_dist = torch.norm(x_pred_token[:, None] - x_pred_token[:, :, None], dim=-1)
        x_gt_decoy_token = x_gt_decoy[:, token_id_to_centre_atom_id]
        x_gt_decoy_token_dist = torch.norm(x_gt_decoy_token[:, None] - x_gt_decoy_token[:, :, None], dim=-1)

        dist_error = torch.square(x_gt_decoy_token_dist - x_pred_dist)
        dist_error = masked_mean(z_mask[None], dist_error, dim=[-1, -2])

        dist_error = dist_error.where(x_gt_decoy_exists > 0, dist_error.max())
        feats["x_gt"] = torch.index_select(feats["x_gt_decoy"], dim=0, index=dist_error.argmin())[0]

    return feats


def one_hot_with_nearest_bin(
        x: torch.Tensor,
        v_bins: torch.Tensor
) -> torch.Tensor:
    """
    Implements AlphaFold3 Algorithm 4.
    Args:
        x (torch.Tensor): input tensor, [*, 1]
        v_bins (torch.Tensor): bins tensor, [no_bins]
    """
    p = torch.zeros(x.shape[:-1] + (len(v_bins),), device=x.device, dtype=x.dtype)
    b = torch.argmin(torch.abs(x[..., None] - v_bins), dim=-1)
    p.scatter_(-1, b, 1)
    return p


def dgram_from_positions(
        pos: torch.Tensor,
        min_bin: float = 3.25,
        max_bin: float = 50.75,
        no_bins: float = 39,
        inf: float = 1e8,
):
    dgram = torch.sum(
        (pos[..., None, :] - pos[..., None, :, :]) ** 2, dim=-1, keepdim=True
    )
    lower = torch.linspace(min_bin, max_bin, no_bins, device=pos.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)

    return dgram


def smooth_dgram_from_positions(
        pos: torch.Tensor,
        min_bin: float = 0.5,
        max_bin: float = 4.5,
        no_bins: float = 9,
        inf: float = 1e8,
):
    dgram = torch.sum(
        (pos[..., None, :] - pos[..., None, :, :]) ** 2, dim=-1, keepdim=True
    )
    dgram = torch.log(dgram)

    lower = torch.linspace(min_bin, max_bin, no_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)
    return dgram


def weighted_rigid_align(
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        weights: torch.Tensor,
) -> torch.Tensor:
    """
    Implements Algorithm 28. Weighted Rigid Align
    Args:
        x_pred: predicted atom positions, torch.Tensor, [..., num_samples, num_atoms, 3]
        x_gt: ground truth atom positions, torch.Tensor, [..., num_atoms, 3]
        weights: weights for each atom, torch.Tensor, [..., num_atoms]
    """
    # Mean-centre positions
    # [*, num_samples, 3]

    with torch.amp.autocast("cuda", enabled=False):
        x_pred = x_pred.float()
        x_gt = x_gt.float()
        weights = weights.float()

        if len(x_gt.shape) == 2:
            x_gt = x_gt[..., None, :, :]

        mu_pred = torch.sum(x_pred * weights[..., None, :, None], dim=-2) / torch.sum(weights[..., None, :], dim=-1,
                                                                                      keepdim=True)
        mu_gt = torch.sum(x_gt * weights[..., None, :, None], dim=-2) / torch.sum(weights[..., None, :],
                                                                                  dim=-1, keepdim=True)

        # [*, num_samples, num_atoms, 3]
        x_pred_hat = x_pred - mu_pred[..., None, :]
        x_gt_hat = x_gt - mu_gt[..., None, :]

        # Find optimal rotation from singular value decomposition
        outer_product = torch.einsum("...ij,...ik->...ijk", x_gt_hat, x_pred_hat)
        H = torch.sum(outer_product * weights[..., None, :, None, None], dim=-3)
        U, _, Vh = torch.linalg.svd(H)

        F = torch.eye(3, device=H.device)
        F[-1, -1] = -1

        R = torch.matmul(U, Vh)
        R_reflection = torch.matmul(U, F).matmul(Vh)

        reflection_mask = torch.det(R) < 0
        R = torch.where(reflection_mask[..., None, None], R_reflection, R)
        R = torch.transpose(R, -1, -2)

        # Apply rotation
        # [*, num_samples, num_atoms, 3]
        x_pred_align_to_gt = torch.einsum("...ij,...kj->...ki", R, x_gt_hat) + mu_pred[..., None, :]

        # Return aligned positions with stopping gradients
        # TODO: Check if detach is necessary
        # [*, num_samples, num_atoms, 3]
    return x_pred_align_to_gt
