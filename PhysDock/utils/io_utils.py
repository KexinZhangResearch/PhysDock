import sys
from typing import Union, Optional, List, Dict
from os import PathLike
import os
import pickle
import gzip
import json
import tqdm
import multiprocessing
import hashlib
from functools import partial
import logging
import glob


def find_files(directory, suffix="pkl", recursive=True):
    pattern = os.path.join(directory, '**', f'*.{suffix}')
    return glob.glob(pattern, recursive=recursive)


def load_txt(filename: Union[str, PathLike]):
    with open(filename, "r") as f:
        return f.read()


def load_json(
        filename: Union[str, PathLike]
):
    with open(filename, "r") as f:
        return json.load(f)


def load_pkl(
        filename: Union[str, PathLike]
) -> dict:
    file, ext = os.path.splitext(filename)
    if ext in [".pkl", ".pickle", ".pt"]:
        with open(filename, "rb") as f:
            return pickle.load(f)
    elif ext == ".gz":
        file, ext2 = os.path.splitext(file)
        if ext2 in [".pkl", ".pickle", ".pt"]:
            with gzip.open(filename, "rb") as f:
                return pickle.load(f)


def dump_pkl(
        obj: object,
        filename: Union[str, PathLike],
        compress: bool = False
):
    if os.path.splitext(filename)[1] == ".gz":
        filename = os.path.splitext(filename)[0]
        # DEBUG
        compress = True
    # assert os.path.splitext(filename)[1] == ".pkl"
    if compress:
        with gzip.GzipFile(filename + ".gz", "wb") as f:
            pickle.dump(obj, f)
    else:
        with open(filename, "wb") as f:
            pickle.dump(obj, f)


def dump_txt(obj: str, filename: Union[str, PathLike], ):
    with open(filename, "w") as f:
        f.write(obj)


def dump_json(obj: Union[List, Dict], filename: Union[str, PathLike], ):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)


def chunk_lists(l, num_workers: int):
    mod = int(len(l) // num_workers)
    res = int(len(l) % num_workers)

    def _csum(ll):
        new_l0 = []
        new_l1 = []
        a = 0
        for lll in ll:
            new_l0.append(a)
            a += lll
            new_l1.append(a)
        return new_l0, new_l1

    lists = [l[i:j] for i, j in
             zip(*list(_csum([mod + int(1 if worker_id < res else 0) for worker_id in range(num_workers)])))]
    return lists


def chunk_split_sizes(l, num_workers: int):
    mod = int(len(l) // num_workers)
    res = int(len(l) % num_workers)
    sizes = [num_workers] * mod if res == 0 else [num_workers] * mod + [res]
    return sizes


def doctorate_worker_progress(callback_fn, worker_id):
    def fn(ls, **kwargs):
        ls = tqdm.tqdm(ls, dynamic_ncols=True, leave=True,
                       # position=worker_id,
                       desc=f"Worker: {worker_id:>3}"
                       )
        sys.stdout.flush()
        out = callback_fn(ls, **kwargs)
        sys.stdout.flush()
        ls.close()
        return out

    return fn


def run_multiple_task(callback_fn, l: List, num_workers=16, merge_out_dict=False):
    ls = chunk_lists(l, num_workers)
    manager = multiprocessing.Manager()
    share_dict = manager.dict()

    def doc_fn(fn, id):
        def _fn(*args, **kwargs):
            # print(f"Rank id {id} begin!!!")
            out = fn(*args, **kwargs)
            # print(f"Rank id {id} end!!!")
            share_dict[id] = out
            return out

        return doctorate_worker_progress(_fn, id)

    # pool = multiprocessing.Pool(processes=num_workers)
    # pool.apply_async(doc_fn(doctorate_worker_progress(callback_fn, worker_id), worker_id), (msg,))
    ps = []
    for worker_id, l_ in enumerate(ls):
        ps.append(multiprocessing.Process(
            target=doc_fn(callback_fn, worker_id),
            args=(l_,))
        )
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    if merge_out_dict:
        new_dict = dict()
        for v in share_dict.values():
            new_dict.update(v)
        return new_dict
    return share_dict


def run_pool_tasks(callback_fn, ls, num_workers=32, desc="Progress", return_dict=False, num_dict=1, **kwargs):
    logging.warning("Initialization of shared arguments ......")
    manager = multiprocessing.Manager()
    out = []
    kwargs = {
        k: manager.dict(v) if isinstance(v, dict) else (manager.list(v) if isinstance(v, list) else v)
        for k, v in kwargs.items()
    }
    logging.warning("Finish initialization of shared arguments among pool processes!")

    with multiprocessing.Pool(num_workers) as pool:
        with tqdm.tqdm(total=len(ls), desc=desc) as pbar:
            fn = partial(callback_fn, **kwargs)
            for d in pool.imap_unordered(fn, ls, chunksize=1):
                out.append(d)
                # share_list.append(d)
                pbar.update(1)
    pool.close()
    pool.join()
    if return_dict and num_dict == 1:
        out_dict = dict()
        for o in out:
            out_dict.update(o)
        return out_dict
    elif return_dict and num_dict > 1:
        out_dicts = [dict() for i in range(num_dict)]
        for o in out:
            for id, i in enumerate(o):
                out_dicts[id].update(i)
        return tuple(out_dicts)
    return out


def run_pool_gpu_tasks(callback_fn, ls, num_workers=8, desc="Progress", return_dict=False, num_dict=1, **kwargs):
    logging.warning("Initialization of shared arguments ......")
    manager = multiprocessing.Manager()
    out = []
    kwargs = {
        k: manager.dict(v) if isinstance(v, dict) else (manager.list(v) if isinstance(v, list) else v)
        for k, v in kwargs.items()
    }
    logging.warning("Finish initialization of shared arguments among pool processes!")

    with multiprocessing.Pool(num_workers) as pool:
        with tqdm.tqdm(total=len(ls), desc=desc) as pbar:
            fn = partial(callback_fn, **kwargs)
            for gpu_id, d in enumerate(pool.imap_unordered(fn, ls, chunksize=1)):
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                out.append(d)
                # share_list.append(d)
                pbar.update(1)
    pool.close()
    pool.join()
    if return_dict and num_dict == 1:
        out_dict = dict()
        for o in out:
            out_dict.update(o)
        return out_dict
    elif return_dict and num_dict > 1:
        out_dicts = [dict() for i in range(num_dict)]
        for o in out:
            for id, i in enumerate(o):
                out_dicts[id].update(i)
        return tuple(out_dicts)
    return out


def convert_md5_string(string):
    md5_string = hashlib.md5(str(string).encode('utf-8')).hexdigest()
    return md5_string
