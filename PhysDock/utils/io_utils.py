"""File helpers (pickle / json / text), md5 names and a process pool with a progress bar."""
import glob
import gzip
import hashlib
import json
import logging
import multiprocessing
import os
import pickle
from functools import partial

import tqdm


def find_files(directory, suffix="pkl", recursive=True):
    """All files *.suffix below directory."""
    return glob.glob(os.path.join(directory, "**", f"*.{suffix}"), recursive=recursive)


def load_txt(filename):
    with open(filename, "r") as f:
        return f.read()


def dump_txt(obj: str, filename):
    with open(filename, "w") as f:
        f.write(obj)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def dump_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)


def load_pkl(filename):
    """Load .pkl / .pickle / .pt, optionally gzipped (.gz); None for other extensions."""
    name, ext = os.path.splitext(filename)
    if ext == ".gz":
        name, ext = os.path.splitext(name)
        opener = gzip.open
    else:
        opener = open
    if ext not in (".pkl", ".pickle", ".pt"):
        return None
    with opener(filename, "rb") as f:
        return pickle.load(f)


def dump_pkl(obj, filename, compress: bool = False):
    """Pickle obj; a name ending in .gz (or compress=True) writes gzip to <name>.gz."""
    if os.path.splitext(filename)[1] == ".gz":
        filename, compress = os.path.splitext(filename)[0], True
    if compress:
        with gzip.GzipFile(filename + ".gz", "wb") as f:
            pickle.dump(obj, f)
    else:
        with open(filename, "wb") as f:
            pickle.dump(obj, f)


def convert_md5_string(string) -> str:
    return hashlib.md5(str(string).encode("utf-8")).hexdigest()


def run_pool_tasks(callback_fn, ls, num_workers=32, desc="Progress", return_dict=False, num_dict=1, **kwargs):
    """callback_fn(item, **kwargs) for every item in a process pool (dict / list kwargs are shared via a Manager).

    Returns the list of results, or with return_dict the merged dict(s) of the results.
    """
    logging.warning("Initialization of shared arguments ......")
    manager = multiprocessing.Manager()
    kwargs = {k: manager.dict(v) if isinstance(v, dict) else manager.list(v) if isinstance(v, list) else v
              for k, v in kwargs.items()}
    logging.warning("Finish initialization of shared arguments among pool processes!")
    out = []
    with multiprocessing.Pool(num_workers) as pool, tqdm.tqdm(total=len(ls), desc=desc) as pbar:
        for d in pool.imap_unordered(partial(callback_fn, **kwargs), ls, chunksize=1):
            out.append(d)
            pbar.update(1)
    if not return_dict:
        return out
    merged = [dict() for _ in range(num_dict)]
    for o in out:
        for i, part in enumerate([o] if num_dict == 1 else o):
            merged[i].update(part)
    return merged[0] if num_dict == 1 else tuple(merged)
