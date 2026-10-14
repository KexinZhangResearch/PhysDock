#!/usr/bin/env python3
"""Download PhysDock model parameters (EMA weights) from Zenodo into ``data/params``.

    python scripts/download/download_params.py                 # params.pt (latest) + params_2418.pt (paper)
    python scripts/download/download_params.py --which latest  # only params.pt
    python scripts/download/download_params.py --link          # also symlink <repo>/params/params.pt -> data/params/params.pt

Downloads are parallel (``--workers``), resumable, and md5-verified against the checksums
published by Zenodo.  Rerunning the command only fetches what is missing or corrupt.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from downloader import FileSpec, add_common_args, log, repo_root, run_from_args, warn, zenodo_url  # noqa: E402

ZENODO_RECORD = 15178859
# name -> (size, md5) as published by https://zenodo.org/api/records/15178859 (Zenodo records are immutable)
PARAMS = {
    "params.pt": (613_186_797, "c3aec80e39003311712f69f08be2e54e", "latest", "latest EMA parameters"),
    "params_2418.pt": (613_186_797, "d53cae4e31fe956ba935ffe23b73a043", "paper", "EMA parameters used in the paper"),
}


def build(data_dir: Path, which: str = "all"):
    dest_dir = Path(data_dir) / "params"
    files = []
    for name, (size, md5, group, desc) in PARAMS.items():
        if which != "all" and group != which:
            continue
        files.append(FileSpec(name=name, url=zenodo_url(ZENODO_RECORD, name), dest=dest_dir / name,
                              size=size, md5=md5, description=desc, group=group))
    return files, []


def link_repo_params(data_dir: Path) -> None:
    """Make ``<repo>/params/params.pt`` (what redocking.py / screening.py load) point at the download."""
    src = Path(data_dir) / "params" / "params.pt"
    dst = repo_root() / "params" / "params.pt"
    if not src.exists():
        return
    if dst.is_symlink() or dst.exists():
        if dst.resolve() == src.resolve():
            return
        warn(f"{dst} already exists and is not the downloaded file; leaving it alone")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(src, dst.parent), dst)
    log(f"linked {dst} -> {src}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=["all", "latest", "paper"], default="all",
                    help="which parameter set to download (default: all)")
    ap.add_argument("--link", action="store_true",
                    help="symlink <repo>/params/params.pt to the downloaded file so the inference scripts find it")
    add_common_args(ap)
    args = ap.parse_args(argv)

    files, actions = build(args.data_dir, args.which)
    rc = run_from_args(args, files, actions)
    if rc == 0 and not args.dry_run:
        if args.link:
            link_repo_params(args.data_dir)
        elif not (repo_root() / "params" / "params.pt").exists():
            log(f"note: redocking.py/screening.py load <repo>/params/params.pt; "
                f"run with --link or copy {args.data_dir / 'params' / 'params.pt'} there")
    return rc


if __name__ == "__main__":
    sys.exit(main())
