#!/usr/bin/env python3
"""Download the preprocessed PhysDock benchmarks (PoseBusters, DeepDockingDare, PhiBench) into ``data/benchmarks``.

    python scripts/download_benchmarks.py

Result layout::

    data/benchmarks/
        PoseBusters/            system .pkl.gz files
        DeepDockingDare/
        PhiBench/
        features/msa_features/           MSA features (md5-named) for all three benchmarks
        features/uniprot_msa_features/
        tmp/physdock_benchmarks.zip      the verified archive (kept unless --delete-archives)

Example::

    python redocking.py -i data/benchmarks/PoseBusters -f data/benchmarks/features \\
        --crop_size 256 --atom_crop_size 2048 --enable_physics_correction --use_pocket --use_key_res --enable_ranking
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from downloader import FileSpec, add_common_args, run_from_args, zenodo_url, zip_extract_action  # noqa: E402

ZENODO_RECORD = 15178859
ARCHIVE = "physdock_benchmarks.zip"
SIZE, MD5 = 3_998_842_107, "ad71e631eb439367667a89de8c41892e"
EXTRACTED_BYTES = 4_010_000_000  # 4300 entries, mostly already-compressed .pkl.gz


def build(data_dir: Path):
    dest_dir = Path(data_dir) / "benchmarks"
    tmp = dest_dir / "tmp"
    archive = FileSpec(name=ARCHIVE, url=zenodo_url(ZENODO_RECORD, ARCHIVE), dest=tmp / ARCHIVE, size=SIZE, md5=MD5,
                       description="PoseBusters + DeepDockingDare + PhiBench systems and MSA features",
                       group="benchmarks", extracted_hint=EXTRACTED_BYTES)
    # the zip has a single top-level folder 'benchmarks/' -> strip it so we get data/benchmarks/PoseBusters/...
    extract = zip_extract_action("extract_benchmarks", archive, dest_dir, marker=tmp / ".extract_benchmarks.done.json",
                                 strip=1, description=f"unzip {ARCHIVE} -> {dest_dir}", group="benchmarks",
                                 outputs=[dest_dir / d for d in ("PoseBusters", "DeepDockingDare", "PhiBench", "features")])
    return [archive], [extract]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args(argv)
    files, actions = build(args.data_dir)
    return run_from_args(args, files, actions)


if __name__ == "__main__":
    sys.exit(main())
