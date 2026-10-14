#!/usr/bin/env python3
"""Download the PhysDock training/validation data (curated from PLINDER) into ``data/v1``.

    python scripts/download/download_dataset.py                         # everything (~145 GB download)
    python scripts/download/download_dataset.py --only train_val weights
    python scripts/download/download_dataset.py --only msa_features uniprot_msa_features

Groups
------
train_val             train_val.zip (11 GB)            -> data/v1/train_val/<system>.pkl.gz   (259k systems)
weights               train_val_samples_weights.json   -> data/v1/train_val_samples_weights.json
msa_features          msa_features_aa..ae (51 GB)      -> data/v1/msa_features/<md5>.pkl.gz
uniprot_msa_features  uniprot_msa_features_aa..ah (83 GB) -> data/v1/uniprot_msa_features/<md5>.pkl.gz

The multi-volume archives are streamed straight from the verified volumes into ``tar``
(``cat aa ab .. | tar xz``) without materialising the concatenated tar.gz.  All source archives
stay in ``data/v1/tmp`` unless ``--delete-archives`` is given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from downloader import (FileSpec, add_common_args, chained_tar_extract_action, filter_groups,  # noqa: E402
                        run_from_args, zenodo_url, zip_extract_action)

GiB = 1 << 30
GROUPS = ["train_val", "weights", "msa_features", "uniprot_msa_features"]

# (record, name, size, md5) as published by the Zenodo API (records are immutable).
TRAIN_VAL = (15178859, "train_val.zip", 11_028_254_975, "c99bc29f105009927708cf04f4ed0bfe")
WEIGHTS = (15220255, "train_val_samples_weights.json", 11_699_912, "5e3db84d9916f6dad6c47b731256b6af")
MSA_PARTS = [
    (15178859, "msa_features_aa", 10_737_418_240, "44a1953edc3cc2bd84ca1a4ba5218f68"),
    (15178859, "msa_features_ab", 10_737_418_240, "3bc71cfe73d7b98539ee7fa5e7e637ce"),
    (15178859, "msa_features_ac", 10_737_418_240, "22cac5e343390d731e7dd2d9b61b811c"),
    (15206943, "msa_features_ad", 10_737_418_240, "1ae1351b6f83843e6c13fad0b122a5a5"),
    (15206943, "msa_features_ae", 7_881_922_601, "edd3c5da124996778b3c13d34e17679d"),
]
UNIPROT_MSA_PARTS = [
    (15206943, "uniprot_msa_features_aa", 10_737_418_240, "6209e15d230cb79bdbb5a1cf3df0abba"),
    (15206943, "uniprot_msa_features_ab", 10_737_418_240, "8dc36b00aa21dd8a10608bf761d4a729"),
    (15209515, "uniprot_msa_features_ac", 10_737_418_240, "97eacc6d92cd8fd5c5af3bbe66bc6622"),
    (15209515, "uniprot_msa_features_ad", 10_737_418_240, "e7d709853ae72fe0f992ed6750391d41"),
    (15209515, "uniprot_msa_features_ae", 10_737_418_240, "30be2b007e60c90f1d4c252d1e54fd2a"),
    (15209515, "uniprot_msa_features_af", 10_737_418_240, "1e4a86beee79b12655c7b3538b579f59"),
    (15210625, "uniprot_msa_features_ag", 10_737_418_240, "d1b45881b643c4e582f9afca9164802b"),
    (15210625, "uniprot_msa_features_ah", 7_710_748_756, "b20666fabb9a86756acaef1773f74562"),
]


def _spec(entry, tmp: Path, group: str, desc: str, extracted_hint: int = 0) -> FileSpec:
    record, name, size, md5 = entry
    return FileSpec(name=name, url=zenodo_url(record, name), dest=tmp / name, size=size, md5=md5,
                    description=desc, group=group, extracted_hint=extracted_hint)


def build(data_dir: Path, only=None):
    v1 = Path(data_dir) / "v1"
    tmp = v1 / "tmp"
    files, actions = [], []

    # -- train_val.zip: entries are 'train_val/<system>.pkl.gz' -> unzip into data/v1/ ---------------------
    tv = _spec(TRAIN_VAL, tmp, "train_val", "curated PLINDER train/val systems", extracted_hint=11 * GiB)
    files.append(tv)
    actions.append(zip_extract_action("extract_train_val", tv, v1, marker=tmp / ".extract_train_val.done.json",
                                      description=f"unzip train_val.zip -> {v1 / 'train_val'} (259,555 systems)",
                                      group="train_val", outputs=[v1 / "train_val"]))

    # -- sample weights (plain json, lives next to train_val) --------------------------------------------
    record, name, size, md5 = WEIGHTS
    files.append(FileSpec(name=name, url=zenodo_url(record, name), dest=v1 / name, size=size, md5=md5, tmp_dir=tmp,
                          description="per-sample training weights", group="weights"))

    # -- split tar.gz volumes: 'cat aa ab .. | tar xz' -> data/v1/msa_features/ -------------------------
    msa = [_spec(e, tmp, "msa_features", "MSA features volume", extracted_hint=11 * GiB) for e in MSA_PARTS]
    files += msa
    actions.append(chained_tar_extract_action("extract_msa_features", msa, v1,
                                              marker=tmp / ".extract_msa_features.done.json",
                                              description=f"cat msa_features_aa..ae | tar xz -> {v1 / 'msa_features'}",
                                              group="msa_features", verifies=msa, outputs=[v1 / "msa_features"]))

    umsa = [_spec(e, tmp, "uniprot_msa_features", "UniProt MSA features volume", extracted_hint=11 * GiB)
            for e in UNIPROT_MSA_PARTS]
    files += umsa
    actions.append(chained_tar_extract_action("extract_uniprot_msa_features", umsa, v1,
                                              marker=tmp / ".extract_uniprot_msa_features.done.json",
                                              description=f"cat uniprot_msa_features_aa..ah | tar xz -> {v1 / 'uniprot_msa_features'}",
                                              group="uniprot_msa_features", verifies=umsa,
                                              outputs=[v1 / "uniprot_msa_features"]))

    return filter_groups(files, only), filter_groups(actions, only)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=GROUPS, metavar="GROUP",
                    help=f"restrict to some groups: {' '.join(GROUPS)}")
    add_common_args(ap)
    args = ap.parse_args(argv)
    files, actions = build(args.data_dir, args.only)
    return run_from_args(args, files, actions)


if __name__ == "__main__":
    sys.exit(main())
