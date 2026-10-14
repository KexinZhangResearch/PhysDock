#!/usr/bin/env python3
"""Download the sequence databases needed for MSA search (jackhmmer / hhblits) into ``data/homo_data``.

    python scripts/download/download_homo_data.py                       # all five databases (~465 GB download, ~2.3 TB extracted)
    python scripts/download/download_homo_data.py --only uniref90 uniprot mgnify
    python scripts/download/download_homo_data.py --only bfd --workers 32

Groups and resulting paths (what ``scripts/prepare/homo_search.py`` expects)
---------------------------------------------------------------
uniref90    EBI      uniref90.fasta.gz (32 GB)                  -> data/homo_data/uniref90.fasta
mgnify      GCS      mgy_clusters_2022_05.fa.gz (73 GB)         -> data/homo_data/mgy_clusters_2022_05.fa
uniprot     EBI      uniprot_trembl + uniprot_sprot .fasta.gz   -> data/homo_data/uniprot.fasta  (TrEMBL then Swiss-Prot)
uniclust30  GWDG     uniclust30_2018_08_hhsuite.tar.gz (27 GB)  -> data/homo_data/uniclust30/uniclust30_2018_08/uniclust30_2018_08*
bfd         GCS      bfd_..._sorted_opt.tar.gz (292 GB)         -> data/homo_data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt*

Verification: UniProt/UniRef md5 from the EBI ``RELEASE.metalink`` (fetched at run time because the
"current_release" changes every eight weeks), BFD md5 from Google Cloud Storage, MGnify crc32c from GCS
(needs ``pip install google-crc32c``; otherwise the gzip stream CRC is validated while decompressing),
Uniclust30 has no published checksum so its gzip CRC is validated during extraction.

A ``DATABASES.json`` with the final paths is written to ``data/homo_data`` for use with ``scripts/prepare/homo_search.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from downloader import (FileSpec, add_common_args, chained_tar_extract_action, filter_groups,  # noqa: E402
                        gcs_object_resolver, gunzip_concat_action, log, run_from_args, uniprot_metalink_resolver)

GiB = 1 << 30
GROUPS = ["uniref90", "mgnify", "uniprot", "uniclust30", "bfd"]

EBI = "https://ftp.ebi.ac.uk/pub/databases/uniprot"
UNIREF90_URL = f"{EBI}/uniref/uniref90/uniref90.fasta.gz"
UNIREF90_METALINK = f"{EBI}/uniref/uniref90/RELEASE.metalink"
KB = f"{EBI}/current_release/knowledgebase/complete"
KB_METALINK = f"{KB}/RELEASE.metalink"

GCS_BUCKET = "alphafold-databases"
MGNIFY_OBJ = "v2.3/mgy_clusters_2022_05.fa.gz"
BFD_OBJ = "casp14_versions/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz"
BFD_PREFIX = "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"

UNICLUST30_URL = "http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz"


def db_paths(data_dir: Path) -> dict:
    hd = Path(data_dir) / "homo_data"
    return {
        "uniref90_database_path": str(hd / "uniref90.fasta"),
        "mgnify_database_path": str(hd / "mgy_clusters_2022_05.fa"),
        "uniprot_database_path": str(hd / "uniprot.fasta"),
        "uniclust30_database_path": str(hd / "uniclust30" / "uniclust30_2018_08" / "uniclust30_2018_08"),
        "bfd_database_path": str(hd / "bfd" / BFD_PREFIX),
    }


def build(data_dir: Path, only=None):
    hd = Path(data_dir) / "homo_data"
    tmp = hd / "tmp"
    paths = db_paths(data_dir)
    files, actions = [], []

    # -- UniRef90 --------------------------------------------------------------------------------------
    uniref = FileSpec("uniref90.fasta.gz", UNIREF90_URL, tmp / "uniref90.fasta.gz",
                      resolver=uniprot_metalink_resolver(UNIREF90_METALINK, "uniref90.fasta.gz"),
                      description="UniRef90 (EBI current release)", group="uniref90", extracted_hint=100 * GiB)
    files.append(uniref)
    actions.append(gunzip_concat_action("gunzip_uniref90", [uniref], Path(paths["uniref90_database_path"]),
                                        marker=tmp / ".gunzip_uniref90.done.json",
                                        description=f"gunzip -> {paths['uniref90_database_path']}",
                                        group="uniref90", verifies=[uniref]))

    # -- MGnify clusters (GCS composite object: crc32c only) -----------------------------------------------
    mgnify = FileSpec("mgy_clusters_2022_05.fa.gz", f"https://storage.googleapis.com/{GCS_BUCKET}/{MGNIFY_OBJ}",
                      tmp / "mgy_clusters_2022_05.fa.gz", resolver=gcs_object_resolver(GCS_BUCKET, MGNIFY_OBJ),
                      description="MGnify clusters 2022_05 (AlphaFold mirror)", group="mgnify", extracted_hint=120 * GiB)
    files.append(mgnify)
    actions.append(gunzip_concat_action("gunzip_mgnify", [mgnify], Path(paths["mgnify_database_path"]),
                                        marker=tmp / ".gunzip_mgnify.done.json",
                                        description=f"gunzip -> {paths['mgnify_database_path']}",
                                        group="mgnify", verifies=[mgnify]))

    # -- UniProt = TrEMBL + Swiss-Prot concatenated (same order as the original script) ---------------------
    trembl = FileSpec("uniprot_trembl.fasta.gz", f"{KB}/uniprot_trembl.fasta.gz", tmp / "uniprot_trembl.fasta.gz",
                      resolver=uniprot_metalink_resolver(KB_METALINK, "uniprot_trembl.fasta.gz"),
                      description="UniProtKB/TrEMBL (EBI current release)", group="uniprot", extracted_hint=120 * GiB)
    sprot = FileSpec("uniprot_sprot.fasta.gz", f"{KB}/uniprot_sprot.fasta.gz", tmp / "uniprot_sprot.fasta.gz",
                     resolver=uniprot_metalink_resolver(KB_METALINK, "uniprot_sprot.fasta.gz"),
                     description="UniProtKB/Swiss-Prot (EBI current release)", group="uniprot")
    files += [trembl, sprot]
    actions.append(gunzip_concat_action("gunzip_uniprot", [trembl, sprot], Path(paths["uniprot_database_path"]),
                                        marker=tmp / ".gunzip_uniprot.done.json",
                                        description=f"gunzip trembl + sprot -> {paths['uniprot_database_path']}",
                                        group="uniprot", verifies=[trembl, sprot]))

    # -- Uniclust30 (hhsuite db; no published checksum) ------------------------------------------------------
    uc30 = FileSpec("uniclust30_2018_08_hhsuite.tar.gz", UNICLUST30_URL, tmp / "uniclust30_2018_08_hhsuite.tar.gz",
                    description="Uniclust30 2018_08 hh-suite database", group="uniclust30", extracted_hint=90 * GiB)
    files.append(uc30)
    actions.append(chained_tar_extract_action("extract_uniclust30", [uc30], hd / "uniclust30",
                                              marker=tmp / ".extract_uniclust30.done.json",
                                              description=f"tar xz -> {hd / 'uniclust30'}", group="uniclust30",
                                              verifies=[uc30], outputs=[Path(paths["uniclust30_database_path"] + "_a3m.ffdata")]))

    # -- BFD ---------------------------------------------------------------------------------------------------
    bfd = FileSpec(f"{BFD_PREFIX}.tar.gz", f"https://storage.googleapis.com/{GCS_BUCKET}/{BFD_OBJ}",
                   tmp / f"{BFD_PREFIX}.tar.gz", resolver=gcs_object_resolver(GCS_BUCKET, BFD_OBJ),
                   description="BFD (AlphaFold mirror)", group="bfd", extracted_hint=1800 * GiB)
    files.append(bfd)
    actions.append(chained_tar_extract_action("extract_bfd", [bfd], hd / "bfd", marker=tmp / ".extract_bfd.done.json",
                                              description=f"tar xz -> {hd / 'bfd'}", group="bfd", verifies=[bfd],
                                              outputs=[Path(paths["bfd_database_path"] + "_a3m.ffdata")]))

    return filter_groups(files, only), filter_groups(actions, only)


def write_db_manifest(data_dir: Path) -> Path:
    """Record which databases are in place (paths usable as scripts/prepare/homo_search.py arguments)."""
    hd = Path(data_dir) / "homo_data"
    paths = db_paths(data_dir)
    present = {}
    for key, p in paths.items():
        probe = Path(p + "_a3m.ffdata") if key in ("uniclust30_database_path", "bfd_database_path") else Path(p)
        present[key] = p if probe.exists() else None
    out = hd / "DATABASES.json"
    hd.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(present, f, indent=2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=GROUPS, metavar="GROUP",
                    help=f"restrict to some databases: {' '.join(GROUPS)}")
    add_common_args(ap)
    args = ap.parse_args(argv)
    files, actions = build(args.data_dir, args.only)
    rc = run_from_args(args, files, actions)
    if not args.dry_run:
        manifest = write_db_manifest(args.data_dir)
        ready = {k: v for k, v in json.load(open(manifest)).items() if v}
        log(f"database manifest written to {manifest} ({len(ready)}/5 databases in place)")
        if ready:
            flags = " \\\n    ".join(f"--{k} {v}" for k, v in ready.items())
            print(f"\nscripts/prepare/homo_search.py arguments:\n    {flags}\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
