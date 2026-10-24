#!/usr/bin/env python3
"""Download the Chemical Component Dictionary (CCD) of the PDB into data/raw/ccd, resumable.

    python scripts/download/download_ccd.py                         # components.cif.gz and the companion files -> data/raw/ccd, from RCSB
    python scripts/download/download_ccd.py --mirror ebi
    python scripts/download/download_ccd.py --split                 # also every component as its own file, ccd/split/<first letter>/<ID>.cif (~45k files)
    python scripts/download/download_ccd.py --only components       # only the main dictionary

Files (in <out>/ccd):
  components.cif.gz        the whole CCD, every chemical component of the PDB (one mmCIF block per component: atoms, bonds, ideal / model coordinates); ~120 MB
  aa-variants-v1.cif.gz    protonation variants of the standard amino acids
  chem_comp_model.cif.gz   the CCD model coordinates from the CSD (component-models)
  prd-all.cif.gz           BIRD: the reference dictionary of peptide-like antibiotics and inhibitors
--split writes the blocks of components.cif.gz as ccd/split/<first character>/<ID>.cif with gemmi (pip install gemmi), so that a folder never lists all of them.
Every download goes to <name>.part, is checked as gzip and is skipped on a re-run if the size and date of the server match (<name>.meta.json). <out>/ccd/RAW_TAG.json
tags the folder as raw data (raw_tag.py). Standard library only (gemmi for --split).
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_tag import use_dns_cache, write_raw_tag  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MIRRORS = {"rcsb": "https://files.rcsb.org/pub/pdb/data", "ebi": "https://ftp.ebi.ac.uk/pub/databases/pdb/data"}
FILES = {                                                                    # name -> path below the data folder of the mirror
    "components": "monomers/components.cif.gz",
    "aa-variants": "monomers/aa-variants-v1.cif.gz",
    "chem_comp_model": "component-models/complete/chem_comp_model.cif.gz",
    "prd": "bird/prd/prd-all.cif.gz",
}
USER_AGENT = "PhysDock-downloader/1.0 (+https://github.com/KexinZhangResearch/PhysDock)"


def head(url: str, timeout: float):
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT}), timeout=timeout) as r:
        return int(r.headers.get("Content-Length") or 0), r.headers.get("Last-Modified", "")


def download(url: str, path: Path, retries: int, timeout: float) -> str:
    """-> "skipped" (present, the same size and date on the server) or "downloaded"; streamed to <path>.part, checked as gzip"""
    meta = path.with_name(path.name + ".meta.json")
    for attempt in range(retries):
        try:
            size, modified = head(url, timeout)
            if path.exists() and meta.exists() and json.loads(meta.read_text()) == {"size": size, "modified": modified} and path.stat().st_size == size:
                return "skipped"
            part = path.with_name(path.name + ".part")
            t0, n = time.time(), 0
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout) as r, open(part, "wb") as fh:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
                    n += len(block)
                    if size:
                        print(f"\r  {path.name}: {n / 2**20:.0f}/{size / 2**20:.0f} MiB  {n / 2**20 / max(time.time() - t0, 1e-9):.1f} MiB/s", end="", flush=True)
            print()
            if size and n != size:
                raise IOError(f"truncated: {n} of {size} bytes")
            with gzip.open(part, "rb") as fh:                               # the whole gzip stream (CRC) checks out
                while fh.read(1 << 24):
                    pass
            part.replace(path)
            meta.write_text(json.dumps({"size": size, "modified": modified}))
            return "downloaded"
        except Exception as e:  # noqa: BLE001
            if isinstance(e, urllib.error.HTTPError) and e.code == 404 or attempt == retries - 1:
                raise RuntimeError(f"{url}: {type(e).__name__}: {e}") from e
            print(f"\n  retry {attempt + 1} after {type(e).__name__}: {e}")
            time.sleep(min(30.0, 2.0 ** attempt))


def split_components(src: Path, dest: Path) -> int:
    """every block of components.cif.gz as dest/<first character>/<ID>.cif (gemmi) -> the number of components"""
    try:
        import gemmi
    except ImportError as e:
        raise SystemExit("--split needs gemmi: pip install -U gemmi -i https://pypi.tuna.tsinghua.edu.cn/simple") from e
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    doc = gemmi.cif.read(str(src))
    for i, block in enumerate(doc, 1):
        sub = tmp / (block.name[0].upper() if block.name else "_")
        sub.mkdir(parents=True, exist_ok=True)
        one = gemmi.cif.Document()
        one.add_copied_block(block)
        one.write_file(str(sub / f"{block.name}.cif"))
        if i % 5000 == 0:
            print(f"\r  split {i}/{len(doc)}", end="", flush=True)
    print(f"\r  split {len(doc)}/{len(doc)}")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.replace(dest)
    return len(doc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"), help="the raw data folder (the files go to <out>/ccd)")
    ap.add_argument("--mirror", choices=sorted(MIRRORS), default="rcsb")
    ap.add_argument("--only", nargs="+", choices=sorted(FILES), default=None, help="only these files")
    ap.add_argument("--split", action="store_true", help="also one file per component, ccd/split/<first character>/<ID>.cif (gemmi)")
    ap.add_argument("--retries", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args(argv)
    use_dns_cache()

    base = MIRRORS[args.mirror]
    root = Path(args.out).expanduser().resolve() / "ccd"
    root.mkdir(parents=True, exist_ok=True)
    status = {}
    for key in args.only or FILES:
        rel = FILES[key]
        print(f"[ccd] {key}: {base}/{rel}")
        status[key] = download(f"{base}/{rel}", root / Path(rel).name, args.retries, args.timeout)
        print(f"  {status[key]}")
    counts = {f"{k}_file": FILES[k].split("/")[-1] for k in status}
    if args.split:
        if not (root / "components.cif.gz").exists():
            raise SystemExit("--split needs components.cif.gz (download it first)")
        counts["components_split"] = split_components(root / "components.cif.gz", root / "split")
    tag = write_raw_tag(root, source=base, description="PDB Chemical Component Dictionary (components.cif.gz) and companion files", **counts)
    print(f"[done] {root}  ({', '.join(f'{k}: {v}' for k, v in status.items())}); tag {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
