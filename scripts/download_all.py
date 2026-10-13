#!/usr/bin/env python3
"""Download everything PhysDock needs in one go (shared worker pool, one progress line).

    python scripts/download_all.py                       # params + benchmarks + dataset  (~150 GB)
    python scripts/download_all.py --all                 # ... plus homo_data (~465 GB sequence databases)
    python scripts/download_all.py --only params benchmarks
    python scripts/download_all.py --skip dataset
    python scripts/download_all.py --dry-run             # show sizes / what is already present

Sections: params, benchmarks, dataset, homo_data.  homo_data is NOT included by default because of its size;
select it with --all or --only homo_data.  Everything goes to <repo>/data unless --data-dir is given.
See the individual download_*.py scripts for details on each section.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_benchmarks  # noqa: E402
import download_dataset  # noqa: E402
import download_homo_data  # noqa: E402
import download_params  # noqa: E402
from downloader import add_common_args, log, run_from_args  # noqa: E402

SECTIONS = ["params", "benchmarks", "dataset", "homo_data"]
DEFAULT_SECTIONS = ["params", "benchmarks", "dataset"]   # homo_data (~465 GB) only on request


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=SECTIONS, metavar="SECTION", help=f"only these sections: {' '.join(SECTIONS)}")
    ap.add_argument("--skip", nargs="+", choices=SECTIONS, metavar="SECTION", default=[], help="skip these sections")
    ap.add_argument("--all", action="store_true", help="include homo_data (the ~465 GB sequence databases)")
    ap.add_argument("--link", action="store_true", help="symlink <repo>/params/params.pt to data/params/params.pt")
    add_common_args(ap)
    args = ap.parse_args(argv)

    base = SECTIONS if args.all else DEFAULT_SECTIONS
    sections = [s for s in (args.only or base) if s not in args.skip]
    files, actions = [], []
    if "params" in sections:
        f, a = download_params.build(args.data_dir)
        files += f
        actions += a
    if "benchmarks" in sections:
        f, a = download_benchmarks.build(args.data_dir)
        files += f
        actions += a
    if "dataset" in sections:
        f, a = download_dataset.build(args.data_dir)
        files += f
        actions += a
    if "homo_data" in sections:
        f, a = download_homo_data.build(args.data_dir)
        files += f
        actions += a

    log(f"sections: {', '.join(sections)}")
    rc = run_from_args(args, files, actions)
    if not args.dry_run:
        if "homo_data" in sections:
            download_homo_data.write_db_manifest(args.data_dir)
        if "params" in sections and args.link and rc == 0:
            download_params.link_repo_params(args.data_dir)
    return rc


if __name__ == "__main__":
    sys.exit(main())
