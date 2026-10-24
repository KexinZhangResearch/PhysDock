#!/usr/bin/env python3
"""Download the whole PDB archive in mmCIF format (~260k entries, ~90 GB of .cif.gz) into data/raw, multi-process and resumable.

    python scripts/download/download_pdb_mmcif.py                                   # all entries -> data/raw/pdb_mmcif, from RCSB
    python scripts/download/download_pdb_mmcif.py --out /2024133006/zkx_temp/PhysDock/data/raw --mirror ebi --procs 32 --threads 8
    python scripts/download/download_pdb_mmcif.py --ids 6u5l 2bkj                   # only these entries (or --id-file ids.txt)
    python scripts/download/download_pdb_mmcif.py --dry-run                         # only list what would be fetched
    python scripts/download/download_pdb_mmcif.py --benchmark                       # measure the stable download speed for several process counts, then stop

Layout: <out>/pdb_mmcif/<middle two letters>/<id>.cif.gz, the "divided" layout of the PDB servers (6u5l -> u5/6u5l.cif.gz): ~1300 folders of ~200 files, so
listing a folder never shows all entries. The files stay gzipped (gemmi reads .cif.gz directly). <out>/pdb_mmcif/RAW_TAG.json tags the folder as raw data
(raw_tag.py), with the source and the counts of the last run.

Parallel: --procs processes x --threads threads (connections = procs x threads); every process downloads whole two-letter folders, so the files of a folder are
fetched together. A file already present and a valid gzip is skipped, so a re-run resumes or updates; every file is written to <name>.part and renamed only after
its gzip stream checked out. Network and DNS errors are retried with back-off; an id the mirror does not have (HTTP 404) is reported and skipped. Failed entries
are listed in <out>/pdb_mmcif_failed.txt. Standard library only.

--benchmark downloads into <out>/.pdb_mmcif_benchmark (deleted afterwards) for --bench-seconds per setting and reports files/s, MB/s and the error rate of each
process count (--bench-procs), the stable rate excluding the first seconds.
"""
from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_tag import use_dns_cache, write_raw_tag  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MIRRORS = {
    "rcsb": "https://files.rcsb.org/pub/pdb/data/structures/divided/mmCIF",
    "ebi": "https://ftp.ebi.ac.uk/pub/databases/pdb/data/structures/divided/mmCIF",
    "pdbj": "https://ftp.pdbj.org/pub/pdb/data/structures/divided/mmCIF",
}
USER_AGENT = "PhysDock-downloader/1.0 (+https://github.com/KexinZhangResearch/PhysDock)"
DIR_RE = re.compile(r'href="([0-9a-z]{2})/"')
FILE_RE = re.compile(r'href="([0-9a-z]{4,}\.cif\.gz)"')
SUBDIR = "pdb_mmcif"


class NotFound(Exception):
    """HTTP 404: the entry is not on the mirror (an obsolete or unknown id); not retried"""


def fetch(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout) as r:
        return r.read()


def with_retries(fn, retries: int, what: str):
    """fn() with exponential back-off on any error (network, DNS, a truncated file) but a 404"""
    for attempt in range(retries):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise NotFound(f"{what}: not on the mirror (HTTP 404)") from e
            if attempt == retries - 1:
                raise RuntimeError(f"{what}: HTTP {e.code}") from e
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"{what}: {type(e).__name__}: {e}") from e
        time.sleep(min(30.0, 2.0 ** attempt))


def gzip_ok(data: bytes) -> bool:
    try:
        gzip.decompress(data)
        return True
    except Exception:  # noqa: BLE001
        return False


def list_entries(base: str, threads: int, timeout: float, retries: int):
    """-> {two-letter dir: [file names]} of the whole archive, from the directory index of the mirror"""
    dirs = sorted(set(DIR_RE.findall(with_retries(lambda: fetch(base + "/", timeout), retries, "index").decode())))
    out = {}
    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(with_retries, lambda d=d: fetch(f"{base}/{d}/", timeout), retries, f"index {d}"): d for d in dirs}
        for i, fut in enumerate(as_completed(futs), 1):
            out[futs[fut]] = sorted(set(FILE_RE.findall(fut.result().decode())))
            if i % 100 == 0 or i == len(dirs):
                print(f"\r[list] {i}/{len(dirs)} folders, {sum(map(len, out.values()))} entries", end="", flush=True)
    print()
    return out


def worker(jobs, results, base: str, root: str, threads: int, retries: int, timeout: float, stop_at: float) -> None:
    """one process: takes (folder, [names]) from jobs, downloads them with `threads` threads; puts (status, bytes, name) on results for every file
    (status ok / missing / failed) and None when it stops"""
    root = Path(root)
    use_dns_cache()

    def one(d: str, name: str):
        path = root / d / name
        part = path.with_name(path.name + ".part")

        def attempt():
            data = fetch(f"{base}/{d}/{name}", timeout)
            if not gzip_ok(data):
                raise ValueError("not a valid gzip")
            return data
        try:
            data = with_retries(attempt, retries, name)
            part.write_bytes(data)
            part.replace(path)
            return "ok", len(data), name
        except NotFound:
            return "missing", 0, name
        except Exception as e:  # noqa: BLE001
            part.unlink(missing_ok=True)
            return "failed", 0, f"{name}\t{e}"

    with ThreadPoolExecutor(threads) as ex:
        while True:
            job = jobs.get()
            if job is None or (stop_at and time.time() > stop_at):
                break
            d, names = job
            (root / d).mkdir(parents=True, exist_ok=True)
            for res in ex.map(lambda nm: one(d, nm), names):
                results.put(res)
    results.put(None)


def run(base: str, root: Path, todo, procs: int, threads: int, retries: int, timeout: float, stop_after: float = 0.0, quiet: bool = False):
    """download todo {folder: [names]} with procs x threads; -> (ok, bytes, missing, failed, seconds, [(t, ok cumulative, bytes cumulative)])"""
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    folders = sorted(todo.items(), key=lambda kv: -len(kv[1]))                    # big folders first: the tail stays short
    for item in folders:
        jobs.put(item)
    for _ in range(procs):
        jobs.put(None)
    t0 = time.time()
    stop_at = t0 + stop_after if stop_after else 0.0
    ps = [ctx.Process(target=worker, args=(jobs, results, base, str(root), threads, retries, timeout, stop_at), daemon=True) for _ in range(procs)]
    for p in ps:
        p.start()
    total = sum(len(v) for v in todo.values())
    ok = nbytes = 0
    missing, failed, trace, alive, last = [], [], [], procs, 0.0
    while alive:
        r = results.get()
        if r is None:
            alive -= 1
            continue
        status, b, item = r
        if status == "ok":
            ok, nbytes = ok + 1, nbytes + b
        elif status == "missing":
            missing.append(item)
        else:
            failed.append(item)
        dt = time.time() - t0
        trace.append((dt, ok, nbytes))
        if not quiet and (dt - last > 1.0 or ok + len(missing) + len(failed) == total):
            last = dt
            n = ok + len(missing) + len(failed)
            print(f"\r[get] {n}/{total}  {nbytes / 2**30:.1f} GiB  {ok / dt:.0f} files/s  {nbytes / 2**20 / dt:.1f} MiB/s  failed {len(failed)}  "
                  f"not found {len(missing)}  eta {(total - n) / max(n / dt, 1e-9) / 60:.0f} min   ", end="", flush=True)
    for p in ps:
        p.join(timeout=60)
    if not quiet:
        print()
    return ok, nbytes, missing, failed, time.time() - t0, trace


def benchmark(base: str, out: Path, entries, args) -> None:
    """the stable download rate for every process count of --bench-procs (threads fixed), each for --bench-seconds on fresh folders"""
    bench = out / ".pdb_mmcif_benchmark"
    shutil.rmtree(bench, ignore_errors=True)
    import random
    folders = sorted(entries.items())
    random.Random(0).shuffle(folders)                                             # every setting: a random mix of small and big folders
    pos, warm = 0, min(10.0, args.bench_seconds / 4)
    print(f"[bench] {args.threads} threads per process, {args.bench_seconds:.0f} s per setting (the first {warm:.0f} s not counted), into {bench}")
    print(f"{'procs':>5} {'conns':>6} {'files/s':>8} {'MiB/s':>7} {'errors':>7}")
    try:
        for procs in args.bench_procs:
            chunk = dict(folders[pos:pos + max(2 * procs, 20)])                     # fresh folders: no file is skipped as present
            pos += len(chunk)
            ok, nbytes, missing, failed, secs, trace = run(base, bench, chunk, procs, args.threads, 2, args.timeout, stop_after=args.bench_seconds, quiet=True)
            after = [(t, o, b) for t, o, b in trace if warm <= t <= args.bench_seconds]  # the stable window: after the warm-up, before the stop
            if len(after) >= 2:
                (t1, o1, b1), (t2, o2, b2) = after[0], after[-1]
                rate, mib = (o2 - o1) / max(t2 - t1, 1e-9), (b2 - b1) / 2**20 / max(t2 - t1, 1e-9)
            else:
                rate, mib = ok / secs, nbytes / 2**20 / secs
            err = len(failed) / max(ok + len(failed), 1)
            print(f"{procs:>5} {procs * args.threads:>6} {rate:>8.0f} {mib:>7.1f} {err:>7.1%}", flush=True)
    finally:
        shutil.rmtree(bench, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"), help="the raw data folder (the files go to <out>/pdb_mmcif)")
    ap.add_argument("--mirror", choices=sorted(MIRRORS), default="rcsb")
    ap.add_argument("--ids", nargs="*", default=None, help="only these PDB ids")
    ap.add_argument("--id-file", default=None, help="only the PDB ids in this file (whitespace separated)")
    ap.add_argument("--procs", type=int, default=16, help="download processes")
    ap.add_argument("--threads", type=int, default=8, help="threads (connections) per process")
    ap.add_argument("--retries", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--benchmark", action="store_true", help="measure the download rate for --bench-procs process counts, then stop")
    ap.add_argument("--bench-procs", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    ap.add_argument("--bench-seconds", type=float, default=60.0)
    args = ap.parse_args(argv)
    use_dns_cache()

    base = MIRRORS[args.mirror]
    out = Path(args.out).expanduser().resolve()
    root = out / SUBDIR
    root.mkdir(parents=True, exist_ok=True)

    ids = [i.lower() for i in (args.ids or [])]
    if args.id_file:
        ids += Path(args.id_file).read_text().lower().split()
    if ids:
        entries = {}
        for i in sorted(set(ids)):
            entries.setdefault(i[1:3], []).append(f"{i}.cif.gz")
    else:
        print(f"[list] {base}")
        entries = list_entries(base, 64, args.timeout, args.retries)

    if args.benchmark:
        benchmark(base, out, entries, args)
        return 0

    todo = {d: [n for n in names if not (root / d / n).exists()] for d, names in entries.items()}
    todo = {d: v for d, v in todo.items() if v}
    n_all, n_todo = sum(map(len, entries.values())), sum(map(len, todo.values()))
    print(f"[plan] {n_all} entries in {len(entries)} folders, {n_all - n_todo} already present, {n_todo} to download -> {root}  "
          f"({args.procs} processes x {args.threads} threads)")
    if args.dry_run:
        return 0

    ok, nbytes, missing, failed, secs, _ = run(base, root, todo, args.procs, args.threads, args.retries, args.timeout)
    if missing:
        print(f"[note] {len(missing)} entries not on the mirror (HTTP 404, e.g. obsolete or unknown ids): {' '.join(sorted(missing)[:20])}"
              + (" ..." if len(missing) > 20 else ""))
    present = sum(1 for d in entries for n in entries[d] if (root / d / n).exists())
    tag = write_raw_tag(root, source=base, description="PDB archive, mmCIF (.cif.gz), divided layout <middle two letters>/<id>.cif.gz",
                        entries_listed=n_all, entries_present=present, downloaded_last_run=ok, not_on_mirror=len(missing), failed_last_run=len(failed))
    report = out / f"{SUBDIR}_failed.txt"
    print(f"[done] {ok} downloaded ({nbytes / 2**30:.1f} GiB, {secs / 60:.1f} min), {present}/{n_all} present; tag {tag}")
    if failed:
        report.write_text("\n".join(sorted(failed)) + "\n")
        print(f"[done] {len(failed)} failed, listed in {report}; re-run to retry")
        return 1
    report.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
