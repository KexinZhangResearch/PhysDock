"""Shared by download_pdb_mmcif.py and download_ccd.py: the `raw` tag of the download folders (data/raw/...: a RAW_TAG.json in every top folder, saying that its
content is raw data as downloaded from the PDB, not to be edited by hand, from where and when, with the counts of the last run) and a DNS cache.

DNS: in a Kubernetes pod (resolv.conf "options ndots:5" with cluster search domains) every lookup of files.rcsb.org first tries the search domains; under many
parallel connections these lookups time out and fail at random. use_dns_cache() resolves every host once as an absolute name (a trailing dot: no search domains),
retrying, and serves socket.getaddrinfo from the cache afterwards (per process; every downloader process calls it)."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

TAG_FILE = "RAW_TAG.json"


def write_raw_tag(folder: Path, source: str, description: str, **counts) -> Path:
    """<folder>/RAW_TAG.json (overwritten at every run; the first download time is kept)"""
    folder = Path(folder)
    path = folder / TAG_FILE
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except ValueError:
            old = {}
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tag = {"tag": "raw", "description": description, "source": source, "first_downloaded": old.get("first_downloaded", now), "last_updated": now,
           "host": socket.gethostname(), "user": os.environ.get("USER", ""), **counts,
           "note": "raw data as downloaded; do not edit by hand (re-run the download script to update)"}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(tag, indent=2) + "\n")
    tmp.replace(path)
    return path


_dns_cache, _dns_lock, _getaddrinfo = {}, threading.Lock(), socket.getaddrinfo


def _cached_getaddrinfo(host, port, *args, **kwargs):
    key = (host, port, args, tuple(sorted(kwargs.items())))
    with _dns_lock:
        hit = _dns_cache.get(key)
    if hit is not None:
        return hit
    name = host + "." if isinstance(host, str) and host and not host.endswith(".") and not host.replace(".", "").isdigit() and host != "localhost" else host
    for attempt in range(8):
        try:
            res = _getaddrinfo(name, port, *args, **kwargs)
            break
        except socket.gaierror:
            if attempt == 7:
                raise
            time.sleep(0.5 * 2 ** attempt)
    with _dns_lock:
        _dns_cache[key] = res
    return res


def use_dns_cache() -> None:
    """socket.getaddrinfo with the absolute name, retried, cached (this process)"""
    socket.getaddrinfo = _cached_getaddrinfo
