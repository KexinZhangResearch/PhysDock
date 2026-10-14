#!/usr/bin/env python3
"""
Shared download engine for the PhysDock data scripts.

Standard-library only.  Optional accelerators: ``google-crc32c`` or ``crc32c``
(used to verify Google Cloud Storage composite objects that only publish a
crc32c checksum; without them the gzip stream CRC is used instead).

Features
--------
* Parallel ranged downloads: every file is split into up to ``--workers``
  byte ranges which are written directly into a pre-allocated sparse file, so
  no merge/copy pass is needed afterwards.
* Resumable: progress of every range is checkpointed into a small sidecar JSON
  (``<tmp>/<file>.dl.json``).  Re-running a script continues where it stopped.
  The remote ETag/size is recorded; if the server file changed (e.g. a new
  UniProt release) the download restarts from scratch instead of producing a
  corrupt mix of two versions.
* Verification: md5 where published (Zenodo, UniProt metalink, GCS), crc32c
  for GCS composite objects, gzip-CRC stream check for archives without any
  published checksum.  A file is only marked ``verified`` after the check
  passes; a mismatch deletes the file and re-downloads it once.
* Post-processing "actions" run as soon as their inputs are verified, while
  other files are still downloading: chained (multi-volume) tar.gz streaming
  extraction without materialising the concatenated archive, multi-process zip
  extraction (idempotent/resumable), streaming gunzip + concatenation.
* Completed actions write a marker JSON so reruns skip them; archives are kept
  in the ``tmp`` directory unless ``--delete-archives`` is given.

Run ``python scripts/download/downloader.py --selftest`` to exercise the engine against
a local HTTP server with fault injection.
"""
from __future__ import annotations

import argparse
import base64
import collections
import errno
import gzip
import hashlib
import http.client
import io
import ipaddress
import itertools
import json
import multiprocessing
import os
import random
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "FileSpec", "Action", "Downloader", "add_common_args", "run_from_args",
    "chained_tar_extract_action", "zip_extract_action", "gunzip_concat_action",
    "zenodo_url", "zenodo_checksum_resolver", "uniprot_metalink_resolver", "gcs_object_resolver",
    "fmt_bytes", "log",
]

KiB, MiB, GiB = 1 << 10, 1 << 20, 1 << 30
USER_AGENT = "PhysDock-downloader/1.0 (+https://github.com/KexinZhangResearch/PhysDock)"
READ_BLOCK = 64 * KiB         # socket read size (small so progress/stop are responsive on slow links)
WRITE_BLOCK = 1 * MiB         # data is written to disk in aligned blocks of this size
SIDECAR_INTERVAL = 3.0        # seconds between sidecar checkpoints per file
HASH_BLOCK = 16 * MiB
PIPE_BLOCK = 8 * MiB
DEFAULT_WORKERS = 96          # parallel HTTP connections / extraction processes (speed on long-haul links scales ~linearly)
DEFAULT_MIN_CHUNK = 8 * MiB   # download chunks are never smaller than this ...
DEFAULT_MAX_CHUNK = 64 * MiB  # ... or larger than this
DEFAULT_RETRIES = 30
DEFAULT_TIMEOUT = 60

# --------------------------------------------------------------------------------------
# Logging / formatting
# --------------------------------------------------------------------------------------
_print_lock = threading.RLock()
_PROGRESS: Optional["Progress"] = None


def fmt_bytes(n: Optional[float]) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1000 or unit == "PB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1000.0
    return f"{n:.2f} PB"


def fmt_time(seconds: Optional[float]) -> str:
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def log(msg: str, level: str = "INFO") -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with _print_lock:
        if _PROGRESS is not None:
            _PROGRESS.clear_line()
        print(f"[{ts}] {level:<5} {msg}", flush=True)


def warn(msg: str) -> None:
    log(msg, "WARN")


def error(msg: str) -> None:
    log(msg, "ERROR")


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w") as f:
        if len(data.get("parts", ())) > 40:
            json.dump(data, f, sort_keys=True, separators=(",", ":"))
        else:
            json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def load_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------------------------------
class Progress:
    """Progress reporter with a 1 Hz speed sampler.

    Shows percent, bytes, current speed (last ``WINDOW`` seconds), average speed, ETA, number of open
    connections and file counts.  On a terminal the line is refreshed every second; in a log file one
    line is printed every ``interval_log`` seconds (the sampler always runs at 1 Hz, so the speed is
    correct regardless of the print interval).
    """
    WINDOW = 10.0

    def __init__(self, total_bytes: int, enabled: bool = True, interval_tty: float = 1.0, interval_log: float = 10.0):
        self.total = total_bytes
        self.done = 0
        self.enabled = enabled
        self.is_tty = sys.stdout.isatty()
        self.interval = interval_tty if self.is_tty else interval_log
        self._lock = threading.Lock()
        self._status: Dict[str, str] = {}
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._t0 = time.monotonic()
        self._samples: "collections.deque" = collections.deque(maxlen=int(self.WINDOW) + 5)
        self._line_len = 0
        self.files_done = 0
        self.files_total = 0
        self.active = 0
        self.peak = 0.0
        self._speed_now = 0.0

    # -- counters --------------------------------------------------------------
    def add(self, n: int) -> None:
        with self._lock:
            self.done += n

    def add_total(self, n: int) -> None:
        with self._lock:
            self.total += n

    def add_active(self, n: int) -> None:
        with self._lock:
            self.active += n

    def set_status(self, key: str, text: Optional[str]) -> None:
        with self._lock:
            if text is None:
                self._status.pop(key, None)
            else:
                self._status[key] = text

    # -- sampling / rendering ----------------------------------------------------
    def _sample(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._samples.append((now, self.done))
            t_old, d_old = self._samples[0]
            dt = now - t_old
            self._speed_now = (self.done - d_old) / dt if dt >= 0.5 else 0.0
            self.peak = max(self.peak, self._speed_now)

    @property
    def average(self) -> float:
        dt = time.monotonic() - self._t0
        return self.done / dt if dt > 0 else 0.0

    def render(self) -> str:
        with self._lock:
            speed, avg = self._speed_now, self.average
            remaining = max(self.total - self.done, 0)
            rate = speed if speed > 0 else avg
            eta = remaining / rate if rate > 0 else None
            pct = 100.0 * self.done / self.total if self.total else 100.0
            parts = [
                f"DL {pct:5.1f}% {fmt_bytes(self.done)}/{fmt_bytes(self.total)}",
                f"{fmt_bytes(speed)}/s (avg {fmt_bytes(avg)}/s)",
                f"ETA {fmt_time(eta)}",
                f"conns {self.active}",
                f"files {self.files_done}/{self.files_total}",
            ]
            parts += [f"{k}: {v}" for k, v in sorted(self._status.items())]
        return " | ".join(parts)

    def clear_line(self) -> None:
        if self.is_tty and self._line_len:
            sys.stdout.write("\r" + " " * self._line_len + "\r")
            sys.stdout.flush()
            self._line_len = 0

    def _sampler_loop(self) -> None:
        while not self._stop.wait(1.0):
            self._sample()

    def _print_loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.tick()

    def tick(self, force: bool = False) -> None:
        if not self.enabled:
            return
        line = self.render()
        with _print_lock:
            if self.is_tty:
                self.clear_line()
                sys.stdout.write(line[:shutil.get_terminal_size((160, 20)).columns - 1])
                sys.stdout.flush()
                self._line_len = len(line)
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] PROG  {line}", flush=True)

    def start(self) -> None:
        global _PROGRESS
        _PROGRESS = self
        self._t0 = time.monotonic()
        self._samples.append((self._t0, self.done))
        t = threading.Thread(target=self._sampler_loop, name="speed-sampler", daemon=True)
        t.start()
        self._threads.append(t)
        if self.enabled:
            t = threading.Thread(target=self._print_loop, name="progress", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        global _PROGRESS
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)
        with _print_lock:
            self.clear_line()
        _PROGRESS = None


# --------------------------------------------------------------------------------------
# HTTP layer: cached DNS, pinned IPs, keep-alive pool, per-host rate-limit pacing
# --------------------------------------------------------------------------------------
class DownloadError(Exception):
    pass


class FatalDownloadError(DownloadError):
    """Errors that retrying will not fix (404, manifest mismatch, ...)."""


class HttpStatusError(DownloadError):
    def __init__(self, status: int, url: str, retry_after: Optional[float] = None):
        super().__init__(f"HTTP {status} for {url}")
        self.status, self.url, self.retry_after = status, url, retry_after


_SSL_CONTEXT: Optional[ssl.SSLContext] = None
_DEFAULT_CONTEXT: Optional[ssl.SSLContext] = None
_DNS_LOCK = threading.Lock()
_DNS_CACHE: Dict[tuple, List[str]] = {}
_RR = itertools.count()
_LIMITERS: Dict[str, "HostLimiter"] = {}
_LIMITERS_LOCK = threading.Lock()
_local = threading.local()


def set_insecure_tls(insecure: bool) -> None:
    global _SSL_CONTEXT
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        _SSL_CONTEXT = ctx
    else:
        _SSL_CONTEXT = None


def _ssl_context() -> ssl.SSLContext:
    global _DEFAULT_CONTEXT
    if _SSL_CONTEXT is not None:
        return _SSL_CONTEXT
    if _DEFAULT_CONTEXT is None:
        _DEFAULT_CONTEXT = ssl.create_default_context()
    return _DEFAULT_CONTEXT


_NO_DATA = {getattr(socket, n) for n in ("EAI_NONAME", "EAI_NODATA", "EAI_ADDRFAMILY") if hasattr(socket, n)}


def invalidate_host(host: str, port: int, ip: Optional[str] = None) -> None:
    """Forget a cached resolution (or just one bad address) so the next lookup is fresh."""
    key = (host, port)
    with _DNS_LOCK:
        cur = _DNS_CACHE.get(key)
        if cur is None:
            return
        rest = [i for i in cur if i != ip] if ip else []
        if rest:
            _DNS_CACHE[key] = rest
        else:
            _DNS_CACHE.pop(key, None)


def resolve_host(host: str, port: int) -> List[str]:
    """Resolve once per process and cache (self-healing, see :func:`_new_connection`).

    * The name is queried as an absolute FQDN (trailing dot).  Kubernetes-style resolv.conf files use
      ``ndots:5``, which otherwise turns every lookup into several failing search-domain queries.
    * IPv4 is preferred.  IPv6 is used ONLY if the resolver definitively answers that the name has no
      A record.  A transient resolver error (timeout / EAI_AGAIN) is retried, never converted into an
      IPv6-only answer: on hosts without an IPv6 route that would make every connection fail (Errno 99).
    """
    key = (host, port)
    with _DNS_LOCK:
        cached = _DNS_CACHE.get(key)
        if cached:
            return cached
    try:
        ipaddress.ip_address(host)
        ips = [host]
    except ValueError:
        name = host.rstrip(".") + "."
        ips, last, v4_definitive_empty = [], None, False
        for attempt in range(10):
            try:
                for info in socket.getaddrinfo(name, port, socket.AF_INET, socket.SOCK_STREAM):
                    if info[4][0] not in ips:
                        ips.append(info[4][0])
            except socket.gaierror as e:
                last = e
                if e.errno in _NO_DATA:
                    v4_definitive_empty = True
            except OSError as e:
                last = e
            if ips:
                break
            if v4_definitive_empty:
                try:
                    for info in socket.getaddrinfo(name, port, socket.AF_INET6, socket.SOCK_STREAM):
                        if info[4][0] not in ips:
                            ips.append(info[4][0])
                except OSError as e:
                    last = e
                break
            time.sleep(min(5.0, 0.5 * (attempt + 1)))
        if not ips:
            raise DownloadError(f"cannot resolve {host}: {last!r}")
    with _DNS_LOCK:
        _DNS_CACHE[key] = ips
    return ips


class _PinnedMixin:
    _ip: str

    def _open_socket(self) -> socket.socket:
        fam = socket.AF_INET6 if ":" in self._ip else socket.AF_INET
        s = socket.socket(fam, socket.SOCK_STREAM)
        try:
            s.settimeout(self.timeout)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            s.connect((self._ip, self.port))
        except BaseException:
            s.close()
            raise
        return s


class PinnedHTTPConnection(_PinnedMixin, http.client.HTTPConnection):
    def __init__(self, host: str, port: int, ip: str, timeout: float):
        super().__init__(host, port, timeout=timeout)
        self._ip = ip

    def connect(self) -> None:
        self.sock = self._open_socket()


class PinnedHTTPSConnection(_PinnedMixin, http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, ip: str, timeout: float):
        super().__init__(host, port, timeout=timeout, context=_ssl_context())
        self._ip = ip

    def connect(self) -> None:
        s = self._open_socket()
        try:
            self.sock = self._context.wrap_socket(s, server_hostname=self.host)
        except BaseException:
            s.close()
            raise


class HostLimiter:
    """Paces request starts using ``X-RateLimit-*`` response headers (Zenodo allows 133 requests/min)."""
    MARGIN = 6

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.remaining: Optional[int] = None
        self.reset: float = 0.0
        self.blocked_until: float = 0.0

    def wait(self, stop: Optional[threading.Event] = None) -> None:
        while True:
            with self.lock:
                now = time.time()
                if now < self.blocked_until:
                    delay = self.blocked_until - now
                elif self.remaining is None or now >= self.reset:
                    self.remaining = None
                    return
                elif self.remaining > self.MARGIN:
                    self.remaining -= 1
                    return
                else:
                    delay = min(max(self.reset - now, 0.0) + 0.5, 65.0)
            if stop is not None:
                if stop.wait(delay):
                    raise InterruptedError("stopped")
            else:
                time.sleep(delay)

    def note(self, headers) -> None:
        rem, reset = headers.get("X-RateLimit-Remaining"), headers.get("X-RateLimit-Reset")
        if rem is None or reset is None:
            return
        try:
            rem_i, reset_f = int(rem), float(reset)
        except ValueError:
            return
        with self.lock:
            if self.remaining is None or abs(reset_f - self.reset) > 1.0:
                self.remaining, self.reset = rem_i, reset_f   # new window
            else:
                self.remaining = min(self.remaining, rem_i)

    def penalize(self, seconds: float) -> None:
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.time() + seconds)


def limiter_for(host: str) -> HostLimiter:
    with _LIMITERS_LOCK:
        return _LIMITERS.setdefault(host, HostLimiter())


def _pool() -> Dict[tuple, http.client.HTTPConnection]:
    pool = getattr(_local, "pool", None)
    if pool is None:
        pool = _local.pool = {}
    return pool


def _new_connection(scheme: str, host: str, port: int, timeout: float) -> http.client.HTTPConnection:
    """Connect to one of the host's addresses (round-robin).  If every cached address fails, drop the cache and
    resolve again once; addresses that fail with 'no route / cannot assign' (Errno 99/101/113) are discarded."""
    last: Optional[Exception] = None
    for round_ in (0, 1):
        ips = resolve_host(host, port)
        start = next(_RR)
        for k in range(len(ips)):
            ip = ips[(start + k) % len(ips)]
            cls = PinnedHTTPSConnection if scheme == "https" else PinnedHTTPConnection
            conn = cls(host, port, ip, timeout)
            try:
                conn.connect()
                return conn
            except OSError as e:
                conn.close()
                last = e
                if e.errno in (errno.EADDRNOTAVAIL, errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EAFNOSUPPORT):
                    invalidate_host(host, port, ip)       # unusable address (e.g. IPv6 without a route)
        invalidate_host(host, port)
    raise DownloadError(f"cannot connect to {host}:{port} ({last!r})")


def _retry_after_seconds(headers) -> Optional[float]:
    ra = headers.get("Retry-After") if headers is not None else None
    if ra:
        try:
            return min(float(ra), 300.0)
        except ValueError:
            return None
    return None


def http_get(url: str, headers: Optional[dict] = None, timeout: float = DEFAULT_TIMEOUT,
             stop: Optional[threading.Event] = None, reuse: bool = True, max_redirects: int = 6):
    """Send a GET and return ``(conn, resp, final_url)`` for a 200/206 response.

    Raises HttpStatusError for other statuses (redirects are followed).  The caller must read the body and
    then call :func:`release` (keeps the connection for the next request in this thread) or close the conn.
    """
    h = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if headers:
        h.update(headers)
    for _hop in range(max_redirects + 1):
        u = urllib.parse.urlsplit(url)
        scheme = u.scheme.lower()
        if scheme not in ("http", "https"):
            raise FatalDownloadError(f"unsupported URL scheme in {url}")
        host = u.hostname or ""
        port = u.port or (443 if scheme == "https" else 80)
        target = (u.path or "/") + (("?" + u.query) if u.query else "")
        limiter = limiter_for(host)
        key = (scheme, host, port)
        pool = _pool()
        for attempt in (0, 1):
            limiter.wait(stop)
            conn = pool.pop(key, None) if reuse else None
            pooled = conn is not None
            if conn is None:
                conn = _new_connection(scheme, host, port, timeout)
            try:
                conn.timeout = timeout
                if conn.sock is not None:
                    conn.sock.settimeout(timeout)
                conn.request("GET", target, headers=h)
                resp = conn.getresponse()
                break
            except (OSError, http.client.HTTPException) as e:
                conn.close()
                if pooled and attempt == 0:
                    continue        # stale keep-alive connection: retry once on a fresh one
                raise DownloadError(f"request to {host} failed: {e!r}") from e
        limiter.note(resp.headers)
        if resp.status in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
            url = urllib.parse.urljoin(url, resp.headers["Location"])
            try:
                resp.read(1 << 16)
            except (OSError, http.client.HTTPException):
                pass
            conn.close()
            continue
        if resp.status in (200, 206):
            return conn, resp, url
        ra = _retry_after_seconds(resp.headers)
        status = resp.status
        conn.close()
        if status == 429:
            limiter.penalize(ra if ra is not None else 30.0)
        raise HttpStatusError(status, url, ra)
    raise DownloadError(f"too many redirects for {url}")


def release(conn: http.client.HTTPConnection, resp: http.client.HTTPResponse, url: str) -> None:
    """Return a fully-read connection to this thread's keep-alive pool, otherwise close it."""
    try:
        if resp.isclosed() and not resp.will_close and conn.sock is not None:
            u = urllib.parse.urlsplit(url)
            port = u.port or (443 if u.scheme == "https" else 80)
            old = _pool().get((u.scheme.lower(), u.hostname or "", port))
            if old is not None and old is not conn:
                old.close()
            _pool()[(u.scheme.lower(), u.hostname or "", port)] = conn
            return
    except Exception:  # noqa: BLE001
        pass
    conn.close()


def _backoff_delay(attempt: int) -> float:
    return min(60.0, 2.0 ** min(attempt, 6)) * (0.75 + 0.5 * random.random())


def fetch_bytes(url: str, timeout: float = DEFAULT_TIMEOUT, retries: int = 5, max_bytes: int = 64 * MiB) -> bytes:
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            conn, resp, _final = http_get(url, timeout=timeout, reuse=False)
            try:
                return resp.read(max_bytes)
            finally:
                conn.close()
        except HttpStatusError as e:
            last = e
            if 400 <= e.status < 500 and e.status not in (408, 429):
                raise FatalDownloadError(str(e)) from e
            time.sleep(e.retry_after or _backoff_delay(attempt))
        except (DownloadError, http.client.HTTPException, OSError) as e:
            last = e
            time.sleep(_backoff_delay(attempt))
    raise DownloadError(f"failed to fetch {url}: {last!r}")


def fetch_text(url: str, **kw) -> str:
    return fetch_bytes(url, **kw).decode("utf-8", "replace")


def fetch_json(url: str, **kw) -> dict:
    return json.loads(fetch_bytes(url, **kw))


@dataclass
class RemoteInfo:
    size: int
    etag: Optional[str]
    last_modified: Optional[str]
    ranges: bool
    final_url: Optional[str] = None


def probe_url(url: str, timeout: float = DEFAULT_TIMEOUT, retries: int = 5) -> RemoteInfo:
    """Learn size / ETag / Range support / final URL with a 1-byte ranged GET (HEAD is unreliable on mirrors)."""
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        conn = None
        try:
            conn, resp, final = http_get(url, {"Range": "bytes=0-0"}, timeout=timeout, reuse=False)
            h = resp.headers
            if resp.status == 206:
                cr = h.get("Content-Range", "")
                m = re.match(r"bytes\s+\d+-\d+/(\d+)", cr)
                if not m:
                    raise DownloadError(f"unparsable Content-Range {cr!r} from {url}")
                size, ranges = int(m.group(1)), True
                resp.read()
            else:
                cl = h.get("Content-Length")
                if cl is None:
                    raise DownloadError(f"no Content-Length for {url}")
                size, ranges = int(cl), False   # body not read: closing aborts the transfer
            return RemoteInfo(size, h.get("ETag"), h.get("Last-Modified"), ranges, final)
        except HttpStatusError as e:
            last = e
            if 400 <= e.status < 500 and e.status not in (408, 429):
                raise FatalDownloadError(str(e)) from e
            time.sleep(e.retry_after or _backoff_delay(attempt))
        except (DownloadError, http.client.HTTPException, OSError) as e:
            last = e
            time.sleep(_backoff_delay(attempt))
        finally:
            if conn is not None:
                conn.close()
    raise DownloadError(f"failed to probe {url}: {last!r}")


# --------------------------------------------------------------------------------------
# Checksum helpers
# --------------------------------------------------------------------------------------
def _crc32c_impl():
    """Return (name, factory) for an incremental crc32c, or (None, None) if no C library is installed."""
    try:
        import google_crc32c  # type: ignore

        class _G:
            def __init__(self):
                self._c = google_crc32c.Checksum()

            def update(self, b):
                self._c.update(b)

            def hexdigest(self):
                return self._c.digest().hex()

        return "google-crc32c", _G
    except Exception:
        pass
    try:
        import crc32c  # type: ignore

        class _C:
            def __init__(self):
                self._v = 0

            def update(self, b):
                self._v = crc32c.crc32c(b, self._v)

            def hexdigest(self):
                return f"{self._v:08x}"

        return "crc32c", _C
    except Exception:
        pass
    return None, None


CRC32C_NAME, _CRC32C_FACTORY = _crc32c_impl()


def crc32c_available() -> bool:
    return _CRC32C_FACTORY is not None


def b64_to_hex(s: str) -> str:
    return base64.b64decode(s).hex()


def hash_file(path: Path, method: str, progress: Optional[Callable[[int], None]] = None,
              stop: Optional[threading.Event] = None) -> str:
    """Return hex digest for ``method`` in {md5, sha256, crc32c, gzip}.

    ``gzip`` streams the member(s) through zlib (validating the embedded CRC32 and ISIZE) and
    returns the CRC32 of the *decompressed* data as reported by the trailer.
    """
    if method in ("md5", "sha256"):
        h = hashlib.new(method)
    elif method == "crc32c":
        if not crc32c_available():
            raise RuntimeError("crc32c requested but no crc32c library installed (pip install google-crc32c)")
        h = _CRC32C_FACTORY()
    elif method == "gzip":
        return gzip_stream_check(path, progress, stop)
    else:
        raise ValueError(f"unknown hash method {method}")
    with open(path, "rb", buffering=0) as f:
        while True:
            if stop is not None and stop.is_set():
                raise InterruptedError("stopped")
            block = f.read(HASH_BLOCK)
            if not block:
                break
            h.update(block)
            if progress:
                progress(len(block))
    return h.hexdigest()


def gzip_stream_check(path: Path, progress: Optional[Callable[[int], None]] = None,
                      stop: Optional[threading.Event] = None) -> str:
    """Decompress a (multi-member) gzip file and discard the output; raises on CRC/size mismatch."""
    import zlib
    crc = 0
    with open(path, "rb") as raw:
        buffered = io.BufferedReader(raw, PIPE_BLOCK)
        gz = gzip.GzipFile(fileobj=buffered, mode="rb")
        last = 0
        while True:
            if stop is not None and stop.is_set():
                raise InterruptedError("stopped")
            block = gz.read(HASH_BLOCK)
            if not block:
                break
            crc = zlib.crc32(block, crc)
            if progress:
                now = buffered.tell()
                progress(now - last)
                last = now
        if progress:
            progress(buffered.tell() - last)
    return f"{crc & 0xFFFFFFFF:08x}"


# --------------------------------------------------------------------------------------
# Checksum resolvers for the various hosts
# --------------------------------------------------------------------------------------
def zenodo_url(record_id: int | str, filename: str) -> str:
    return f"https://zenodo.org/records/{record_id}/files/{urllib.parse.quote(filename)}?download=1"


def zenodo_checksum_resolver(record_id: int | str, filename: str) -> Callable[[], dict]:
    """Fetch size + md5 of a Zenodo file from the records API (fallback for files not hardcoded)."""

    def resolve() -> dict:
        data = fetch_json(f"https://zenodo.org/api/records/{record_id}")
        for f in data.get("files", []):
            if f.get("key") == filename:
                algo, _, value = f.get("checksum", "").partition(":")
                out = {"size": int(f["size"])}
                if algo == "md5":
                    out["md5"] = value.lower()
                return out
        raise FatalDownloadError(f"{filename} not found in Zenodo record {record_id}")

    return resolve


def parse_metalink(text: str) -> tuple[dict, Optional[str]]:
    """Parse a Metalink 3.0 document -> ({filename: {"size": int, "md5": hex}}, version)."""
    files: Dict[str, dict] = {}
    version = None
    try:
        root = ET.fromstring(text)
        ns = {"m": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        q = (lambda t: f"m:{t}") if ns else (lambda t: t)
        v = root.find(q("version"), ns)
        version = v.text.strip() if v is not None and v.text else None
        for fe in root.iter(f"{{{ns['m']}}}file" if ns else "file"):
            name = fe.get("name")
            if not name:
                continue
            entry: dict = {}
            se = fe.find(q("size"), ns)
            if se is not None and se.text:
                entry["size"] = int(se.text)
            for he in fe.iter(f"{{{ns['m']}}}hash" if ns else "hash"):
                if he.get("type", "").lower() == "md5" and he.text:
                    entry["md5"] = he.text.strip().lower()
            files[name] = entry
    except ET.ParseError:
        for m in re.finditer(r'<file\s+name="([^"]+)"\s*>(.*?)</file>', text, re.S):
            name, body = m.group(1), m.group(2)
            entry = {}
            sm = re.search(r"<size>(\d+)</size>", body)
            hm = re.search(r'<hash type="md5">([0-9a-fA-F]+)</hash>', body)
            if sm:
                entry["size"] = int(sm.group(1))
            if hm:
                entry["md5"] = hm.group(1).lower()
            files[name] = entry
        vm = re.search(r"<version>([^<]+)</version>", text)
        version = vm.group(1).strip() if vm else None
    return files, version


_METALINK_CACHE: Dict[str, tuple] = {}
_METALINK_LOCK = threading.Lock()


def uniprot_metalink_resolver(metalink_url: str, filename: str) -> Callable[[], dict]:
    def resolve() -> dict:
        with _METALINK_LOCK:
            if metalink_url not in _METALINK_CACHE:
                _METALINK_CACHE[metalink_url] = parse_metalink(fetch_text(metalink_url, max_bytes=8 * MiB))
            files, version = _METALINK_CACHE[metalink_url]
        if filename not in files:
            raise FatalDownloadError(f"{filename} not listed in {metalink_url}")
        out = dict(files[filename])
        if version:
            out["release"] = version
        return out

    return resolve


def gcs_object_resolver(bucket: str, object_name: str) -> Callable[[], dict]:
    """Size / md5 / crc32c of a public Google Cloud Storage object via the JSON API."""

    def resolve() -> dict:
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(object_name, safe='')}"
        j = fetch_json(url, retries=8)
        out = {"size": int(j["size"])}
        if j.get("md5Hash"):
            out["md5"] = b64_to_hex(j["md5Hash"])
        if j.get("crc32c"):
            out["crc32c"] = b64_to_hex(j["crc32c"])
        if j.get("updated"):
            out["release"] = j["updated"]
        return out

    return resolve


# --------------------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------------------
@dataclass
class FileSpec:
    """A file to download.  ``dest`` is the final path; the sidecar lives in ``tmp_dir``."""
    name: str
    url: str
    dest: Path
    size: Optional[int] = None
    md5: Optional[str] = None
    crc32c: Optional[str] = None            # hex (big-endian), as published by GCS
    resolver: Optional[Callable[[], dict]] = None  # lazily fills size/md5/crc32c/release
    description: str = ""
    group: str = ""
    extracted_hint: int = 0                 # rough on-disk size after extraction (disk-space estimate)
    tmp_dir: Optional[Path] = None

    def __post_init__(self):
        self.dest = Path(self.dest)
        if self.tmp_dir is None:
            self.tmp_dir = self.dest.parent / "tmp" if self.dest.parent.name != "tmp" else self.dest.parent
        self.tmp_dir = Path(self.tmp_dir)
        if self.md5:
            self.md5 = self.md5.lower()
        if self.crc32c:
            self.crc32c = self.crc32c.lower()


@dataclass
class Action:
    """Post-processing step that runs once all ``requires`` files are verified."""
    name: str
    requires: List[FileSpec]
    run: Callable[["Downloader", "Action"], dict]   # returns extra info to store in the marker
    marker: Path
    description: str = ""
    group: str = ""
    verifies: List[FileSpec] = field(default_factory=list)  # 'gzip'-method files validated by running this action
    consumes: bool = True                                  # archives may be deleted after success (--delete-archives)
    outputs: List[Path] = field(default_factory=list)      # for the final summary

    def __post_init__(self):
        self.marker = Path(self.marker)


# --------------------------------------------------------------------------------------
# Runtime state
# --------------------------------------------------------------------------------------
class S:  # file states
    PLANNED, PENDING, DOWNLOADING, COMPLETE, VERIFYING, VERIFIED, CONSUMED, FAILED = (
        "planned", "pending", "downloading", "complete", "verifying", "verified", "consumed", "failed")


@dataclass
class Part:
    start: int
    end: int    # inclusive
    done: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class FileState:
    def __init__(self, spec: FileSpec):
        self.spec = spec
        self.state = S.PLANNED
        self.size: Optional[int] = spec.size
        self.remote: Optional[RemoteInfo] = None
        self.parts: List[Part] = []
        self.lock = threading.Lock()
        self.pending_parts = 0
        self.method: str = "size"          # md5 | crc32c | gzip | size
        self.expected: Optional[str] = None
        self.verified_value: Optional[str] = None
        self.deferred_to: Optional[Action] = None  # gzip verification delegated to an extraction action
        self.attempts = 0
        self.error: Optional[str] = None
        self.extra: dict = {}
        self.sidecar: dict = load_json(self.sidecar_path) or {}
        self._last_save = 0.0

    @property
    def dest(self) -> Path:
        return self.spec.dest

    @property
    def part_path(self) -> Path:
        return self.spec.dest.with_name(self.spec.dest.name + ".part")

    @property
    def sidecar_path(self) -> Path:
        return self.spec.tmp_dir / (self.spec.name + ".dl.json")

    @property
    def downloaded(self) -> int:
        return sum(p.done for p in self.parts)

    def recorded_hash(self) -> Optional[tuple]:
        v = self.sidecar.get("verified")
        if isinstance(v, dict) and v.get("method"):
            if v.get("value") or v["method"] in ("gzip", "size"):
                return v["method"], v.get("value")
        return None

    def identity(self) -> dict:
        """What a marker stores to recognise this exact file later (name + strongest known fingerprint)."""
        if self.method not in ("size", "gzip") and self.expected:
            return {"name": self.spec.name, "method": self.method, "value": self.expected}
        if self.method not in ("size", "gzip") and self.verified_value:
            return {"name": self.spec.name, "method": self.method, "value": self.verified_value}
        rec = self.recorded_hash()
        if rec and rec[1] and rec[0] not in ("gzip", "size"):
            return {"name": self.spec.name, "method": rec[0], "value": rec[1]}
        etag = self.sidecar.get("etag") or (self.remote.etag if self.remote else None)
        size = self.size if self.size is not None else self.sidecar.get("size")
        return {"name": self.spec.name, "method": "size", "value": f"{size}:{etag or ''}"}

    def save_sidecar(self, force: bool = True, **updates) -> None:
        """Persist progress.  Periodic callers pass force=False and are rate limited per file."""
        now = time.monotonic()
        with self.lock:
            self.sidecar.update(updates)
            if not force and not updates and now - self._last_save < SIDECAR_INTERVAL:
                return
            self._last_save = now
            self.sidecar["parts"] = [[p.start, p.end, p.done] for p in self.parts]
            self.sidecar["updated"] = time.time()
            atomic_write_json(self.sidecar_path, self.sidecar)


# --------------------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------------------
class Downloader:
    def __init__(self, *, workers: int = DEFAULT_WORKERS, connections: Optional[int] = None,
                 min_split: int = DEFAULT_MIN_CHUNK, max_chunk: int = DEFAULT_MAX_CHUNK,
                 retries: int = DEFAULT_RETRIES, timeout: float = DEFAULT_TIMEOUT,
                 verify: bool = True, extract: bool = True, force_verify: bool = False,
                 delete_archives: bool = False, quiet: bool = False, fail_fast: bool = False):
        self.workers = max(1, workers)
        self.connections = max(1, connections if connections else self.workers)
        self.min_split = max(64 * KiB, min_split)
        self.max_chunk = max(self.min_split, max_chunk)
        self.retries = retries
        self.timeout = timeout
        self.verify = verify
        self.extract = extract
        self.force_verify = force_verify
        self.delete_archives = delete_archives
        self.quiet = quiet
        self.fail_fast = fail_fast
        self.stop = threading.Event()
        self.progress: Optional[Progress] = None
        self.files: Dict[str, FileState] = {}
        self.actions: List[Action] = []
        self.action_state: Dict[str, str] = {}   # name -> pending|running|done|failed|skipped
        self.action_error: Dict[str, str] = {}
        self._post_pool: Optional[ThreadPoolExecutor] = None
        self._pending_post = 0
        self._cv = threading.Condition()
        self._action_lock = threading.Lock()
        self._dl_pool: Optional[ThreadPoolExecutor] = None
        self._dl_futures: list = []
        self.t_start = 0.0
        self._dry_run = False

    # ----------------------------------------------------------------------------------
    # Planning
    # ----------------------------------------------------------------------------------
    def _resolve(self, fs: FileState) -> None:
        spec = fs.spec
        if spec.resolver is None:
            return
        try:
            info = spec.resolver()
        except Exception as e:  # noqa: BLE001 - checksum lookup is best effort
            warn(f"{spec.name}: could not fetch published checksum ({e!r}); "
                 f"falling back to {'gzip-CRC' if spec.name.endswith(('.gz', '.tgz')) else 'size'} verification")
            return
        if info.get("size") is not None:
            spec.size = int(info["size"])
            fs.size = spec.size
        if info.get("md5"):
            spec.md5 = info["md5"].lower()
        if info.get("crc32c"):
            spec.crc32c = info["crc32c"].lower()
        if info.get("release"):
            fs.extra["release"] = info["release"]

    def _choose_method(self, fs: FileState) -> None:
        spec = fs.spec
        if spec.md5:
            fs.method, fs.expected = "md5", spec.md5
        elif spec.crc32c and crc32c_available():
            fs.method, fs.expected = "crc32c", spec.crc32c
        elif spec.name.endswith((".gz", ".tgz")):
            fs.method, fs.expected = "gzip", None
            if spec.crc32c:
                warn(f"{spec.name}: server publishes only crc32c and no crc32c library is installed "
                     f"(pip install google-crc32c); verifying gzip stream CRC instead")
        else:
            fs.method, fs.expected = "size", None

    def _marker_valid(self, act: Action) -> bool:
        data = load_json(act.marker)
        if not data or data.get("action") != act.name or not data.get("ok"):
            return False
        got = {s.get("name"): s for s in data.get("sources", [])}
        for spec in act.requires:
            ident = self.files[spec.name].identity()
            g = got.get(ident["name"])
            if not g or g.get("method") != ident["method"] or g.get("value") != ident["value"]:
                return False
        return True

    def _plan_file(self, fs: FileState) -> None:
        spec = fs.spec
        side = fs.sidecar
        dest, part = fs.dest, fs.part_path
        if not self._dry_run:
            spec.tmp_dir.mkdir(parents=True, exist_ok=True)
            dest.parent.mkdir(parents=True, exist_ok=True)

        # 1. Already consumed by extraction and intentionally deleted?
        if side.get("consumed") and not dest.exists() and not part.exists():
            users = [a for a in self.actions if spec in a.requires]
            if users and all(self.action_state.get(a.name) == "done" for a in users):
                fs.state = S.CONSUMED
                fs.size = side.get("size", fs.size)
                rec = fs.recorded_hash()
                if rec:
                    fs.method, fs.verified_value = rec
                return

        # 2. Final file present.
        if dest.exists():
            actual = dest.stat().st_size
            if fs.size is not None and actual != fs.size:
                warn(f"{spec.name}: size on disk {actual} != expected {fs.size}; re-downloading")
                dest.unlink()
            else:
                fs.size = actual if fs.size is None else fs.size
                rec = fs.recorded_hash()
                if rec and not self.force_verify and side.get("size") == actual and \
                        (fs.expected is None or (rec[0] == fs.method and rec[1] == fs.expected)):
                    fs.method, fs.verified_value = rec
                    fs.state = S.VERIFIED
                    return
                if not self.verify:
                    fs.state = S.VERIFIED
                    fs.verified_value = rec[1] if rec else None
                    return
                fs.state = S.COMPLETE   # needs (re-)verification
                return

        # 3. Need to download (fresh or resume).
        remote = probe_url(spec.url, self.timeout, min(self.retries, 5))
        fs.remote = remote
        if fs.size is not None and remote.size != fs.size:
            raise FatalDownloadError(
                f"{spec.name}: server reports {remote.size} bytes but manifest/checksum source says {fs.size}. "
                f"The remote file probably changed; refusing to download something the checksum cannot cover.")
        fs.size = remote.size

        resumable = (
            part.exists() and side.get("parts") and side.get("size") == remote.size
            and (not side.get("etag") or not remote.etag or side.get("etag") == remote.etag)
            and side.get("url") == spec.url
        )
        if resumable:
            try:
                if part.stat().st_size != remote.size:
                    resumable = False
            except OSError:
                resumable = False
        if resumable:
            fs.parts = [Part(s, e, d) for s, e, d in side["parts"]]
            if any(p.done > p.length or p.done < 0 for p in fs.parts):
                resumable = False
            elif remote.ranges:
                fs.parts = self._resplit(fs.parts, self._chunk_size(remote.size))
            elif remote.ranges:
                fs.parts = self._resplit(fs.parts, self._chunk_size(remote.size))
        if not resumable:
            if part.exists() or side.get("parts"):
                why = "remote file changed" if (part.exists() and side.get("etag") and remote.etag and side.get("etag") != remote.etag) else "incompatible partial state"
                warn(f"{spec.name}: {why}; restarting download from scratch")
            fs.parts = self._split(remote.size, remote.ranges)
            if self._dry_run:
                fs.state = S.PENDING
                return
            with open(part, "wb") as f:
                f.truncate(remote.size)   # sparse pre-allocation
            fs.sidecar = {}
        if not self._dry_run:
            fs.save_sidecar(url=spec.url, size=remote.size, etag=remote.etag, last_modified=remote.last_modified,
                            method=fs.method, expected=fs.expected, complete=False, verified=None, consumed=False,
                            extra=fs.extra)
        fs.state = S.PENDING

    def _chunk_size(self, size: int) -> int:
        """Small chunks balance load across connections (fast connections simply take more of them) and make
        a failure cost little; large chunks keep the request rate low (Zenodo: 133 requests/minute)."""
        target = size // max(1, int(self.connections * 1.5))
        return max(self.min_split, min(self.max_chunk, target))

    @staticmethod
    def _resplit(parts: List[Part], chunk: int) -> List[Part]:
        """Re-cut unfinished parts (e.g. from an older, coarser layout) into ``chunk``-sized pieces."""
        out: List[Part] = []
        for p in parts:
            remaining = p.length - p.done
            if remaining <= chunk * 3 // 2:
                out.append(p)
                continue
            if p.done:
                out.append(Part(p.start, p.start + p.done - 1, p.done))
            start = p.start + p.done
            while start <= p.end:
                end = min(p.end, start + chunk - 1)
                out.append(Part(start, end))
                start = end + 1
        return out

    @staticmethod
    def _resplit(parts: List[Part], chunk: int) -> List[Part]:
        """Re-cut unfinished parts (e.g. from an older, coarser layout) into ``chunk``-sized pieces."""
        out: List[Part] = []
        for p in parts:
            remaining = p.length - p.done
            if remaining <= chunk * 3 // 2:
                out.append(p)
                continue
            if p.done:
                out.append(Part(p.start, p.start + p.done - 1, p.done))
            start = p.start + p.done
            while start <= p.end:
                end = min(p.end, start + chunk - 1)
                out.append(Part(start, end))
                start = end + 1
        return out

    def _split(self, size: int, ranges_ok: bool) -> List[Part]:
        if size <= 0:
            return [Part(0, -1, 0)]
        if not ranges_ok:
            return [Part(0, size - 1)]
        chunk = self._chunk_size(size)
        # align chunk boundaries to 1 MiB so writes are block aligned on parallel file systems (GPFS/Lustre)
        chunk = max(WRITE_BLOCK, (chunk // WRITE_BLOCK) * WRITE_BLOCK) if chunk >= WRITE_BLOCK else chunk
        parts, start = [], 0
        while start < size:
            end = min(size - 1, start + chunk - 1)
            parts.append(Part(start, end))
            start = end + 1
        return parts

    # ----------------------------------------------------------------------------------
    # Downloading
    # ----------------------------------------------------------------------------------
    def _download_part(self, fs: FileState, part: Part) -> None:
        spec = fs.spec
        remote = fs.remote
        use_range = remote.ranges if remote is not None else True
        url = (remote.final_url if remote is not None and remote.final_url else spec.url)
        attempt = 0
        buf = getattr(_local, "wbuf", None)
        if buf is None:
            buf = _local.wbuf = bytearray(WRITE_BLOCK)
        mv = memoryview(buf)

        def backoff(reason: str, delay: float) -> None:
            nonlocal attempt
            attempt += 1
            if attempt > self.retries:
                raise DownloadError(f"{spec.name}: giving up after {attempt} attempts ({reason})")
            if attempt in (1, 3, 10) or attempt % 10 == 0:
                warn(f"{spec.name} [{part.start}-{part.end}]: {reason}; retry {attempt} in {delay:.0f}s")
            fs.save_sidecar(force=False)
            self._sleep(delay)

        while not self.stop.is_set():
            if part.done >= part.length:
                return
            if not use_range and part.done:
                # server cannot resume: this (single) range restarts from zero
                if self.progress:
                    self.progress.add(-part.done)
                part.done = 0
            start = part.start + part.done
            headers = {"Range": f"bytes={start}-{part.end}"} if use_range else {}
            conn = None
            progressed = False
            try:
                conn, resp, final = http_get(url, headers, timeout=self.timeout, stop=self.stop)
                if use_range:
                    m = re.match(r"bytes\s+(\d+)-(\d+)/(\d+)", resp.headers.get("Content-Range", ""))
                    if resp.status != 206 or not m or int(m.group(1)) != start or int(m.group(3)) != fs.size:
                        raise DownloadError(f"unexpected response (status {resp.status}, "
                                            f"Content-Range {resp.headers.get('Content-Range')!r})")
                    expect = int(m.group(2)) - start + 1
                else:
                    expect = fs.size or 0
                if resp.length is not None and resp.length != expect and use_range:
                    raise DownloadError(f"Content-Length {resp.length} != requested {expect}")
                if self.progress:
                    self.progress.add_active(1)
                fd = os.open(fs.part_path, os.O_WRONLY)
                filled = 0
                try:
                    while not self.stop.is_set() and part.done + filled < part.length:
                        want = min(READ_BLOCK, part.length - part.done - filled, WRITE_BLOCK - filled)
                        n = resp.readinto(mv[filled:filled + want])
                        if not n:
                            break
                        filled += n
                        if filled >= WRITE_BLOCK or part.done + filled >= part.length:
                            os.pwrite(fd, mv[:filled], part.start + part.done)
                            part.done += filled
                            if self.progress:
                                self.progress.add(filled)
                            progressed = True
                            filled = 0
                            fs.save_sidecar(force=False)
                finally:
                    if filled:      # keep what was received before an error / stop
                        os.pwrite(fd, mv[:filled], part.start + part.done)
                        part.done += filled
                        if self.progress:
                            self.progress.add(filled)
                        progressed = True
                    os.close(fd)
                    if self.progress:
                        self.progress.add_active(-1)
                if part.done >= part.length:
                    release(conn, resp, final)
                    conn = None
                    fs.save_sidecar(force=False)
                    return
                if self.stop.is_set():
                    return
                if progressed:
                    attempt = 0
                backoff(f"connection closed after {part.done}/{part.length} bytes", min(30.0, 2.0 ** min(attempt, 5)))
            except InterruptedError:
                return
            except HttpStatusError as e:
                if e.status == 416 and part.done >= part.length:
                    return
                if e.status in (408, 429) or e.status >= 500:
                    backoff(f"HTTP {e.status}", e.retry_after or _backoff_delay(attempt + 1))
                    continue
                raise FatalDownloadError(f"{spec.name}: {e}") from e
            except FatalDownloadError:
                raise
            except (http.client.HTTPException, OSError, DownloadError) as e:
                if progressed:
                    attempt = 0
                backoff(f"{e.__class__.__name__}: {e}", _backoff_delay(attempt + 1))
            finally:
                if conn is not None:
                    conn.close()
                fs.save_sidecar(force=False)

    def _sleep(self, seconds: float) -> None:
        self.stop.wait(seconds)

    def _part_task(self, fs: FileState, part: Part) -> None:
        if fs.state == S.FAILED:
            return
        try:
            with fs.lock:
                fs.state = S.DOWNLOADING
            self._download_part(fs, part)
        except Exception as e:  # noqa: BLE001
            self._fail_file(fs, str(e))
            return
        if self.stop.is_set():
            return
        with fs.lock:
            fs.pending_parts -= 1
            last = fs.pending_parts == 0 and fs.state == S.DOWNLOADING
        if last:
            self._file_downloaded(fs)

    def _fail_file(self, fs: FileState, msg: str) -> None:
        with fs.lock:
            if fs.state == S.FAILED:
                return
            fs.state = S.FAILED
            fs.error = msg
        error(f"{fs.spec.name}: FAILED - {msg}")
        if self.fail_fast:
            self.stop.set()

    def _file_downloaded(self, fs: FileState) -> None:
        if fs.downloaded != fs.size:
            self._fail_file(fs, f"downloaded {fs.downloaded} of {fs.size} bytes")
            return
        try:
            os.replace(fs.part_path, fs.dest)
        except OSError as e:
            self._fail_file(fs, f"cannot finalise file: {e}")
            return
        fs.save_sidecar(complete=True)
        with fs.lock:
            fs.state = S.COMPLETE
        log(f"{fs.spec.name}: download complete ({fmt_bytes(fs.size)})")
        self._submit_post(self._verify_task, fs)

    # ----------------------------------------------------------------------------------
    # Verification
    # ----------------------------------------------------------------------------------
    def _verify_task(self, fs: FileState) -> None:
        spec = fs.spec
        if not self.verify:
            self._mark_verified(fs, None)
            return
        if fs.deferred_to is not None and self.extract:
            # gzip-stream verification happens implicitly while the action extracts the archive
            with self._action_lock:
                already_done = self.action_state.get(fs.deferred_to.name) == "done"
            if already_done:
                self._mark_verified(fs, None, via=fs.deferred_to.name)
                return
            log(f"{spec.name}: no published checksum; integrity will be validated by '{fs.deferred_to.name}' (gzip CRC)")
            self._check_actions()
            return
        with fs.lock:
            fs.state = S.VERIFYING
        key = f"verify {spec.name}"
        done = [0]
        total = fs.size or 1

        def prog(n: int) -> None:
            done[0] += n
            if self.progress:
                self.progress.set_status(key, f"{100.0 * done[0] / total:4.1f}%")

        try:
            if self.progress:
                self.progress.set_status(key, "0.0%")
            if fs.method == "size":
                value = None
            else:
                t0 = time.monotonic()
                value = hash_file(fs.dest, fs.method, prog, self.stop)
                dt = time.monotonic() - t0
                log(f"{spec.name}: {fs.method} = {value} ({fmt_time(dt)}, {fmt_bytes(fs.size / max(dt, 1e-6))}/s)")
        except InterruptedError:
            return
        except Exception as e:  # noqa: BLE001
            self._verification_failed(fs, f"{fs.method} check raised {e!r}")
            return
        finally:
            if self.progress:
                self.progress.set_status(key, None)
        if fs.expected is not None and value != fs.expected:
            self._verification_failed(fs, f"{fs.method} mismatch: got {value}, expected {fs.expected}")
            return
        self._mark_verified(fs, value)

    def _mark_verified(self, fs: FileState, value: Optional[str], via: Optional[str] = None) -> None:
        with fs.lock:
            fs.verified_value = value
            fs.state = S.VERIFIED
        fs.save_sidecar(complete=True, size=fs.size, method=fs.method, expected=fs.expected,
                        verified={"method": fs.method, "value": value, "time": time.time(), "via": via},
                        extra=fs.extra)
        if self.progress:
            self.progress.files_done += 1
        suffix = f" (via {via})" if via else ""
        log(f"{fs.spec.name}: verified [{fs.method}]{suffix}")
        self._check_actions()

    def _verification_failed(self, fs: FileState, msg: str) -> None:
        fs.attempts += 1
        if fs.attempts >= 2 or self.stop.is_set():
            self._fail_file(fs, msg)
            return
        warn(f"{fs.spec.name}: {msg}; deleting and downloading again")
        try:
            fs.dest.unlink()
        except OSError:
            pass
        fs.sidecar = {}
        try:
            fs.sidecar_path.unlink()
        except OSError:
            pass
        try:
            self._plan_file(fs)
        except Exception as e:  # noqa: BLE001
            self._fail_file(fs, f"re-plan failed: {e}")
            return
        if self.progress:
            self.progress.add_total(fs.size or 0)
        self._submit_download(fs)

    # ----------------------------------------------------------------------------------
    # Actions
    # ----------------------------------------------------------------------------------
    def _requirements_met(self, act: Action) -> bool:
        for spec in act.requires:
            fs = self.files[spec.name]
            if fs.state == S.VERIFIED:
                continue
            if fs.state == S.COMPLETE and fs.deferred_to is act:
                continue
            return False
        return True

    def _check_actions(self) -> None:
        if not self.extract:
            return
        to_run = []
        with self._action_lock:
            for act in self.actions:
                if self.action_state.get(act.name) == "pending" and self._requirements_met(act):
                    self.action_state[act.name] = "running"
                    to_run.append(act)
        for act in to_run:
            self._submit_post(self._action_task, act)

    def _action_task(self, act: Action) -> None:
        log(f"{act.name}: starting{' - ' + act.description if act.description else ''}")
        t0 = time.monotonic()
        try:
            extra = act.run(self, act) or {}
        except InterruptedError:
            with self._action_lock:
                self.action_state[act.name] = "pending"
            return
        except Exception as e:  # noqa: BLE001
            with self._action_lock:
                self.action_state[act.name] = "failed"
                self.action_error[act.name] = f"{e.__class__.__name__}: {e}"
            error(f"{act.name}: FAILED - {e}")
            if self.fail_fast:
                self.stop.set()
            return
        if self.stop.is_set():
            with self._action_lock:
                self.action_state[act.name] = "pending"
            return
        # files whose integrity was validated by this action
        for spec in act.verifies:
            fs = self.files[spec.name]
            if fs.state != S.VERIFIED:
                self._mark_verified(fs, None, via=act.name)
        sources = [self.files[s.name].identity() for s in act.requires]
        atomic_write_json(act.marker, {"action": act.name, "ok": True, "sources": sources,
                                       "finished": time.time(), "elapsed_s": round(time.monotonic() - t0, 1),
                                       "outputs": [str(p) for p in act.outputs], **extra})
        with self._action_lock:
            self.action_state[act.name] = "done"
        log(f"{act.name}: done ({fmt_time(time.monotonic() - t0)})")
        if self.delete_archives:
            self._cleanup_consumed()

    def _cleanup_consumed(self) -> None:
        """With --delete-archives: remove archives whose every consuming action has finished."""
        for fs in self.files.values():
            if fs.state != S.VERIFIED or not fs.dest.exists():
                continue
            users = [a for a in self.actions if fs.spec in a.requires]
            if not users or not all(a.consumes for a in users):
                continue
            with self._action_lock:
                done = all(self.action_state.get(a.name) == "done" for a in users)
            if not done:
                continue
            try:
                fs.dest.unlink()
            except OSError as e:
                warn(f"{fs.spec.name}: could not delete archive: {e}")
                continue
            with fs.lock:
                fs.state = S.CONSUMED
            fs.save_sidecar(consumed=True)
            log(f"{fs.spec.name}: archive deleted (--delete-archives)")

    # ----------------------------------------------------------------------------------
    # Pools
    # ----------------------------------------------------------------------------------
    def _submit_post(self, fn: Callable, *args) -> None:
        with self._cv:
            self._pending_post += 1

        def wrapper():
            try:
                fn(*args)
            except Exception as e:  # noqa: BLE001 - never lose a worker silently
                error(f"internal error in {fn.__name__}: {e!r}")
            finally:
                with self._cv:
                    self._pending_post -= 1
                    self._cv.notify_all()

        assert self._post_pool is not None
        self._post_pool.submit(wrapper)

    def _submit_download(self, fs: FileState) -> None:
        assert self._dl_pool is not None
        todo = [p for p in fs.parts if p.done < p.length]
        with fs.lock:
            fs.pending_parts = len(todo)
            fs.state = S.PENDING
        if not todo:
            self._file_downloaded(fs)
            return
        for p in todo:
            self._dl_futures.append(self._dl_pool.submit(self._part_task, fs, p))

    # ----------------------------------------------------------------------------------
    # Main entry
    # ----------------------------------------------------------------------------------
    def run(self, files: Sequence[FileSpec], actions: Sequence[Action] = (), dry_run: bool = False) -> int:
        self.t_start = time.monotonic()
        self._dry_run = dry_run
        self.files = {}
        for spec in files:
            if spec.name in self.files:
                raise ValueError(f"duplicate file name {spec.name}")
            self.files[spec.name] = FileState(spec)
        self.actions = list(actions)
        for act in self.actions:
            for spec in act.requires:
                if spec.name not in self.files:
                    raise ValueError(f"action {act.name} requires unknown file {spec.name}")
            for spec in act.verifies:
                if spec not in act.requires:
                    raise ValueError(f"action {act.name} verifies {spec.name} but does not require it")

        # -- resolve checksums (network) & choose verification methods -------------------------
        log(f"Planning {len(self.files)} file(s), {len(self.actions)} post-processing step(s); "
            f"connections={self.connections}, workers={self.workers}, crc32c={CRC32C_NAME or 'unavailable'}")
        with ThreadPoolExecutor(min(8, max(1, len(self.files)))) as ex:
            list(ex.map(self._resolve, self.files.values()))
        for fs in self.files.values():
            self._choose_method(fs)
        for act in self.actions:
            for spec in act.verifies:
                fs = self.files[spec.name]
                if fs.method == "gzip":
                    fs.deferred_to = act

        # -- which actions are already done? ----------------------------------------------------
        for act in self.actions:
            self.action_state[act.name] = "done" if self._marker_valid(act) else "pending"

        # -- plan files (probes the servers; parallel) -----------------------------------------
        def plan(fs: FileState):
            try:
                self._plan_file(fs)
            except Exception as e:  # noqa: BLE001
                fs.state = S.FAILED
                fs.error = str(e)
                error(f"{fs.spec.name}: cannot plan download - {e}")

        with ThreadPoolExecutor(min(8, max(1, len(self.files)))) as ex:
            list(ex.map(plan, self.files.values()))

        # markers may now be judged with hashes recorded in sidecars
        for act in self.actions:
            if self.action_state[act.name] == "pending" and self._marker_valid(act):
                self.action_state[act.name] = "done"

        self._print_plan(dry_run)
        if dry_run:
            return 0
        if any(fs.state == S.FAILED for fs in self.files.values()) and self.fail_fast:
            return 1

        # -- execute ------------------------------------------------------------------------------
        to_download = [fs for fs in self.files.values() if fs.state == S.PENDING]
        total_remaining = sum((fs.size or 0) - fs.downloaded for fs in to_download)
        self.progress = Progress(total_remaining, enabled=not self.quiet)
        self.progress.files_total = len([fs for fs in self.files.values() if fs.state not in (S.FAILED, S.CONSUMED)])
        self.progress.files_done = len([fs for fs in self.files.values() if fs.state == S.VERIFIED])
        self.progress.start()

        post_workers = max(2, min(self.workers, 8))
        self._post_pool = ThreadPoolExecutor(post_workers, thread_name_prefix="post")
        self._dl_pool = ThreadPoolExecutor(self.connections, thread_name_prefix="dl")
        rc = 0
        try:
            # verified files may already unlock actions; complete-but-unverified ones need hashing
            for fs in list(self.files.values()):
                if fs.state == S.COMPLETE:
                    self._submit_post(self._verify_task, fs)
            self._check_actions()
            for fs in to_download:
                self._submit_download(fs)

            # wait for downloads (the list can grow while re-downloading after a hash mismatch)
            # and for the post-processing queue to drain
            i = 0
            while not self.stop.is_set():
                while i < len(self._dl_futures) and not self.stop.is_set():
                    try:
                        self._dl_futures[i].result(timeout=0.5)
                        i += 1
                    except FuturesTimeout:
                        continue
                with self._cv:
                    if i >= len(self._dl_futures) and self._pending_post == 0:
                        break
                    self._cv.wait(timeout=0.5)
            if not self.stop.is_set() and self.delete_archives:
                self._cleanup_consumed()
        except KeyboardInterrupt:
            warn("interrupted - saving progress; rerun the same command to resume")
            self.stop.set()
            rc = 130
        finally:
            self._dl_pool.shutdown(wait=True, cancel_futures=True)
            self._post_pool.shutdown(wait=True, cancel_futures=True)
            self.progress.stop()
            for fs in self.files.values():          # make sure the latest chunk progress is on disk
                if fs.parts and fs.state in (S.PENDING, S.DOWNLOADING, S.FAILED) and fs.part_path.exists():
                    fs.save_sidecar()
        return self._summary(rc)

    # ----------------------------------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------------------------------
    def _print_plan(self, dry_run: bool) -> None:
        rows = []
        need = 0
        for fs in self.files.values():
            st = fs.state
            if st == S.PENDING:
                left = (fs.size or 0) - fs.downloaded
                need += left
                what = "resume" if fs.downloaded else "download"
                detail = f"{what} {fmt_bytes(left)} in {len([p for p in fs.parts if p.done < p.length])} range(s)"
            elif st == S.COMPLETE:
                detail = f"present, verify {fs.method}"
            elif st == S.VERIFIED:
                detail = "present, verified"
            elif st == S.CONSUMED:
                detail = "extracted earlier, archive deleted"
            elif st == S.FAILED:
                detail = f"ERROR: {fs.error}"
            else:
                detail = st
            rows.append((fs.spec.name, fmt_bytes(fs.size), fs.method + (f"={fs.expected[:12]}…" if fs.expected else ""), detail))
        name_w = max(len(r[0]) for r in rows) if rows else 10
        log("Plan:")
        for name, size, method, detail in rows:
            print(f"    {name:<{name_w}}  {size:>10}  {method:<22}  {detail}")
        for act in self.actions:
            print(f"    [{self.action_state[act.name]:<7}] {act.name}: {act.description}")
        if need:
            # disk space estimate
            dests = {fs.spec.dest.parent for fs in self.files.values() if fs.state == S.PENDING}
            extracted = sum(fs.spec.extracted_hint for fs in self.files.values()
                            if any(self.action_state.get(a.name) == "pending" for a in self.actions if fs.spec in a.requires))
            try:
                free = min(shutil.disk_usage(d).free for d in dests)
            except OSError:
                free = None
            msg = f"To download: {fmt_bytes(need)}; estimated extracted size: {fmt_bytes(extracted)}"
            if free is not None:
                msg += f"; free disk: {fmt_bytes(free)}"
                if need + extracted > free:
                    warn(msg + "  <-- INSUFFICIENT DISK SPACE (estimate)")
                    return
            log(msg)
        elif not dry_run:
            log("Nothing to download.")

    def _summary(self, rc: int) -> int:
        elapsed = time.monotonic() - self.t_start
        failed = [fs for fs in self.files.values() if fs.state == S.FAILED]
        incomplete = [fs for fs in self.files.values() if fs.state in (S.PENDING, S.DOWNLOADING, S.COMPLETE, S.VERIFYING)]
        bad_actions = [a for a in self.actions if self.action_state.get(a.name) != "done"] if self.extract else []
        downloaded = self.progress.done if self.progress else 0
        log(f"Summary: {fmt_bytes(downloaded)} downloaded in {fmt_time(elapsed)}"
            + (f" ({fmt_bytes(downloaded / elapsed)}/s)" if elapsed > 0 and downloaded else ""))
        for fs in self.files.values():
            mark = {S.VERIFIED: "OK ", S.CONSUMED: "OK ", S.FAILED: "ERR"}.get(fs.state, "...")
            extra = f" - {fs.error}" if fs.error else ""
            print(f"    [{mark}] {fs.spec.name:<32} {fs.state:<10} {fs.method}{extra}")
        for act in self.actions:
            st = self.action_state.get(act.name, "pending")
            mark = {"done": "OK ", "failed": "ERR"}.get(st, "...")
            extra = f" - {self.action_error[act.name]}" if act.name in self.action_error else ""
            print(f"    [{mark}] {act.name:<32} {st:<10}{extra}")
        if failed or bad_actions or incomplete:
            if rc == 130:
                log("Interrupted. Rerun the same command to resume.")
            else:
                error(f"{len(failed)} file(s) failed, {len(incomplete)} incomplete, {len(bad_actions)} step(s) not done. "
                      f"Rerun the same command to retry/resume.")
            return rc or 1
        log("All files verified" + (" and extracted." if self.actions and self.extract else "."))
        return rc


# --------------------------------------------------------------------------------------
# Archive helpers used by actions
# --------------------------------------------------------------------------------------
class ChainedReader(io.RawIOBase):
    """Read several files back-to-back as one stream (for split archives: cat a b c | ...)."""

    def __init__(self, paths: Sequence[Path]):
        super().__init__()
        self.paths = [Path(p) for p in paths]
        self._idx = 0
        self._fh = open(self.paths[0], "rb") if self.paths else None
        self.consumed = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while self._fh is not None:
            n = self._fh.readinto(b)
            if n:
                self.consumed += n
                return n
            self._fh.close()
            self._idx += 1
            self._fh = open(self.paths[self._idx], "rb") if self._idx < len(self.paths) else None
        return 0

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        super().close()


def strip_components(path: str, n: int) -> Optional[str]:
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    if path.startswith("/") or any(p == ".." for p in parts):
        raise ValueError(f"unsafe path inside archive: {path!r}")
    if len(parts) <= n:
        return None
    return "/".join(parts[n:])


def _tar_cmd() -> Optional[str]:
    return shutil.which("tar")


def chained_tar_extract_action(name: str, parts: List[FileSpec], dest_dir: Path, marker: Path,
                               strip: int = 0, description: str = "", group: str = "",
                               verifies: Optional[List[FileSpec]] = None, outputs: Optional[List[Path]] = None) -> Action:
    """Extract ``cat part1 part2 ... | tar -xzf -`` into ``dest_dir`` without creating the merged archive."""
    dest_dir = Path(dest_dir)

    def run(dl: Downloader, act: Action) -> dict:
        paths = [dl.files[p.name].dest for p in parts]
        total = sum(p.stat().st_size for p in paths)
        dest_dir.mkdir(parents=True, exist_ok=True)
        key = f"extract {name}"
        reader = ChainedReader(paths)
        tar = _tar_cmd()
        n_members = None
        t0 = time.monotonic()
        try:
            if tar:
                cmd = [tar, "-x", "-z", "-f", "-", "-C", str(dest_dir), "--no-same-owner"]
                if strip:
                    cmd.append(f"--strip-components={strip}")
                log_path = act.marker.with_suffix(".tar.log")
                with open(log_path, "wb") as errlog:
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=errlog, stderr=errlog)
                    try:
                        assert proc.stdin is not None
                        while True:
                            if dl.stop.is_set():
                                proc.kill()
                                raise InterruptedError("stopped")
                            block = reader.read(PIPE_BLOCK)
                            if not block:
                                break
                            try:
                                proc.stdin.write(block)
                            except BrokenPipeError:
                                break
                            if dl.progress:
                                dl.progress.set_status(key, f"{100.0 * reader.consumed / max(total, 1):4.1f}%")
                        try:
                            proc.stdin.close()
                        except BrokenPipeError:
                            pass
                        rc = proc.wait()
                    finally:
                        if proc.poll() is None:
                            proc.kill()
                if rc != 0:
                    tail = ""
                    try:
                        tail = log_path.read_text(errors="replace")[-2000:]
                    except OSError:
                        pass
                    raise RuntimeError(f"tar exited with {rc} (stream {reader.consumed}/{total} bytes){': ' + tail if tail else ''}")
                if reader.consumed != total:
                    raise RuntimeError(f"tar stopped early after {reader.consumed}/{total} bytes (corrupt archive?)")
            else:
                warn(f"{name}: system 'tar' not found, using Python tarfile (slower)")
                n_members = 0
                buffered = io.BufferedReader(reader, PIPE_BLOCK)
                with tarfile.open(fileobj=buffered, mode="r|gz") as tf:
                    for m in tf:
                        if dl.stop.is_set():
                            raise InterruptedError("stopped")
                        rel = strip_components(m.name, strip)
                        if rel is None:
                            continue
                        m.name = rel
                        try:
                            tf.extract(m, path=dest_dir, set_attrs=False, filter="data")  # type: ignore[call-arg]
                        except TypeError:
                            tf.extract(m, path=dest_dir, set_attrs=False)
                        n_members += 1
                        if dl.progress and n_members % 200 == 0:
                            dl.progress.set_status(key, f"{100.0 * reader.consumed / max(total, 1):4.1f}%")
        finally:
            reader.close()
            if dl.progress:
                dl.progress.set_status(key, None)
        out = {"bytes_in": total, "elapsed_s": round(time.monotonic() - t0, 1), "dest": str(dest_dir)}
        if n_members is not None:
            out["members"] = n_members
        return out

    return Action(name=name, requires=list(parts), run=run, marker=marker, description=description, group=group,
                  verifies=list(verifies or []), outputs=list(outputs or [dest_dir]))


# ---- zip -----------------------------------------------------------------------------
_ZIP_COUNTER = None


def _zip_init(counter):
    global _ZIP_COUNTER
    _ZIP_COUNTER = counter


def _zip_extract_slice(zip_path: str, dest: str, strip: int, lo: int, hi: int) -> tuple:
    extracted = skipped = 0
    dest_p = Path(dest)
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()[lo:hi]
        for i, info in enumerate(infos):
            rel = strip_components(info.filename, strip)
            if rel is None:
                continue
            target = dest_p / rel
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            try:
                if target.stat().st_size == info.file_size:
                    skipped += 1
                    continue
            except OSError:
                pass
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            with zf.open(info) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 * MiB)   # zipfile validates the member CRC on EOF
            os.replace(tmp, target)
            extracted += 1
            if _ZIP_COUNTER is not None and (i + 1) % 200 == 0:
                with _ZIP_COUNTER.get_lock():
                    _ZIP_COUNTER.value += 200
    return extracted, skipped


def zip_extract_action(name: str, archive: FileSpec, dest_dir: Path, marker: Path, strip: int = 0,
                       description: str = "", group: str = "", outputs: Optional[List[Path]] = None) -> Action:
    """Multi-process, idempotent zip extraction (members already present with the right size are skipped)."""
    dest_dir = Path(dest_dir)

    def run(dl: Downloader, act: Action) -> dict:
        zip_path = dl.files[archive.name].dest
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            n = len(zf.infolist())
        workers = max(1, min(dl.workers, 16, -(-n // 2000) or 1))
        slices = []
        step = -(-n // workers)
        for lo in range(0, n, step):
            slices.append((lo, min(n, lo + step)))
        key = f"extract {name}"
        ctx = multiprocessing.get_context("spawn")   # never fork() a multi-threaded process
        counter = ctx.Value("q", 0)
        extracted = skipped = 0
        t0 = time.monotonic()
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_zip_init, initargs=(counter,)) as ex:
            futs = [ex.submit(_zip_extract_slice, str(zip_path), str(dest_dir), strip, lo, hi) for lo, hi in slices]
            pending = set(futs)
            while pending:
                if dl.stop.is_set():
                    for f in pending:
                        f.cancel()
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise InterruptedError("stopped")
                done = {f for f in pending if f.done()}
                for f in done:
                    e, s = f.result()
                    extracted += e
                    skipped += s
                pending -= done
                if dl.progress:
                    dl.progress.set_status(key, f"{100.0 * counter.value / max(n, 1):4.1f}% ({n} entries)")
                if pending:
                    time.sleep(0.5)
        if dl.progress:
            dl.progress.set_status(key, None)
        log(f"{name}: {extracted} file(s) extracted, {skipped} already present, {n} entries total")
        return {"entries": n, "extracted": extracted, "skipped": skipped, "elapsed_s": round(time.monotonic() - t0, 1),
                "dest": str(dest_dir)}

    return Action(name=name, requires=[archive], run=run, marker=marker, description=description, group=group,
                  outputs=list(outputs or [dest_dir]))


# ---- gunzip --------------------------------------------------------------------------
def gunzip_concat_action(name: str, sources: List[FileSpec], dest_file: Path, marker: Path,
                         description: str = "", group: str = "", verifies: Optional[List[FileSpec]] = None) -> Action:
    """Stream-decompress one or more .gz files and concatenate them (in order) into ``dest_file``."""
    dest_file = Path(dest_file)

    def run(dl: Downloader, act: Action) -> dict:
        paths = [dl.files[s.name].dest for s in sources]
        total = sum(p.stat().st_size for p in paths)
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest_file.with_name(dest_file.name + ".tmp")
        key = f"gunzip {name}"
        consumed = 0
        written = 0
        t0 = time.monotonic()
        try:
            with open(tmp, "wb", buffering=PIPE_BLOCK) as out:
                for p in paths:
                    with open(p, "rb") as raw:
                        buffered = io.BufferedReader(raw, PIPE_BLOCK)
                        gz = gzip.GzipFile(fileobj=buffered, mode="rb")
                        last = 0
                        while True:
                            if dl.stop.is_set():
                                raise InterruptedError("stopped")
                            block = gz.read(HASH_BLOCK)
                            if not block:
                                break
                            out.write(block)
                            written += len(block)
                            now = buffered.tell()
                            consumed += now - last
                            last = now
                            if dl.progress:
                                dl.progress.set_status(key, f"{100.0 * consumed / max(total, 1):4.1f}%")
                        consumed += buffered.tell() - last
            os.replace(tmp, dest_file)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        finally:
            if dl.progress:
                dl.progress.set_status(key, None)
        return {"bytes_in": total, "bytes_out": written, "elapsed_s": round(time.monotonic() - t0, 1), "dest": str(dest_file)}

    return Action(name=name, requires=list(sources), run=run, marker=marker, description=description, group=group,
                  verifies=list(verifies or []), outputs=[dest_file])


# --------------------------------------------------------------------------------------
# CLI plumbing shared by the entry scripts
# --------------------------------------------------------------------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_common_args(parser: argparse.ArgumentParser, default_data_dir: Optional[Path] = None) -> None:
    g = parser.add_argument_group("download options")
    g.add_argument("--data-dir", type=Path, default=default_data_dir or repo_root() / "data",
                   help="root data directory (default: <repo>/data)")
    g.add_argument("-j", "--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"parallel connections / processes (default {DEFAULT_WORKERS}). On high-latency links the "
                        f"download speed grows almost linearly with this number")
    g.add_argument("-c", "--connections", type=int, default=None,
                   help="parallel HTTP connections if different from --workers")
    g.add_argument("--chunk-size", type=int, default=DEFAULT_MAX_CHUNK // MiB, metavar="MiB",
                   help="maximum size of one download chunk (default 64; small files use 8 MiB chunks)")
    g.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="retries per range on transient errors")
    g.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="socket timeout in seconds")
    g.add_argument("--dry-run", action="store_true", help="only show what would be downloaded / extracted")
    g.add_argument("--no-verify", action="store_true", help="skip checksum verification (not recommended)")
    g.add_argument("--force-verify", action="store_true", help="re-hash files that were already verified")
    g.add_argument("--no-extract", action="store_true", help="download + verify only, skip merging/extraction")
    g.add_argument("--delete-archives", action="store_true",
                   help="delete archives from tmp/ after successful extraction (default: keep)")
    g.add_argument("--fail-fast", action="store_true", help="stop everything on the first failure")
    g.add_argument("--insecure", action="store_true", help="disable TLS certificate verification")
    g.add_argument("--quiet", action="store_true", help="no progress line")


def run_from_args(args: argparse.Namespace, files: Sequence[FileSpec], actions: Sequence[Action] = ()) -> int:
    set_insecure_tls(bool(getattr(args, "insecure", False)))
    dl = Downloader(workers=args.workers, connections=args.connections or args.workers, max_chunk=args.chunk_size * MiB,
                    retries=args.retries, timeout=args.timeout,
                    verify=not args.no_verify, extract=not args.no_extract, force_verify=args.force_verify,
                    delete_archives=args.delete_archives, quiet=args.quiet, fail_fast=args.fail_fast)
    if not files:
        error("nothing selected")
        return 2
    return dl.run(files, actions, dry_run=args.dry_run)


def filter_groups(items: Iterable, only: Optional[Sequence[str]]) -> list:
    items = list(items)
    if not only:
        return items
    return [it for it in items if it.group in only]


# --------------------------------------------------------------------------------------
# Self test: local range-capable HTTP server with fault injection
# --------------------------------------------------------------------------------------
def _selftest() -> int:  # pragma: no cover - exercised manually
    import random
    import http.server
    import socketserver

    random.seed(1234)
    files: Dict[str, bytes] = {
        "/big.bin": random.randbytes(5 * MiB + 12345),
        "/mid.bin": random.randbytes(2 * MiB + 1),
        "/small.bin": random.randbytes(100 * KiB),
    }
    # split tar.gz
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for i in range(50):
            data = random.randbytes(random.randint(100, 50_000))
            ti = tarfile.TarInfo(f"msa_features/{i:03d}.pkl.gz")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    tgz = buf.getvalue()
    cut = len(tgz) // 2
    files["/feat_aa"], files["/feat_ab"] = tgz[:cut], tgz[cut:]
    # zip with a top-level dir to strip
    zbuf = io.BytesIO()
    zip_payload = {}
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("benchmarks/", b"")
        for i in range(30):
            data = random.randbytes(random.randint(10, 20_000))
            zip_payload[f"sub/{i}.bin"] = data
            zf.writestr(f"benchmarks/sub/{i}.bin", data)
    files["/bench.zip"] = zbuf.getvalue()
    # gz files for gunzip-concat (no checksum -> gzip method deferred to the action)
    raw1, raw2 = b"A" * 300_000 + b"\n", b"B" * 200_000 + b"\n"
    files["/p1.fasta.gz"] = gzip.compress(raw1)
    files["/p2.fasta.gz"] = gzip.compress(raw2)

    faults = {"/big.bin": {"drop_after": 700_000, "n": 4}, "/mid.bin": {"status": 429, "n": 1}, "/feat_aa": {"status": 503, "n": 2}}
    stats = {"requests": 0}

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence
            pass

        def _serve(self, head: bool):
            stats["requests"] += 1
            path = urllib.parse.urlparse(self.path).path
            data = files.get(path)
            if data is None:
                self.send_error(404)
                return
            f = faults.get(path)
            if f and f.get("status") and f["n"] > 0:
                f["n"] -= 1
                self.send_response(f["status"])
                self.send_header("Retry-After", "1")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            rng = self.headers.get("Range")
            start, end = 0, len(data) - 1
            if rng:
                m = re.match(r"bytes=(\d+)-(\d*)", rng)
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), len(data) - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            else:
                self.send_response(200)
            body = data[start:end + 1]
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", f'"{hashlib.md5(data).hexdigest()[:8]}"')
            self.end_headers()
            if head:
                return
            if f and f.get("drop_after") and f["n"] > 0 and len(body) > f["drop_after"]:
                f["n"] -= 1
                self.wfile.write(body[:f["drop_after"]])
                self.wfile.flush()
                self.connection.close()  # simulate network drop mid-transfer
                return
            self.wfile.write(body)

        def do_GET(self):
            self._serve(False)

        def do_HEAD(self):
            self._serve(True)

    class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    srv = Srv(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    tmp_root = Path(tempfile.mkdtemp(prefix="physdock_dl_selftest_"))
    data = tmp_root / "data"
    print(f"selftest: server {base}, data dir {data}")

    def specs():
        v1 = data / "v1"
        hd = data / "homo"
        f_big = FileSpec("big.bin", base + "/big.bin", data / "params" / "big.bin", md5=hashlib.md5(files["/big.bin"]).hexdigest())
        f_mid = FileSpec("mid.bin", base + "/mid.bin", data / "params" / "mid.bin", md5=hashlib.md5(files["/mid.bin"]).hexdigest())
        f_small = FileSpec("small.bin", base + "/small.bin", data / "params" / "small.bin", md5=hashlib.md5(files["/small.bin"]).hexdigest())
        f_aa = FileSpec("feat_aa", base + "/feat_aa", v1 / "tmp" / "feat_aa", md5=hashlib.md5(files["/feat_aa"]).hexdigest())
        f_ab = FileSpec("feat_ab", base + "/feat_ab", v1 / "tmp" / "feat_ab", md5=hashlib.md5(files["/feat_ab"]).hexdigest())
        f_zip = FileSpec("bench.zip", base + "/bench.zip", data / "bench" / "tmp" / "bench.zip", md5=hashlib.md5(files["/bench.zip"]).hexdigest())
        f_p1 = FileSpec("p1.fasta.gz", base + "/p1.fasta.gz", hd / "tmp" / "p1.fasta.gz")  # no checksum -> gzip
        f_p2 = FileSpec("p2.fasta.gz", base + "/p2.fasta.gz", hd / "tmp" / "p2.fasta.gz")
        acts = [
            chained_tar_extract_action("extract_feat", [f_aa, f_ab], v1, v1 / "tmp" / ".extract_feat.done.json"),
            zip_extract_action("extract_bench", f_zip, data / "bench", data / "bench" / "tmp" / ".extract_bench.done.json", strip=1),
            gunzip_concat_action("gunzip_p", [f_p1, f_p2], hd / "p.fasta", hd / "tmp" / ".gunzip_p.done.json", verifies=[f_p1, f_p2]),
        ]
        return [f_big, f_mid, f_small, f_aa, f_ab, f_zip, f_p1, f_p2], acts

    def check_outputs():
        assert (data / "params" / "big.bin").read_bytes() == files["/big.bin"]
        assert (data / "params" / "mid.bin").read_bytes() == files["/mid.bin"]
        assert (data / "params" / "small.bin").read_bytes() == files["/small.bin"]
        n = len(list((data / "v1" / "msa_features").glob("*.pkl.gz")))
        assert n == 50, n
        for rel, payload in zip_payload.items():
            assert (data / "bench" / rel).read_bytes() == payload, rel
        assert not (data / "bench" / "benchmarks").exists()
        assert (data / "homo" / "p.fasta").read_bytes() == raw1 + raw2

    # 1) fresh run with fault injection (connection drops, 429, 503)
    fl, acts = specs()
    rc = Downloader(workers=4, connections=4, min_split=1 * MiB, max_chunk=1 * MiB, retries=10, timeout=10).run(fl, acts)
    assert rc == 0, rc
    check_outputs()
    print(f"selftest: pass 1 OK ({stats['requests']} requests)")

    # 2) rerun: everything should be skipped quickly, no new downloads
    before = stats["requests"]
    fl, acts = specs()
    rc = Downloader(workers=4, connections=4, min_split=1 * MiB, max_chunk=1 * MiB).run(fl, acts)
    assert rc == 0 and stats["requests"] == before, (rc, stats["requests"], before)
    print("selftest: pass 2 (idempotent rerun) OK")

    # 3) simulate an interrupted download: partial .part + sidecar with reduced progress
    big = data / "params" / "big.bin"
    side = load_json(data / "params" / "tmp" / "big.bin.dl.json")
    big.rename(big.with_name("big.bin.part"))
    for p in side["parts"]:
        p[2] = max(0, p[2] - 300_000)
    side["complete"] = False
    side["verified"] = None
    atomic_write_json(data / "params" / "tmp" / "big.bin.dl.json", side)
    fl, acts = specs()
    dlr = Downloader(workers=4, connections=4, min_split=1 * MiB, max_chunk=1 * MiB)
    rc = dlr.run(fl, acts)
    assert rc == 0, rc
    assert dlr.progress.done <= len(side['parts']) * 300_000 + 10, dlr.progress.done
    check_outputs()
    print(f"selftest: pass 3 (resume, re-downloaded only {fmt_bytes(dlr.progress.done)}) OK")

    # 4) corrupt a verified file -> hash mismatch -> automatic re-download
    mid = data / "params" / "mid.bin"
    b = bytearray(mid.read_bytes())
    b[1000] ^= 0xFF
    mid.write_bytes(b)
    fl, acts = specs()
    rc = Downloader(workers=4, connections=4, min_split=1 * MiB, max_chunk=1 * MiB, force_verify=True).run(fl, acts)
    assert rc == 0, rc
    check_outputs()
    print("selftest: pass 4 (corruption detected and repaired) OK")

    # 5) delete archives after extraction, then rerun -> consumed, nothing downloaded
    fl, acts = specs()
    rc = Downloader(workers=4, connections=4, min_split=1 * MiB, max_chunk=1 * MiB, delete_archives=True).run(fl, acts)
    assert rc == 0 and not (data / "v1" / "tmp" / "feat_aa").exists()
    before = stats["requests"]
    fl, acts = specs()
    rc = Downloader(workers=4, connections=4, min_split=1 * MiB, max_chunk=1 * MiB).run(fl, acts)
    assert rc == 0 and stats["requests"] == before, (rc, stats["requests"], before)
    check_outputs()
    print("selftest: pass 5 (--delete-archives + consumed state) OK")

    # 6) remote file changed (new ETag/size) while a partial download exists -> restart from scratch
    files["/small.bin"] = random.randbytes(120 * KiB)
    small = data / "params" / "small.bin"
    small.rename(small.with_name("small.bin.part"))   # stale partial download of the OLD file
    side = load_json(data / "params" / "tmp" / "small.bin.dl.json")
    side["complete"] = False
    side["verified"] = None
    side["parts"] = [[0, side["size"] - 1, 1000]]
    atomic_write_json(data / "params" / "tmp" / "small.bin.dl.json", side)
    fl, acts = specs()
    rc = Downloader(workers=4, connections=4, min_split=1 * MiB, max_chunk=1 * MiB).run(fl, acts)
    assert rc == 0
    assert small.read_bytes() == files["/small.bin"]
    print("selftest: pass 6 (changed remote file) OK")

    # 7) 404 is fatal for that file but the rest proceeds
    fl, acts = specs()
    fl.append(FileSpec("missing.bin", base + "/nope", data / "params" / "missing.bin"))
    rc = Downloader(workers=4, connections=4, min_split=1 * MiB, max_chunk=1 * MiB).run(fl, acts)
    assert rc == 1
    print("selftest: pass 7 (404 handling) OK")

    srv.shutdown()
    shutil.rmtree(tmp_root, ignore_errors=True)
    print("selftest: ALL OK")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run the built-in self test")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    ap.print_help()
