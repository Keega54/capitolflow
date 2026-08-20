"""Polite HTTP session: retries with backoff, on-disk caching, rate limiting."""
from __future__ import annotations
import hashlib, logging, time, threading
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import SETTINGS

log = logging.getLogger(__name__)
_lock = threading.Lock()
_last_call = {"t": 0.0}


class HostUnavailable(RuntimeError):
    """Raised instead of re-attempting a host that is already known to be down."""


class _CircuitBreaker:
    """Stop hammering a host that is clearly unreachable.

    A run may fetch several hundred PDFs from one server. If that server is down
    or blocked, every request burns its full retry-and-backoff budget — roughly
    20 seconds each — and a job that should fail in a minute instead grinds for
    hours and gets killed by the scheduler. After `threshold` consecutive
    connection failures a host is tripped and further requests fail instantly,
    until `reset_after` seconds have passed and one probe is allowed through.
    """

    def __init__(self, threshold: int = 2, reset_after: float = 300.0):
        self.threshold = threshold
        self.reset_after = reset_after
        self._fails: dict[str, int] = {}
        self._tripped_at: dict[str, float] = {}
        self._lk = threading.Lock()

    @staticmethod
    def _host(url: str) -> str:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower()

    def check(self, url: str) -> None:
        h = self._host(url)
        with self._lk:
            t = self._tripped_at.get(h)
            if t is None:
                return
            if time.time() - t < self.reset_after:
                raise HostUnavailable(
                    f"{h} unreachable after {self._fails.get(h, 0)} consecutive failures; "
                    f"skipping for {int(self.reset_after - (time.time() - t))}s")
            # half-open: let a single probe through
            self._tripped_at.pop(h, None)
            self._fails[h] = self.threshold - 1

    def record_failure(self, url: str) -> None:
        h = self._host(url)
        with self._lk:
            n = self._fails.get(h, 0) + 1
            self._fails[h] = n
            if n >= self.threshold and h not in self._tripped_at:
                self._tripped_at[h] = time.time()
                log.error("circuit breaker tripped for %s after %d consecutive failures", h, n)

    def record_success(self, url: str) -> None:
        h = self._host(url)
        with self._lk:
            self._fails.pop(h, None)
            self._tripped_at.pop(h, None)

    def reset(self) -> None:
        with self._lk:
            self._fails.clear()
            self._tripped_at.clear()


BREAKER = _CircuitBreaker()


def make_session(extra_headers: dict | None = None) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": SETTINGS.user_agent,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    if extra_headers:
        s.headers.update(extra_headers)
    # Retry throttling and server errors patiently; do NOT retry a connection
    # failure four times — an unreachable host or a blocking proxy will not fix
    # itself, and backing off through every attempt turns an offline run into a
    # multi-minute hang instead of a fast, clear failure.
    retry = Retry(
        total=SETTINGS.max_retries,
        connect=1,
        read=2,
        status=SETTINGS.max_retries,
        backoff_factor=1.0,
        backoff_max=20,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "HEAD"]),
        respect_retry_after_header=True,
    )
    ad = HTTPAdapter(max_retries=retry, pool_maxsize=8)
    s.mount("https://", ad)
    s.mount("http://", ad)
    return s


def throttle() -> None:
    with _lock:
        wait = SETTINGS.request_delay_s - (time.time() - _last_call["t"])
        if wait > 0:
            time.sleep(wait)
        _last_call["t"] = time.time()


def _cache_path(url: str, suffix: str) -> Path:
    SETTINGS.ensure_dirs()
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    return SETTINGS.cache_dir / f"{h}{suffix}"


def get_bytes(session: requests.Session, url: str, *, cache: bool = True,
              suffix: str = ".bin", max_age_s: float | None = None, **kw) -> bytes:
    """GET returning raw bytes, with an optional on-disk cache.

    Immutable artifacts (a filed PDF never changes) should use cache=True with no
    max_age; search endpoints should pass max_age_s or cache=False.
    """
    p = _cache_path(url, suffix)
    if cache and p.exists():
        if max_age_s is None or (time.time() - p.stat().st_mtime) < max_age_s:
            return p.read_bytes()
    BREAKER.check(url)
    throttle()
    try:
        r = session.get(url, timeout=(SETTINGS.connect_timeout_s, SETTINGS.timeout_s), **kw)
        r.raise_for_status()
    except requests.exceptions.RequestException:
        BREAKER.record_failure(url)
        raise
    BREAKER.record_success(url)
    if cache:
        p.write_bytes(r.content)
    return r.content


def get_text(session: requests.Session, url: str, **kw) -> str:
    return get_bytes(session, url, suffix=".txt", **kw).decode("utf-8", errors="replace")


def get_json(session: requests.Session, url: str, **kw):
    import json
    return json.loads(get_bytes(session, url, suffix=".json", **kw))


def post(session: requests.Session, url: str, data=None, **kw):
    BREAKER.check(url)
    throttle()
    try:
        r = session.post(url, data=data,
                         timeout=(SETTINGS.connect_timeout_s, SETTINGS.timeout_s), **kw)
        r.raise_for_status()
    except requests.exceptions.RequestException:
        BREAKER.record_failure(url)
        raise
    BREAKER.record_success(url)
    return r
