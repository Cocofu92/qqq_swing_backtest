"""IG REST API client — read-only.

Auth flow (V2 session):
    POST {base}/session
    Headers: X-IG-API-KEY, Version: 2, Content-Type, Accept
    Body:    {"identifier": <username>, "password": <password>}
    -> Response headers carry CST and X-SECURITY-TOKEN; body has
       currentAccountId / accounts.

Read-only by design. The following endpoints are deliberately NOT
implemented because Adam's standing rule forbids any trading op from
this codebase:
    POST   /positions/otc
    PUT    /positions/otc/{dealId}
    DELETE /positions/otc
    POST   /workingorders/otc
    PUT    /workingorders/otc/{dealId}
    DELETE /workingorders/otc/{dealId}

Endpoint versions used:
    /accounts                 v1
    /positions, /positions/{} v2
    /workingorders            v2
    /history/activity         v3
    /history/transactions     v2
    /markets/{epic}           v3
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Any, Optional

import requests

log = logging.getLogger("ig_client")

LIVE_BASE = "https://api.ig.com/gateway/deal"
DEMO_BASE = "https://demo-api.ig.com/gateway/deal"


class IGError(RuntimeError):
    """Raised for unrecoverable IG REST errors."""


class _TokenBucket:
    """Simple sliding-window rate limiter (default 30 req/min)."""

    def __init__(self, max_calls: int = 30, window_s: float = 60.0):
        self.max_calls = max_calls
        self.window_s = window_s
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= self.window_s:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                sleep_for = self.window_s - (now - self._calls[0]) + 0.05
                if sleep_for > 0:
                    log.debug("rate limit: sleeping %.2fs", sleep_for)
                    print(f"  [ig] rate-limit sleep {sleep_for:.1f}s", flush=True)
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.window_s:
                    self._calls.popleft()
            self._calls.append(now)


class IGClient:
    """Read-only IG REST client.

    Tokens (CST / X-SECURITY-TOKEN) are kept in memory only — never
    written to disk or logs. Password is held only on the instance for
    the lifetime of the process.
    """

    def __init__(
        self,
        api_key: str,
        username: str,
        password: str,
        account_type: str = "LIVE",
        timeout: float = 30.0,
        user_agent: str = "ig-show-book/0.1",
    ):
        if not api_key or not username or not password:
            raise IGError("api_key, username and password are all required")
        self._api_key = api_key
        self._username = username
        self._password = password
        self.account_type = (account_type or "LIVE").upper()
        self.base = DEMO_BASE if self.account_type == "DEMO" else LIVE_BASE
        self.timeout = timeout
        self.user_agent = user_agent
        self._cst: Optional[str] = None
        self._xst: Optional[str] = None
        self.account_id: Optional[str] = None
        self.session_info: dict[str, Any] = {}
        self._rate = _TokenBucket(max_calls=30, window_s=60.0)
        self._sess = requests.Session()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _base_headers(self, version: str = "1") -> dict[str, str]:
        h = {
            "X-IG-API-KEY": self._api_key,
            "Accept": "application/json; charset=UTF-8",
            "Content-Type": "application/json; charset=UTF-8",
            "Version": str(version),
            "User-Agent": self.user_agent,
        }
        if self._cst and self._xst:
            h["CST"] = self._cst
            h["X-SECURITY-TOKEN"] = self._xst
        return h

    def _request(
        self,
        method: str,
        path: str,
        version: str = "1",
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
        _is_retry: bool = False,
    ) -> Any:
        if method.upper() != "GET" and not path.startswith("/session"):
            # Defensive guard: this client is read-only.
            raise IGError(
                f"refusing non-GET to {path}: this client is read-only"
            )

        url = f"{self.base}{path}"
        headers = self._base_headers(version=version)
        attempts = 0
        max_429_retries = 3
        while True:
            self._rate.acquire()
            try:
                resp = self._sess.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    data=json.dumps(json_body) if json_body is not None else None,
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                raise IGError(f"network error on {method} {path}: {e}") from e

            if resp.status_code in (200, 201, 204):
                if resp.status_code == 204 or not resp.content:
                    return {}
                try:
                    return resp.json()
                except ValueError:
                    return resp.text

            body_text = resp.text or ""
            if resp.status_code in (401, 403) and not _is_retry and not path.startswith("/session"):
                # Token may have expired — re-login once and retry.
                log.info("ig %s on %s -> re-login and retry", resp.status_code, path)
                self._cst = None
                self._xst = None
                self.login()
                return self._request(
                    method, path, version=version, params=params,
                    json_body=json_body, _is_retry=True,
                )

            if resp.status_code == 429:
                if attempts >= max_429_retries:
                    raise IGError(
                        f"HTTP 429 on {method} {path} after {attempts} retries: "
                        f"{body_text[:300]}"
                    )
                wait = (2 ** attempts) * 2.0 + 1.0
                log.warning("ig 429 on %s -> backoff %.1fs", path, wait)
                print(f"  [ig] 429 backoff {wait:.1f}s on {path}", flush=True)
                time.sleep(wait)
                attempts += 1
                continue

            if 500 <= resp.status_code < 600:
                if attempts < 1:
                    log.warning("ig %s on %s -> retry in 5s", resp.status_code, path)
                    time.sleep(5.0)
                    attempts += 1
                    continue
                raise IGError(
                    f"HTTP {resp.status_code} on {method} {path}: {body_text[:300]}"
                )

            # Other 4xx
            raise IGError(
                f"HTTP {resp.status_code} on {method} {path}: {body_text[:500]}"
            )

    # ------------------------------------------------------------------
    # session
    # ------------------------------------------------------------------

    def login(self) -> None:
        """POST /session V2. Caches CST + X-SECURITY-TOKEN in memory."""
        url = f"{self.base}/session"
        headers = {
            "X-IG-API-KEY": self._api_key,
            "Accept": "application/json; charset=UTF-8",
            "Content-Type": "application/json; charset=UTF-8",
            "Version": "2",
            "User-Agent": self.user_agent,
        }
        body = {"identifier": self._username, "password": self._password}
        self._rate.acquire()
        try:
            resp = self._sess.post(
                url, headers=headers, data=json.dumps(body), timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise IGError(f"network error on login: {e}") from e

        if resp.status_code != 200:
            # Sanitise body so it never leaks creds even if IG echoes them
            snippet = (resp.text or "")[:300]
            raise IGError(f"login failed: HTTP {resp.status_code}: {snippet}")

        cst = resp.headers.get("CST")
        xst = resp.headers.get("X-SECURITY-TOKEN")
        if not cst or not xst:
            raise IGError("login: missing CST or X-SECURITY-TOKEN in response")
        self._cst = cst
        self._xst = xst
        try:
            data = resp.json()
        except ValueError:
            data = {}
        self.session_info = data
        self.account_id = data.get("currentAccountId")
        log.info(
            "ig login ok: account=%s type=%s",
            self.account_id, self.account_type,
        )

    def logout(self) -> None:
        """DELETE /session. Best-effort; clears tokens regardless."""
        if not self._cst or not self._xst:
            return
        url = f"{self.base}/session"
        headers = self._base_headers(version="1")
        try:
            self._rate.acquire()
            self._sess.delete(url, headers=headers, timeout=self.timeout)
        except requests.RequestException as e:
            log.info("ig logout: network error (ignored): %s", e)
        finally:
            self._cst = None
            self._xst = None
            self.account_id = None

    def __enter__(self) -> "IGClient":
        self.login()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.logout()

    # ------------------------------------------------------------------
    # read endpoints
    # ------------------------------------------------------------------

    def get_accounts(self) -> dict:
        """GET /accounts (v1) -> {"accounts": [...]}"""
        return self._request("GET", "/accounts", version="1")

    def get_positions(self) -> list[dict]:
        """GET /positions (v2) -> list of position dicts."""
        data = self._request("GET", "/positions", version="2")
        return list(data.get("positions") or [])

    def get_position(self, deal_id: str) -> dict:
        """GET /positions/{dealId} (v2)."""
        if not deal_id:
            raise IGError("get_position: deal_id required")
        return self._request("GET", f"/positions/{deal_id}", version="2")

    def get_working_orders(self) -> list[dict]:
        """GET /workingorders (v2)."""
        data = self._request("GET", "/workingorders", version="2")
        return list(data.get("workingOrders") or [])

    @staticmethod
    def _fmt_activity_date(value: str) -> str:
        """Coerce an ISO datetime into IG's expected /history/activity v3
        format: yyyy-MM-dd'T'HH:mm:ss with NO timezone suffix and NO
        millis. IG silently returns 0 events if a `Z` or `+00:00` is
        appended.
        """
        if value is None:
            return ""
        s = str(value).strip()
        # Strip a trailing Z (UTC indicator) if present.
        if s.endswith("Z") or s.endswith("z"):
            s = s[:-1]
        # Strip an explicit +00:00 / -00:00 offset if present.
        if len(s) >= 6 and s[-6] in "+-" and s[-3] == ":":
            s = s[:-6]
        # Strip fractional seconds if present.
        if "." in s:
            head, _, _frac = s.partition(".")
            s = head
        return s

    def get_activity(self, from_date: str, to_date: str) -> list[dict]:
        """GET /history/activity (v3).

        Dates are ISO 8601 (YYYY-MM-DDTHH:MM:SS, NO timezone) per IG.
        Anything else (Z suffix, +00:00, millis) is silently rejected by
        IG and returns an empty `activities` list — so we sanitise here.

        The endpoint returns paginated results — we follow paging.next
        links until exhausted.
        """
        path = "/history/activity"
        params = {
            "from": self._fmt_activity_date(from_date),
            "to": self._fmt_activity_date(to_date),
            "detailed": "true",
            "pageSize": 500,
        }
        # One-shot diagnostic of the actual request shape so empty
        # responses can be debugged without enabling --debug-activity.
        try:
            print(
                f"DIAG: GET {self.base}{path} params={params} version=3",
                flush=True,
            )
        except Exception:
            pass
        all_rows: list[dict] = []
        next_path: Optional[str] = None
        while True:
            if next_path is None:
                data = self._request("GET", path, version="3", params=params)
            else:
                data = self._request("GET", next_path, version="3")
            all_rows.extend(data.get("activities") or [])
            metadata = data.get("metadata") or {}
            paging = metadata.get("paging") or {}
            nxt = paging.get("next")
            if not nxt:
                break
            # next is typically "/gateway/deal/history/activity?..."; strip prefix
            next_path = nxt.replace("/gateway/deal", "", 1) if nxt.startswith("/gateway/deal") else nxt
            # Defensive cap
            if len(all_rows) > 5000:
                break
        return all_rows

    def get_transactions(
        self,
        from_date: str,
        to_date: str,
        page_size: int = 500,
    ) -> list[dict]:
        """GET /history/transactions (v2).

        Dates are ISO 8601 (YYYY-MM-DDTHH:MM:SS, NO timezone) per IG —
        same quirk as /history/activity (a `Z` or `+00:00` suffix returns
        an empty result silently). We sanitise here so callers can pass
        whatever they like.

        Paging follows `metadata.pageData.totalPages`. We default to the
        max pageSize of 500 to minimise round-trips for the deposits
        sync (transactions are infrequent, so one page is normally
        enough).
        """
        path = "/history/transactions"
        params = {
            "from": self._fmt_activity_date(from_date),
            "to": self._fmt_activity_date(to_date),
            "pageSize": int(page_size),
            "pageNumber": 1,
        }
        all_rows: list[dict] = []
        page = 1
        while True:
            params["pageNumber"] = page
            data = self._request("GET", path, version="2", params=params)
            all_rows.extend(data.get("transactions") or [])
            metadata = data.get("metadata") or {}
            paging = metadata.get("pageData") or {}
            total_pages = int(paging.get("totalPages") or 1)
            if page >= total_pages:
                break
            page += 1
            if page > 100:
                break
        return all_rows

    def get_market(self, epic: str) -> dict:
        """GET /markets/{epic} (v3)."""
        if not epic:
            raise IGError("get_market: epic required")
        return self._request("GET", f"/markets/{epic}", version="3")


    # ------------------------------------------------------------------
    # market discovery + historical prices  (added for backtesting)
    # ------------------------------------------------------------------

    def search_markets(self, search_term: str) -> list[dict]:
        """GET /markets?searchTerm=...   -> list of {epic, instrumentName, ...}"""
        if not search_term:
            raise IGError("search_markets: search_term required")
        data = self._request("GET", "/markets", version="1",
                             params={"searchTerm": search_term})
        return data.get("markets") or []

    def get_prices(
        self,
        epic: str,
        resolution: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_points: int = 10000,
        page_size: int = 0,
    ) -> dict:
        """GET /prices/{epic} (v3) — historical candles.

        resolution: SECOND, MINUTE, MINUTE_2, MINUTE_3, MINUTE_5, MINUTE_10,
                    MINUTE_15, MINUTE_30, HOUR, HOUR_2, HOUR_3, HOUR_4,
                    DAY, WEEK, MONTH
        start / end: yyyy-MM-dd'T'HH:mm:ss (no timezone — IG treats as UTC)
        max_points: cap (IG returns at most ~10k per call)
        page_size: 0 = single page (IG default; honour max_points instead)

        Returns the raw IG response: {prices: [...], allowance: {...}, metadata: {...}}
        """
        if not epic or not resolution:
            raise IGError("get_prices: epic and resolution required")
        params = {"resolution": resolution, "max": str(max_points), "pageSize": str(page_size)}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        return self._request("GET", f"/prices/{epic}", version="3", params=params)

    def get_allowance(self) -> dict:
        """Make a tiny /prices call to read remaining historical-data allowance.
        IG returns allowance/{remainingAllowance, totalAllowance, allowanceExpiry}
        as part of every /prices response. We piggyback on a 1-point fetch for
        a known ETF epic. Caller can also just check `allowance` from any
        get_prices response.
        """
        # Small probe: one daily bar, doesn't burn meaningful allowance.
        try:
            data = self._request("GET", "/prices/IX.D.NASDAQ.IFD.IP", version="3",
                                 params={"resolution": "DAY", "max": "1"})
            return data.get("allowance") or {}
        except IGError:
            return {}


__all__ = ["IGClient", "IGError"]
