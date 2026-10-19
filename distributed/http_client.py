from __future__ import annotations

import threading
import time
from typing import Any

import requests


class CoordinatorHttpClient:
    def __init__(self, base_url: str, token: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {token}"}
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def request(self, method: str, path: str, *, retries: int = 3, **kwargs: Any) -> dict:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.session.request(
                    method,
                    f"{self.base_url}{path}",
                    headers=self.headers,
                    timeout=self.timeout,
                    **kwargs,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(min(2 ** attempt, 5))
        raise ConnectionError(f"coordinator request failed: {method} {path}: {last_error}")

    def post(self, path: str, payload: dict, *, retries: int = 3) -> dict:
        return self.request("POST", path, json=payload, retries=retries)

    def get(self, path: str, *, retries: int = 3) -> dict:
        return self.request("GET", path, retries=retries)


class HttpEventSink:
    """Queue-compatible adapter used by the existing local trainer."""

    def __init__(self, client: CoordinatorHttpClient):
        self.client = client

    def put(self, item: dict) -> None:
        kind = item.get("kind")
        if kind == "heartbeat":
            self.client.post("/heartbeat", item)
        elif kind == "telemetry":
            self.client.post("/telemetry", item["telemetry"])
        elif kind == "event":
            self.client.post("/event", item)
        else:
            raise ValueError(f"unsupported network event kind: {kind}")
