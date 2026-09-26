"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store a request start record and return its correlation id."""
        correlation_id = request_id or str(uuid4())
        self.logs.append(
            {
                "request_id": correlation_id,
                "user_id": user_id,
                "input": text,
                "started_at": utc_now_iso(),
                "blocked": False,
                "layer": None,
            }
        )
        self._open[correlation_id] = monotonic()
        return correlation_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete the matching request with its result and latency."""
        correlation_id = request_id
        if correlation_id is None:
            correlation_id = next(
                (
                    entry["request_id"]
                    for entry in reversed(self.logs)
                    if entry["user_id"] == user_id
                    and entry.get("completed_at") is None
                ),
                None,
            )

        started = self._open.pop(correlation_id, None) if correlation_id else None
        latency_ms = round((monotonic() - started) * 1000, 3) if started is not None else None
        entry = next(
            (
                row for row in reversed(self.logs)
                if correlation_id and row["request_id"] == correlation_id
            ),
            None,
        )
        if entry is None:
            entry = {
                "request_id": correlation_id or str(uuid4()),
                "user_id": user_id,
                "input": None,
                "started_at": None,
            }
            self.logs.append(entry)
        entry.update(
            {
                "output": text,
                "blocked": bool(blocked),
                "layer": layer,
                "latency_ms": latency_ms,
                "completed_at": utc_now_iso(),
            }
        )
        return entry["request_id"]

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
