"""Persistent daily quotas for initiated model requests."""

import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

_QUOTA_TZ = timezone(timedelta(hours=8))
_EXHAUSTED = {
    "user": "今日个人解卦次数已达上限，请明天再试。",
    "group": "今日群聊解卦次数已达上限，请明天再试。",
}


class QuotaExceeded(Exception):
    """A daily quota was exhausted."""


class QuotaStore:
    def __init__(self, path: Path, *, now: Callable[[], datetime] | None = None):
        self._path = path
        self._now = now if now is not None else lambda: datetime.now(_QUOTA_TZ)

    def consume(
        self,
        *,
        platform: str,
        bot: str,
        user: str,
        group: str,
        user_limit: int,
        group_limit: int,
    ) -> None:
        """Count one request atomically. An empty group denotes a private chat."""
        for name, limit in (("user_limit", user_limit), ("group_limit", group_limit)):
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                raise ValueError(f"{name} 必须是非负整数（0 表示不限）。")

        buckets = [("user", user, user_limit)]
        if group:
            buckets.append(("group", group, group_limit))

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path, timeout=10)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            now = self._now()
            if now.utcoffset() is None:
                raise ValueError(
                    "The quota clock must return a timezone-aware datetime."
                )
            day = now.astimezone(_QUOTA_TZ).date().isoformat()
            db.execute("""
                CREATE TABLE IF NOT EXISTS quota (
                    day TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    bot TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (day, platform, bot, scope, scope_id)
                )
            """)
            for scope, scope_id, limit in buckets:
                key = (day, platform, bot, scope, scope_id)
                if limit:
                    row = db.execute(
                        "SELECT count FROM quota WHERE day = ? AND platform = ? AND bot = ? "
                        "AND scope = ? AND scope_id = ?",
                        key,
                    ).fetchone()
                    if row is not None and row[0] >= limit:
                        raise QuotaExceeded(_EXHAUSTED[scope])
                db.execute(
                    "INSERT INTO quota (day, platform, bot, scope, scope_id, count) "
                    "VALUES (?, ?, ?, ?, ?, 1) "
                    "ON CONFLICT (day, platform, bot, scope, scope_id) "
                    "DO UPDATE SET count = count + 1",
                    key,
                )
