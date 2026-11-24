import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_meihua.quota import QuotaExceeded, QuotaStore


class QuotaTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "data" / "quota.db"
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.store = self.new_store()

    def new_store(self):
        return QuotaStore(self.path, now=lambda: self.now)

    def consume(self, store=None, **overrides):
        kwargs = dict(
            platform="platform-1",
            bot="bot-1",
            user="user-1",
            group="group-1",
            user_limit=1,
            group_limit=1,
        )
        kwargs.update(overrides)
        return (self.store if store is None else store).consume(**kwargs)

    def test_user_quota_spans_groups_and_private(self):
        self.assertIsNone(self.consume(user_limit=3))
        self.consume(user_limit=3, group="group-2")
        self.consume(user_limit=3, group="")
        with self.assertRaisesRegex(QuotaExceeded, "个人"):
            self.consume(user_limit=3, group="group-3")

    def test_group_quota_is_shared_by_members(self):
        self.consume()
        with self.assertRaisesRegex(QuotaExceeded, "群聊"):
            self.consume(user="user-2")
        self.consume(user="user-2", group="group-2")

    def test_private_requests_skip_group_quota(self):
        self.consume(group="")
        self.consume(group="", user="user-2")
        with self.assertRaisesRegex(QuotaExceeded, "个人"):
            self.consume(group="")

    def test_platform_and_bot_isolate_both_quotas(self):
        for identity in ({}, {"platform": "platform-2"}, {"bot": "bot-2"}):
            with self.subTest(identity=identity):
                self.consume(**identity)
                with self.assertRaisesRegex(QuotaExceeded, "个人"):
                    self.consume(**identity)
                with self.assertRaisesRegex(QuotaExceeded, "群聊"):
                    self.consume(user="user-2", **identity)

    def test_zero_limits_allow_calls_and_keep_counts(self):
        for _ in range(3):
            self.consume(user_limit=0, group_limit=0)
        with self.assertRaisesRegex(QuotaExceeded, "个人"):
            self.consume(user_limit=3, group_limit=0)
        with self.assertRaisesRegex(QuotaExceeded, "群聊"):
            self.consume(user_limit=0, group_limit=3)

    def test_invalid_limits_raise_even_in_private(self):
        for name in ("user_limit", "group_limit"):
            for limit in (-1, 1.0, "1", None, False, True):
                with self.subTest(name=name, limit=limit):
                    with self.assertRaisesRegex(ValueError, name):
                        self.consume(group="", **{name: limit})
        self.consume(group="")

    def test_counts_persist_in_a_new_store(self):
        self.consume()
        restarted = self.new_store()
        with self.assertRaisesRegex(QuotaExceeded, "个人"):
            self.consume(restarted)
        with self.assertRaisesRegex(QuotaExceeded, "群聊"):
            self.consume(restarted, user="user-2")

    def test_both_quotas_reset_at_utc_plus_eight_midnight(self):
        self.now = datetime(2026, 1, 1, 15, 59, 59, tzinfo=timezone.utc)
        self.consume()
        with self.assertRaisesRegex(QuotaExceeded, "个人"):
            self.consume()
        with self.assertRaisesRegex(QuotaExceeded, "群聊"):
            self.consume(user="user-2")

        self.now = datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc)
        self.consume()

    def test_rejection_leaves_other_bucket_unchanged(self):
        self.consume()
        with self.assertRaisesRegex(QuotaExceeded, "群聊"):
            self.consume(user="user-2")
        with self.assertRaisesRegex(QuotaExceeded, "个人"):
            self.consume(group="group-2")
        self.consume(user="user-2", group="group-2")

    def test_concurrent_requests_share_the_last_group_slot(self):
        self.consume(group_limit=2)
        ready = Barrier(2, timeout=5)

        def attempt(user):
            store = self.new_store()
            ready.wait()
            try:
                self.consume(store, user=user, group_limit=2)
            except QuotaExceeded as exc:
                return str(exc)
            return "accepted"

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, ("user-2", "user-3")))
        self.assertEqual(results.count("accepted"), 1)
        rejection = next(result for result in results if result != "accepted")
        self.assertIn("群聊", rejection)
