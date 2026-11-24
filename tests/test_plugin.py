import asyncio
import importlib
import logging
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_core import CLIPBOARD_SAMPLE

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
PACKAGE = Path(__file__).resolve().parents[1].name


class Client:
    api_key = "synthetic-key"

    def with_options(self):
        return Client()


class FakeProvider:
    def __init__(self):
        self.provider_config = {
            "type": "openai_chat_completion",
            "model": "test-model",
            "custom_extra_body": {"reasoning_effort": "low", "temperature": 0.5},
        }
        self.api_keys = ["synthetic-key"]
        self.client = Client()
        self.calls = []
        self.failure = None
        self.answer = "分层解读：合成测试正文"

    async def text_chat(self, **kwargs):
        await asyncio.sleep(0)
        self.calls.append((kwargs, deepcopy(self.provider_config)))
        self.client.api_key = "request-local-key"
        if self.failure:
            raise self.failure
        return SimpleNamespace(completion_text=self.answer, reasoning_content="HIDDEN")

    def _ollama_disable_thinking_enabled(self):
        return bool(self.provider_config.get("ollama_disable_thinking"))


class NativeProvider:
    instances = []

    def __init__(self, config, settings):
        self.provider_config = config
        self.provider_settings = settings
        self.client = Client()
        self.closed = False
        self.calls = []
        self.instances.append(self)

    def get_model(self):
        return "synthetic-native-model"

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        self.client = Client()
        self.client.api_key = "rotated-local-key"
        if self.provider_settings.get("fail"):
            raise RuntimeError("synthetic failure")
        return SimpleNamespace(completion_text="native-ok")

    async def terminate(self):
        self.closed = True


class NativeAnthropic(NativeProvider):
    pass


class NativeGemini(NativeProvider):
    pass


class Plain:
    def __init__(self, text):
        self.text = text


def stub_modules():
    modules = {
        name: ModuleType(name)
        for name in (
            "astrbot",
            "astrbot.api",
            "astrbot.api.event",
            "astrbot.api.message_components",
            "astrbot.api.star",
            "astrbot.core",
            "astrbot.core.star",
            "astrbot.core.star.session_llm_manager",
            "astrbot.core.provider",
            "astrbot.core.provider.sources",
            "astrbot.core.provider.sources.openai_source",
            "astrbot.core.provider.sources.anthropic_source",
            "astrbot.core.provider.sources.gemini_source",
        )
    }
    modules["astrbot.api"].AstrBotConfig = dict
    modules["astrbot.api"].logger = logging.getLogger("meihua-test")
    modules["astrbot.api.event"].AstrMessageEvent = object
    modules["astrbot.api.event"].filter = SimpleNamespace(
        command=lambda *args, **kwargs: lambda function: function,
        on_llm_request=lambda **kwargs: lambda function: function,
        on_decorating_result=lambda **kwargs: lambda function: function,
    )
    modules["astrbot.api.message_components"].Plain = Plain
    modules[
        "astrbot.core.star.session_llm_manager"
    ].SessionServiceManager = SimpleNamespace(
        should_process_llm_request=AsyncMock(return_value=True)
    )

    class Star:
        def __init__(self, context):
            self.context = context

    modules["astrbot.api.star"].Star = Star
    modules["astrbot.api.star"].Context = object
    modules["astrbot.api.star"].StarTools = SimpleNamespace()
    modules["astrbot.api.star"].register = lambda *args: lambda cls: cls
    modules[
        "astrbot.core.provider.sources.openai_source"
    ].ProviderOpenAIOfficial = FakeProvider
    modules[
        "astrbot.core.provider.sources.anthropic_source"
    ].ProviderAnthropic = NativeAnthropic
    modules[
        "astrbot.core.provider.sources.gemini_source"
    ].ProviderGoogleGenAI = NativeGemini
    return modules


class Event:
    def __init__(
        self,
        text,
        *,
        origin="test:GroupMessage:group-1",
        sender="user-1",
        bot="bot-1",
        role="member",
    ):
        self.text = text
        self.unified_msg_origin = origin
        self.sender = sender
        self.bot = bot
        self.role = role
        self.sent = []
        self.stopped = False
        self.call_llm = False
        self.extras = {}
        self.result = None

    def get_message_str(self):
        return self.text

    def get_sender_id(self):
        return self.sender

    def get_self_id(self):
        return self.bot

    def get_platform_id(self):
        return self.unified_msg_origin.split(":", 1)[0]

    def get_group_id(self):
        parts = self.unified_msg_origin.split(":", 2)
        return parts[2] if parts[1] == "GroupMessage" else ""

    def should_call_llm(self, value):
        self.call_llm = value

    def stop_event(self):
        self.stopped = True

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def get_result(self):
        return self.result

    def plain_result(self, text):
        return text

    async def send(self, result):
        self.sent.append(result)


class PluginTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, stub_modules())
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.main = importlib.import_module(PACKAGE + ".main")
        self.llm = importlib.import_module(PACKAGE + ".llm")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        data_dir = Path(temporary.name)
        star_tools = patch.object(
            self.main, "StarTools", SimpleNamespace(get_data_dir=lambda _: data_dir)
        )
        star_tools.start()
        self.addCleanup(star_tools.stop)
        self.provider = FakeProvider()
        self.manager = SimpleNamespace(
            get_provider_by_id=AsyncMock(return_value=self.provider)
        )
        self.chat_config = {"agent_runner": {"runner_type": "local"}}
        self.context = SimpleNamespace(
            provider_manager=self.manager, get_config=lambda **kwargs: self.chat_config
        )
        self.plugin = self.main.MeihuaPlugin(
            self.context,
            {"provider_id": "selected-model", "reasoning_effort": "xhigh"},
        )

    async def test_multiline_command_full_references_and_no_history(self):
        event = Event("完整起卦 占问: 合成测试事项\n本卦: 水泽节\n变卦: 坎为水")
        await self.plugin.full_qigua(event)
        self.assertTrue(event.stopped)
        self.assertTrue(event.call_llm)
        self.manager.get_provider_by_id.assert_awaited_once_with("selected-model")
        kwargs, config = self.provider.calls[0]
        self.assertEqual(kwargs["contexts"], [])
        self.assertIsNone(kwargs["session_id"])
        self.assertIsNone(kwargs["func_tool"])
        self.assertIn("合成测试事项", kwargs["prompt"])
        self.assertEqual(config["custom_extra_body"]["reasoning_effort"], "xhigh")
        output = "".join(event.sent)
        for name in ("水泽节", "山雷颐", "坎为水"):
            self.assertIn(name, output)
        for field in (
            "象意",
            "签名与等级",
            "卦辞",
            "推断",
            "大象",
            "运势",
            "爱情",
            "疾病",
            "失物",
            "诉讼",
        ):
            self.assertEqual(output.count("• " + field + "："), 3)
        self.assertTrue(output.endswith("分层解读：合成测试正文"))
        self.assertNotIn("HIDDEN", output)

    async def test_clipboard_command_cleans_markdown_before_splitting(self):
        event = Event("完整起卦 " + CLIPBOARD_SAMPLE)
        answer = "### 分层解读\n\n**本卦**：" + "保留正文。" * 500 + "\n\n1. 建议"
        with patch.object(
            self.main, "interpret", AsyncMock(return_value=answer)
        ) as call:
            await self.plugin.full_qigua(event)
        self.assertTrue(event.stopped)
        self.assertTrue(event.call_llm)
        self.assertIn(CLIPBOARD_SAMPLE, call.await_args.args[3])
        from astrbot_plugin_meihua.core import prepare_reading
        from astrbot_plugin_meihua.formatting import plain_text

        reading = prepare_reading(CLIPBOARD_SAMPLE)
        expected = "".join(
            map(plain_text, (reading.summary, reading.references, answer))
        )
        output = "".join(event.sent)
        self.assertEqual(output, expected)
        self.assertNotIn("###", output)
        self.assertNotIn("**", output)
        self.assertNotIn("`", output)
        self.assertTrue(all(len(part) <= 1800 for part in event.sent))

    async def test_missing_input_does_not_call_model_and_has_no_pending_state(self):
        event = Event(
            "完整起卦 占问: 合成事项 2026 年预算 5000\n时间: 2026年11月22日17时"
        )
        await self.plugin.full_qigua(event)
        self.assertFalse(self.provider.calls)
        self.assertTrue(event.sent)
        self.assertTrue(event.stopped)
        next_event = Event("完整起卦 11 12")
        await self.plugin.full_qigua(next_event)
        self.assertNotIn("5000", self.provider.calls[0][0]["prompt"])

    async def test_followups_include_full_reading_and_only_its_successful_turns(self):
        original = Event("完整起卦 " + CLIPBOARD_SAMPLE)
        await self.plugin.full_qigua(original)
        original_prompt = self.provider.calls[-1][0]["prompt"]
        self.provider.answer = "### 补充解读\n\n**合成背景分析**"
        first = Event("问卦 补充背景：合成项目预算有限\n希望先测试")
        await self.plugin.wengua(first)
        kwargs, config = self.provider.calls[-1]
        self.assertTrue(kwargs["prompt"].startswith(original_prompt))
        self.assertIn("分层解读：合成测试正文", kwargs["prompt"])
        self.assertIn("合成项目预算有限", kwargs["prompt"])
        self.assertEqual(kwargs["contexts"], [])
        self.assertIsNone(kwargs["session_id"])
        self.assertIsNone(kwargs["func_tool"])
        self.assertEqual(config["custom_extra_body"]["reasoning_effort"], "xhigh")
        self.assertIn(self.main.FOLLOWUP_RULES, kwargs["system_prompt"])
        self.assertEqual("".join(first.sent), "补充解读\n\n合成背景分析")
        self.assertTrue(first.stopped)
        self.assertTrue(first.call_llm)
        second = Event("问卦 哪个环节应当优先？")
        await self.plugin.wengua(second)
        prompt = self.provider.calls[-1][0]["prompt"]
        self.assertTrue(prompt.startswith(original_prompt))
        self.assertIn("合成项目预算有限", prompt)
        self.assertIn(self.provider.answer.replace("\n", "\\n"), prompt)
        self.assertIn("哪个环节应当优先？", prompt)
        self.assertNotIn("HIDDEN", prompt)

    async def test_new_cast_replaces_reading_and_history_but_invalid_input_does_not(
        self,
    ):
        await self.plugin.full_qigua(Event("完整起卦 报数:11 12\n占问:旧合成事项"))
        await self.plugin.wengua(Event("问卦 旧合成背景"))
        invalid = Event("完整起卦 占问:不完整的新输入")
        await self.plugin.full_qigua(invalid)
        self.assertIn("补充报数", "".join(invalid.sent))
        await self.plugin.wengua(Event("问卦 接续检查"))
        self.assertIn("旧合成事项", self.provider.calls[-1][0]["prompt"])
        self.assertNotIn("不完整的新输入", self.provider.calls[-1][0]["prompt"])
        self.provider.answer = "新的首次解读"
        await self.plugin.full_qigua(Event("完整起卦 报数:3 4 5\n占问:新合成事项"))
        new_prompt = self.provider.calls[-1][0]["prompt"]
        await self.plugin.wengua(Event("问卦 新的补充问题"))
        prompt = self.provider.calls[-1][0]["prompt"]
        self.assertTrue(prompt.startswith(new_prompt))
        self.assertIn("新的首次解读", prompt)
        for old in ("旧合成事项", "旧合成背景", "接续检查", "分层解读：合成测试正文"):
            self.assertNotIn(old, prompt)

    async def test_followup_scope_isolates_users_conversations_platforms_and_bots(self):
        await self.plugin.full_qigua(Event("完整起卦 11 12"))
        for scope in (
            {"sender": "user-2"},
            {"origin": "test:GroupMessage:group-2"},
            {"origin": "other:GroupMessage:group-1"},
            {"origin": "test:FriendMessage:user-1"},
            {"bot": "bot-2"},
        ):
            with self.subTest(scope=scope):
                event = Event("问卦 合成背景", **scope)
                await self.plugin.wengua(event)
                self.assertIn("没有你的卦象记录", "".join(event.sent))
                self.assertTrue(event.stopped)
                self.assertTrue(event.call_llm)
        self.assertEqual(len(self.provider.calls), 1)
        await self.plugin.wengua(Event("问卦 同一人的追问"))
        self.assertEqual(len(self.provider.calls), 2)

    async def test_empty_unknown_identity_and_cleared_sessions_do_not_call_model(self):
        for command in ("问卦", "问卦 合成问题"):
            event = Event(command)
            await self.plugin.wengua(event)
            self.assertTrue(event.sent)
            self.assertTrue(event.stopped)
            self.assertTrue(event.call_llm)
        for command, handler in (
            ("完整起卦 11 12", self.plugin.full_qigua),
            ("问卦 合成问题", self.plugin.wengua),
        ):
            event = Event(command, sender="")
            await handler(event)
            self.assertIn("无法识别", "".join(event.sent))
        self.assertFalse(self.provider.calls)
        await self.plugin.full_qigua(Event("完整起卦 11 12"))
        await self.plugin.terminate()
        event = Event("问卦 重载后的追问")
        await self.plugin.wengua(event)
        self.assertIn("没有你的卦象记录", "".join(event.sent))
        self.assertEqual(len(self.provider.calls), 1)

    async def test_failed_followup_does_not_pollute_history(self):
        await self.plugin.full_qigua(Event("完整起卦 11 12"))
        await self.plugin.wengua(Event("问卦 成功的补充背景"))
        self.provider.failure = RuntimeError("SECRET")
        failed = Event("问卦 失败的补充背景")
        with self.assertLogs("meihua-test", level="WARNING"):
            await self.plugin.wengua(failed)
        self.assertIn("模型调用失败", "".join(failed.sent))
        self.provider.failure = None
        await self.plugin.wengua(Event("问卦 再试一个问题"))
        prompt = self.provider.calls[-1][0]["prompt"]
        self.assertIn("成功的补充背景", prompt)
        for text in ("失败的补充背景", "模型调用失败", "SECRET"):
            self.assertNotIn(text, prompt)

    async def test_model_failure_on_new_cast_still_replaces_old_reading(self):
        await self.plugin.full_qigua(Event("完整起卦 报数:11 12\n占问:旧合成事项"))
        self.provider.failure = RuntimeError("SECRET")
        with self.assertLogs("meihua-test", level="WARNING"):
            await self.plugin.full_qigua(Event("完整起卦 报数:3 4 5\n占问:新合成事项"))
        self.provider.failure = None
        await self.plugin.wengua(Event("问卦 请继续分析"))
        prompt = self.provider.calls[-1][0]["prompt"]
        self.assertIn("新合成事项", prompt)
        self.assertIn('"首次解读": ""', prompt)
        self.assertNotIn("旧合成事项", prompt)
        self.assertNotIn("SECRET", prompt)

    async def test_inflight_followup_cannot_restore_replaced_reading(self):
        await self.plugin.full_qigua(Event("完整起卦 报数:11 12\n占问:旧合成事项"))
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow_interpret(
            provider, effort, system_prompt, model_input, *, before_request=None
        ):
            if before_request:
                await before_request()
            if "等待中的追问" in model_input:
                entered.set()
                await release.wait()
                return "旧问卦回答"
            return "新的合成解读"

        tasks = []
        with patch.object(self.main, "interpret", side_effect=slow_interpret):
            try:
                tasks.append(
                    asyncio.create_task(self.plugin.wengua(Event("问卦 等待中的追问")))
                )
                await asyncio.wait_for(entered.wait(), timeout=2)
                new_event = Event("完整起卦 报数:3 4 5\n占问:新合成事项")
                tasks.append(asyncio.create_task(self.plugin.full_qigua(new_event)))
                other_event = Event("完整起卦 11 12", sender="user-2")
                await asyncio.wait_for(self.plugin.full_qigua(other_event), timeout=2)
                self.assertTrue(other_event.sent)
                self.assertTrue(new_event.stopped)
                self.assertFalse(tasks[-1].done())
            finally:
                release.set()
                await asyncio.gather(*tasks)
        await self.plugin.wengua(Event("问卦 接续新卦"))
        prompt = self.provider.calls[-1][0]["prompt"]
        self.assertIn("新合成事项", prompt)
        self.assertIn("新的合成解读", prompt)
        for old in ("旧合成事项", "等待中的追问", "旧问卦回答"):
            self.assertNotIn(old, prompt)

    async def test_daily_quota_shared_by_commands_and_rejected_cast_keeps_old_reading(
        self,
    ):
        self.plugin.config.update(daily_user_limit=2, daily_group_limit=10)
        await self.plugin.full_qigua(Event("完整起卦 报数:11 12\n占问:保留的合成事项"))
        await self.plugin.wengua(Event("问卦 补充背景"))
        blocked = Event("完整起卦 报数:3 4 5\n占问:额度不足的新事项")
        await self.plugin.full_qigua(blocked)
        self.assertIn("上限", "".join(blocked.sent))
        self.assertEqual(len(self.provider.calls), 2)
        self.assertTrue(blocked.stopped)
        self.assertTrue(blocked.call_llm)
        self.plugin.config["daily_user_limit"] = 3
        await self.plugin.wengua(Event("问卦 仍然接续原卦"))
        prompt = self.provider.calls[-1][0]["prompt"]
        self.assertIn("保留的合成事项", prompt)
        self.assertNotIn("额度不足的新事项", prompt)

    async def test_configured_and_global_admins_bypass_both_quotas_without_consuming(
        self,
    ):
        self.plugin.config.update(
            daily_user_limit=1, daily_group_limit=1, admin_ids=["plugin-admin"]
        )
        await self.plugin.full_qigua(Event("完整起卦 11 12", sender="plugin-admin"))
        await self.plugin.wengua(Event("问卦 管理员追问", sender="plugin-admin"))
        await self.plugin.full_qigua(
            Event("完整起卦 11 12", sender="global-admin", role="admin")
        )
        normal = Event("完整起卦 11 12")
        await self.plugin.full_qigua(normal)
        self.assertIn("采用规则", "".join(normal.sent))
        await self.plugin.wengua(Event("问卦 群额度满后仍可用", sender="plugin-admin"))
        for role in ("member", "owner", "group_admin"):
            event = Event("完整起卦 11 12", sender="another-user", role=role)
            event.is_admin = True
            await self.plugin.full_qigua(event)
            self.assertIn("上限", "".join(event.sent))
        self.assertEqual(len(self.provider.calls), 5)

    async def test_invalid_input_and_local_configuration_do_not_consume_quota(self):
        self.plugin.config.update(daily_user_limit=1, daily_group_limit=1)
        await self.plugin.full_qigua(Event("完整起卦 缺少排盘"))
        await self.plugin.wengua(Event("问卦 尚无记录"))
        self.plugin.config["provider_id"] = ""
        await self.plugin.full_qigua(Event("完整起卦 11 12"))
        self.plugin.config["provider_id"] = "selected-model"
        self.plugin.config["reasoning_effort"] = "invalid-effort"
        await self.plugin.full_qigua(Event("完整起卦 11 12"))
        self.assertFalse(self.provider.calls)
        self.plugin.config["reasoning_effort"] = "high"
        event = Event("完整起卦 11 12")
        await self.plugin.full_qigua(event)
        self.assertIn("分层解读", "".join(event.sent))
        self.assertEqual(len(self.provider.calls), 1)

    async def test_attempted_call_counts_after_failure_and_plugin_restart(self):
        self.plugin.config.update(daily_user_limit=1, daily_group_limit=1)
        self.provider.failure = RuntimeError("synthetic failure")
        with self.assertLogs("meihua-test", level="WARNING"):
            await self.plugin.full_qigua(Event("完整起卦 11 12"))
        self.provider.failure = None
        blocked = Event("问卦 再问一次")
        await self.plugin.wengua(blocked)
        self.assertIn("上限", "".join(blocked.sent))
        await self.plugin.terminate()
        restarted = self.main.MeihuaPlugin(self.context, self.plugin.config)
        new_event = Event("完整起卦 11 12")
        await restarted.full_qigua(new_event)
        self.assertIn("上限", "".join(new_event.sent))
        self.assertEqual(len(self.provider.calls), 1)

    async def test_quota_storage_failure_stops_model_request(self):
        self.plugin.config["daily_user_limit"] = 1
        event = Event("完整起卦 11 12")
        with patch.object(
            self.plugin.quota, "consume", side_effect=OSError("private path")
        ):
            with self.assertLogs("meihua-test", level="WARNING") as captured:
                await self.plugin.full_qigua(event)
        self.assertIn("额度记录暂不可用", "".join(event.sent))
        self.assertNotIn("private path", "".join(event.sent) + str(captured.output))
        self.assertFalse(self.provider.calls)

    async def test_concurrent_group_last_slot_allows_only_one_model_call(self):
        self.plugin.config.update(daily_user_limit=0, daily_group_limit=1)
        events = [Event("完整起卦 11 12", sender=user) for user in ("user-a", "user-b")]
        await asyncio.gather(*(self.plugin.full_qigua(event) for event in events))
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(sum("上限" in "".join(event.sent) for event in events), 1)

    async def test_concurrent_requests_do_not_mutate_provider(self):
        original = deepcopy(self.provider.provider_config)
        await asyncio.gather(
            self.llm.interpret(self.provider, "high", "rules", "first"),
            self.llm.interpret(self.provider, "xhigh", "rules", "second"),
        )
        self.assertEqual(self.provider.provider_config, original)
        self.assertEqual(self.provider.client.api_key, "synthetic-key")
        by_prompt = {call[0]["prompt"]: call[1] for call in self.provider.calls}
        self.assertEqual(
            by_prompt["first"]["custom_extra_body"]["reasoning_effort"], "high"
        )
        self.assertEqual(
            by_prompt["second"]["custom_extra_body"]["reasoning_effort"], "xhigh"
        )

    async def test_native_request_retains_chat_context_and_charges_only_once(self):
        self.plugin.config.update(provider_id="", daily_user_limit=2)
        event = Event("起卦 " + CLIPBOARD_SAMPLE)
        await self.plugin.qigua(event)
        self.assertFalse(event.sent)
        self.assertFalse(event.stopped)
        self.assertFalse(event.call_llm)
        self.assertFalse(event.get_extra("enable_streaming"))
        reading = event.get_extra(self.main.NATIVE_PENDING)
        request = SimpleNamespace(
            prompt="原始命令与其他插件资料",
            system_prompt="原有人设和记忆",
            contexts=[{"role": "user", "content": "原有聊天历史"}],
            func_tool=object(),
            conversation=object(),
        )
        unchanged = {
            key: value for key, value in vars(request).items() if key != "prompt"
        }
        event.set_extra("provider_request", request)
        await self.plugin.on_native_request(event, request)
        self.assertTrue(request.prompt.startswith("原始命令与其他插件资料"))
        self.assertIn(reading.model_input, request.prompt)
        self.assertNotIn(self.main.REQUEST_RULES, request.prompt)
        for key, value in unchanged.items():
            self.assertIs(getattr(request, key), value)
        self.assertEqual("".join(event.sent), self.main.plain_text(reading.summary))
        self.assertFalse(event.stopped)
        self.assertFalse(self.provider.calls)
        self.manager.get_provider_by_id.assert_not_awaited()
        first_prompt, first_output = request.prompt, list(event.sent)
        await self.plugin.on_native_request(event, request)
        self.assertEqual(request.prompt, first_prompt)
        self.assertEqual(event.sent, first_output)
        self.plugin.config["provider_id"] = "selected-model"
        await self.plugin.full_qigua(Event("完整起卦 11 12"))
        self.assertEqual(len(self.provider.calls), 1)
        blocked = Event("问卦 本次应超额")
        await self.plugin.wengua(blocked)
        self.assertIn("上限", "".join(blocked.sent))

    async def test_native_cast_does_not_replace_independent_reading(self):
        await self.plugin.full_qigua(Event("解卦 报数:11 12\n占问:独立事项"))
        event = Event("起卦 报数:3 4 5\n占问:日常聊天事项")
        await self.plugin.qigua(event)
        request = SimpleNamespace(prompt=event.text)
        event.set_extra("provider_request", request)
        await self.plugin.on_native_request(event, request)
        await self.plugin.wengua(Event("问卦 继续独立事项"))
        prompt = self.provider.calls[-1][0]["prompt"]
        self.assertIn("独立事项", prompt)
        self.assertNotIn("日常聊天事项", prompt)

    async def test_native_quota_denial_stops_chat_without_modifying_request(self):
        self.plugin.config["daily_user_limit"] = 1
        await self.plugin.full_qigua(Event("完整起卦 11 12"))
        event = Event("起卦 3 4 5")
        await self.plugin.qigua(event)
        request = SimpleNamespace(prompt="untouched")
        event.set_extra("provider_request", request)
        await self.plugin.on_native_request(event, request)
        self.assertTrue(event.stopped)
        self.assertTrue(event.call_llm)
        self.assertIn("上限", "".join(event.sent))
        self.assertEqual(request.prompt, "untouched")
        self.assertIsNone(event.get_extra(self.main.NATIVE_ACCEPTED))
        self.assertIsNone(event.get_extra(self.main.NATIVE_PENDING))

    async def test_native_hook_ignores_unrelated_chat_and_explicit_plugin_requests(
        self,
    ):
        ordinary = Event("普通聊天")
        request = SimpleNamespace(prompt="untouched")
        ordinary.set_extra("provider_request", request)
        with patch.object(self.plugin, "consume_quota", AsyncMock()) as consume:
            await self.plugin.on_native_request(ordinary, request)
            event = Event("起卦 11 12")
            await self.plugin.qigua(event)
            event.set_extra("provider_request", SimpleNamespace(prompt="other"))
            await self.plugin.on_native_request(event, request)
            consume.assert_not_awaited()
        self.assertEqual(request.prompt, "untouched")
        self.assertFalse(ordinary.sent)
        self.assertFalse(event.sent)
        self.assertIsNotNone(event.get_extra(self.main.NATIVE_PENDING))

    async def test_native_cleaning_only_changes_accepted_llm_plain_components(self):
        event = Event("起卦 11 12")
        text = Plain("### 标题\n\n**解读**与 `010011`")
        attachment = object()
        event.result = SimpleNamespace(
            chain=[text, attachment], is_llm_result=lambda: True
        )
        await self.plugin.clean_native_result(event)
        self.assertIn("###", text.text)
        event.set_extra(self.main.NATIVE_ACCEPTED, True)
        event.result.is_llm_result = lambda: False
        await self.plugin.clean_native_result(event)
        self.assertIn("**", text.text)
        event.result.is_llm_result = lambda: True
        await self.plugin.clean_native_result(event)
        self.assertEqual(text.text, "标题\n\n解读与 010011")
        self.assertIs(event.result.chain[1], attachment)

    async def test_native_invalid_input_and_unsupported_runner_stop_without_charge(
        self,
    ):
        with patch.object(self.plugin, "consume_quota", AsyncMock()) as consume:
            invalid = Event("起卦 占问:没有数字和卦象")
            await self.plugin.qigua(invalid)
            self.assertIn("补充报数", "".join(invalid.sent))
            self.chat_config["agent_runner"] = {"runner_type": "dify"}
            unsupported = Event("起卦 11 12")
            await self.plugin.qigua(unsupported)
            self.assertIn("完整起卦", "".join(unsupported.sent))
            self.chat_config["agent_runner"] = {"runner_type": "local"}
            self.chat_config["provider_settings"] = {"enable": False}
            disabled = Event("起卦 11 12")
            await self.plugin.qigua(disabled)
            self.assertIn("已完成排盘校验", "".join(disabled.sent))
            self.assertIn("采用规则", "".join(disabled.sent))
            self.chat_config["provider_settings"]["enable"] = True
            session_disabled = Event("起卦 11 12")
            with patch.object(
                self.main.SessionServiceManager,
                "should_process_llm_request",
                AsyncMock(return_value=False),
            ):
                await self.plugin.qigua(session_disabled)
            self.assertIn("已完成排盘校验", "".join(session_disabled.sent))
            for event in (invalid, unsupported, disabled, session_disabled):
                self.assertTrue(event.stopped)
                self.assertTrue(event.call_llm)
                self.assertIsNone(event.get_extra(self.main.NATIVE_PENDING))
            consume.assert_not_awaited()

    async def test_extra_body_cannot_override_context(self):
        self.provider.provider_config["custom_extra_body"].update(
            {
                "messages": [{"role": "system", "content": "external persona"}],
                "previous_response_id": "prior",
                "instructions": "old",
                "tools": ["memory"],
                "reasoning": {"effort": "low"},
            }
        )
        await self.llm.interpret(self.provider, "high", "rules", "new")
        extra = self.provider.calls[0][1]["custom_extra_body"]
        self.assertFalse(self.llm.PROTECTED_BODY_KEYS.intersection(extra))
        self.assertNotIn("reasoning", extra)
        self.assertIn("messages", self.provider.provider_config["custom_extra_body"])

    async def test_provider_default_and_unsupported_custom_effort(self):
        other = SimpleNamespace(
            provider_config={"type": "unknown"},
            text_chat=AsyncMock(return_value=SimpleNamespace(completion_text="ok")),
        )
        with self.assertRaises(self.llm.ModelConfigError):
            await self.llm.interpret(other, "default", "rules", "new")
        with self.assertRaises(self.llm.ModelConfigError):
            await self.llm.interpret(other, "high", "rules", "new")
        self.assertEqual(other.text_chat.await_count, 0)

    async def test_native_default_isolation_and_cleanup(self):
        for cls, provider_type in (
            (NativeAnthropic, "anthropic_chat_completion"),
            (NativeGemini, "googlegenai_chat_completion"),
        ):
            for fail in (False, True):
                with self.subTest(provider=provider_type, fail=fail):
                    original = cls(
                        {
                            "type": provider_type,
                            "gm_native_search": True,
                            "gm_native_coderunner": True,
                            "gm_url_context": True,
                            "custom_extra_body": {
                                "system": "OLD",
                                "messages": ["OLD"],
                                "tools": ["search"],
                            },
                        },
                        {"fail": fail},
                    )
                    before = deepcopy(original.provider_config)
                    if fail:
                        with self.assertRaises(RuntimeError):
                            await self.llm.interpret(
                                original, "default", "rules", "new"
                            )
                    else:
                        self.assertEqual(
                            await self.llm.interpret(
                                original, "default", "rules", "new"
                            ),
                            "native-ok",
                        )
                    request = cls.instances[-1]
                    self.assertIsNot(request, original)
                    self.assertTrue(request.closed)
                    self.assertFalse(original.closed)
                    self.assertFalse(original.calls)
                    self.assertEqual(original.provider_config, before)
                    self.assertEqual(original.client.api_key, "synthetic-key")
                    self.assertEqual(request.provider_config["custom_extra_body"], {})
                    self.assertFalse(
                        any(
                            request.provider_config[flag]
                            for flag in self.llm.NATIVE_TOOL_FLAGS
                        )
                    )
                    self.assertEqual(request.calls[0]["contexts"], [])
                    with self.assertRaises(self.llm.ModelConfigError):
                        await self.llm.interpret(original, "high", "rules", "new")

    async def test_xai_native_tools_disabled_in_request_only(self):
        self.provider.provider_config["xai_native_search"] = True
        self.provider.provider_config["custom_extra_body"]["search_parameters"] = {
            "mode": "auto"
        }
        await self.llm.interpret(self.provider, "default", "rules", "new")
        config = self.provider.calls[-1][1]
        self.assertFalse(config["xai_native_search"])
        self.assertNotIn("search_parameters", config["custom_extra_body"])
        self.assertTrue(self.provider.provider_config["xai_native_search"])

    async def test_error_preserves_reference_text_and_does_not_leak(self):
        self.provider.failure = RuntimeError("SECRET and full private request")
        original = deepcopy(self.provider.provider_config)
        event = Event("完整起卦 11 12")
        with self.assertLogs("meihua-test", level="WARNING") as captured:
            await self.plugin.full_qigua(event)
        output = "".join(event.sent)
        self.assertIn("《灵签详解》", output)
        self.assertIn("模型调用失败", output)
        self.assertNotIn("SECRET", output + str(captured.output))
        self.assertEqual(self.provider.provider_config, original)

    async def test_timeout_keeps_command_consumed(self):
        event = Event("完整起卦 11 12")

        async def time_out(coroutine, timeout):
            coroutine.close()
            raise asyncio.TimeoutError

        with patch.object(self.main.asyncio, "wait_for", time_out):
            await self.plugin.full_qigua(event)
        self.assertTrue(event.stopped)
        self.assertIn("超时", "".join(event.sent))

    def test_split_message_preserves_every_character(self):
        text = "## 原文\n\n" + "长行" * 2000 + "\n\n末尾。\n"
        parts = list(self.main.split_message(text))
        self.assertEqual("".join(parts), text)
        self.assertTrue(all(len(part) <= 1800 for part in parts))


if __name__ == "__main__":
    unittest.main()
