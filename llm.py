"""One-shot provider calls, without AstrBot's conversation or agent pipeline."""

from copy import copy, deepcopy

EFFORTS = {"default", "none", "minimal", "low", "medium", "high", "xhigh"}
PROTECTED_BODY_KEYS = {
    "messages",
    "input",
    "instructions",
    "system",
    "system_prompt",
    "contexts",
    "previous_response_id",
    "conversation",
    "conversation_id",
    "session_id",
    "tools",
    "tool_choice",
    "prompt",
    "store",
    "search_parameters",
    "web_search_options",
    "enable_search",
    "plugins",
    "tool_config",
    "cached_content",
    "system_instruction",
}
NATIVE_TOOL_FLAGS = (
    "xai_native_search",
    "gm_native_search",
    "gm_native_coderunner",
    "gm_url_context",
)


class ModelConfigError(ValueError):
    pass


def isolated_provider(provider, effort):
    from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial

    if effort not in EFFORTS:
        raise ModelConfigError("推理等级无效，请在插件设置中重新选择。")
    is_openai = isinstance(provider, ProviderOpenAIOfficial)
    config = deepcopy(provider.provider_config)
    if not is_openai and effort != "default":
        raise ModelConfigError(
            "自定义推理等级目前支持 OpenAI 兼容 Chat Completions / Responses 供应商；"
            "Anthropic / Gemini 原生供应商请将推理等级设为 default。"
        )
    extra = config.get("custom_extra_body") or {}
    if not isinstance(extra, dict):
        raise ModelConfigError("供应商的 custom_extra_body 必须是 JSON 对象。")
    for key in PROTECTED_BODY_KEYS:
        extra.pop(key, None)
    for key in NATIVE_TOOL_FLAGS:
        config[key] = False
    config["custom_extra_body"] = extra
    if not is_openai:
        provider_type = config.get("type")
        if provider_type == "anthropic_chat_completion":
            from astrbot.core.provider.sources.anthropic_source import ProviderAnthropic

            provider_class = ProviderAnthropic
        elif provider_type == "googlegenai_chat_completion":
            from astrbot.core.provider.sources.gemini_source import ProviderGoogleGenAI

            provider_class = ProviderGoogleGenAI
        else:
            raise ModelConfigError(
                "当前供应商尚未适配独立请求，请选择 OpenAI 兼容、Anthropic 或 Gemini 供应商。"
            )
        if type(provider) is not provider_class:
            raise ModelConfigError(
                "当前自定义供应商尚未验证请求隔离，请选择内置供应商。"
            )
        config["model"] = provider.get_model()
        # Gemini key rotation replaces its client; use and close an owned instance.
        return provider_class(config, deepcopy(provider.provider_settings)), True

    # AstrBot 4.29.4 reads these parameters from provider_config, not text_chat kwargs.
    request_provider = copy(provider)
    request_provider.provider_config = config
    request_provider.api_keys = list(provider.api_keys)
    request_provider.client = provider.client.with_options()
    if effort != "default":
        if (
            request_provider.provider_config.get("provider") == "ollama"
            and request_provider._ollama_disable_thinking_enabled()
            and effort != "none"
        ):
            raise ModelConfigError(
                "供应商已禁用 Ollama 思考，请先取消禁用或选择 none。"
            )
        extra.pop("reasoning", None)
        extra["reasoning_effort"] = effort
    return request_provider, False


async def interpret(
    provider, effort, system_prompt, model_input, *, before_request=None
):
    request_provider, owned = isolated_provider(provider, effort)
    try:
        if before_request is not None:
            await before_request()
        response = await request_provider.text_chat(
            prompt=model_input,
            system_prompt=system_prompt,
            contexts=[],
            session_id=None,
            image_urls=[],
            func_tool=None,
        )
    finally:
        if owned:
            await request_provider.terminate()
    text = response.completion_text
    if not isinstance(text, str) or not text.strip():
        raise ModelConfigError("模型未返回解读正文，请检查模型及推理等级设置后重试。")
    return text.strip()
