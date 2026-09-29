"""Settings are built from an explicit mapping, never from a live os.environ."""

import pytest

from mapta.config import LLMSettings, SandboxSettings, Settings, TelegramSettings
from mapta.domain import ConfigurationError


def test_missing_key_is_refused():
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY is not set"):
        LLMSettings.from_env({})


def test_openrouter_defaults():
    settings = LLMSettings.from_env({"OPENROUTER_API_KEY": "sk-or"})
    assert settings.base_url == "https://openrouter.ai/api/v1"
    assert settings.model == "openai/gpt-5"
    assert settings.track_cost is True


def test_any_openrouter_slug_is_accepted():
    settings = LLMSettings.from_env(
        {"OPENROUTER_API_KEY": "sk-or", "MODEL": "anthropic/claude-3.7-sonnet"}
    )
    assert settings.model == "anthropic/claude-3.7-sonnet"


def test_attribution_headers():
    settings = LLMSettings.from_env(
        {
            "OPENROUTER_API_KEY": "sk-or",
            "OPENROUTER_APP_NAME": "recon",
            "OPENROUTER_SITE_URL": "https://example.test",
        }
    )
    assert settings.default_headers == {
        "X-Title": "recon",
        "HTTP-Referer": "https://example.test",
    }


def test_site_url_header_is_omitted_when_unset():
    settings = LLMSettings.from_env({"OPENROUTER_API_KEY": "sk-or"})
    assert settings.default_headers == {"X-Title": "MAPTA"}


def test_reasoning_none_disables_the_field():
    settings = LLMSettings.from_env({"OPENROUTER_API_KEY": "sk-or"}, reasoning="none")
    assert settings.sends_reasoning is False


def test_cli_overrides_beat_the_environment():
    settings = LLMSettings.from_env(
        {"OPENROUTER_API_KEY": "sk-or", "MODEL": "env-model"}, model="cli-model"
    )
    assert settings.model == "cli-model"


def test_base_url_can_point_at_a_proxy():
    settings = LLMSettings.from_env(
        {"OPENROUTER_API_KEY": "sk-or", "OPENROUTER_BASE_URL": "https://proxy.test/v1/"}
    )
    assert settings.base_url == "https://proxy.test/v1"


def test_provider_routing_preferences_are_parsed():
    settings = LLMSettings.from_env(
        {
            "OPENROUTER_API_KEY": "sk-or",
            "OPENROUTER_PROVIDER_ORDER": "anthropic, google-vertex ,",
            "OPENROUTER_ALLOW_FALLBACKS": "0",
        }
    )
    assert settings.provider_order == ("anthropic", "google-vertex")
    assert settings.allow_fallbacks is False


def test_telegram_needs_both_a_token_and_a_chat():
    assert TelegramSettings.from_env({"TELEGRAM_BOT_TOKEN": "123:abc"}).enabled is False
    assert TelegramSettings.from_env({"TELEGRAM_CHAT_ID": "-100"}).enabled is False
    assert (
        TelegramSettings.from_env(
            {"TELEGRAM_BOT_TOKEN": "123:abc", "TELEGRAM_CHAT_ID": "-100"}
        ).enabled
        is True
    )


def test_telegram_forum_topic_is_parsed():
    settings = TelegramSettings.from_env({"TELEGRAM_MESSAGE_THREAD_ID": "7"})
    assert settings.message_thread_id == 7


def test_telegram_ignores_a_non_numeric_topic():
    settings = TelegramSettings.from_env({"TELEGRAM_MESSAGE_THREAD_ID": "abc"})
    assert settings.message_thread_id is None


def test_sandbox_factory_env_var_is_still_honoured():
    settings = SandboxSettings.from_env({"SANDBOX_FACTORY": "my_provider:create"})
    assert settings.provider == "my_provider:create"


def test_full_settings_assemble():
    settings = Settings.from_env({"OPENROUTER_API_KEY": "sk-or", "MAX_ROUNDS": "7"})
    assert settings.scan.max_rounds == 7
    assert settings.telegram.enabled is False
