"""Configuration: every environment variable MAPTA reads, in one place.

Settings objects are immutable and built explicitly from a mapping, so tests
never have to mutate ``os.environ`` and the rest of the code never calls
``os.getenv``.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from .domain import ConfigurationError

__all__ = [
    "LLMSettings",
    "ReportSettings",
    "SandboxSettings",
    "ScanSettings",
    "Settings",
    "TelegramSettings",
    "load_dotenv",
]

type Env = Mapping[str, str]

#: OpenRouter's OpenAI-compatible endpoint.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
#: Any slug from https://openrouter.ai/models works; override with $MODEL.
DEFAULT_MODEL = "openai/gpt-5"


def _number[T: (int, float)](env: Env, name: str, default: T, cast: type[T]) -> T:
    """Read a numeric setting, reporting a bad value as a configuration error."""
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise ConfigurationError(
            f"{name}={raw!r} is not a valid {cast.__name__}."
        ) from None


def _flag(env: Env, name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _csv(raw: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (raw or "").split(",") if part.strip())


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Everything the OpenRouter adapter needs to issue a request."""

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    reasoning_effort: str = "high"
    #: Output cap per answer. None leaves the provider's default in place;
    #: raise it when long tool-call arguments are being cut off.
    max_tokens: int | None = None
    #: Sent as X-Title / HTTP-Referer, which OpenRouter uses for its rankings.
    app_name: str = "MAPTA"
    site_url: str | None = None
    #: Ask OpenRouter to return the generation's cost alongside token counts.
    track_cost: bool = True
    #: Some vendors behind OpenRouter reject strict tool schemas.
    strict_tools: bool = True
    #: Optional upstream routing preference, e.g. ("anthropic", "google-vertex").
    provider_order: tuple[str, ...] = ()
    allow_fallbacks: bool = True

    @property
    def sends_reasoning(self) -> bool:
        """False for models that reject a ``reasoning`` field."""
        return self.reasoning_effort not in ("none", "off", "")

    @property
    def default_headers(self) -> dict[str, str]:
        headers = {"X-Title": self.app_name}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        return headers

    @classmethod
    def from_env(
        cls,
        env: Env | None = None,
        *,
        model: str | None = None,
        reasoning: str | None = None,
    ) -> Self:
        """Build settings from the environment, with CLI overrides on top."""
        env = os.environ if env is None else env

        api_key = env.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is not set. Create a key at "
                "https://openrouter.ai/keys, then export it or add it to a .env "
                "file (see .env.example)."
            )

        return cls(
            api_key=api_key,
            model=model or env.get("MODEL") or DEFAULT_MODEL,
            base_url=(env.get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            reasoning_effort=(reasoning or env.get("REASONING_EFFORT", "high")).strip().lower(),
            # 0 and unset both mean "leave the provider default alone".
            max_tokens=_number(env, "MAX_TOKENS", 0, int) or None,
            app_name=env.get("OPENROUTER_APP_NAME", "MAPTA"),
            site_url=env.get("OPENROUTER_SITE_URL") or None,
            track_cost=_flag(env, "OPENROUTER_TRACK_COST", True),
            strict_tools=_flag(env, "OPENROUTER_STRICT_TOOLS", True),
            provider_order=_csv(env.get("OPENROUTER_PROVIDER_ORDER")),
            allow_fallbacks=_flag(env, "OPENROUTER_ALLOW_FALLBACKS", True),
        )


@dataclass(frozen=True, slots=True)
class SandboxSettings:
    """How to build the per-scan sandbox."""

    #: ``docker``, ``none``, or ``module:function`` for a third-party provider.
    provider: str = "docker"
    image: str = "python:3.14-slim"
    workdir: str = "/home/user"
    network: str = "bridge"
    memory: str = "2g"
    cpus: str = "2"
    pids_limit: str = "512"
    apt_packages: str = ""
    pip_packages: str = ""
    bootstrap: bool = True
    default_timeout: float = 120.0
    #: Ceiling for a timeout an agent asks for, so one command cannot hang a scan.
    max_timeout: float = 900.0

    @classmethod
    def from_env(cls, env: Env | None = None) -> Self:
        env = os.environ if env is None else env
        # SANDBOX_FACTORY stays supported for third-party providers.
        provider = env.get("SANDBOX_PROVIDER") or env.get("SANDBOX_FACTORY") or "docker"
        return cls(
            provider=provider.strip(),
            image=env.get("SANDBOX_IMAGE", "python:3.14-slim"),
            workdir=env.get("SANDBOX_WORKDIR", "/home/user"),
            network=env.get("SANDBOX_NETWORK", "bridge"),
            memory=env.get("SANDBOX_MEMORY", "2g"),
            cpus=env.get("SANDBOX_CPUS", "2"),
            pids_limit=env.get("SANDBOX_PIDS_LIMIT", "512"),
            apt_packages=env.get("SANDBOX_PACKAGES", ""),
            pip_packages=env.get("SANDBOX_PIP_PACKAGES", ""),
            bootstrap=not _flag(env, "SANDBOX_SKIP_BOOTSTRAP", False),
            default_timeout=_number(env, "SANDBOX_TIMEOUT", 120.0, float),
            max_timeout=_number(env, "SANDBOX_MAX_TIMEOUT", 900.0, float),
        )


@dataclass(frozen=True, slots=True)
class TelegramSettings:
    """Telegram Bot API reporting. Disabled when the bot is not configured."""

    bot_token: str | None = None
    chat_id: str | None = None
    #: Forum-topic id, for supergroups that use topics.
    message_thread_id: int | None = None
    #: Deliver without a notification sound.
    silent: bool = False
    api_root: str = "https://api.telegram.org"

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    @classmethod
    def from_env(cls, env: Env | None = None) -> Self:
        env = os.environ if env is None else env
        thread = env.get("TELEGRAM_MESSAGE_THREAD_ID", "").strip()
        return cls(
            bot_token=env.get("TELEGRAM_BOT_TOKEN") or None,
            chat_id=env.get("TELEGRAM_CHAT_ID") or None,
            message_thread_id=int(thread) if thread.lstrip("-").isdigit() else None,
            silent=_flag(env, "TELEGRAM_SILENT", False),
            api_root=env.get("TELEGRAM_API_ROOT", "https://api.telegram.org").rstrip("/"),
        )


@dataclass(frozen=True, slots=True)
class ReportSettings:
    """Where reports and usage logs are written."""

    output_dir: str = "scan-results"

    @classmethod
    def from_env(cls, env: Env | None = None) -> Self:
        env = os.environ if env is None else env
        return cls(output_dir=env.get("OUTPUT_DIR", "scan-results"))


@dataclass(frozen=True, slots=True)
class ScanSettings:
    """Prompt and round budgets for a run."""

    system_prompt: str | None = None
    user_prompt: str | None = None
    sandbox_prompt: str | None = None
    validator_prompt: str | None = None
    #: Model rounds the whole scan may spend, nested agents included.
    max_rounds: int = 100
    max_concurrency: int = 0
    mailbox_enabled: bool = True

    @classmethod
    def from_env(cls, env: Env | None = None) -> Self:
        env = os.environ if env is None else env
        return cls(
            system_prompt=env.get("SYSTEM_PROMPT"),
            user_prompt=env.get("USER_PROMPT"),
            sandbox_prompt=env.get("SANDBOX_SYSTEM_PROMPT"),
            validator_prompt=env.get("VALIDATOR_SYSTEM_PROMPT"),
            max_rounds=_number(env, "MAX_ROUNDS", 100, int),
            max_concurrency=_number(env, "MAX_CONCURRENCY", 0, int),
            mailbox_enabled=_flag(env, "MAILBOX_ENABLED", True),
        )


@dataclass(frozen=True, slots=True)
class Settings:
    """The whole configuration, assembled once in the composition root."""

    llm: LLMSettings
    sandbox: SandboxSettings
    telegram: TelegramSettings
    reports: ReportSettings
    scan: ScanSettings

    @classmethod
    def from_env(
        cls,
        env: Env | None = None,
        *,
        model: str | None = None,
        reasoning: str | None = None,
    ) -> Self:
        env = os.environ if env is None else env
        return cls(
            llm=LLMSettings.from_env(env, model=model, reasoning=reasoning),
            sandbox=SandboxSettings.from_env(env),
            telegram=TelegramSettings.from_env(env),
            reports=ReportSettings.from_env(env),
            scan=ScanSettings.from_env(env),
        )


def load_dotenv() -> None:
    """Load a local .env when python-dotenv is installed (optional dependency)."""
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        return
    _load()
