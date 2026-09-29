import argparse
import os
import json
import re
import asyncio
from typing import Any, Dict, Optional, List
from openai import AsyncOpenAI
from datetime import datetime, UTC
import sys
import threading
import logging
import importlib

from function_tool import function_tool
import json as json_module
import httpx
import aiohttp
# from core.config import SLACK_WEBHOOK_URL, SLACK_CHANNEL

# Load a local .env file when python-dotenv is installed (optional dependency).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "#security-alerts")


# --- Provider configuration -------------------------------------------------
# The agents talk to the OpenAI Responses API. Any endpoint that implements it
# works, so OpenRouter (and self-hosted gateways) can be used instead of OpenAI.
_PROVIDERS = {
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": None,  # SDK default
        "model": "gpt-5",
    },
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        # OpenRouter namespaces model ids by vendor.
        "model": "openai/gpt-5",
    },
    # Any other OpenAI-Responses-compatible endpoint (LiteLLM, vLLM, ...).
    "custom": {
        "key_env": "LLM_API_KEY",
        "base_url": None,  # must be given via LLM_BASE_URL
        "model": None,     # must be given via MODEL
    },
}


def _detect_provider() -> str:
    """Pick the provider from LLM_PROVIDER, else infer it from the keys present."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in _PROVIDERS:
            raise RuntimeError(
                f"Unknown LLM_PROVIDER '{explicit}'. Choose one of: {', '.join(_PROVIDERS)}."
            )
        return explicit
    if os.getenv("LLM_BASE_URL"):
        return "custom"
    if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        return "openrouter"
    return "openai"


LLM_PROVIDER = _detect_provider()
_PROVIDER = _PROVIDERS[LLM_PROVIDER]

# Model used by the main agent and by the nested sandbox/validator agents.
MODEL = os.getenv("MODEL") or _PROVIDER["model"]

# Reasoning effort passed to the model. Set to "none" for models that do not
# support reasoning (many non-OpenAI models on OpenRouter reject the field).
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "high").strip().lower()

# Per-request metadata is an OpenAI feature; gateways may reject unknown fields.
SEND_METADATA = os.getenv("SEND_METADATA", "1" if LLM_PROVIDER == "openai" else "0").lower() in ("1", "true", "yes")


# --- Setup ---
# The client is created lazily so importing this module (e.g. to inspect the
# tool schemas) does not require an API key to be set.
_client: Optional[AsyncOpenAI] = None
_client_lock = threading.Lock()


def get_client() -> AsyncOpenAI:
    """Return the shared AsyncOpenAI-compatible client, creating it on first use."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                key_env = _PROVIDER["key_env"]
                api_key = os.getenv(key_env)
                if not api_key:
                    raise RuntimeError(
                        f"{key_env} is not set (provider '{LLM_PROVIDER}'). Export it or add "
                        "it to a .env file (see .env.example)."
                    )

                base_url = os.getenv("LLM_BASE_URL") or _PROVIDER["base_url"]
                if LLM_PROVIDER == "custom" and not base_url:
                    raise RuntimeError(
                        "LLM_PROVIDER=custom requires LLM_BASE_URL pointing at an "
                        "OpenAI-Responses-compatible endpoint."
                    )
                if not MODEL:
                    raise RuntimeError(
                        f"MODEL is not set and provider '{LLM_PROVIDER}' has no default."
                    )

                kwargs: Dict[str, Any] = {"api_key": api_key}
                if base_url:
                    kwargs["base_url"] = base_url
                if LLM_PROVIDER == "openrouter":
                    # Optional attribution headers OpenRouter uses for its rankings.
                    headers = {}
                    if os.getenv("OPENROUTER_SITE_URL"):
                        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL")
                    headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "MAPTA")
                    kwargs["default_headers"] = headers

                logging.info(
                    "LLM provider: %s | model: %s | base_url: %s",
                    LLM_PROVIDER, MODEL, base_url or "default (api.openai.com)",
                )
                _client = AsyncOpenAI(**kwargs)
    return _client


async def create_response(tools_subset: List[Dict[str, Any]], input_list: List[Any], metadata: Dict[str, str]):
    """Issue one Responses API call, adapted to the configured provider."""
    request: Dict[str, Any] = {
        "model": MODEL,
        "tools": tools_subset,
        "input": input_list,
    }

    if REASONING_EFFORT not in ("none", "off", ""):
        request["reasoning"] = {"effort": REASONING_EFFORT}

    extra_body: Dict[str, Any] = {}
    if SEND_METADATA:
        extra_body["metadata"] = metadata
    if LLM_PROVIDER == "openrouter":
        # OpenRouter is stateless and rejects `store: true`; the agents already
        # resend the full conversation each round, so be explicit.
        extra_body["store"] = False
    if extra_body:
        request["extra_body"] = extra_body

    return await get_client().responses.create(**request)


# Global sandbox configuration (sanitized for open release)
# Provide a factory via env var SANDBOX_FACTORY="your_module:create_sandbox" that returns a sandbox instance
SANDBOX_FACTORY = os.getenv("SANDBOX_FACTORY")

# Thread-local storage for sandbox instances
_thread_local = threading.local()

def get_current_sandbox():
    """Get the sandbox instance for the current thread/scan."""
    return getattr(_thread_local, 'sandbox', None)

def set_current_sandbox(sandbox):
    """Set the sandbox instance for the current thread/scan."""
    _thread_local.sandbox = sandbox

def create_sandbox_from_env():
    """Create a sandbox instance using a user-provided factory specified in SANDBOX_FACTORY.

    SANDBOX_FACTORY should be in the form "module_path:function_name" and must return an
    object exposing .files.write(path, content), .commands.run(cmd, timeout=..., user=...),
    and optional .set_timeout(ms) and .kill().

    Returns None if not configured.
    """
    factory_path = SANDBOX_FACTORY
    if not factory_path:
        logging.info("Sandbox factory not configured; running without a sandbox.")
        return None
    try:
        module_name, func_name = factory_path.rsplit(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, func_name)
        sandbox = factory()
        # Optionally extend timeout if provider supports it
        if hasattr(sandbox, "set_timeout"):
            try:
                sandbox.set_timeout(timeout=12000)
            except TypeError:
                # Some providers may use milliseconds
                sandbox.set_timeout(12000)
        return sandbox
    except Exception as exc:
        logging.warning(f"Failed to create sandbox from SANDBOX_FACTORY: {exc}")
        return None

# Usage tracking
def _usage_to_dict(usage_data) -> Any:
    """Normalise an SDK usage object into plain JSON-serialisable data.

    Without this the usage objects fall through `json.dump(default=str)` and land
    in the log as Python reprs, which no tool can read back.
    """
    if usage_data is None or isinstance(usage_data, (dict, int, float, str)):
        return usage_data
    for attr in ("model_dump", "dict", "to_dict"):
        method = getattr(usage_data, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:
                continue
    return str(usage_data)


class UsageTracker:
    def __init__(self):
        self.main_agent_usage = []
        self.sandbox_agent_usage = []
        self.start_time = datetime.now(UTC)

    def log_main_agent_usage(self, usage_data, target_url=""):
        """Log usage data from main agent responses."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "target_url": target_url,
            "agent_type": "main_agent",
            "usage": _usage_to_dict(usage_data)
        }
        self.main_agent_usage.append(entry)
        logging.info(f"Main Agent Usage - Target: {target_url}, Usage: {usage_data}")
    
    def log_sandbox_agent_usage(self, usage_data, target_url=""):
        """Log usage data from sandbox agent responses."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "target_url": target_url,
            "agent_type": "sandbox_agent", 
            "usage": _usage_to_dict(usage_data)
        }
        self.sandbox_agent_usage.append(entry)
        logging.info(f"Sandbox Agent Usage - Target: {target_url}, Usage: {usage_data}")
    
    def get_summary(self):
        """Get usage summary for all agents."""
        return {
            "scan_duration": str(datetime.now(UTC) - self.start_time),
            "main_agent_calls": len(self.main_agent_usage),
            "sandbox_agent_calls": len(self.sandbox_agent_usage),
            "total_calls": len(self.main_agent_usage) + len(self.sandbox_agent_usage),
            "main_agent_usage": self.main_agent_usage,
            "sandbox_agent_usage": self.sandbox_agent_usage
        }
    
    def save_to_file(self, filename_prefix=""):
        """Save usage data to JSON file."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}usage_log_{timestamp}.json"
        
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(self.get_summary(), f, indent=2, default=str)
        
        logging.info(f"Usage data saved to {filename}")
        return filename

# Thread-local storage for usage trackers
def get_current_usage_tracker():
    """Get the usage tracker for the current thread/scan."""
    return getattr(_thread_local, 'usage_tracker', None)

def set_current_usage_tracker(tracker):
    """Set the usage tracker for the current thread/scan."""
    _thread_local.usage_tracker = tracker


# Create tasks for parallel execution
def parse_tool_arguments(raw: Any) -> Dict[str, Any]:
    """Parse the `arguments` payload of a function call, tolerating sloppy models.

    Models -- especially smaller ones behind gateways -- routinely emit arguments
    that are not strictly valid JSON. Raising here would abort the whole scan, so
    recover where possible and raise ValueError only when the payload is truly
    unusable.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ValueError(f"unsupported arguments type {type(raw).__name__}")

    text = raw.strip()
    # Some providers wrap the JSON in a markdown fence.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    attempts = (
        lambda s: json.loads(s),
        # Allow raw newlines/tabs inside string values (very common).
        lambda s: json.loads(s, strict=False),
        # Take the first complete object and ignore trailing garbage or a
        # second concatenated object.
        lambda s: json.JSONDecoder(strict=False).raw_decode(s)[0],
    )
    first_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            parsed = attempt(text)
        except Exception as exc:
            first_error = first_error or exc
            continue
        if isinstance(parsed, dict):
            return parsed
        first_error = first_error or ValueError(f"arguments decoded to {type(parsed).__name__}, not an object")

    # Double-encoded JSON: a string whose content is itself JSON.
    try:
        inner = json.loads(text, strict=False)
        if isinstance(inner, str):
            reparsed = json.loads(inner, strict=False)
            if isinstance(reparsed, dict):
                return reparsed
    except Exception:
        pass

    raise ValueError(str(first_error or "could not parse arguments"))


async def execute_function_call(function_call):
    name = getattr(function_call, "name", "<unknown>")
    raw_arguments = getattr(function_call, "arguments", None)

    try:
        function_call_arguments = parse_tool_arguments(raw_arguments)
    except ValueError as exc:
        # Never let one malformed tool call abort the scan: report the problem
        # back to the model so it can retry with valid JSON.
        preview = raw_arguments if isinstance(raw_arguments, str) else repr(raw_arguments)
        logging.error(
            "Malformed arguments for tool '%s' (%s). Length=%s. Raw payload: %r",
            name, exc, len(preview) if preview is not None else 0, preview,
        )
        result = json.dumps({
            "error": "invalid_tool_arguments",
            "detail": f"Your `arguments` for `{name}` were not valid JSON ({exc}). "
                      "Re-issue the call with a single well-formed JSON object. "
                      "Escape newlines as \\n inside string values.",
        })
        return {
            "type": "function_call_output",
            "call_id": getattr(function_call, "call_id", None),
            "output": result,
        }

    # Execute the function logic
    result = await execute_tool(name, function_call_arguments)

    return {
        "type": "function_call_output",
        "call_id": function_call.call_id,
        "output": result,
    }



# In-memory store: email -> JWT token (for mail.tm API)
email_token_store = {}

MAIL_TM_API = "https://api.mail.tm"


@function_tool
async def create_email_account(address: Optional[str] = None, password: Optional[str] = None):
    """
    Create a disposable email account on mail.tm and store its JWT for later use.
    Use it to receive account activation emails, password resets or credentials.
    Returns JSON: {address, password} on success.

    Args:
        address: Optional full address to claim (e.g. "someone@domain.tld"). A random
            local part on an available domain is generated when omitted.
        password: Optional password. A random one is generated when omitted.
    """
    import secrets

    password = password or secrets.token_urlsafe(16)
    try:
        with httpx.Client(timeout=30) as http:
            if not address:
                resp = http.get(f"{MAIL_TM_API}/domains")
                if resp.status_code != 200:
                    return f"Failed to list mail.tm domains. Status: {resp.status_code}, Response: {resp.text}"
                domains = [d.get("domain") for d in resp.json().get("hydra:member", []) if d.get("domain")]
                if not domains:
                    return "No mail.tm domains are currently available."
                address = f"mapta{secrets.token_hex(6)}@{domains[0]}"

            resp = http.post(f"{MAIL_TM_API}/accounts", json={"address": address, "password": password})
            if resp.status_code not in (200, 201):
                return f"Failed to create account {address}. Status: {resp.status_code}, Response: {resp.text}"

            resp = http.post(f"{MAIL_TM_API}/token", json={"address": address, "password": password})
            if resp.status_code not in (200, 201):
                return f"Account {address} created but token request failed. Status: {resp.status_code}, Response: {resp.text}"

            token = resp.json().get("token")
            if not token:
                return f"Account {address} created but no token was returned: {resp.text}"

            email_token_store[address] = token
            return json_module.dumps({"address": address, "password": password})
    except Exception as e:
        return f"Request failed: {e}"


@function_tool
async def set_email_jwt_token(email: str, jwt_token: str):
    """
    Store a mail.tm JWT for an email account you already own, so the
    list_account_messages and get_message_by_id tools can use it.

    Args:
        email: The email address the token belongs to
        jwt_token: The mail.tm JWT for that account
    """
    if not email or not jwt_token:
        return "Both email and jwt_token are required."
    email_token_store[email] = jwt_token
    return json_module.dumps({"success": True, "email": email})


@function_tool
async def get_registered_emails():
    """
    Return the list of email accounts in case you need to use them to receive emails such as account activation emails, credentials, etc.
    """
    return json_module.dumps(list(email_token_store.keys()))



@function_tool
async def list_account_messages(email: str, limit: int = 50):
    """
    List recent messages for the given email account.
    Returns JSON list: [{id, subject, from, intro, seen, createdAt}]
    
    Args:
        email: The email account to fetch messages for
        limit: Maximum number of messages to return (default: 50)
    """
    jwt = email_token_store.get(email)
    if not jwt:
        return f"No JWT token stored for {email}. Call set_email_jwt_token(email, jwt_token) first."

    headers = {"Authorization": f"Bearer {jwt}"}
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get("https://api.mail.tm/messages", headers=headers)
            if resp.status_code != 200:
                return f"Failed to fetch messages. Status: {resp.status_code}, Response: {resp.text}"
            data = resp.json()
            messages = data.get("hydra:member", [])
            items = []
            for m in messages[:limit]:
                sender = m.get("from") or {}
                items.append(
                    {
                        "id": m.get("id"),
                        "subject": m.get("subject"),
                        "from": sender.get("address") or sender.get("name") or "",
                        "intro": m.get("intro", ""),
                        "seen": m.get("seen", False),
                        "createdAt": m.get("createdAt", ""),
                    }
                )
            return json_module.dumps(items)
    except Exception as e:
        return f"Request failed: {e}"



@function_tool
async def get_message_by_id(email: str, message_id: str):
    """
    Fetch a specific message by id for the given email account using its stored JWT.
    Returns JSON: {id, subject, from, text, html}
    
    Args:
        email: The email account to fetch the message from
        message_id: The ID of the message to fetch
    """
    jwt = email_token_store.get(email)
    if not jwt:
        return f"No JWT token stored for {email}. Call set_email_jwt_token(email, jwt_token) first."

    headers = {"Authorization": f"Bearer {jwt}"}
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"https://api.mail.tm/messages/{message_id}", headers=headers
            )
            if resp.status_code != 200:
                return f"Failed to fetch message. Status: {resp.status_code}, Response: {resp.text}"
            msg = resp.json()
            sender = msg.get("from") or {}
            result = {
                "id": msg.get("id"),
                "subject": msg.get("subject"),
                "from": sender.get("address") or sender.get("name") or "",
                "text": msg.get("text", ""),
                "html": msg.get("html", ""),
            }
            return json_module.dumps(result)
    except Exception as e:
        return f"Request failed: {e}"


@function_tool(name_override="send_slack_alert")
async def send_slack_security_alert(
    vulnerability_type: str,
    severity: str,
    target_url: str,
    description: str,
    evidence: Optional[str] = None,
    recommendation: Optional[str] = None,
    thread_ts: Optional[str] = None
):
    """
    Send a security vulnerability alert to Slack channel.
    
    Args:
        vulnerability_type: Type of vulnerability (e.g., "XSS", "SQL Injection", "IDOR")
        severity: Severity level ("Critical", "High", "Medium", "Low", "Info")
        target_url: The affected URL or endpoint
        description: Detailed description of the vulnerability
        evidence: Optional proof-of-concept or evidence details
        recommendation: Optional remediation recommendation
        thread_ts: Optional thread timestamp to reply to existing thread
    """
    
    # Severity color mapping
    severity_colors = {
        "Critical": "#FF0000",  # Red
        "High": "#FF6600",      # Orange
        "Medium": "#FFB84D",    # Yellow-Orange
        "Low": "#FFCC00",       # Yellow
        "Info": "#0099FF"       # Blue
    }
    
    # Severity emoji mapping
    severity_emojis = {
        "Critical": "🚨",
        "High": "⚠️",
        "Medium": "⚡",
        "Low": "📝",
        "Info": "ℹ️"
    }
    
    color = severity_colors.get(severity, "#808080")
    emoji = severity_emojis.get(severity, "📌")
    
    # Build Slack message with blocks for rich formatting
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {vulnerability_type} Vulnerability Detected",
                "emoji": True
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Severity:*\n{severity}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Target:*\n<{target_url}|{target_url}>"
                }
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Description:*\n{description}"
            }
        }
    ]
    
    # Add evidence if provided
    if evidence:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Evidence/PoC:*\n```{evidence[:500]}```"  # Limit evidence length
            }
        })
    
    # Add recommendation if provided
    if recommendation:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Recommendation:*\n{recommendation}"
            }
        })
    
    # Add timestamp
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"Detected at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            }
        ]
    })
    
    # Fallback text for notifications
    fallback_text = f"{emoji} {severity} {vulnerability_type} vulnerability found at {target_url}"
    
    # Send via webhook
    if SLACK_WEBHOOK_URL:
        payload = {
            "channel": SLACK_CHANNEL,
            "username": "Security Scanner Bot",
            "icon_emoji": ":shield:",
            "text": fallback_text,
            "blocks": blocks,
            "attachments": [
                {
                    "color": color,
                    "fallback": fallback_text
                }
            ]
        }
        
        if thread_ts:
            payload["thread_ts"] = thread_ts
        
        async with aiohttp.ClientSession() as session:
            async with session.post(SLACK_WEBHOOK_URL, json=payload) as response:
                if response.status == 200:
                    return json_module.dumps({"success": True, "message": "Alert sent to Slack successfully"})
                else:
                    error_text = await response.text()
                    return json_module.dumps({"success": False, "error": f"Failed to send to Slack: {error_text}"})
    else:
        return json_module.dumps({
            "success": False, 
            "error": "No Slack webhook configured. Set SLACK_WEBHOOK_URL in .env file"
        })


@function_tool(name_override="send_slack_summary")
async def send_slack_scan_summary(
    target_url: str,
    total_findings: int,
    critical_count: int = 0,
    high_count: int = 0,
    medium_count: int = 0,
    low_count: int = 0,
    scan_duration: Optional[str] = None
):
    """
    Send a summary of the security scan to Slack.
    
    Args:
        target_url: The target that was scanned
        total_findings: Total number of vulnerabilities found
        critical_count: Number of critical severity findings
        high_count: Number of high severity findings
        medium_count: Number of medium severity findings
        low_count: Number of low severity findings
        scan_duration: Optional duration of the scan
    """
    
    # Determine overall status
    if critical_count > 0:
        status_emoji = "🔴"
        status_text = "Critical Issues Found"
        color = "#FF0000"
    elif high_count > 0:
        status_emoji = "🟠"
        status_text = "High Risk Issues Found"
        color = "#FF6600"
    elif medium_count > 0:
        status_emoji = "🟡"
        status_text = "Medium Risk Issues Found"
        color = "#FFB84D"
    elif low_count > 0:
        status_emoji = "🟢"
        status_text = "Low Risk Issues Found"
        color = "#00FF00"
    else:
        status_emoji = "✅"
        status_text = "No Issues Found"
        color = "#00FF00"
    
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{status_emoji} Security Scan Summary",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Target:* <{target_url}|{target_url}>\n*Status:* {status_text}\n*Total Findings:* {total_findings}"
            }
        }
    ]
    
    # Add findings breakdown if any exist
    if total_findings > 0:
        findings_text = []
        if critical_count > 0:
            findings_text.append(f"🚨 Critical: {critical_count}")
        if high_count > 0:
            findings_text.append(f"⚠️ High: {high_count}")
        if medium_count > 0:
            findings_text.append(f"⚡ Medium: {medium_count}")
        if low_count > 0:
            findings_text.append(f"📝 Low: {low_count}")
        
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Findings Breakdown:*\n" + "\n".join(findings_text)
            }
        })
    
    # Add scan duration if provided
    if scan_duration:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Scan Duration: {scan_duration} | Completed: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
                }
            ]
        })
    
    fallback_text = f"{status_emoji} Security scan completed for {target_url}: {total_findings} findings"
    
    # Send via webhook
    if SLACK_WEBHOOK_URL:
        payload = {
            "channel": SLACK_CHANNEL,
            "username": "Security Scanner Bot",
            "icon_emoji": ":shield:",
            "text": fallback_text,
            "blocks": blocks,
            "attachments": [
                {
                    "color": color,
                    "fallback": fallback_text
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(SLACK_WEBHOOK_URL, json=payload) as response:
                if response.status == 200:
                    return json_module.dumps({"success": True, "message": "Summary sent to Slack successfully"})
                else:
                    error_text = await response.text()
                    return json_module.dumps({"success": False, "error": f"Failed to send to Slack: {error_text}"})
    else:
        return json_module.dumps({
            "success": False,
            "error": "No Slack webhook configured. Set SLACK_WEBHOOK_URL in .env file"
        })


@function_tool(name_override="sandbox_agent")
async def run_sandbox_agent(instruction: str, max_rounds: int = 100):
    """
    Nested agent loop that uses only sandbox execution tools to fulfill the provided instruction.
    Returns the final textual response when the model stops requesting tools or when max_rounds is hit.
    
    Args:
        instruction: The instruction for the sandbox agent to execute
        max_rounds: Maximum number of execution rounds (default: 100)
    """
    sandbox_system_prompt = os.getenv(
        "SANDBOX_SYSTEM_PROMPT",
        (
            "You are an agent that autonomously interacts with an isolated sandbox using two tools: "
            "`sandbox_run_command` (bash) and `sandbox_run_python` (Python). Keep responses within 30,000 "
            "characters; chunk large outputs. Think step-by-step before taking actions."
        ),
    )

    sandbox_input_list = [
        {
            "role": "developer",
            "content": [
                {"type": "input_text", "text": sandbox_system_prompt},
            ],
        },
        {"role": "user", "content": instruction},
    ]

    # Restrict to the low-level sandbox tools to avoid recursive nesting
    sandbox_tools = [
        t for t in tools if t.get("name") in ("sandbox_run_command", "sandbox_run_python")
    ]

    # print(f"[debug] Sandbox input list: {sandbox_input_list}")

    rounds_completed = 0
    while True:
        response = await create_response(
            sandbox_tools,
            sandbox_input_list,
            {"name": "sandbox_agent"},
        )

        # Log sandbox agent usage
        usage_tracker = get_current_usage_tracker()
        if usage_tracker and hasattr(response, 'usage'):
            usage_tracker.log_sandbox_agent_usage(response.usage, getattr(_thread_local, 'current_target_url', ''))

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        # print(f"[debug] Function calls: {function_calls}")

        if not function_calls:
            output_text = ""
            for item in response.output:
                if item.type == "message" and hasattr(item, 'content'):
                    for content_item in item.content:
                        if hasattr(content_item, 'text'):
                            output_text += content_item.text
            # print(output_text)
            return output_text or ""

        # Record model tool requests and execute them in parallel
        sandbox_input_list.extend(response.output)
        tasks = [
            execute_function_call(function_call) for function_call in function_calls
        ]
        results = await asyncio.gather(*tasks)

        sandbox_input_list.extend(results)
        rounds_completed += 1

        if max_rounds and rounds_completed >= max_rounds:
            return f"[sandbox_agent] Reached max rounds limit: {max_rounds}"
        
@function_tool(name_override="validator_agent")
async def run_validator_agent(instruction: str, max_rounds: int = 50):
    """
    Agent loop specialized for validating Proofs-of-Concept (PoCs) in the sandbox.
    Use only sandbox tools, keep outputs concise, and return a clear verdict.

    Args:
        instruction: Validation instruction that includes the PoC and expected outcome
        max_rounds: Maximum number of execution rounds (default: 50)
    """
    validator_system_prompt = os.getenv(
        "VALIDATOR_SYSTEM_PROMPT",
        (
            "You validate security PoCs inside an isolated sandbox using two tools: "
            "`sandbox_run_command` (bash) and `sandbox_run_python` (Python). Your goal is to: "
            "(1) Reproduce the PoC minimally and safely, (2) Capture evidence (stdout, file diffs, HTTP responses), "
            "(3) Decide if the PoC reliably demonstrates a real vulnerability with impact, (4) Provide a concise verdict. "
            "Always think step-by-step before actions. Keep outputs within 30,000 characters and chunk large outputs. "
            "Avoid destructive actions unless explicitly required for validation."
        ),
    )

    validator_input_list = [
        {
            "role": "developer",
            "content": [
                {"type": "input_text", "text": validator_system_prompt},
            ],
        },
        {"role": "user", "content": instruction},
    ]

    validator_tools = [
        t for t in tools if t.get("name") in ("sandbox_run_command", "sandbox_run_python")
    ]

    rounds_completed = 0
    while True:
        response = await create_response(
            validator_tools,
            validator_input_list,
            {"name": "validator_agent"},
        )

        # Reuse sandbox usage tracker for validator agent
        usage_tracker = get_current_usage_tracker()
        if usage_tracker and hasattr(response, 'usage'):
            usage_tracker.log_sandbox_agent_usage(response.usage, getattr(_thread_local, 'current_target_url', ''))

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        if not function_calls:
            output_text = ""
            for item in response.output:
                if item.type == "message" and hasattr(item, 'content'):
                    for content_item in item.content:
                        if hasattr(content_item, 'text'):
                            output_text += content_item.text
            return output_text or ""

        validator_input_list.extend(response.output)
        tasks = [
            execute_function_call(function_call) for function_call in function_calls
        ]
        results = await asyncio.gather(*tasks)

        validator_input_list.extend(results)
        rounds_completed += 1

        if max_rounds and rounds_completed >= max_rounds:
            return f"[validator_agent] Reached max rounds limit: {max_rounds}"
        
@function_tool
async def sandbox_run_python(python_code: str, timeout: int = 120):
    """
    Run Python code inside a Docker sandbox and return stdout/stderr/exit code. If the output exceeds 30000 characters, output will be truncated before being returned to you.

    Args:
        python_code: Python code to execute (e.g., "print('Hello World')").
        timeout: Max seconds to wait before timing out the code execution.

    Returns:
        A string containing exit code, stdout, and stderr.
    """

    print(f"Running Python code: {python_code[:100]}...")
    try:
        # Get the current sandbox instance
        sbx = get_current_sandbox()
        if sbx is None:
            return "Error: No sandbox instance available for this scan"
            
        import uuid
        # Generate a random script name
        script_name = f"temp_script_{uuid.uuid4().hex[:8]}.py"
        script_path = f"/home/user/{script_name}"
        
        # Write Python code to a temporary file with random name
        sbx.files.write(script_path, python_code)
        
        # Execute the Python script using configured sandbox
        result = sbx.commands.run(f"source .venv/bin/activate && python3 {script_path}", timeout=timeout, user="root")

        stdout_raw = (
            result.stdout
            if hasattr(result, "stdout") and result.stdout is not None
            else ""
        )
        stderr_raw = (
            result.stderr
            if hasattr(result, "stderr") and result.stderr is not None
            else ""
        )
        exit_code = result.exit_code if hasattr(result, "exit_code") else "unknown"

        output = f"Exit code: {exit_code}\n\nSTDOUT\n{stdout_raw}\n\nSTDERR\n{stderr_raw}"

        # Truncate output if it exceeds 30000 characters
        if len(output) > 30000:
            output = (
                output[:30000]
                + "\n...[OUTPUT TRUNCATED - EXCEEDED 30000 CHARACTERS]"
            )

        return output
    except Exception as e:
        return f"Failed to run Python code in sandbox: {e}"


@function_tool
async def sandbox_run_command(command: str, timeout: int = 120):
    """
    Run a shell command inside an ephemeral sandbox and return stdout/stderr/exit code.

    Arguments:
        command: Shell command to execute (e.g., "ls -la").
        timeout: Max seconds to wait before timing out the command.

    Returns:
        A string containing exit code, stdout, and stderr.
    """

    print(f"Running command: {command}")
    try:
        # Get the current sandbox instance
        sbx = get_current_sandbox()
        if sbx is None:
            return "Error: No sandbox instance available for this scan"
            
        # Use the current sandbox instance
        result = sbx.commands.run(command, timeout=timeout, user="root")

        def clip_to_max_lines(text: str, max_lines: int = 100) -> str:
            if not text:
                return ""
            lines = text.splitlines()
            if len(lines) <= max_lines:
                return "\n".join(lines)
            visible = "\n".join(lines[:max_lines])
            remaining = len(lines) - max_lines
            return f"{visible}\n...[TRUNCATED {remaining} more lines]"

        stdout_raw = (
            result.stdout
            if hasattr(result, "stdout") and result.stdout is not None
            else ""
        )
        stderr_raw = (
            result.stderr
            if hasattr(result, "stderr") and result.stderr is not None
            else ""
        )
        # stdout = clip_to_max_lines(stdout_raw, 50)
        # stderr = clip_to_max_lines(stderr_raw, 50)
        exit_code = result.exit_code if hasattr(result, "exit_code") else "unknown"

        return f"Exit code: {exit_code}\n\nSTDOUT\n{stdout_raw}\n\nSTDERR\n{stderr_raw}"
    except Exception as e:
        return f"Failed to run command in sandbox: {e}"

# Collect all function tools that were decorated
_function_tools = {
    "sandbox_run_command": sandbox_run_command,
    "sandbox_run_python": sandbox_run_python,
    "sandbox_agent": run_sandbox_agent,
    "validator_agent": run_validator_agent,
    "get_message_by_id": get_message_by_id,
    "list_account_messages": list_account_messages,
    "get_registered_emails": get_registered_emails,
    "create_email_account": create_email_account,
    "set_email_jwt_token": set_email_jwt_token,
    "send_slack_alert": send_slack_security_alert,
    "send_slack_summary": send_slack_scan_summary,
}

async def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    try:
        if name in _function_tools:
            func_tool = _function_tools[name]
            if name == "sandbox_agent":
                # Handle legacy 'input' parameter or new 'instruction' parameter
                instruction = arguments.get("instruction", arguments.get("input", ""))
                max_rounds = arguments.get("max_rounds", 100)
                out = await func_tool(instruction, max_rounds)
            else:
                out = await func_tool(**arguments)
        else:
            out = {"error": f"Unknown tool: {name}", "args": arguments}
    except Exception as e:
        out = {"error": str(e), "args": arguments}
    return json.dumps(out)


def generate_tools_from_function_tools():
    """Auto-generate tools list from decorated functions."""
    tools = []
    
    for _, func_tool in _function_tools.items():
        # Each function tool should have the FunctionTool attributes
        if hasattr(func_tool, 'name') and hasattr(func_tool, 'description') and hasattr(func_tool, 'params_json_schema'):
            tool_def = {
                "type": "function",
                "name": func_tool.name,
                "description": func_tool.description,
                "parameters": func_tool.params_json_schema,
                "strict": getattr(func_tool, 'strict_json_schema', True),
            }
            tools.append(tool_def)
    
    return tools

# Generate tools automatically from decorated functions
tools = generate_tools_from_function_tools()


# Default scan instruction. `{target_url}` is substituted per target.
DEFAULT_USER_PROMPT = (
    "I need you to do a full vulnerability scan of {target_url}, you must critically "
    "analyse the code and identify every single vulnerability, for identified "
    "vulnerabilities a PoC must be provided, focus on critical vulnerabilities, i m only "
    "insterested in real world vulnerabilities, not theoretical ones"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are MAPTA, an autonomous penetration-testing agent. You operate only against "
    "targets the operator has explicitly authorised. Delegate hands-on work to the "
    "`sandbox_agent` tool and confirm every finding with the `validator_agent` tool before "
    "reporting it. Never report a vulnerability you have not reproduced. Produce a final "
    "Markdown report listing each confirmed vulnerability with severity, affected "
    "endpoint, a reproducible PoC, observed impact and a remediation recommendation."
)


def read_targets_from_file(file_path: str) -> List[str]:
    """
    Read target URLs from a text file, one per line.
    Ignores empty lines and lines starting with #.
    """
    targets = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    targets.append(line)
        return targets
    except FileNotFoundError:
        print(f"Error: Target file '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading target file: {e}")
        return []

async def run_continuously(max_rounds: int = 100, user_prompt: str = "", system_prompt: str = "", target_url: str = "", sandbox_instance=None):
    """
    Keep prompting the model and executing any requested tool calls in parallel
    until the model stops requesting tools or the optional max_rounds is reached.

    max_rounds: 0 means unlimited; otherwise, loop up to max_rounds tool-execution rounds.
    target_url: The target URL being scanned (used for metadata)
    sandbox_instance: Specific sandbox instance to use for this scan
    """
    # Create sandbox instance if not provided
    if sandbox_instance is None:
        sandbox_instance = create_sandbox_from_env()
    
    # Set the sandbox for this thread/scan
    set_current_sandbox(sandbox_instance)
    
    # Set target URL for usage tracking
    _thread_local.current_target_url = target_url
    
    rounds_completed = 0

    input_list = [
    {"role": "developer", "content": [{"type": "input_text", "text": system_prompt}]},
    {
        "role": "user",
        "content": user_prompt,
    }]

    # Everything except the low-level sandbox tools, which only the nested
    # sandbox/validator agents are allowed to call.
    main_agent_tools = [
        t for t in tools
        if t.get("name") not in ("sandbox_run_command", "sandbox_run_python")
    ]

    # Extract site name from URL for metadata
    site_name = target_url.replace("https://", "").replace("http://", "").split('/')[0] if target_url else "unknown"

    try:
        while True:
            # 1) Ask the model what to do next
            response = await create_response(
                main_agent_tools,
                input_list,
                {
                    "name": "security_scan",
                    "site_name": site_name,
                    "target_url": target_url,
                },
            )

            # Log main agent usage
            usage_tracker = get_current_usage_tracker()
            if usage_tracker and hasattr(response, 'usage'):
                usage_tracker.log_main_agent_usage(response.usage, target_url)

            # 2) Check for function calls
            function_calls = [
                item for item in response.output if item.type == "function_call"
            ] 

            # If there are no tool calls, print whatever the model returned and stop
            if not function_calls:
                output_text = ""
                for item in response.output:
                    if item.type == "message" and hasattr(item, 'content'):
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                output_text += content_item.text
                        break
                print(output_text)
                print(response.id)
                return output_text

            # 3) Record the function calls in the conversation and execute them in parallel
            input_list.extend(response.output)
            print(f"[debug] Executing {len(function_calls)} function calls in parallel...")

            tasks = [
                execute_function_call(function_call) for function_call in function_calls
            ]
            results = await asyncio.gather(*tasks)

            # 4) Add tool results for the next round
            input_list.extend(results)
            rounds_completed += 1

            # 5) Safety valve for infinite loops
            if max_rounds and rounds_completed >= max_rounds:
                print(f"[debug] Reached max rounds limit: {max_rounds}")
                return f"[main_agent] Reached max rounds limit: {max_rounds}"
    finally:
        # Kill the sandbox when scan is done
        if sandbox_instance and hasattr(sandbox_instance, "kill"):
            sandbox_instance.kill()

def slugify_target(target_url: str) -> str:
    """Turn a target URL into a string usable as a filename."""
    slug = target_url.split("://", 1)[-1]
    slug = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in slug)
    return slug.strip("_") or "target"


async def run_single_target_scan(target_url: str, system_prompt: str, base_user_prompt: str, max_rounds: int = 100, output_dir: str = "scan-results"):
    """
    Run a security scan for a single target URL.
    Returns the scan result and saves it to a file.
    Each scan gets its own isolated sandbox instance.
    """
    print(f"Starting scan for: {target_url}")
    os.makedirs(output_dir, exist_ok=True)

    # Create a dedicated sandbox instance for this scan (if configured)
    sandbox_instance = create_sandbox_from_env()
    
    # Create usage tracker for this scan
    usage_tracker = UsageTracker()
    set_current_usage_tracker(usage_tracker)
    
    # Format the user prompt with the target URL
    user_prompt = base_user_prompt.format(target_url=target_url)
    
    try:
        # Run the scan with dedicated sandbox
        result = await run_continuously(
            user_prompt=user_prompt, 
            system_prompt=system_prompt, 
            target_url=target_url,
            max_rounds=max_rounds,
            sandbox_instance=sandbox_instance
        )
        
        # Generate filename from target URL
        filename = os.path.join(output_dir, slugify_target(target_url) + ".md")

        # Save result to file (the agent may finish without any text output)
        with open(filename, "w", encoding='utf-8') as f:
            f.write(result or "[no report produced by the agent]")

        # Save usage data
        host = target_url.split("://", 1)[-1].split("/")[0]
        usage_filename = usage_tracker.save_to_file(os.path.join(output_dir, f"{slugify_target(host)}_"))
        
        print(f"Scan completed for {target_url} - Results saved to {filename}")
        print(f"Usage data saved to {usage_filename}")
        
        return {
            "target": target_url,
            "filename": filename,
            "usage_filename": usage_filename,
            "status": "completed",
            "result": result,
            "usage_summary": usage_tracker.get_summary()
        }
        
    except Exception as e:
        print(f"Error scanning {target_url}: {e}")
        # The tokens were already paid for, so persist the usage data even though
        # the scan failed, instead of discarding it with the exception.
        usage_filename = None
        try:
            host = target_url.split("://", 1)[-1].split("/")[0]
            usage_filename = usage_tracker.save_to_file(
                os.path.join(output_dir, f"{slugify_target(host)}_failed_")
            )
            print(f"Usage data saved to {usage_filename}")
        except Exception as save_error:
            print(f"Could not save usage data: {save_error}")

        return {
            "target": target_url,
            "filename": None,
            "usage_filename": usage_filename,
            "status": "error",
            "error": str(e),
            "usage_summary": usage_tracker.get_summary(),
        }

async def run_parallel_scans(targets: List[str], system_prompt: str, base_user_prompt: str, max_rounds: int = 100, output_dir: str = "scan-results"):
    """
    Run security scans for multiple targets in parallel.
    """
    print(f"Starting parallel scans for {len(targets)} targets...")
    
    # Create tasks for all targets
    tasks = [
        run_single_target_scan(target, system_prompt, base_user_prompt, max_rounds, output_dir)
        for target in targets
    ]
    
    # Run all scans in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    completed = 0
    errors = 0
    
    for result in results:
        if isinstance(result, Exception):
            print(f"Task failed with exception: {result}")
            errors += 1
        elif result.get("status") == "completed":
            completed += 1
        else:
            errors += 1
    
    print(f"\nScan Summary:")
    print(f"Total targets: {len(targets)}")
    print(f"Completed successfully: {completed}")
    print(f"Failed: {errors}")
    
    # Create overall usage summary
    total_main_calls = 0
    total_sandbox_calls = 0
    usage_files = []
    
    # Count failed scans too: their tokens were billed all the same.
    for result in results:
        if isinstance(result, dict):
            summary = result.get("usage_summary", {})
            total_main_calls += summary.get("main_agent_calls", 0)
            total_sandbox_calls += summary.get("sandbox_agent_calls", 0)
            if result.get("usage_filename"):
                usage_files.append(result["usage_filename"])
    
    print(f"\nUsage Summary:")
    print(f"Total Main Agent API calls: {total_main_calls}")
    print(f"Total Sandbox Agent API calls: {total_sandbox_calls}")
    print(f"Total API calls: {total_main_calls + total_sandbox_calls}")
    print(f"Usage files created: {len(usage_files)}")
    for uf in usage_files:
        print(f"  - {uf}")
    
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapta",
        description="MAPTA - autonomous multi-agent penetration testing. "
                    "Only run this against targets you are authorised to test.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "-t", "--target",
        action="append",
        metavar="URL",
        help="Target URL to scan. Repeat the flag to scan several targets in parallel.",
    )
    source.add_argument(
        "-f", "--targets-file",
        default="targets.txt",
        metavar="PATH",
        help="File with one target URL per line ('#' starts a comment). Default: targets.txt",
    )
    parser.add_argument(
        "-p", "--prompt",
        default=os.getenv("USER_PROMPT", DEFAULT_USER_PROMPT),
        help="Scan instruction template. '{target_url}' is replaced with each target. "
             "Defaults to $USER_PROMPT or the built-in full-scan prompt.",
    )
    parser.add_argument(
        "-s", "--system-prompt",
        default=os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        help="System prompt for the main agent. Defaults to $SYSTEM_PROMPT or the built-in one.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=os.getenv("OUTPUT_DIR", "scan-results"),
        help="Directory for reports and usage logs. Default: scan-results",
    )
    parser.add_argument(
        "-r", "--max-rounds",
        type=int,
        default=int(os.getenv("MAX_ROUNDS", "100")),
        help="Maximum tool-execution rounds for the main agent (0 = unlimited). Default: 100",
    )
    parser.add_argument(
        "--log-file",
        default="scan_usage.log",
        help="Path to the run log. Default: scan_usage.log",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help=f"Model id to use. Defaults to $MODEL or the provider default ({MODEL}).",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDERS),
        default=None,
        help=f"LLM provider. Defaults to $LLM_PROVIDER, else inferred from the API keys "
             f"present (currently: {LLM_PROVIDER}).",
    )
    parser.add_argument(
        "--reasoning",
        choices=["high", "medium", "low", "none"],
        default=None,
        help=f"Reasoning effort. Use 'none' for models that reject a reasoning field. "
             f"Defaults to $REASONING_EFFORT (currently: {REASONING_EFFORT}).",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the generated tool schemas and exit (no API key required).",
    )
    return parser


def main() -> int:
    global LLM_PROVIDER, _PROVIDER, MODEL, SEND_METADATA, REASONING_EFFORT

    args = build_arg_parser().parse_args()

    if args.list_tools:
        print(json.dumps(tools, indent=2))
        return 0

    # --provider / --model override the environment-derived defaults.
    if args.provider and args.provider != LLM_PROVIDER:
        LLM_PROVIDER = args.provider
        _PROVIDER = _PROVIDERS[LLM_PROVIDER]
        if not args.model and not os.getenv("MODEL"):
            MODEL = _PROVIDER["model"]
        if not os.getenv("SEND_METADATA"):
            SEND_METADATA = LLM_PROVIDER == "openai"
    if args.model:
        MODEL = args.model
    if args.reasoning:
        REASONING_EFFORT = args.reasoning

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )

    # Resolve the target list: explicit --target flags win over the targets file.
    if args.target:
        targets = args.target
        print(f"Scanning {len(targets)} target(s) from the command line")
    elif os.path.exists(args.targets_file):
        print(f"Found targets file: {args.targets_file}")
        targets = read_targets_from_file(args.targets_file)
        if not targets:
            print(f"No valid targets found in '{args.targets_file}'.")
            return 1
        print(f"Found {len(targets)} targets to scan")
    else:
        print(
            f"No targets given: pass --target URL, or create '{args.targets_file}' with one "
            "URL per line. See --help."
        )
        return 1

    if "{target_url}" not in args.prompt:
        print(
            "[warn] the scan prompt does not contain '{target_url}'; the agent will not be "
            "told which target to scan."
        )

    if not os.getenv("SANDBOX_FACTORY"):
        print(
            "[warn] SANDBOX_FACTORY is not set, so the sandbox tools will fail. "
            "See README.md for how to plug in a sandbox provider."
        )

    print(f"Provider: {LLM_PROVIDER} | model: {MODEL}")
    try:
        get_client()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 2

    results = asyncio.run(
        run_parallel_scans(
            targets,
            args.system_prompt,
            args.prompt,
            max_rounds=args.max_rounds,
            output_dir=args.output_dir,
        )
    )

    print("\nAll scans completed!")
    failed = sum(
        1 for r in results
        if isinstance(r, Exception) or not (isinstance(r, dict) and r.get("status") == "completed")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
