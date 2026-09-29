"""Default prompts and the agent profiles that use them."""

from dataclasses import dataclass

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_PROMPT",
    "MAIN_AGENT",
    "SANDBOX_AGENT",
    "SANDBOX_TOOLS",
    "VALIDATOR_AGENT",
    "AgentProfile",
]

#: Low-level tools only the nested agents may call.
SANDBOX_TOOLS: tuple[str, ...] = ("sandbox_run_command", "sandbox_run_python")

DEFAULT_USER_PROMPT = (
    "I need you to do a full vulnerability scan of {target_url}, you must critically "
    "analyse the code and identify every single vulnerability, for identified "
    "vulnerabilities a PoC must be provided, focus on critical vulnerabilities, I am only "
    "interested in real world vulnerabilities, not theoretical ones"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are MAPTA, an autonomous penetration-testing agent. You operate only against "
    "targets the operator has explicitly authorised. Delegate hands-on work to the "
    "`sandbox_agent` tool and confirm every finding with the `validator_agent` tool before "
    "reporting it. Never report a vulnerability you have not reproduced. Produce a final "
    "Markdown report listing each confirmed vulnerability with severity, affected "
    "endpoint, a reproducible PoC, observed impact and a remediation recommendation."
)

DEFAULT_SANDBOX_PROMPT = (
    "You are an agent that autonomously interacts with an isolated sandbox using two tools: "
    "`sandbox_run_command` (bash) and `sandbox_run_python` (Python). Keep responses within "
    "30,000 characters; chunk large outputs. Think step-by-step before taking actions."
)

DEFAULT_VALIDATOR_PROMPT = (
    "You validate security PoCs inside an isolated sandbox using two tools: "
    "`sandbox_run_command` (bash) and `sandbox_run_python` (Python). Your goal is to: "
    "(1) Reproduce the PoC minimally and safely, (2) Capture evidence (stdout, file diffs, "
    "HTTP responses), (3) Decide if the PoC reliably demonstrates a real vulnerability with "
    "impact, (4) Provide a concise verdict. Always think step-by-step before actions. Keep "
    "outputs within 30,000 characters and chunk large outputs. Avoid destructive actions "
    "unless explicitly required for validation."
)


@dataclass(frozen=True, slots=True)
class AgentProfile:
    """Static description of one agent loop: who it is and what it may call."""

    name: str
    system_prompt: str
    max_rounds: int
    #: ``None`` means "every registered tool".
    allowed_tools: tuple[str, ...] | None = None
    denied_tools: tuple[str, ...] = ()

    def with_(
        self, *, system_prompt: str | None = None, max_rounds: int | None = None
    ) -> AgentProfile:
        """Copy with the operator's overrides applied."""
        return AgentProfile(
            name=self.name,
            system_prompt=system_prompt or self.system_prompt,
            max_rounds=self.max_rounds if max_rounds is None else max_rounds,
            allowed_tools=self.allowed_tools,
            denied_tools=self.denied_tools,
        )


MAIN_AGENT = AgentProfile(
    name="main_agent",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    max_rounds=100,
    # The main agent delegates execution; it never drives the sandbox itself.
    denied_tools=SANDBOX_TOOLS,
)

SANDBOX_AGENT = AgentProfile(
    name="sandbox_agent",
    system_prompt=DEFAULT_SANDBOX_PROMPT,
    max_rounds=100,
    allowed_tools=SANDBOX_TOOLS,
)

VALIDATOR_AGENT = AgentProfile(
    name="validator_agent",
    system_prompt=DEFAULT_VALIDATOR_PROMPT,
    max_rounds=50,
    allowed_tools=SANDBOX_TOOLS,
)
