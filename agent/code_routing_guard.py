"""Hard tool-level guard enforcing Claude/Fable code routing (Phase 2).

Phase 1 (commit ``dae9d03bd``) added *prompt* guidance: a stable-tier system
prompt block telling Hermes to delegate code implementation / debugging /
refactor / deploy / technical-audit work to Claude (Fable) via the Claude Agent
SDK, orchestrate and verify rather than editing code directly. Prompt guidance
is advisory — the model can still ignore it and reach for ``write_file`` or
``terminal`` on the primary path.

This module adds *enforcement* at the tool-execution layer. When the active
user request looks like code work and Hermes tries to use a code-mutation /
code-execution tool directly, the call is blocked with an actionable message
unless one of the accepted exceptions applies:

  * a Claude/Fable delegation or audit has already been attempted this
    turn/session (evidenced in the conversation/tool history), OR
  * Jose explicitly asked Hermes to edit directly / skip Fable / emergency, OR
  * Claude/Fable is genuinely unavailable/failing after a real attempt — which
    itself shows up as delegation evidence in the message history.

The controller is intentionally pure: ``evaluate_code_routing_guard`` inspects
the tool name, the parsed args, and the working message list, and returns a
block decision or ``None``. Runtime callers in ``agent/tool_executor.py`` turn
a block into a synthetic ``role=tool`` error result, exactly like a plugin
pre-tool-call block. Subagents (the delegated Claude/Fable implementers) are
never guarded — guarding them would block the very work the delegation routes
to.

Gated by ``config.yaml`` ``agent.hard_code_routing_guard`` (default True).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


# Tools whose direct use is code-mutation or code-execution regardless of the
# argument detail. ``terminal`` is handled separately because only build /
# deploy / code commands count as code work (``ls`` during a code task is fine).
_ALWAYS_CODE_TOOLS = frozenset(
    {
        "write_file",
        "patch",
        "execute_code",
        "process",
        # Filesystem MCP mutators that bypass the native write_file/patch path.
        "mcp_filesystem_write_file",
        "mcp_filesystem_edit_file",
        "mcp_filesystem_move_file",
        "mcp_filesystem_create_directory",
    }
)

# ``terminal`` is guarded only when the command looks like build / deploy /
# code work. Kept as its own constant so callers/tests can introspect it.
_TERMINAL_TOOL = "terminal"

# Default set of tool names the guard may act on. ``terminal`` is included so
# config can drop it, but the per-command check still applies.
DEFAULT_GUARDED_TOOLS = frozenset(_ALWAYS_CODE_TOOLS | {_TERMINAL_TOOL})


# Source/code file extensions used to corroborate that a write_file/patch
# target is code (only consulted as a secondary signal; request intent is the
# primary gate).
_CODE_FILE_EXTENSIONS = frozenset(
    {
        ".py", ".pyi", ".ipynb",
        ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
        ".go", ".rs", ".rb", ".php", ".java", ".kt", ".kts", ".scala",
        ".c", ".h", ".cc", ".cpp", ".hpp", ".cxx", ".m", ".mm",
        ".cs", ".swift", ".dart", ".lua", ".pl", ".pm", ".r",
        ".sh", ".bash", ".zsh", ".fish", ".ps1",
        ".sql", ".graphql", ".gql", ".proto",
        ".vue", ".svelte",
        ".tf", ".tfvars",
    }
)


# Words that, on their own, signal code-implementation/technical intent.
_STRONG_CODE_INTENT = (
    r"implement(?:s|ed|ing|ation)?",
    r"re-?factor(?:s|ed|ing)?",
    r"debug(?:s|ged|ging)?",
    r"deploy(?:s|ed|ing|ment)?",
    r"recompile|compil(?:e|es|ed|ing|ation)",
    r"hotfix",
    r"code\s*review",
    r"technical\s+audit",
    r"code\s+audit",
)

# Verbs that are code work only when paired with a code noun (below).
_WEAK_CODE_VERBS = (
    r"fix(?:es|ed|ing)?",
    r"build(?:s|ing)?",
    r"creat(?:e|es|ed|ing)",
    r"add(?:s|ed|ing)?",
    r"updat(?:e|es|ed|ing)",
    r"writ(?:e|es|ing)|wrote",
    r"chang(?:e|es|ed|ing)",
    r"modif(?:y|ies|ied|ying)",
    r"patch(?:es|ed|ing)?",
    r"rewrit(?:e|es|ing)|rewrote",
    r"migrat(?:e|es|ed|ing|ion)",
    r"upgrad(?:e|es|ed|ing)",
    r"wire\s*up|integrat(?:e|es|ed|ing|ion)",
)

# Single-word nouns that mark a code/engineering context for the weak verbs
# above.  Matched with word boundaries so "report" does not match "repo",
# "rapid" does not match "api", etc.
_CODE_NOUN_WORDS = (
    "bug", "bugs", "code", "codebase", "function", "functions", "method",
    "methods", "class", "classes", "module", "modules", "script", "scripts",
    "endpoint", "endpoints", "api", "apis", "route", "routes", "test", "tests",
    "build", "compiler", "package", "packages", "dependency", "dependencies",
    "migration", "migrations", "schema", "component", "components", "service",
    "services", "feature", "repo", "repository", "branch", "pr", "regression",
    "traceback", "exception", "error", "errors", "linter", "lint", "ci",
    "pipeline", "dockerfile",
)

# Multi-word / non-word-bounded code-context patterns (file extensions etc.).
_CODE_NOUN_PATTERNS = (
    r"unit\s*tests?",
    r"pull\s*requests?",
    r"stack\s*traces?",
    r"type\s*errors?",
    r"\.py\b", r"\.ts\b", r"\.js\b", r"\.go\b", r"\.rs\b",
    r"\.rb\b", r"\.java\b", r"\.cpp\b",
)


def _compile_alt(parts: Iterable[str]) -> re.Pattern[str]:
    return re.compile(r"(?:%s)" % "|".join(parts), re.IGNORECASE)


_STRONG_CODE_RE = _compile_alt(_STRONG_CODE_INTENT)
_WEAK_VERB_RE = _compile_alt(_WEAK_CODE_VERBS)
_CODE_NOUN_RE = re.compile(
    r"\b(?:%s)\b|(?:%s)" % ("|".join(_CODE_NOUN_WORDS), "|".join(_CODE_NOUN_PATTERNS)),
    re.IGNORECASE,
)


# Explicit "edit directly / skip Fable / emergency" phrases. Any match lifts the
# guard for the turn — Jose is overriding the delegate-first policy on purpose.
#
# These must be *deliberate override* phrasings. Merely addressing Hermes by name
# ("Hermes, implement the login endpoint") is an ordinary code request and must
# NOT bypass the guard; the override requires direct/yourself/skip-delegation
# language. Likewise a bare "emergency" token does not count — only explicit
# emergency-override phrases do, so "implement the emergency-stop endpoint" still
# routes to Fable.
_EXPLICIT_DIRECT_EDIT_RE = _compile_alt(
    (
        r"edit\s+(?:it|this|the\s+\w+)?\s*(?:yourself|directly)",
        r"(?:edit|do|fix|implement|change|patch|write)\s+(?:it|this|that)?\s*yourself",
        r"directly\s+edit",
        r"skip\s+(?:fable|claude|delegation|the\s+delegation)",
        r"no\s+delegation|without\s+delegat(?:ing|ion)",
        r"do\s*n['o]?t\s+delegate|do\s+not\s+delegate",
        r"bypass\s+(?:fable|claude|delegation)",
        r"don['o]?t\s+(?:use|route\s+to)\s+(?:fable|claude)",
        # Explicit emergency override — not a bare "emergency" occurrence.
        r"this\s+is\s+(?:a\s+real\s+|an?\s+)?emergency",
        r"it'?s\s+(?:a\s+real\s+|an?\s+)?emergency",
        r"\breal\s+emergency\b",
        r"emergency\s+override",
    )
)


# Phrases showing a delegation attempt was made and Claude/Fable failed or is
# down — counts as delegation evidence so Hermes can apply/verify directly.
_DELEGATION_FAILURE_RE = _compile_alt(
    (
        r"(?:claude|fable)[^.\n]{0,40}(?:unavailable|is\s+down|failed|failing|error|timed?\s*out|not\s+available|cannot)",
        r"delegat(?:ion|ing|ed|e)[^.\n]{0,40}(?:failed|failing|unavailable|error|timed?\s*out)",
        r"claude\s+agent\s+sdk[^.\n]{0,40}(?:failed|unavailable|error)",
        r"fell\s+back\s+from\s+(?:claude|fable)",
    )
)

# Tool names that, when called, prove a delegation/audit was attempted.
_DELEGATION_TOOL_NAMES = frozenset({"delegate_task", "mixture_of_agents"})


@dataclass(frozen=True)
class CodeRoutingGuardConfig:
    """Configuration for the hard code-routing guard.

    Built from the ``config.yaml`` ``agent.hard_code_routing_guard`` value:
    a bool (the common case) or a mapping for finer control.
    """

    enabled: bool = True
    guarded_tools: frozenset[str] = field(default_factory=lambda: DEFAULT_GUARDED_TOOLS)

    @classmethod
    def from_value(cls, value: Any) -> "CodeRoutingGuardConfig":
        """Coerce a config value (bool or mapping) into a config object."""
        if value is None:
            return cls()
        if isinstance(value, bool):
            return cls(enabled=value)
        if isinstance(value, Mapping):
            enabled = _as_bool(value.get("enabled"), True)
            tools = value.get("guarded_tools")
            if isinstance(tools, (list, tuple, set, frozenset)):
                names = frozenset(str(t).strip() for t in tools if str(t).strip())
                if names:
                    return cls(enabled=enabled, guarded_tools=names)
            return cls(enabled=enabled)
        # Strings like "true"/"false"; anything else falls back to default-on.
        return cls(enabled=_as_bool(value, True))


@dataclass(frozen=True)
class CodeRoutingDecision:
    """Result of evaluating the guard for one tool call."""

    blocked: bool
    code: str = "allow"
    message: str = ""
    tool_name: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "code": self.code,
            "tool_name": self.tool_name,
        }


def is_code_request(text: str | None) -> bool:
    """Heuristic: does the user request look like code work?

    True for implementation / debugging / refactor / deploy / technical-audit
    intent. Conservative on weak verbs (``fix``, ``build``, ``add`` …): they
    only count when paired with a code noun, so "add a calendar invite" or
    "build a campaign" do not trip the guard while "fix the build" or
    "add an endpoint" do.
    """
    if not text or not isinstance(text, str):
        return False
    if _STRONG_CODE_RE.search(text):
        return True
    verb = _WEAK_VERB_RE.search(text)
    if not verb:
        return False
    # The code noun must be a distinct token from the verb itself — otherwise a
    # word that is both a verb and a noun ("build") would self-satisfy and trip
    # on "build a marketing campaign".
    return any(noun.start() != verb.start() for noun in _CODE_NOUN_RE.finditer(text))


def is_explicit_direct_edit(text: str | None) -> bool:
    """True when the user explicitly told Hermes to edit directly / skip Fable."""
    if not text or not isinstance(text, str):
        return False
    return bool(_EXPLICIT_DIRECT_EDIT_RE.search(text))


def is_code_terminal_command(command: str | None) -> bool:
    """True when a terminal command looks like build / deploy / code work.

    Read-only or housekeeping commands (``ls``, ``cat``, ``cd``, ``git status``)
    are not code work; compilers, test runners, package managers, build tools,
    git mutations, and deploy commands are.
    """
    if not command or not isinstance(command, str):
        return False
    lowered = command.lower()
    for segment in re.split(r"&&|\|\||;|\|", lowered):
        token = segment.strip()
        if not token:
            continue
        head = token.split()
        if not head:
            continue
        first = head[0]
        # Tool runners that may prefix the real command.
        if first in {"sudo", "env", "time", "nice", "nohup", "xargs"} and len(head) > 1:
            first = head[1]
        if _is_code_command_token(first, token):
            return True
    return False


def _is_code_command_token(first: str, full: str) -> bool:
    # Build systems / compilers / bundlers.
    code_commands = {
        "make", "cmake", "ninja", "bazel", "buck", "gradle", "gradlew",
        "mvn", "ant", "cargo", "go", "gcc", "g++", "clang", "clang++",
        "rustc", "javac", "tsc", "webpack", "vite", "rollup", "esbuild",
        "babel", "swc", "dotnet", "msbuild", "xcodebuild",
        # Package / dependency managers (mutate the build).
        "npm", "pnpm", "yarn", "bun", "pip", "pip3", "poetry", "uv",
        "pipenv", "bundle", "gem", "composer", "nuget",
        # Test runners.
        "pytest", "tox", "nox", "jest", "vitest", "mocha", "phpunit",
        "rspec", "ctest",
        # Code intelligence / quality.
        "ruff", "black", "isort", "flake8", "mypy", "pyright", "pylint",
        "eslint", "prettier", "clippy", "rubocop", "golangci-lint",
        # Deploy / orchestration.
        "docker", "docker-compose", "kubectl", "helm", "terraform",
        "ansible", "vagrant", "serverless", "sls", "vercel", "netlify",
        "fly", "flyctl", "heroku", "pulumi", "skaffold",
        # Migrations / app runners commonly used during code work.
        "alembic", "flask", "django-admin", "rails", "rake", "node",
        "deno", "python", "python3", "ruby", "php",
    }
    if first in code_commands:
        return True
    # python -m build/pytest/pip etc. is already caught by the python head.
    # git is only code work when it mutates (commit/push/merge/rebase/...).
    if first == "git":
        return bool(
            re.search(
                r"\bgit\s+(commit|push|merge|rebase|cherry-pick|revert|reset"
                r"|tag|apply|am|am\b|stash|checkout\s+-b|switch\s+-c)\b",
                full,
            )
        )
    if first in {"./gradlew", "./mvnw"}:
        return True
    # Inline interpreter execution (python -c, node -e) is running code.
    if first in {"python", "python3", "node", "deno", "ruby", "php"} and (
        " -c " in f" {full} " or " -e " in f" {full} " or " -m " in f" {full} "
    ):
        return True
    return False


def _is_code_path(path: Any) -> bool:
    if not isinstance(path, str) or not path.strip():
        return False
    lowered = path.strip().lower()
    for ext in _CODE_FILE_EXTENSIONS:
        if lowered.endswith(ext):
            return True
    return False


def _tool_targets_code_work(
    tool_name: str,
    args: Mapping[str, Any],
    config: CodeRoutingGuardConfig,
) -> bool:
    """Tool-specific necessary condition for the guard to apply.

    Request intent (``is_code_request``) is the primary gate; this answers the
    second question: *is this particular tool call code-shaped at all?*
    """
    if tool_name not in config.guarded_tools:
        return False
    if tool_name == _TERMINAL_TOOL:
        return is_code_terminal_command(args.get("command"))
    if tool_name in _ALWAYS_CODE_TOOLS:
        return True
    return False


def _current_turn_messages(
    messages: Sequence[Mapping[str, Any]],
) -> Sequence[Mapping[str, Any]]:
    """Slice ``messages`` to the current turn: everything after the latest user
    message.

    Delegation evidence only counts when it was produced in response to the
    request now being evaluated. A delegation from an earlier turn (before a
    newer user request) is stale and must not unlock a direct edit for the new
    request.
    """
    last_user_idx = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, Mapping) and msg.get("role") == "user":
            last_user_idx = idx
    if last_user_idx < 0:
        return messages
    return messages[last_user_idx + 1 :]


def has_delegation_evidence(messages: Sequence[Mapping[str, Any]] | None) -> bool:
    """True when a Claude/Fable delegation or audit was attempted this turn.

    Scoped to the current turn (messages after the latest user request) so that
    a stale delegation from an earlier request does not unlock a direct edit for
    a newer one. Evidence is any of:
      * an assistant ``tool_calls`` entry naming ``delegate_task`` /
        ``mixture_of_agents`` (a delegation/audit was dispatched), OR
      * message text saying Claude/Fable/delegation failed or is unavailable
        (a real attempt was made and fell back).
    """
    if not messages:
        return False
    for msg in _current_turn_messages(messages):
        if not isinstance(msg, Mapping):
            continue
        for tc in msg.get("tool_calls") or []:
            name = _tool_call_name(tc)
            if name in _DELEGATION_TOOL_NAMES:
                return True
        content = msg.get("content")
        if isinstance(content, str) and content:
            if _DELEGATION_FAILURE_RE.search(content):
                return True
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, Mapping):
                    text = block.get("text")
                    if isinstance(text, str) and _DELEGATION_FAILURE_RE.search(text):
                        return True
    return False


def _tool_call_name(tc: Any) -> str:
    if isinstance(tc, Mapping):
        fn = tc.get("function")
        if isinstance(fn, Mapping):
            return str(fn.get("name") or "")
        return str(tc.get("name") or "")
    fn = getattr(tc, "function", None)
    if fn is not None:
        return str(getattr(fn, "name", "") or "")
    return str(getattr(tc, "name", "") or "")


def latest_user_request(messages: Sequence[Mapping[str, Any]] | None) -> str:
    """Return the most recent user message text from the working message list."""
    if not messages:
        return ""
    for msg in reversed(messages):
        if not isinstance(msg, Mapping) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, Mapping):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
    return ""


def build_block_message(tool_name: str) -> str:
    """Actionable block message shown to the model as a synthetic tool result."""
    return (
        f"Direct code edit blocked by the hard code-routing guard "
        f"(agent.hard_code_routing_guard). You tried to use `{tool_name}` "
        "directly, but this request looks like code implementation/debug/"
        "refactor/deploy/technical-audit work and no Claude (Fable) delegation "
        "has been attempted yet this turn.\n"
        "Delegate the actual coding to Claude (Fable) via the Claude Agent SDK "
        "first (e.g. the `delegate_task` tool). Hermes orchestrates and "
        "verifies — scope the work, hand implementation to Claude/Fable, then "
        "check and report the result — rather than editing code on the primary "
        "path.\n"
        "Accepted exceptions (any one lifts this block):\n"
        "  1. Delegate to Claude/Fable first — once a delegation or audit "
        "attempt is in this turn's tool history, direct edits are allowed to "
        "apply and verify its result.\n"
        "  2. Jose explicitly asks you to edit directly / skip Fable / "
        "treats it as an emergency.\n"
        "  3. Claude/Fable is genuinely unavailable or failing after a real "
        "delegation attempt (the failed attempt itself counts as evidence).\n"
        "Whenever you take an exception, say so plainly in your reply and state "
        "why delegation was skipped."
    )


def evaluate_code_routing_guard(
    *,
    tool_name: str,
    args: Mapping[str, Any] | None,
    messages: Sequence[Mapping[str, Any]] | None,
    config: CodeRoutingGuardConfig | None = None,
    is_subagent: bool = False,
    user_request: str | None = None,
) -> CodeRoutingDecision:
    """Decide whether a direct code-tool call should be blocked.

    Returns a :class:`CodeRoutingDecision`. ``blocked`` is True only when the
    guard is enabled, the caller is not a delegated subagent, the request looks
    like code work, the tool is a code-mutation/execution tool (and, for
    ``terminal``, the command is build/deploy/code), and no exception applies.
    """
    config = config or CodeRoutingGuardConfig()
    args = args if isinstance(args, Mapping) else {}

    allow = CodeRoutingDecision(blocked=False, tool_name=tool_name)

    if not config.enabled:
        return allow
    # The delegated Claude/Fable implementer must be free to edit — guarding it
    # would block the very work delegation routes to.
    if is_subagent:
        return allow
    if not _tool_targets_code_work(tool_name, args, config):
        return allow

    request = user_request if user_request is not None else latest_user_request(messages)

    # Explicit "edit directly / skip Fable / emergency" override.
    if is_explicit_direct_edit(request):
        return CodeRoutingDecision(
            blocked=False, code="explicit_direct_edit_exception", tool_name=tool_name
        )

    if not is_code_request(request):
        return allow

    # A delegation/audit attempt (or a recorded Claude/Fable failure) lifts the
    # block so Hermes can apply and verify the delegated result.
    if has_delegation_evidence(messages):
        return CodeRoutingDecision(
            blocked=False, code="delegation_evidence_exception", tool_name=tool_name
        )

    return CodeRoutingDecision(
        blocked=True,
        code="code_routing_block",
        message=build_block_message(tool_name),
        tool_name=tool_name,
    )


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default
