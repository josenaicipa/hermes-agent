"""Deterministic structured digests for oversized tool outputs.

Large tool results (CI logs, test runs, CodeQL scans, long poll output) are
the dominant cause of context-compaction pain: they balloon the summarizer
prompt until the auxiliary model times out, and when the LLM summary fails
they survive in the protected tail and leave the "compressed" transcript at
100K+ tokens.

This module builds a bounded, deterministic digest of such outputs — no LLM
call required — that preserves what matters for failure diagnosis:

  * tool name and exit code / status when detectable
  * the first lines (command echo, headers)
  * the last lines (final test summary, exit trailer)
  * lines matching failure/error patterns from the omitted middle
  * file paths / URLs mentioned on those failure lines
  * length statistics (lines / chars / rough tokens) so the model knows how
    much bulk was dropped

The digest is used by the context compressor when serializing tool results
into the summarizer prompt, and as the deterministic shrink applied to
protected tool outputs when the LLM summary fails (fallback compaction).
It never touches the live tool-result path for normal-sized outputs.
"""

from __future__ import annotations

import re

# Outputs at or below this size pass through unchanged — small results carry
# their own context and digesting them would lose more than it saves.
DIGEST_TRIGGER_CHARS = 6_000

# Default ceiling for a produced digest.  Callers can lower it (summarizer
# prompt serialization) or keep it (fallback shrink of protected outputs).
DIGEST_MAX_CHARS = 4_000

_DIGEST_HEADER_PREFIX = "[tool output digest"

# Per-line clip so a single minified-JS / base64 line can't eat the budget.
_LINE_CLIP_CHARS = 240

_HEAD_LINES = 15
_TAIL_LINES = 25
_MAX_FAILURE_LINES = 40

# Failure/error line matcher.  Tuned for CI logs, pytest/jest output, CodeQL
# scan results, compiler output, and shell traces.  Deliberately excludes
# bare "warning" — CI logs emit thousands of benign warnings and they would
# crowd out the actual failures.
_FAILURE_LINE_RE = re.compile(
    r"(?:\b(?:error|errors|err!|fail|failed|failure|failures|failing|fatal|"
    r"exception|traceback|panic(?:ked)?|assert|assertion|denied|refused|"
    r"unauthorized|forbidden|timed.?out|timeout|segfault|segmentation fault|"
    r"core dumped|killed|oom|out of memory|critical|severity|unreachable|"
    r"cannot|could not|unable to|not found|no such file|missing|broken|"
    r"rejected|exit code [1-9]|non-zero)\b|✗|✖|❌|FAILED|ERROR)",
    re.IGNORECASE,
)

# Exit code extraction: JSON tool envelopes ({"exit_code": 1, ...}) and
# plain-text trailers ("exit code 1", "exited with code 2", "Exit status 1").
_EXIT_CODE_RES = (
    re.compile(r'"exit_code"\s*:\s*(-?\d+)'),
    re.compile(r"\bexit(?:ed)?(?:\s+with)?(?:\s+(?:code|status))\s*[:=]?\s*(-?\d+)", re.IGNORECASE),
)

# Path / URL handles worth carrying into the digest.  Restricted to failure
# lines so the digest doesn't enumerate every file a build touched.
_PATH_OR_URL_RE = re.compile(
    r"(?:https?://[^\s`'\")\]}<>]+|(?:/|~/|[A-Za-z]:\\)[^\s`'\")\]}<>:]+)"
)


def is_tool_output_digest(text: str) -> bool:
    """True when *text* is already a digest produced by this module."""
    return isinstance(text, str) and text.lstrip().startswith(_DIGEST_HEADER_PREFIX)


def extract_exit_code(content: str) -> str | None:
    """Best-effort exit code / status extraction from a tool result body."""
    if not content:
        return None
    for pattern in _EXIT_CODE_RES:
        match = pattern.search(content)
        if match:
            return match.group(1)
    return None


def extract_failure_lines(content: str, *, limit: int = 10) -> list[str]:
    """Return up to *limit* unique failure/error lines from *content*.

    Deterministic: lines are returned in order of first appearance, clipped
    to ``_LINE_CLIP_CHARS``, with exact duplicates collapsed.
    """
    if not content:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped in seen:
            continue
        if _FAILURE_LINE_RE.search(stripped):
            seen.add(stripped)
            out.append(_clip_line(stripped))
            if len(out) >= limit:
                break
    return out


def _clip_line(line: str) -> str:
    if len(line) <= _LINE_CLIP_CHARS:
        return line
    return line[: _LINE_CLIP_CHARS - 12] + " ...[clipped]"


def _collect_failure_matches(
    lines: list[str], start_line_no: int
) -> tuple[list[tuple[int, str]], int]:
    """Collect unique (1-based line number, clipped text) failure matches.

    Returns ``(unique_matches, total_match_count)``.  Duplicate line bodies
    (poll loops, repeated retry errors) count toward the total but appear
    only once, annotated at first occurrence.
    """
    matches: list[tuple[int, str]] = []
    seen: set[str] = set()
    total = 0
    for offset, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if not _FAILURE_LINE_RE.search(stripped):
            continue
        total += 1
        if stripped in seen:
            continue
        seen.add(stripped)
        matches.append((start_line_no + offset, _clip_line(stripped)))
    return matches, total


def _select_spread(items: list, limit: int) -> list:
    """Keep the first and last halves of *items* when over *limit*.

    Compile errors cluster early, test summaries cluster late — keeping both
    ends preserves more diagnostic signal than a plain head slice.
    """
    if len(items) <= limit:
        return items
    first = limit // 2
    last = limit - first
    return items[:first] + items[-last:]


def build_tool_output_digest(
    content: str,
    *,
    tool_name: str = "",
    max_chars: int = DIGEST_MAX_CHARS,
    head_lines: int = _HEAD_LINES,
    tail_lines: int = _TAIL_LINES,
    max_failure_lines: int = _MAX_FAILURE_LINES,
) -> str:
    """Build a bounded deterministic digest of a large tool output.

    Always returns a string no longer than ``max_chars`` (modulo a tiny
    constant for the truncation marker).  Same input → same output.
    """
    lines = content.splitlines()
    total_lines = len(lines)
    total_chars = len(content)
    rough_tokens = total_chars // 4
    exit_code = extract_exit_code(content)

    def _assemble(n_head: int, n_tail: int, n_matches: int) -> str:
        head = [_clip_line(l) for l in lines[:n_head]]
        if total_lines > n_head + n_tail:
            tail = [_clip_line(l) for l in lines[total_lines - n_tail:]]
            middle = lines[n_head: total_lines - n_tail]
            middle_start = n_head + 1
        else:
            tail = []
            middle = []
            middle_start = n_head + 1
        matches, total_matches = _collect_failure_matches(middle, middle_start)
        kept = _select_spread(matches, n_matches)

        # Path/URL handles from failure lines only — keeps the list relevant
        # to diagnosis instead of enumerating every artifact the run touched.
        handles: list[str] = []
        seen_handles: set[str] = set()
        for _, match_line in kept:
            for handle in _PATH_OR_URL_RE.findall(match_line):
                handle = handle.rstrip(".,:;")
                if handle and handle not in seen_handles and len(handles) < 8:
                    seen_handles.add(handle)
                    handles.append(handle)

        omitted_lines = max(0, total_lines - len(head) - len(tail))
        header_bits = [
            f"{_DIGEST_HEADER_PREFIX}: {tool_name or 'unknown'}",
            f"{total_lines:,} lines / {total_chars:,} chars (~{rough_tokens:,} tokens)",
        ]
        if exit_code is not None:
            header_bits.append(f"exit_code={exit_code}")
        header_bits.append(
            f"kept first {len(head)} + last {len(tail)} lines"
            + (f" + {len(kept)} of {total_matches} failure/error line(s)" if total_matches else "")
            + f"; {omitted_lines:,} middle lines omitted"
        )
        parts = [" | ".join(header_bits) + "]"]
        if head:
            parts.append("--- first lines ---")
            parts.extend(head)
        if kept:
            parts.append(
                f"--- failure/error lines from omitted middle "
                f"({len(kept)} of {total_matches} matches, deduplicated) ---"
            )
            parts.extend(f"L{line_no}: {text}" for line_no, text in kept)
        if handles:
            parts.append("--- paths/urls on failure lines ---")
            parts.append(", ".join(handles))
        if tail:
            parts.append("--- last lines ---")
            parts.extend(tail)
        return "\n".join(parts)

    # Progressively tighter layouts; the first one that fits wins.  All
    # steps are pure functions of the input, so the result is deterministic.
    layouts = (
        (head_lines, tail_lines, max_failure_lines),
        (max(3, head_lines // 2), max(8, tail_lines // 2), max(10, max_failure_lines // 2)),
        (3, 8, 6),
        (2, 5, 3),
    )
    digest = ""
    for layout in layouts:
        digest = _assemble(*layout)
        if len(digest) <= max_chars:
            return digest
    # Pathological case (e.g. max_chars very small): hard-truncate but keep
    # the header so the stats survive.
    return digest[: max(0, max_chars - 22)].rstrip() + "\n...[digest truncated]"


def maybe_digest_tool_output(
    content,
    *,
    tool_name: str = "",
    trigger_chars: int = DIGEST_TRIGGER_CHARS,
    max_chars: int = DIGEST_MAX_CHARS,
):
    """Digest *content* when it is a large string; return it unchanged otherwise.

    Non-string content (multimodal lists, dict envelopes) and small outputs
    pass through untouched, as do outputs that are already digests.
    """
    if not isinstance(content, str):
        return content
    if len(content) <= trigger_chars:
        return content
    if is_tool_output_digest(content):
        return content
    return build_tool_output_digest(content, tool_name=tool_name, max_chars=max_chars)


__all__ = [
    "DIGEST_TRIGGER_CHARS",
    "DIGEST_MAX_CHARS",
    "build_tool_output_digest",
    "extract_exit_code",
    "extract_failure_lines",
    "is_tool_output_digest",
    "maybe_digest_tool_output",
]
