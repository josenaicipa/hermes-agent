"""Tests for agent/tool_output_precompaction.py — deterministic digests of
oversized tool outputs, and their integration with the context compressor
(summarizer-prompt serialization and the summary-failure fallback shrink)."""

import pytest
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor
from agent.tool_output_precompaction import (
    DIGEST_MAX_CHARS,
    DIGEST_TRIGGER_CHARS,
    build_tool_output_digest,
    extract_exit_code,
    extract_failure_lines,
    is_tool_output_digest,
    maybe_digest_tool_output,
)


def _make_ci_log() -> str:
    """A realistic oversized CI/test log: command header, thousands of noise
    lines, a few failure lines buried in the middle, and a summary trailer."""
    head = [f"$ pytest tests/ -q --maxfail=0   # header {i}" for i in range(5)]
    noise = [f"tests/test_mod_{i}.py::test_case_{i} PASSED" for i in range(3000)]
    noise[1200] = (
        "FAILED tests/test_alpha.py::test_compaction - AssertionError: digest mismatch"
    )
    noise[1800] = (
        "ERROR tests/test_beta.py::test_timeout - TimeoutError: deadline exceeded"
    )
    noise[2200] = "error: cannot open /var/log/ci/build.log: No such file or directory"
    tail = [
        "=========== 2 failed, 2998 passed in 61.20s ===========",
        "exit code 1",
    ]
    return "\n".join(head + noise + tail)


class TestBuildToolOutputDigest:
    def test_digest_shrinks_large_output_and_is_bounded(self):
        log = _make_ci_log()
        assert len(log) > 100_000  # genuinely large input

        digest = build_tool_output_digest(log, tool_name="terminal", max_chars=4_000)

        assert len(digest) <= 4_000
        assert len(digest) < len(log) * 0.05

    def test_digest_preserves_first_lines(self):
        digest = build_tool_output_digest(_make_ci_log(), tool_name="terminal")
        assert "# header 0" in digest
        assert "--- first lines ---" in digest

    def test_digest_preserves_tail(self):
        digest = build_tool_output_digest(_make_ci_log(), tool_name="terminal")
        assert "2 failed, 2998 passed in 61.20s" in digest
        assert "--- last lines ---" in digest

    def test_digest_preserves_failure_lines_from_omitted_middle(self):
        digest = build_tool_output_digest(_make_ci_log(), tool_name="terminal")
        assert "FAILED tests/test_alpha.py::test_compaction" in digest
        assert "TimeoutError: deadline exceeded" in digest
        assert "/var/log/ci/build.log" in digest
        # The benign bulk between head and tail is dropped.
        assert "tests/test_mod_1500.py::test_case_1500 PASSED" not in digest

    def test_digest_preserves_exit_status_metadata(self):
        digest = build_tool_output_digest(_make_ci_log(), tool_name="terminal")
        assert "exit_code=1" in digest

    def test_digest_includes_size_stats_and_tool_name(self):
        log = _make_ci_log()
        digest = build_tool_output_digest(log, tool_name="terminal")
        assert digest.startswith("[tool output digest: terminal")
        assert f"{len(log):,} chars" in digest
        assert f"{len(log.splitlines()):,} lines" in digest
        assert "tokens" in digest
        assert "middle lines omitted" in digest

    def test_digest_is_deterministic(self):
        log = _make_ci_log()
        first = build_tool_output_digest(log, tool_name="terminal")
        second = build_tool_output_digest(log, tool_name="terminal")
        assert first == second

    def test_digest_is_recognized_by_is_tool_output_digest(self):
        digest = build_tool_output_digest(_make_ci_log(), tool_name="terminal")
        assert is_tool_output_digest(digest) is True
        assert is_tool_output_digest("plain tool output") is False

    def test_digest_bounded_even_for_single_giant_line(self):
        # Minified JS / base64 blob: one line, no newlines to slice on.
        blob = "error " + ("x" * 50_000)
        digest = build_tool_output_digest(blob, tool_name="fetch", max_chars=2_000)
        assert len(digest) <= 2_000
        assert digest.startswith("[tool output digest: fetch")


class TestMaybeDigestToolOutput:
    def test_small_output_passes_through_unchanged(self):
        small = "ok: 3 files changed, 1 insertion(+)\nexit code 0"
        assert maybe_digest_tool_output(small, tool_name="terminal") is small

    def test_output_at_trigger_size_passes_through_unchanged(self):
        at_trigger = "x" * DIGEST_TRIGGER_CHARS
        assert maybe_digest_tool_output(at_trigger) is at_trigger

    def test_large_output_is_digested(self):
        log = _make_ci_log()
        result = maybe_digest_tool_output(log, tool_name="terminal")
        assert is_tool_output_digest(result)
        assert len(result) <= DIGEST_MAX_CHARS

    def test_non_string_content_passes_through_unchanged(self):
        multimodal = [{"type": "image_url", "image_url": {"url": "data:..."}}]
        assert maybe_digest_tool_output(multimodal) is multimodal

    def test_existing_digest_is_not_redigested(self):
        digest = build_tool_output_digest(_make_ci_log(), tool_name="terminal")
        padded = digest + "\n" + ("pad " * 3000)  # back over the trigger
        # A real digest never exceeds the trigger, but defend against double
        # processing: a digest-prefixed body must pass through untouched.
        assert maybe_digest_tool_output(padded) is padded


class TestExtractHelpers:
    def test_extract_exit_code_from_json_envelope(self):
        assert extract_exit_code('{"exit_code": 2, "stdout": "..."}') == "2"

    def test_extract_exit_code_from_plain_trailer(self):
        assert extract_exit_code("Process exited with code 137") == "137"
        assert extract_exit_code("command finished\nexit status: 1\n") == "1"

    def test_extract_exit_code_absent(self):
        assert extract_exit_code("all good, nothing numeric here") is None
        assert extract_exit_code("") is None

    def test_extract_failure_lines_dedupes_and_limits(self):
        content = "\n".join(
            ["ok line"]
            + ["ConnectionError: connection refused"] * 50
            + [f"FAILED tests/test_{i}.py::t - AssertionError" for i in range(10)]
        )
        lines = extract_failure_lines(content, limit=3)
        assert len(lines) == 3
        # Order of first appearance, exact duplicates collapsed.
        assert lines[0] == "ConnectionError: connection refused"
        assert lines[1].startswith("FAILED tests/test_0.py")
        assert lines[2].startswith("FAILED tests/test_1.py")

    def test_extract_failure_lines_empty_when_no_failures(self):
        assert extract_failure_lines("all 100 tests passed\ndone") == []


def _make_compressor(**kwargs):
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100_000
    ):
        return ContextCompressor(model="test/model", quiet_mode=True, **kwargs)


def _make_giant_tool_log() -> str:
    """>8K chars (above the fallback digest trigger) with a buried failure."""
    lines = [
        f"bulk filler line {i} with padding text to inflate the output size"
        for i in range(400)
    ]
    lines[200] = "MIDDLE-MARKER-SHOULD-BE-DROPPED with more padding text here"
    lines[210] = "FAILED tests/test_core.py::test_main - RuntimeError: boom"
    lines.append("exit code 1")
    return "\n".join(lines)


class TestSerializeForSummaryUsesDigest:
    """The summarizer prompt must receive a structured digest of oversized
    tool results — not the raw bulk (aux-model timeouts) and not a blind
    head/tail slice (loses the failing-test lines)."""

    def _turns(self, tool_content):
        return [
            {
                "role": "assistant",
                "content": "Running the tests",
                "tool_calls": [
                    {
                        "id": "call-9",
                        "function": {"name": "terminal", "arguments": '{"command":"pytest"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-9", "content": tool_content},
        ]

    def test_oversized_tool_result_serialized_as_labeled_digest(self):
        c = _make_compressor()
        giant = _make_giant_tool_log()
        assert len(giant) > c._CONTENT_MAX

        text = c._serialize_for_summary(self._turns(giant))

        assert "[tool output digest: terminal" in text
        assert "FAILED tests/test_core.py::test_main" in text
        assert "exit_code=1" in text
        assert "MIDDLE-MARKER-SHOULD-BE-DROPPED" not in text

    def test_small_tool_result_serialized_verbatim(self):
        c = _make_compressor()
        small = "3 passed in 0.12s\nexit code 0"
        text = c._serialize_for_summary(self._turns(small))
        assert small in text
        assert "[tool output digest" not in text


class TestFallbackShrinksProtectedToolOutputs:
    """When the LLM summary fails, oversized tool outputs surviving in the
    protected head/tail must be digested deterministically so the fallback
    result is bounded — a giant log must not survive compaction verbatim."""

    def _messages(self, giant):
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "investigate the CI failure"},
            {"role": "assistant", "content": "looking into it"},
            {"role": "user", "content": "any progress?"},
            {"role": "assistant", "content": "narrowing it down"},
            {"role": "user", "content": "run the tests"},
            {
                "role": "assistant",
                "content": "running",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "terminal", "arguments": '{"command":"pytest"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": giant},
        ]

    def test_summary_failure_digests_giant_tool_output_in_protected_tail(self):
        c = _make_compressor(protect_first_n=1, protect_last_n=2)
        giant = _make_giant_tool_log()
        msgs = self._messages(giant)

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=Exception("aux model timeout"),
        ):
            result = c.compress(msgs)

        assert c._last_summary_fallback_used is True
        tool_msg = next(m for m in result if m.get("role") == "tool")
        # Bounded digest, not the verbatim log.
        assert is_tool_output_digest(tool_msg["content"])
        assert len(tool_msg["content"]) < len(giant) // 4
        assert "MIDDLE-MARKER-SHOULD-BE-DROPPED" not in tool_msg["content"]
        # Diagnostic signal survives the shrink.
        assert "FAILED tests/test_core.py::test_main" in tool_msg["content"]
        assert "exit_code=1" in tool_msg["content"]
        # The whole fallback transcript is bounded well below the input.
        total = sum(
            len(m["content"]) for m in result if isinstance(m.get("content"), str)
        )
        assert total < len(giant)

    def test_summary_failure_does_not_mutate_input_messages(self):
        c = _make_compressor(protect_first_n=1, protect_last_n=2)
        giant = _make_giant_tool_log()
        msgs = self._messages(giant)

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=Exception("aux model timeout"),
        ):
            c.compress(msgs)

        assert msgs[-1]["content"] == giant

    def test_successful_summary_leaves_protected_tool_output_verbatim(self):
        # The digest pass is a fallback-only shrink; with a working
        # summarizer the protected tail keeps its raw tool output.
        c = _make_compressor(protect_first_n=1, protect_last_n=2)
        giant = _make_giant_tool_log()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"
        with patch(
            "agent.context_compressor.call_llm", return_value=mock_response
        ):
            result = c.compress(self._messages(giant))

        assert c._last_summary_fallback_used is False
        tool_msg = next(m for m in result if m.get("role") == "tool")
        assert tool_msg["content"] == giant

    def test_shrink_helper_skips_small_and_non_string_tool_outputs(self):
        c = _make_compressor()
        msgs = [
            {"role": "tool", "tool_call_id": "a", "content": "short result"},
            {"role": "tool", "tool_call_id": "b", "content": [{"type": "text"}]},
            {"role": "user", "content": "x" * 50_000},  # non-tool: untouched
        ]
        shrunk_msgs, count = c._shrink_tool_outputs_for_fallback(msgs)
        assert count == 0
        assert shrunk_msgs == msgs

    def test_shrink_helper_labels_digest_with_tool_name(self):
        c = _make_compressor()
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call-7", "function": {"name": "codeql_scan", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call-7", "content": _make_giant_tool_log()},
        ]
        shrunk_msgs, count = c._shrink_tool_outputs_for_fallback(msgs)
        assert count == 1
        assert shrunk_msgs[1]["content"].startswith("[tool output digest: codeql_scan")
