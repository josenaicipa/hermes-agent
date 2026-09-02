"""Lean tail must not hoard a long session of cheap stub messages.

After tool-result pruning, Discord-style turns are individually tiny. The
lean 25K token budget then fits hundreds of them, the backward walk
protects the whole transcript, the summarizer is left an empty/tiny
middle, and the ineffective breaker trips while real provider usage stays
over threshold.
"""

from __future__ import annotations

from unittest.mock import patch

from agent.context_compressor import (
    LEAN_TAIL_MAX_MESSAGES,
    ContextCompressor,
    estimate_messages_tokens_rough,
)


def _stub_session(n_turns: int) -> list[dict]:
    """Build a long transcript of already-pruned stubs plus a compaction handoff."""
    messages = [{
        "role": "user",
        "content": (
            "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
            "compacted into the summary below. This is a handoff from a "
            "previous compression boundary.\n\n## Summary\nPrior work."
        ),
    }]
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"status on NAI-{i}?"})
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": "agent_router",
                    "arguments": '{"action":"status"}',
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_name": "agent_router",
            "tool_call_id": f"call_{i}",
            "content": f"[agent_router] args=status lar_{i:04d} (80 chars result)",
        })
        messages.append({
            "role": "assistant",
            "content": f"NAI-{i} is running.",
        })
    return messages


def _lean_compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=1_000_000,
    ):
        cc = ContextCompressor(
            model="claude-fable-5-1",
            threshold_percent=0.45,
            protect_first_n=3,
            protect_last_n=6,
            summary_target_ratio=0.2,
            provider="anthropic",
            api_mode="",
            config_context_length=1_000_000,
            threshold_tokens_cap=150_000,
            min_tail_user_messages=3,
            tail_mode="lean",
            quiet_mode=True,
        )
    cc.context_length = 1_000_000
    cc._tail_token_budget = None
    cc.compression_count = 1  # decay protect_first_n like a live compacted session
    return cc


def test_lean_tail_caps_cheap_stub_session_and_leaves_middle():
    messages = _stub_session(55)  # 1 handoff + 220 turns ≈ 221 messages
    assert len(messages) > LEAN_TAIL_MAX_MESSAGES * 2
    cc = _lean_compressor()
    head_end = cc._protect_head_size(messages)
    cut = cc._find_tail_cut_by_tokens(messages, head_end)
    tail_n = len(messages) - cut
    middle_n = max(0, cut - head_end)
    middle_tokens = estimate_messages_tokens_rough(messages[head_end:cut])

    assert tail_n <= LEAN_TAIL_MAX_MESSAGES + 8, (
        f"lean tail hoarded {tail_n} messages (cut={cut}, n={len(messages)})"
    )
    assert middle_n >= 20, (
        f"expected a compressible middle, got {middle_n} messages "
        f"(cut={cut}, head_end={head_end})"
    )
    assert middle_tokens >= 1_000, (
        f"middle too small to summarize: {middle_tokens} tokens"
    )


def test_lean_tail_cap_does_not_shrink_short_sessions():
    messages = _stub_session(4)  # well under the cap
    cc = _lean_compressor()
    cc.compression_count = 0
    head_end = cc._protect_head_size(messages)
    cut = cc._find_tail_cut_by_tokens(messages, head_end)
    assert 0 < cut <= len(messages)
