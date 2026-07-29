"""Coercion hardening for the absolute compression token cap.

Upstream 0.19 independently shipped the same "hard pre-API compression cap"
feature this commit set out to add — ``compression.threshold_tokens`` ->
``ContextCompressor(threshold_tokens_cap=...)``, applied as
``effective = min(normal_threshold_tokens, cap)`` and re-applied by
``update_model()`` so a fallback/switch cannot silently drop the ceiling.
That behavioral matrix (cap-wins, ratio-wins, clamped-to-context-length,
survives model switch, default-disabled, ...) is already covered by
``tests/agent/test_context_compressor.py::TestThresholdTokensCap`` and is
NOT duplicated here.

The one real gap this commit found in upstream's coercion,
``ContextCompressor._coerce_threshold_tokens_cap``, is that plain
``int(value)`` silently accepts booleans (``int(True) == 1``, so YAML
``threshold_tokens: true`` would cap compression at a single token) and
silently truncates fractional floats instead of rejecting them. This file
regression-tests that hardening against the actual shipped mechanism.
"""

from unittest.mock import patch

import agent.context_compressor as cc
from agent.context_compressor import ContextCompressor


class TestCoerceThresholdTokensCapRejectsBadValues:
    def test_rejects_bools(self):
        # YAML `threshold_tokens: true` must not become cap=1 via int(True).
        assert ContextCompressor._coerce_threshold_tokens_cap(True) is None
        assert ContextCompressor._coerce_threshold_tokens_cap(False) is None

    def test_rejects_non_integer_floats(self):
        # Fractional numbers are invalid/unset, not silently truncated.
        assert ContextCompressor._coerce_threshold_tokens_cap(80.5) is None
        assert ContextCompressor._coerce_threshold_tokens_cap(1.9) is None

    def test_accepts_integer_floats_and_digit_strings(self):
        assert ContextCompressor._coerce_threshold_tokens_cap(80_000) == 80_000
        assert ContextCompressor._coerce_threshold_tokens_cap(80_000.0) == 80_000
        assert ContextCompressor._coerce_threshold_tokens_cap("80000") == 80_000
        assert ContextCompressor._coerce_threshold_tokens_cap("1") == 1

    def test_non_positive_still_unset(self):
        assert ContextCompressor._coerce_threshold_tokens_cap(0) is None
        assert ContextCompressor._coerce_threshold_tokens_cap(-5) is None
        assert ContextCompressor._coerce_threshold_tokens_cap("0") is None


class TestThresholdTokensCapBoolConstructionIsSafe:
    def test_bool_cap_does_not_collapse_threshold_to_one_token(self):
        # End-to-end: a `threshold_tokens: true` config value reaching the
        # constructor must behave exactly like "no cap configured", not
        # like a 1-token cap that would make compression fire constantly.
        with patch.object(cc, "get_model_context_length", return_value=200_000):
            capped_true = ContextCompressor(
                model="test/model", threshold_percent=0.50, quiet_mode=True,
                threshold_tokens_cap=True,
            )
            baseline = ContextCompressor(
                model="test/model", threshold_percent=0.50, quiet_mode=True,
                threshold_tokens_cap=None,
            )
        assert capped_true.threshold_tokens_cap is None
        assert capped_true.threshold_tokens == baseline.threshold_tokens
