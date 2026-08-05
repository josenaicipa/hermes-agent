"""Byte-safe UTF-8 chunking for CLI-backed auxiliary providers.

The kernel caps a single ``execve`` argument at ~128 KiB, so a full
checkpoint prompt must be split before it ever reaches ``agy`` or the Kimi
Code CLI. Splitting on *bytes* is what makes that safe, and splitting on
*codepoint boundaries* is what keeps it correct.
"""

from __future__ import annotations

import pytest

from agent.cli_prompt_chunking import CLI_PROMPT_CHUNK_BYTES, split_text_utf8


class TestChunkSize:
    def test_canonical_chunk_size_is_81920_bytes(self):
        assert CLI_PROMPT_CHUNK_BYTES == 81_920

    def test_agy_adapter_shares_the_same_primitive(self):
        from agent import agy_cli_client

        assert agy_cli_client._CHUNK_PAYLOAD_BYTES == CLI_PROMPT_CHUNK_BYTES
        # Same function, not a copy that can drift.
        assert agy_cli_client._split_text_utf8("abc") == split_text_utf8("abc")


class TestByteSafety:
    def test_empty_text_yields_one_empty_chunk(self):
        assert split_text_utf8("") == [""]

    def test_short_text_is_single_chunk(self):
        assert split_text_utf8("hello") == ["hello"]

    def test_chunks_never_exceed_max_bytes(self):
        text = "é" * 10_000  # 2 bytes each
        chunks = split_text_utf8(text, 1024)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.encode("utf-8")) <= 1024

    def test_roundtrip_is_lossless(self):
        text = "áé íóú 漢字 🌍 " * 3_000
        assert "".join(split_text_utf8(text, 997)) == text

    def test_multibyte_codepoints_are_never_split(self):
        # 4-byte codepoints straddling every plausible cut point.
        text = "🌍" * 5_000
        for max_bytes in (8, 9, 10, 11, 100, 4096):
            chunks = split_text_utf8(text, max_bytes)
            assert "".join(chunks) == text
            for chunk in chunks:
                # Decodes cleanly and contains only whole emoji.
                assert chunk.encode("utf-8").decode("utf-8") == chunk
                assert len(chunk.encode("utf-8")) % 4 == 0

    def test_mixed_width_boundary_alignment(self):
        text = "aé中\U0001f600" * 4_000  # 1 + 2 + 3 + 4 bytes
        chunks = split_text_utf8(text, 1000)
        assert "".join(chunks) == text
        for chunk in chunks:
            assert len(chunk.encode("utf-8")) <= 1000

    def test_rejects_unusably_small_limit(self):
        with pytest.raises(ValueError, match="at least 4"):
            split_text_utf8("🌍", 3)


class TestLargeRealisticPrompt:
    def test_prompt_over_128kb_is_split_below_arg_limit(self):
        # >128 KiB is the point at which a single execve argument fails with
        # OSError E2BIG / "Argument list too long".
        text = "The quick brown fox. " * 10_000
        assert len(text.encode("utf-8")) > 128 * 1024

        chunks = split_text_utf8(text)
        assert len(chunks) >= 2
        assert "".join(chunks) == text
        for chunk in chunks:
            assert len(chunk.encode("utf-8")) <= CLI_PROMPT_CHUNK_BYTES
            # Comfortably under the ~128 KiB MAX_ARG_STRLEN ceiling.
            assert len(chunk.encode("utf-8")) < 128 * 1024

    def test_164k_character_checkpoint_prompt_shape(self):
        # Mirrors the measured 164,711-character synthetic checkpoint prompt.
        text = "x" * 164_711
        chunks = split_text_utf8(text)
        assert "".join(chunks) == text
        assert len(chunks) == 3  # 81920 + 81920 + 871
        assert [len(c.encode("utf-8")) for c in chunks] == [81_920, 81_920, 871]

    def test_multibyte_164k_prompt_stays_lossless(self):
        text = ("漢字テスト " * 20_000)[:164_711]
        chunks = split_text_utf8(text)
        assert "".join(chunks) == text
        assert len(chunks) > 2
        for chunk in chunks:
            assert len(chunk.encode("utf-8")) <= CLI_PROMPT_CHUNK_BYTES
