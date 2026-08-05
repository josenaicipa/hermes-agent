"""Byte-safe prompt chunking for CLI-backed auxiliary providers.

External-process providers (``google-gemini-cli``/``agy``,
``kimi-code-cli``) receive their prompt as an ``execve`` argument.  Linux
caps a *single* argument at ``MAX_ARG_STRLEN`` (32 pages ≈ 128 KiB) even
when ``ARG_MAX`` is far larger, so a one-shot 164 k-character checkpoint
prompt fails with ``OSError: [Errno 7] Argument list too long`` *before*
the provider ever runs.

This module owns the one primitive both adapters share: split text into
UTF-8-correct pieces that stay below that boundary.  Splitting happens on
the encoded bytes (never on characters), and a cut is walked back off any
UTF-8 continuation byte, so a multi-byte codepoint is never torn in half
and every chunk round-trips through ``bytes.decode("utf-8")`` exactly.
"""

from __future__ import annotations

# 80 KiB.  Comfortably under the 128 KiB per-argument kernel limit with room
# for the wrapper text each adapter adds around the payload (ordering
# preamble, SOURCE_DATA tags, acknowledgement instruction).
CLI_PROMPT_CHUNK_BYTES = 80 * 1024

__all__ = ["CLI_PROMPT_CHUNK_BYTES", "split_text_utf8"]


def split_text_utf8(
    text: str, max_bytes: int = CLI_PROMPT_CHUNK_BYTES
) -> list[str]:
    """Split *text* into pieces of at most *max_bytes* UTF-8 bytes.

    Guarantees:

    * every returned piece encodes to ``<= max_bytes`` bytes;
    * concatenating the pieces reproduces *text* byte-for-byte;
    * no multi-byte codepoint is split across pieces;
    * empty input yields ``[""]`` (one empty chunk) so callers always have
      at least one message to send.

    Raises:
        ValueError: if *max_bytes* is too small to hold a single codepoint
            of the input, or if a safe cut point cannot be found.
    """
    if max_bytes < 4:
        # A single UTF-8 codepoint is up to 4 bytes; anything smaller can
        # never produce a valid chunk.
        raise ValueError("max_bytes must be at least 4 to split UTF-8 text")

    data = text.encode("utf-8")
    if not data:
        return [""]

    chunks: list[str] = []
    start = 0
    total = len(data)
    while start < total:
        end = min(start + max_bytes, total)
        # Walk back off UTF-8 continuation bytes (0b10xxxxxx) so the cut
        # lands on a codepoint boundary.
        while end < total and end > start and data[end] & 0xC0 == 0x80:
            end -= 1
        if end <= start:
            raise ValueError("unable to split UTF-8 prompt safely")
        chunks.append(data[start:end].decode("utf-8"))
        start = end
    return chunks
