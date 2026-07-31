"""Live-agent recovery across a multi-generation compression rotation.

``SessionDB.find_live_compression_child`` proves the lineage; these tests cover
the seam that actually moves a stale live agent onto it,
``_adopt_live_compression_child``. The 2026-07-30 incident showed the two are
not interchangeable: a stale agent whose parent had been rotated more than once
resolved no continuation at all, logged "no unique live child could be
adopted", and then failed its next ``append_message`` against the closed
parent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from agent.conversation_compression import _adopt_live_compression_child
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


class _FakeContextEngine:
    """Records the compaction-boundary rebind the real context engine does."""

    def __init__(self) -> None:
        self.session_id: Optional[str] = None
        self.boundary_calls: List[Dict[str, Any]] = []

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        self.session_id = session_id
        self.boundary_calls.append({"session_id": session_id, **kwargs})


class _FakeAgent:
    """Minimal stand-in for the stale live AIAgent.

    Deliberately NOT a MagicMock: this module's own guards document that
    auto-created truthy attributes silently hijack compression branches, and an
    adoption test that passes because every attribute is truthy proves nothing.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.platform = "telegram"
        self.context_compressor = _FakeContextEngine()
        self._gateway_session_key = "telegram:u1:c1"
        self._memory_manager = None
        self._session_db_created = False
        self._cached_system_prompt: Optional[str] = None
        self._last_flushed_db_idx = 0
        self._flushed_db_message_session_id: Optional[str] = session_id
        self._flushed_db_message_ids: set = set()


def _rotated_parent(db: SessionDB, session_id: str = "P") -> None:
    db.create_session(session_id, source="telegram")
    db.append_message(session_id, "user", "before split")
    db.end_session(session_id, "compression")


def _compressed_hop(db: SessionDB, child: str, parent: str) -> None:
    db.create_session(child, source="telegram", parent_session_id=parent)
    db.end_session(child, "compression")


def _live_tip(db: SessionDB, child: str, parent: str) -> None:
    db.create_session(
        child,
        source="telegram",
        parent_session_id=parent,
        system_prompt="compressed system",
    )
    db.append_message(child, "user", "[CONTEXT COMPACTION] summary")
    db.append_message(child, "user", "tail")


def test_adopts_unique_direct_child_one_hop(db: SessionDB) -> None:
    """The pre-existing single-rotation contract must keep working."""
    _rotated_parent(db)
    _live_tip(db, "C1", "P")
    agent = _FakeAgent("P")

    recovered = _adopt_live_compression_child(agent, db, "P")

    assert recovered is not None
    assert [message["content"] for message in recovered] == [
        "[CONTEXT COMPACTION] summary",
        "tail",
    ]
    assert agent.session_id == "C1"
    assert agent.context_compressor.session_id == "C1"


def test_stale_agent_adopts_live_tip_across_compression_generations(
    db: SessionDB,
) -> None:
    """The incident: P was rotated three times while this agent was stalled.

    P ended compression, an unrelated non-canonical tool child ended
    agent_close, C1 and C2 each ended compression, and C3 is the live tip. The
    stale agent must land on C3 — the transcript that owns subsequent
    messages — and rebind its context engine and flush bookkeeping with it,
    rather than resolving nothing and writing into the closed parent.
    """
    _rotated_parent(db)
    db.create_session("sa-tool-child", source="tool", parent_session_id="P")
    db.end_session("sa-tool-child", "agent_close")
    _compressed_hop(db, "C1", "P")
    _compressed_hop(db, "C2", "C1")
    _live_tip(db, "C3", "C2")
    agent = _FakeAgent("P")

    recovered = _adopt_live_compression_child(agent, db, "P")

    assert recovered is not None
    assert [message["content"] for message in recovered] == [
        "[CONTEXT COMPACTION] summary",
        "tail",
    ]
    assert agent.session_id == "C3"
    assert agent.context_compressor.session_id == "C3"

    # The compaction boundary must be announced with the lineage endpoints and
    # the routing identity, or context-engine plugins attribute the adopted
    # turns to the wrong conversation.
    assert len(agent.context_compressor.boundary_calls) == 1
    boundary = agent.context_compressor.boundary_calls[0]
    assert boundary["session_id"] == "C3"
    assert boundary["old_session_id"] == "P"
    assert boundary["boundary_reason"] == "compression"
    assert boundary["conversation_id"] == "telegram:u1:c1"
    assert boundary["platform"] == "telegram"
    assert boundary["session_db"] is db

    # Flush bookkeeping must follow the adoption, otherwise the next flush
    # replays the recovered handoff back into the DB as new rows.
    assert agent._session_db_created is True
    assert agent._cached_system_prompt == "compressed system"
    assert agent._flushed_db_message_session_id == "C3"
    assert agent._last_flushed_db_idx == len(recovered)


def test_adoption_fails_closed_on_deep_ambiguity_and_leaves_agent_pinned(
    db: SessionDB,
) -> None:
    """Ambiguity below the direct child must not be resolved by guessing.

    Two live canonical children of C1 mean no continuation is uniquely proven,
    so the agent must stay on its original session id. Staying pinned raises
    the loud "closed by compression" error on the next append instead of
    silently splicing this agent's turns into one of two competing forks.
    """
    _rotated_parent(db)
    _compressed_hop(db, "C1", "P")
    _live_tip(db, "C2-a", "C1")
    _live_tip(db, "C2-b", "C1")
    agent = _FakeAgent("P")

    assert _adopt_live_compression_child(agent, db, "P") is None
    assert agent.session_id == "P"
    assert agent.context_compressor.session_id is None
    assert agent.context_compressor.boundary_calls == []
    assert agent._session_db_created is False
