"""Regression tests for stale writes after a compression session split."""

from __future__ import annotations

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _compression_parent(db: SessionDB, session_id: str = "parent") -> None:
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "before split")
    db.end_session(session_id, "compression")


def test_find_live_compression_child_returns_unique_direct_child(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("child", source="webui", parent_session_id="parent")

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "child"
    assert child["parent_session_id"] == "parent"
    assert child["ended_at"] is None


def test_find_live_compression_child_fails_closed_when_ambiguous(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("child-a", source="webui", parent_session_id="parent")
    db.create_session("child-b", source="webui", parent_session_id="parent")

    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_ignores_ended_children(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("ended-child", source="webui", parent_session_id="parent")
    db.end_session("ended-child", "agent_close")

    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_ignores_non_continuation_children(
    db: SessionDB,
) -> None:
    _compression_parent(db)
    db.create_session("canonical", source="webui", parent_session_id="parent")
    db.create_session(
        "branch",
        source="webui",
        parent_session_id="parent",
        model_config={"_branched_from": "parent"},
    )
    db.create_session(
        "delegate",
        source="webui",
        parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )
    db.create_session("tool-child", source="tool", parent_session_id="parent")

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "canonical"


def test_find_live_compression_child_walks_multiple_compression_generations(
    db: SessionDB,
) -> None:
    """Reproduces the 2026-07-30 vpsclone incident lineage exactly.

    P ended compression; a direct NONcanonical tool child ended agent_close
    (must be ignored, same as the single-generation case); the direct
    canonical child C1 itself ended compression; C1's canonical child C2 also
    ended compression; C2's canonical child C3 is the unique live tip. A stale
    agent still holding ``P`` must be able to resolve all the way to C3, not
    just the direct child C1.
    """
    _compression_parent(db, "P")
    db.create_session("sa-tool-child", source="tool", parent_session_id="P")
    db.end_session("sa-tool-child", "agent_close")

    db.create_session("C1", source="webui", parent_session_id="P")
    db.end_session("C1", "compression")

    db.create_session("C2", source="webui", parent_session_id="C1")
    db.end_session("C2", "compression")

    db.create_session("C3", source="webui", parent_session_id="C2")

    child = db.find_live_compression_child("P")

    assert child is not None
    assert child["id"] == "C3"
    assert child["ended_at"] is None


def test_find_live_compression_child_fails_closed_on_ambiguity_two_generations_deep(
    db: SessionDB,
) -> None:
    """Ambiguity must fail closed no matter which generation it appears at."""
    _compression_parent(db, "P")
    db.create_session("C1", source="webui", parent_session_id="P")
    db.end_session("C1", "compression")

    # C1 has two live canonical children: a genuine fork one hop below the
    # direct child the old single-generation guard checked.
    db.create_session("C2-a", source="webui", parent_session_id="C1")
    db.create_session("C2-b", source="webui", parent_session_id="C1")

    assert db.find_live_compression_child("P") is None


def test_find_live_compression_child_fails_closed_on_unrelated_end_reason_mid_chain(
    db: SessionDB,
) -> None:
    """A non-compression end reason mid-chain must not be walked past.

    Even though C1's own child C2 is live, C1 itself was NOT closed by
    compression (e.g. a crashed/legacy cleanup reason), so the canonical
    continuation guarantee is broken at C1. Adopting C2 anyway would risk
    landing on the wrong transcript.
    """
    _compression_parent(db, "P")
    db.create_session("C1", source="webui", parent_session_id="P")
    db.end_session("C1", "agent_close")

    db.create_session("C2", source="webui", parent_session_id="C1")

    assert db.find_live_compression_child("P") is None


def test_find_live_compression_child_tolerates_dead_sibling_across_generations(
    db: SessionDB,
) -> None:
    """A stale closed sibling is a dead end, not a fork.

    ``ws_orphan_reap`` siblings are routine (a dropped websocket reaps a row
    that never owned the transcript). The one-hop guard already ignored them,
    and the multi-generation walk must keep ignoring them instead of counting
    them as ambiguity — otherwise every real recovery with a reaped sibling
    would fail closed and strand the agent on the compressed parent.
    """
    _compression_parent(db, "P")
    db.create_session("stale-sibling", source="webui", parent_session_id="P")
    db.end_session("stale-sibling", "ws_orphan_reap")

    db.create_session("C1", source="webui", parent_session_id="P")
    db.end_session("C1", "compression")
    db.create_session("C2", source="webui", parent_session_id="C1")

    child = db.find_live_compression_child("P")

    assert child is not None
    assert child["id"] == "C2"


def test_find_live_compression_child_fails_closed_on_cycle(db: SessionDB) -> None:
    """A corrupted lineage cycle must fail closed, never spin forever.

    The cycle is closed by corrupting ``P``'s OWN parent pointer to its
    grandchild ``C2``. That matters: ``parent_session_id`` is a single column,
    so a cycle that excludes ``P`` is unrepresentable, and rewriting a
    descendant's parent instead would merely detach the chain from ``P`` and
    make the assertion pass vacuously without ever walking. Here every hop
    stays reachable — P -> C1 -> C2 -> P -> ... — so a walk without cycle
    detection genuinely revisits ``P`` forever, and none of the three rows is
    live, so the only correct answer is to fail closed.
    """
    _compression_parent(db, "P")
    db.create_session("C1", source="webui", parent_session_id="P")
    db.end_session("C1", "compression")
    db.create_session("C2", source="webui", parent_session_id="C1")
    db.end_session("C2", "compression")

    # Close the loop: P becomes a child of its own grandchild. Both rows
    # already exist, so this stays a representable (if corrupt) lineage.
    db._conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?", ("C2", "P")
    )
    db._conn.commit()

    assert db.get_session("P")["parent_session_id"] == "C2"
    assert db.find_live_compression_child("P") is None


def test_append_message_rejects_compression_ended_parent_atomically(db: SessionDB) -> None:
    _compression_parent(db)
    before = db.get_session("parent")["message_count"]

    with pytest.raises(RuntimeError, match="closed by compression"):
        db.append_message("parent", "assistant", "must not land on parent")

    assert db.get_session("parent")["message_count"] == before
    assert [m["content"] for m in db.get_messages("parent")] == ["before split"]


def test_append_message_preserves_legacy_behavior_for_other_end_reasons(db: SessionDB) -> None:
    db.create_session("ended", source="test")
    db.end_session("ended", "agent_close")

    message_id = db.append_message("ended", "user", "legacy append")

    assert isinstance(message_id, int)
    assert db.get_messages("ended")[-1]["content"] == "legacy append"


def test_replace_messages_rejects_compression_ended_parent_atomically(
    db: SessionDB,
) -> None:
    _compression_parent(db)

    with pytest.raises(RuntimeError, match="closed by compression"):
        db.replace_messages("parent", [{"role": "user", "content": "rewrite"}])

    assert [m["content"] for m in db.get_messages("parent")] == ["before split"]


def test_publish_compression_child_is_atomic_on_handoff_failure(
    db: SessionDB, monkeypatch
) -> None:
    db.create_session("atomic-parent", source="webui")
    db.append_message("atomic-parent", "user", "original")
    assert db.try_acquire_compression_lock("atomic-parent", "winner", ttl_seconds=60)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("handoff insert failed")

    monkeypatch.setattr(db, "_insert_message_rows", _boom)
    with pytest.raises(RuntimeError, match="handoff insert failed"):
        db.publish_compression_child(
            parent_session_id="atomic-parent",
            child_session_id="atomic-child",
            source="webui",
            messages=[{"role": "user", "content": "summary"}],
            compression_lock_holder="winner",
        )

    parent = db.get_session("atomic-parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert db.get_session("atomic-child") is None


def test_publish_compression_child_exposes_complete_child(db: SessionDB) -> None:
    db.create_session("atomic-parent", source="webui")
    db.append_message("atomic-parent", "user", "original")
    assert db.try_acquire_compression_lock("atomic-parent", "winner", ttl_seconds=60)

    db.publish_compression_child(
        parent_session_id="atomic-parent",
        child_session_id="atomic-child",
        source="webui",
        system_prompt="compressed system",
        messages=[{"role": "user", "content": "summary"}],
        compression_lock_holder="winner",
    )

    assert db.get_session("atomic-parent")["end_reason"] == "compression"
    child = db.find_live_compression_child("atomic-parent")
    assert child is not None
    assert child["id"] == "atomic-child"
    assert child["system_prompt"] == "compressed system"
    assert [m["content"] for m in db.get_messages("atomic-child")] == ["summary"]


def test_publish_compression_child_rejects_lost_or_expired_lease(db: SessionDB) -> None:
    db.create_session("lease-parent", source="webui")
    db.append_message("lease-parent", "user", "new durable turn")
    assert db.try_acquire_compression_lock("lease-parent", "new-winner", ttl_seconds=60)

    with pytest.raises(RuntimeError, match="lease lost"):
        db.publish_compression_child(
            parent_session_id="lease-parent",
            child_session_id="stale-child",
            source="webui",
            messages=[{"role": "user", "content": "stale summary"}],
            compression_lock_holder="old-loser",
        )

    parent = db.get_session("lease-parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert db.get_session("stale-child") is None
    assert [m["content"] for m in db.get_messages("lease-parent")] == [
        "new durable turn"
    ]


def test_compression_lease_blocks_non_owner_but_allows_owner_flush(
    db: SessionDB,
) -> None:
    db.create_session("leased", source="webui")
    assert db.try_acquire_compression_lock("leased", "winner", ttl_seconds=60)

    with pytest.raises(RuntimeError, match="being compressed"):
        db.append_message("leased", "user", "late stale turn")

    db.append_message(
        "leased",
        "assistant",
        "winner flush",
        compression_lock_holder="winner",
    )
    assert [m["content"] for m in db.get_messages("leased")] == ["winner flush"]
