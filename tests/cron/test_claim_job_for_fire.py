"""Tests for the store-level CAS fire claim (Phase 4C).

`claim_job_for_fire` gives multi-machine at-most-once semantics when an external
scheduler (Chronos) fires a job: across N gateway replicas, exactly ONE wins the
claim for a given fire. Single-machine deployments always win (unaffected).

These exercise the real store against a temp HERMES_HOME (no mocks) per the
E2E-over-mocks discipline for file-touching code.
"""
import pytest


@pytest.fixture
def temp_home(tmp_path):
    """Route cron storage explicitly so import order cannot touch the real store."""
    from cron.jobs import use_cron_store

    with use_cron_store(tmp_path):
        yield tmp_path


def test_claim_succeeds_once_then_blocks(temp_home):
    """First claim for a fire wins; a second claim for the same fire loses, and
    next_run_at is advanced (a re-delivery for the old time can't re-fire)."""
    from cron.jobs import create_job, claim_job_for_fire, get_job

    job = create_job(prompt="x", schedule="every 5m", name="t")
    jid = job["id"]
    before = get_job(jid)["next_run_at"]

    assert claim_job_for_fire(jid) is True
    assert claim_job_for_fire(jid) is False
    assert get_job(jid)["next_run_at"] != before


def test_claim_oneshot_cannot_be_double_claimed(temp_home):
    """A one-shot can't be double-claimed (the fresh claim blocks the retry)."""
    from cron.jobs import create_job, claim_job_for_fire

    job = create_job(prompt="x", schedule="30m", name="o")
    assert claim_job_for_fire(job["id"]) is True
    assert claim_job_for_fire(job["id"]) is False


def test_claim_unknown_job_returns_false(temp_home):
    from cron.jobs import claim_job_for_fire

    assert claim_job_for_fire("nope-does-not-exist") is False


def test_claim_paused_job_returns_false(temp_home):
    """A paused job can't be claimed."""
    from cron.jobs import create_job, claim_job_for_fire, pause_job

    job = create_job(prompt="x", schedule="every 5m", name="p")
    pause_job(job["id"])
    assert claim_job_for_fire(job["id"]) is False


def test_stale_claim_is_reclaimable(temp_home, monkeypatch):
    """A claim older than the TTL is overwritten — the fire isn't stuck forever
    if the winning machine crashed before mark_job_run cleared the claim."""
    from cron.jobs import create_job, claim_job_for_fire

    job = create_job(prompt="x", schedule="every 5m", name="s")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    # With a 0s TTL, the existing claim is always considered stale.
    assert claim_job_for_fire(jid, claim_ttl_seconds=0) is True


def test_mark_job_run_clears_claim(temp_home):
    """After a recurring job completes, its claim is cleared so the next fire
    can be claimed again."""
    from cron.jobs import create_job, claim_job_for_fire, mark_job_run, get_job

    job = create_job(prompt="x", schedule="every 5m", name="c")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    assert get_job(jid).get("fire_claim") is not None

    mark_job_run(jid, success=True)
    assert get_job(jid).get("fire_claim") is None
    # …and the re-armed recurring job is claimable again.
    assert claim_job_for_fire(jid) is True


def test_same_machine_reclaim_uses_unique_fencing_tokens(temp_home, monkeypatch):
    """Same-machine reclaim must mint a new generation fencing token.

    ``_machine_id`` is stable (host:pid / HERMES_MACHINE_ID). If fire_claim.by
    were only that string, a reclaim by the same gateway process would look
    identical to the stale generation and heartbeat/mark could not fence it.
    Tokens keep machine attribution but must differ per generation.
    """
    from cron.jobs import create_job, claim_job_for_fire, get_job

    monkeypatch.setattr("cron.jobs._machine_id", lambda: "fixed-machine")

    job = create_job(prompt="x", schedule="every 5m", name="gen")
    jid = job["id"]

    assert claim_job_for_fire(jid) is True
    by1 = get_job(jid)["fire_claim"]["by"]

    # Expire the first claim and reclaim on the same machine id.
    assert claim_job_for_fire(jid, claim_ttl_seconds=0) is True
    by2 = get_job(jid)["fire_claim"]["by"]

    assert by1 != by2, "reclaim must mint a distinct fencing token generation"
    assert by1.startswith("fixed-machine:"), by1
    assert by2.startswith("fixed-machine:"), by2
    # uuid4 hex suffix (opaque generation id) after the machine prefix.
    s1 = by1[len("fixed-machine:") :]
    s2 = by2[len("fixed-machine:") :]
    assert len(s1) == 32 and all(c in "0123456789abcdef" for c in s1), by1
    assert len(s2) == 32 and all(c in "0123456789abcdef" for c in s2), by2


def test_claim_without_owner_generates_unique_token_internally(temp_home, monkeypatch):
    """Legacy bool-only callers omit claim_owner; a unique token is stamped."""
    from cron.jobs import create_job, claim_job_for_fire, get_job

    monkeypatch.setattr("cron.jobs._machine_id", lambda: "legacy-machine")

    job = create_job(prompt="x", schedule="every 5m", name="legacy")
    jid = job["id"]

    # No claim_owner kwarg — public bool API stays usable.
    assert claim_job_for_fire(jid) is True
    by = get_job(jid)["fire_claim"]["by"]
    assert by.startswith("legacy-machine:")
    suffix = by[len("legacy-machine:") :]
    assert len(suffix) == 32 and all(c in "0123456789abcdef" for c in suffix)


def test_claim_with_supplied_owner_stamps_exact_token(temp_home, monkeypatch):
    """When claim_owner is supplied, stamp exactly that token (no replace)."""
    from cron.jobs import create_job, claim_job_for_fire, get_job, new_fire_claim_owner

    monkeypatch.setattr("cron.jobs._machine_id", lambda: "caller-machine")
    token = new_fire_claim_owner()
    assert token.startswith("caller-machine:")

    job = create_job(prompt="x", schedule="every 5m", name="exact")
    jid = job["id"]
    assert claim_job_for_fire(jid, claim_owner=token) is True
    assert get_job(jid)["fire_claim"]["by"] == token


def test_claim_blank_owner_fails_closed_without_mutation(temp_home):
    """Non-None blank claim_owner must fail closed before any claim mutation.

    Only claim_owner=None mints internally. Whitespace/empty strings are
    invalid supplied tokens and must raise ValueError without creating or
    advancing a claim (or next_run_at).
    """
    from cron.jobs import create_job, claim_job_for_fire, get_job

    job = create_job(prompt="x", schedule="every 5m", name="blank-owner")
    jid = job["id"]
    before = get_job(jid)
    assert before is not None
    before_next = before.get("next_run_at")
    assert before.get("fire_claim") in (None, {})

    for blank in ("", "   ", "\t"):
        with pytest.raises(ValueError, match="claim_owner"):
            claim_job_for_fire(jid, claim_owner=blank)
        after = get_job(jid)
        assert after is not None
        assert after.get("fire_claim") in (None, {})
        assert after.get("next_run_at") == before_next

    # None still mints a claim successfully (control).
    assert claim_job_for_fire(jid, claim_owner=None) is True
    claimed = get_job(jid)
    assert claimed is not None
    assert claimed.get("fire_claim") is not None
    assert claimed["fire_claim"].get("by")
