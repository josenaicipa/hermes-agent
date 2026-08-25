"""E2E for the independent agent network against a temp HERMES_HOME.

Exercises real ``create_profile`` imports (not mocks): roster routing,
pinned models, profile isolation, async dispatch with mandatory Linear,
and brokered credentials that never persist secret values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.independent_network.broker import DispatchBroker
from hermes_cli.independent_network.credentials import CredentialBroker
from hermes_cli.independent_network.provision import provision_roster, read_pinned_model
from hermes_cli.independent_network.roster import list_roster
from hermes_cli.independent_network.routing import resolve_agent
from hermes_cli.profiles import get_profile_dir, list_profiles


SECRET = "e2e-secret-must-not-leak"


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    import hermes_constants as hc

    hc._default_hermes_root_memo = None
    return default_home


def test_e2e_core_fleet_isolation_dispatch_and_credentials(profile_env):
    started = []

    def runner(job, child_env):
        started.append({"id": job.id, "keys": sorted(child_env), "env": dict(child_env)})
        return 99

    results = provision_roster(core_only=True, no_skills=True, home=profile_env)
    assert len(results) == 6
    assert {row.agent.alias for row in results} == {
        "Oscar",
        "Ada",
        "Sebastian",
        "Juan",
        "Frank",
        "Nerd",
    }

    names = {p.name for p in list_profiles() if not p.is_default}
    for agent in list_roster(core_only=True):
        assert agent.profile in names
        profile_dir = get_profile_dir(agent.profile)
        provider, model = read_pinned_model(profile_dir)
        assert provider == agent.provider
        assert model == agent.model
        assert (profile_dir / ".env").is_file()
        assert SECRET not in (profile_dir / ".env").read_text(encoding="utf-8")
        assert (profile_dir / "SOUL.md").read_text(encoding="utf-8").startswith(f"# {agent.alias}")

    oscar_dir = get_profile_dir("oscar")
    ada_dir = get_profile_dir("ada")
    (oscar_dir / "workspace" / "note.txt").parent.mkdir(parents=True, exist_ok=True)
    (oscar_dir / "workspace" / "note.txt").write_text("oscar-workspace", encoding="utf-8")
    assert not (ada_dir / "workspace" / "note.txt").exists()
    assert oscar_dir.resolve() != ada_dir.resolve()

    creds = CredentialBroker(
        home=profile_env,
        fetcher=lambda _ref, _name: SECRET,
        allowlist={"ada": ["ANTHROPIC_API_KEY", "LINEAR_API_KEY"]},
    )
    broker = DispatchBroker(home=profile_env, credentials=creds, runner=runner)
    job = broker.dispatch(
        "critico/Ada",
        "Review the landing page against NAI-68 acceptance.",
        "https://linear.app/naicipa/issue/NAI-68/piloto-desplegar-seis-agentes-nucleo",
    )

    assert job.status == "running"
    assert job.pid == 99
    assert job.linear["identifier"] == "NAI-68"
    assert resolve_agent(job.target).profile == "ada"
    assert job.model == "claude-opus-5"
    assert set(job.credential_names) <= {"ANTHROPIC_API_KEY", "LINEAR_API_KEY"}
    assert started and started[0]["id"] == job.id
    assert "ANTHROPIC_API_KEY" in started[0]["keys"]
    assert started[0]["env"]["ANTHROPIC_API_KEY"] == SECRET

    job_path = profile_env / "independent-agent-network" / "jobs" / f"{job.id}.json"
    blob = job_path.read_text(encoding="utf-8")
    assert SECRET not in blob
    stored = json.loads(blob)
    assert stored["linear"]["identifier"] == "NAI-68"
    assert "value" not in json.dumps(stored["credential_receipts"])
    for receipt in stored["credential_receipts"]:
        assert "granted" in receipt
        assert SECRET not in json.dumps(receipt)

    audit = (profile_env / "independent-agent-network" / "audit" / "credentials.jsonl").read_text()
    assert SECRET not in audit
    assert "ANTHROPIC_API_KEY" in audit

    listed = broker.list_jobs()
    assert listed[0].id == job.id
    assert listed[0].status == "running"


def test_e2e_full_roster_models_stay_pinned(profile_env):
    results = provision_roster(no_skills=True, home=profile_env)
    assert len(results) == 12
    for row in results:
        provider, model = read_pinned_model(row.profile_dir)
        assert (provider, model) == (row.provider, row.model)
        # Re-provision is idempotent and does not clone secrets.
        again = provision_roster(names=[row.agent.alias], no_skills=True, home=profile_env)
        assert again[0].created is False
        assert read_pinned_model(again[0].profile_dir) == (row.provider, row.model)
