"""Focal tests for the independent hybrid agent network (NAI-68)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.independent_network.broker import DispatchBroker
from hermes_cli.independent_network.cli import network_command
from hermes_cli.independent_network.credentials import (
    DEFAULT_REFERENCES,
    CredentialBroker,
    SecretRevealedError,
    assert_no_secret_values,
)
from hermes_cli.independent_network.linear import LinearLinkError, require_linear_issue
from hermes_cli.independent_network.provision import (
    provision_roster,
    read_pinned_model,
)
from hermes_cli.independent_network.roster import (
    CANONICAL_ROSTER,
    list_roster,
    provider_for_model,
)
from hermes_cli.independent_network.routing import (
    AmbiguousRosterError,
    UnknownAgentError,
    assert_roster_unambiguous,
    build_routing_index,
    resolve_agent,
)
from hermes_cli.independent_network.roster import AgentSpec


SECRET_VALUE = "s3cret-value-that-must-never-leak"


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME and Path.home() like tests/hermes_cli/test_profiles.py."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    import hermes_constants as hc

    hc._default_hermes_root_memo = None
    return default_home


def _fake_fetcher(_reference: str, _name: str) -> str:
    return SECRET_VALUE


def _quiet_runner(started):
    def _run(job, child_env):
        started.append({"job": job, "env": dict(child_env)})
        return 4242

    return _run


class TestRosterAndRouting:
    def test_canonical_roster_has_twelve_agents_and_six_core(self):
        roster = list_roster()
        assert len(roster) == 12
        assert len(list_roster(core_only=True)) == 6
        by_handle = {agent.handle: agent for agent in roster}
        assert by_handle["producto/Oscar"].model == "grok-4.6"
        assert by_handle["critico/Ada"].model == "claude-opus-5"
        assert by_handle["visual/Sebastian"].model == "claude-sonnet-5"
        assert by_handle["growth/Juan"].model == "grok-4.6"
        assert by_handle["crm/CRM"].model == "grok-4.6"
        assert by_handle["revenue/Revenue"].model == "gpt-5.6-terra"
        assert by_handle["commerce/Commerce"].model == "grok-4.6"
        assert by_handle["educacion/Edu"].model == "grok-4.6"
        assert by_handle["contenido/Content"].model == "claude-sonnet-5"
        assert by_handle["infra/Frank"].model == "grok-4.6"
        assert by_handle["research/Nerd"].model == "gpt-5.6-terra"
        assert by_handle["finanzas/Mat"].model == "gpt-5.6-terra"

    def test_each_roster_model_has_a_provider(self):
        for agent in CANONICAL_ROSTER:
            assert agent.provider == provider_for_model(agent.model)
            assert agent.provider in {"xai", "anthropic", "openai"}

    def test_routing_is_deterministic_for_alias_lane_profile_and_handle(self):
        oscar = resolve_agent("Oscar")
        assert oscar is resolve_agent("producto")
        assert oscar is resolve_agent("oscar")
        assert oscar is resolve_agent("producto/Oscar")
        assert oscar.profile == "oscar"
        ada = resolve_agent("ADA")
        assert ada.alias == "Ada"
        assert ada.model == "claude-opus-5"

    def test_unknown_alias_fails_closed(self):
        with pytest.raises(UnknownAgentError):
            resolve_agent("nobody")
        with pytest.raises(UnknownAgentError):
            resolve_agent("")

    def test_roster_index_rejects_colliding_agents(self):
        dup = (
            AgentSpec("lane-a", "Alpha", "alpha", "grok-4.6", "xai", "A", True),
            AgentSpec("lane-b", "Alpha", "beta", "grok-4.6", "xai", "B", True),
        )
        with pytest.raises(AmbiguousRosterError):
            build_routing_index(dup)

    def test_canonical_roster_is_unambiguous(self):
        assert_roster_unambiguous()


class TestLinearLink:
    def test_parses_id_and_url(self):
        link = require_linear_issue("NAI-68")
        assert link.identifier == "NAI-68"
        assert link.url.endswith("/NAI-68")
        from_url = require_linear_issue(
            "https://linear.app/naicipa/issue/NAI-68/piloto-desplegar-seis-agentes-nucleo"
        )
        assert from_url.identifier == "NAI-68"

    def test_missing_or_garbage_is_rejected(self):
        with pytest.raises(LinearLinkError):
            require_linear_issue(None)
        with pytest.raises(LinearLinkError):
            require_linear_issue("not-an-issue")
        with pytest.raises(LinearLinkError):
            require_linear_issue("https://example.com/NAI-68")


class TestProvisionIsolation:
    def test_provision_pins_models_and_isolates_profiles(self, profile_env):
        results = provision_roster(names=["Oscar", "Ada"], no_skills=True)
        assert len(results) == 2
        oscar = next(row for row in results if row.agent.profile == "oscar")
        ada = next(row for row in results if row.agent.profile == "ada")
        assert oscar.profile_dir != ada.profile_dir
        assert oscar.profile_dir.is_dir()
        assert ada.profile_dir.is_dir()
        assert read_pinned_model(oscar.profile_dir) == ("xai", "grok-4.6")
        assert read_pinned_model(ada.profile_dir) == ("anthropic", "claude-opus-5")

        marker = oscar.profile_dir / "memories" / "PRIVATE.md"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("oscar-only", encoding="utf-8")
        assert not (ada.profile_dir / "memories" / "PRIVATE.md").exists()

        oscar_env = (oscar.profile_dir / ".env").read_text(encoding="utf-8")
        ada_env = (ada.profile_dir / ".env").read_text(encoding="utf-8")
        assert SECRET_VALUE not in oscar_env
        assert SECRET_VALUE not in ada_env
        for ref in DEFAULT_REFERENCES.values():
            assert ref.startswith("op://")

        soul = (oscar.profile_dir / "SOUL.md").read_text(encoding="utf-8")
        assert "Oscar" in soul
        assert "grok-4.6" in soul


class TestAsyncDispatch:
    def test_dispatch_requires_linear_and_starts_without_waiting(self, profile_env):
        provision_roster(names=["Ada"], no_skills=True)
        started = []
        broker = DispatchBroker(
            home=profile_env,
            credentials=CredentialBroker(home=profile_env, fetcher=_fake_fetcher),
            runner=_quiet_runner(started),
        )
        with pytest.raises(LinearLinkError):
            broker.dispatch("Ada", "review copy", "")
        with pytest.raises(LinearLinkError):
            broker.dispatch("Ada", "review copy", "nope")

        job = broker.dispatch("Ada", "review copy", "NAI-68")
        assert job.status == "running"
        assert job.pid == 4242
        assert job.linear["identifier"] == "NAI-68"
        assert job.profile == "ada"
        assert len(started) == 1
        assert started[0]["job"].id == job.id

        stored = json.loads((profile_env / "independent-agent-network" / "jobs" / f"{job.id}.json").read_text())
        assert stored["status"] == "running"
        assert SECRET_VALUE not in json.dumps(stored)
        assert SECRET_VALUE not in job.prompt()

    def test_unknown_target_does_not_create_a_job(self, profile_env):
        broker = DispatchBroker(
            home=profile_env,
            credentials=CredentialBroker(home=profile_env, fetcher=_fake_fetcher),
            runner=_quiet_runner([]),
        )
        with pytest.raises(UnknownAgentError):
            broker.dispatch("ghost", "do a thing", "NAI-68")
        jobs_path = profile_env / "independent-agent-network" / "jobs"
        if jobs_path.exists():
            assert list(jobs_path.glob("*.json")) == []


class TestBrokeredCredentials:
    def test_receipt_and_audit_omit_secret_values(self, profile_env):
        broker = CredentialBroker(home=profile_env, fetcher=_fake_fetcher)
        receipt = broker.request("oscar", "OPENAI_API_KEY")
        assert receipt.granted is True
        payload = receipt.to_dict()
        assert SECRET_VALUE not in json.dumps(payload)
        assert "OPENAI_API_KEY" in payload["secret_name"]
        assert payload["reference"].startswith("op://")
        assert broker.peek_grant(receipt.request_id) == SECRET_VALUE

        audit = (profile_env / "independent-agent-network" / "audit" / "credentials.jsonl").read_text()
        assert SECRET_VALUE not in audit
        row = json.loads(audit.strip().splitlines()[-1])
        assert row["granted"] is True
        assert "value" not in row
        assert row["request_id"] == receipt.request_id

    def test_out_of_scope_name_is_denied(self, profile_env):
        broker = CredentialBroker(
            home=profile_env,
            fetcher=_fake_fetcher,
            allowlist={"oscar": ["OPENAI_API_KEY"]},
        )
        denied = broker.request("oscar", "ANTHROPIC_API_KEY")
        assert denied.granted is False
        assert denied.error == "secret not in profile scope"
        assert broker.peek_grant(denied.request_id) is None

    def test_assert_no_secret_values_catches_leaks(self):
        with pytest.raises(SecretRevealedError):
            assert_no_secret_values({"note": SECRET_VALUE}, [SECRET_VALUE])
        assert_no_secret_values({"reference": "op://Naicipa/OpenAI/credential"}, [SECRET_VALUE])

    def test_fetch_failure_is_audited_without_values(self, profile_env):
        def _boom(_ref, _name):
            raise RuntimeError(f"op failed with {SECRET_VALUE}")

        broker = CredentialBroker(home=profile_env, fetcher=_boom)
        receipt = broker.request("ada", "ANTHROPIC_API_KEY")
        assert receipt.granted is False
        assert SECRET_VALUE not in (receipt.error or "")
        assert SECRET_VALUE not in json.dumps(receipt.to_dict())


class TestCliSurface:
    def test_roster_json(self, capsys):
        rc = network_command(SimpleNamespace(network_command="roster", json=True))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert {row["alias"] for row in payload} >= {"Oscar", "Ada", "Frank", "Nerd"}

    def test_route_and_unknown(self, capsys):
        rc = network_command(SimpleNamespace(network_command="route", name="Juan", json=True))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["profile"] == "juan"
        rc = network_command(SimpleNamespace(network_command="route", name="nope", json=True))
        assert rc == 1
