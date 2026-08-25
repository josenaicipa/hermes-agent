"""Provision isolated Hermes profiles for the canonical roster.

Each agent gets its own profile directory (config, .env, memory, sessions)
with a pinned model. Profiles are created through the existing
``create_profile`` path so isolation, HOME anchoring, and empty .env
seeding stay intact. Secrets are never copied between profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from hermes_cli.independent_network.credentials import write_default_catalog
from hermes_cli.independent_network.roster import AgentSpec, list_roster


@dataclass(frozen=True)
class ProvisionResult:
    """Outcome of provisioning one roster agent."""

    agent: AgentSpec
    profile_dir: Path
    created: bool
    model: str
    provider: str

    def to_dict(self) -> dict:
        return {
            "lane": self.agent.lane,
            "alias": self.agent.alias,
            "profile": self.agent.profile,
            "handle": self.agent.handle,
            "core": self.agent.core,
            "profile_dir": str(self.profile_dir),
            "created": self.created,
            "model": self.model,
            "provider": self.provider,
        }


def pin_profile_model(profile_dir: Path, provider: str, model: str) -> None:
    """Write the pinned model into the profile's config.yaml."""
    import yaml

    config_path = profile_dir / "config.yaml"
    cfg = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            cfg = loaded
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    model_cfg["provider"] = provider
    model_cfg["default"] = model
    cfg["model"] = model_cfg
    config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def write_agent_soul(profile_dir: Path, agent: AgentSpec) -> None:
    """Write a stable, role-specific SOUL.md (byte-stable for prompt cache)."""
    soul = (
        f"# {agent.alias}\n\n"
        f"You are {agent.alias}, the {agent.role} agent in the Naicipa "
        f"independent-agent network.\n\n"
        f"- Lane: `{agent.lane}`\n"
        f"- Profile: `{agent.profile}`\n"
        f"- Model: `{agent.model}` (pinned; do not switch)\n"
        f"- Linear: every task you accept must already be linked to a Linear issue.\n"
        f"- Secrets: request credentials through `hermes network credentials`; "
        f"never print, paste, or remember secret values.\n"
    )
    (profile_dir / "SOUL.md").write_text(soul, encoding="utf-8")


def _ensure_empty_env(profile_dir: Path) -> None:
    """Keep .env present and free of copied fleet secrets."""
    env_path = profile_dir / ".env"
    if env_path.exists():
        return
    env_path.write_text(
        "# Per-profile secrets for this independent agent.\n"
        "# Do not paste shared fleet credentials here. Request them through\n"
        "# `hermes network credentials` (1Password broker).\n",
        encoding="utf-8",
    )
    try:
        env_path.chmod(0o600)
    except OSError:
        pass


def provision_agent(
    agent: AgentSpec,
    *,
    no_skills: bool = True,
    exist_ok: bool = True,
) -> ProvisionResult:
    """Create or update one isolated profile for ``agent``."""
    from hermes_cli.profiles import (
        create_profile,
        get_profile_dir,
        write_profile_meta,
    )

    profile_dir = get_profile_dir(agent.profile)
    created = False
    if not profile_dir.exists():
        create_profile(
            agent.profile,
            no_alias=True,
            no_skills=no_skills,
            description=f"{agent.handle} — {agent.role} (model {agent.model})",
        )
        created = True
    elif not exist_ok:
        raise FileExistsError(f"profile {agent.profile!r} already exists at {profile_dir}")

    pin_profile_model(profile_dir, agent.provider, agent.model)
    write_agent_soul(profile_dir, agent)
    _ensure_empty_env(profile_dir)
    try:
        write_profile_meta(
            profile_dir,
            description=f"{agent.handle} — {agent.role} (model {agent.model})",
            display_name=agent.alias,
            description_auto=False,
        )
    except Exception:
        pass
    return ProvisionResult(
        agent=agent,
        profile_dir=profile_dir,
        created=created,
        model=agent.model,
        provider=agent.provider,
    )


def provision_roster(
    *,
    core_only: bool = False,
    names: Optional[Sequence[str]] = None,
    no_skills: bool = True,
    exist_ok: bool = True,
    home: Optional[Path] = None,
) -> List[ProvisionResult]:
    """Provision isolated profiles for the canonical (or requested) roster."""
    from hermes_cli.independent_network.routing import resolve_agent

    if names:
        agents: Iterable[AgentSpec] = [resolve_agent(name) for name in names]
    else:
        agents = list_roster(core_only=core_only)

    results = [
        provision_agent(agent, no_skills=no_skills, exist_ok=exist_ok)
        for agent in agents
    ]
    write_default_catalog(home)
    return results


def read_pinned_model(profile_dir: Path) -> tuple[str, str]:
    """Return ``(provider, model)`` from a profile config.yaml."""
    import yaml

    config_path = profile_dir / "config.yaml"
    if not config_path.exists():
        return "", ""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        return "", ""
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        return str(model_cfg.get("provider") or ""), str(model_cfg.get("default") or "")
    if isinstance(model_cfg, str):
        return "", model_cfg
    return "", ""
