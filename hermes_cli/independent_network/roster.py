"""Canonical NAI-68 independent-agent roster.

The roster is the source of truth for lane, alias, isolated profile name,
and pinned model. Routing, provision, and dispatch all read from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class AgentSpec:
    """One independent agent in the hybrid network."""

    lane: str
    alias: str
    profile: str
    model: str
    provider: str
    role: str
    core: bool = False

    @property
    def handle(self) -> str:
        return f"{self.lane}/{self.alias}"


def provider_for_model(model: str) -> str:
    """Return the native Hermes provider key for a pinned roster model."""
    name = (model or "").strip().lower()
    if name.startswith("grok"):
        return "xai"
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3"):
        return "openai"
    raise ValueError(f"unsupported roster model: {model!r}")


def _spec(
    lane: str,
    alias: str,
    model: str,
    role: str,
    *,
    core: bool,
    profile: Optional[str] = None,
) -> AgentSpec:
    profile_name = (profile or alias).strip().lower()
    return AgentSpec(
        lane=lane,
        alias=alias,
        profile=profile_name,
        model=model,
        provider=provider_for_model(model),
        role=role,
        core=core,
    )


# Exact NAI-68 roster. Core six are the named specialists in the pilot.
CANONICAL_ROSTER: Tuple[AgentSpec, ...] = (
    _spec("producto", "Oscar", "grok-4.6", "Product", core=True),
    _spec("critico", "Ada", "claude-opus-5", "Critic", core=True),
    _spec("visual", "Sebastian", "claude-sonnet-5", "Visual", core=True),
    _spec("growth", "Juan", "grok-4.6", "Growth", core=True),
    _spec("crm", "CRM", "grok-4.6", "CRM", core=False),
    _spec("revenue", "Revenue", "gpt-5.6-terra", "Revenue", core=False),
    _spec("commerce", "Commerce", "grok-4.6", "Commerce", core=False),
    _spec("educacion", "Edu", "grok-4.6", "Education", core=False),
    _spec("contenido", "Content", "claude-sonnet-5", "Content", core=False),
    _spec("infra", "Frank", "grok-4.6", "Infrastructure", core=True),
    _spec("research", "Nerd", "gpt-5.6-terra", "Research", core=True),
    _spec("finanzas", "Mat", "gpt-5.6-terra", "Finance", core=False),
)


def list_roster(*, core_only: bool = False) -> Tuple[AgentSpec, ...]:
    """Return the canonical roster, optionally restricted to core agents."""
    if not core_only:
        return CANONICAL_ROSTER
    return tuple(agent for agent in CANONICAL_ROSTER if agent.core)


def get_agent(profile: str) -> Optional[AgentSpec]:
    """Return the spec for an exact profile name, or None."""
    key = (profile or "").strip().lower()
    for agent in CANONICAL_ROSTER:
        if agent.profile == key:
            return agent
    return None


def iter_routing_keys(agent: AgentSpec) -> Iterable[str]:
    """Yield every deterministic routing key for ``agent`` (already casefolded)."""
    yield agent.profile.casefold()
    yield agent.alias.casefold()
    yield agent.lane.casefold()
    yield agent.handle.casefold()
