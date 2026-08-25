"""Deterministic alias/name routing onto the canonical roster.

Matching is exact after casefold. Unknown names fail closed — they never
fall through to a default profile. Duplicate keys in the roster are a
load-time error so routing cannot become ambiguous later.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

from hermes_cli.independent_network.roster import (
    AgentSpec,
    CANONICAL_ROSTER,
    iter_routing_keys,
    list_roster,
)


class UnknownAgentError(KeyError):
    """Raised when no roster agent matches the supplied alias or name."""


class AmbiguousRosterError(RuntimeError):
    """Raised when two roster entries claim the same routing key."""


def build_routing_index(
    roster: Optional[Iterable[AgentSpec]] = None,
) -> Dict[str, AgentSpec]:
    """Build the casefolded alias/name index.

    Duplicate keys that point at *different* agents are rejected. Keys that
    collapse onto the same agent (profile == alias.casefold(), etc.) are
    expected and kept.
    """
    index: Dict[str, AgentSpec] = {}
    for agent in roster if roster is not None else CANONICAL_ROSTER:
        for key in iter_routing_keys(agent):
            existing = index.get(key)
            if existing is not None and existing.profile != agent.profile:
                raise AmbiguousRosterError(
                    f"routing key {key!r} maps to both {existing.handle} and {agent.handle}"
                )
            index[key] = agent
    return index


_INDEX: Optional[Dict[str, AgentSpec]] = None


def routing_index() -> Mapping[str, AgentSpec]:
    """Return the process-wide canonical index, building it once."""
    global _INDEX
    if _INDEX is None:
        _INDEX = build_routing_index()
    return _INDEX


def resolve_agent(
    name: str,
    *,
    roster: Optional[Iterable[AgentSpec]] = None,
) -> AgentSpec:
    """Resolve an alias, lane, profile, or ``lane/Alias`` handle.

    Resolution is deterministic: one name, one agent. Unknown names raise
    :class:`UnknownAgentError` instead of falling back.
    """
    query = (name or "").strip()
    if not query:
        raise UnknownAgentError("agent name is required")
    index = build_routing_index(roster) if roster is not None else routing_index()
    agent = index.get(query.casefold())
    if agent is None:
        raise UnknownAgentError(f"unknown agent {name!r}")
    return agent


def assert_roster_unambiguous(roster: Optional[Iterable[AgentSpec]] = None) -> None:
    """Fail if the roster would produce ambiguous routing."""
    build_routing_index(roster if roster is not None else list_roster())
