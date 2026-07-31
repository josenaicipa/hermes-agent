"""Per-agent iteration budget — thread-safe consume/refund counter.

Extracted from ``run_agent.py``.  Each ``AIAgent`` instance (parent or
subagent) holds an :class:`IterationBudget`; the parent's cap comes from
``max_iterations`` (default 500), each subagent's cap comes from
``delegation.max_iterations`` (default 50).

A cap may also be :data:`UNLIMITED_ITERATIONS` — an explicit sentinel for
"no iteration ceiling at all".  Unlimited is deliberately *not* modelled as
``0`` (which historically meant "exhausted") nor as a very large integer
(which is still a ceiling and silently truncates long autonomous runs).
Callers that need to branch on it use :func:`is_unlimited`; callers that
need to persist or log it use :func:`iteration_cap_repr`.

``run_agent`` re-exports ``IterationBudget`` and ``UNLIMITED_ITERATIONS`` so
existing ``from run_agent import IterationBudget`` imports keep working
unchanged.
"""

from __future__ import annotations

import threading
from typing import Any, Union


class _UnlimitedIterations:
    """Singleton sentinel meaning "no iteration ceiling".

    Ordered above every real iteration count so the historical comparisons
    (``api_call_count < max_iterations``, ``remaining <= 0``) keep their
    meaning, and closed under ``+``/``-`` so near-the-limit guards such as
    ``max_iterations - 1`` stay unlimited instead of raising ``TypeError``.
    """

    __slots__ = ()
    _instance: "_UnlimitedIterations | None" = None

    def __new__(cls) -> "_UnlimitedIterations":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # -- ordering ---------------------------------------------------------
    def __lt__(self, other: Any) -> bool:
        if isinstance(other, _UnlimitedIterations):
            return False
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return False
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        if isinstance(other, _UnlimitedIterations):
            return True
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return False
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, _UnlimitedIterations):
            return False
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return True
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, _UnlimitedIterations):
            return True
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return True
        return NotImplemented

    def __eq__(self, other: Any) -> bool:
        return other is self

    def __hash__(self) -> int:
        return hash("hermes.unlimited_iterations")

    # -- arithmetic -------------------------------------------------------
    def __add__(self, other: Any) -> "_UnlimitedIterations":
        return self

    __radd__ = __add__

    def __sub__(self, other: Any) -> "_UnlimitedIterations":
        return self

    def __bool__(self) -> bool:
        return True

    # -- display ----------------------------------------------------------
    def __repr__(self) -> str:
        return "UNLIMITED_ITERATIONS"

    def __str__(self) -> str:
        return "unlimited"


#: Explicit "no iteration ceiling" sentinel (never ``0``, never a big int).
UNLIMITED_ITERATIONS = _UnlimitedIterations()

IterationCap = Union[int, _UnlimitedIterations]


def is_unlimited(value: Any) -> bool:
    """True only for the :data:`UNLIMITED_ITERATIONS` sentinel itself."""
    return value is UNLIMITED_ITERATIONS


def iteration_cap_repr(value: Any) -> Any:
    """Return a JSON-safe form of an iteration cap.

    The sentinel becomes the string ``"unlimited"``; everything else is
    returned unchanged.  Used wherever a cap is persisted (session
    ``model_config``) or handed to a diagnostics payload.
    """
    return "unlimited" if is_unlimited(value) else value


class IterationBudget:
    """Thread-safe iteration counter for an agent.

    Each agent (parent or subagent) gets its own ``IterationBudget``.
    The parent's budget is capped at ``max_iterations`` (default 500).
    Each subagent gets an independent budget capped at
    ``delegation.max_iterations`` (default 50) — this means total
    iterations across parent + subagents can exceed the parent's cap.
    Users control the per-subagent limit via ``delegation.max_iterations``
    in config.yaml.

    ``max_total`` may be :data:`UNLIMITED_ITERATIONS`, in which case
    :meth:`consume` never refuses and :attr:`remaining` never reaches zero.
    ``used`` keeps counting so diagnostics still report real work done.

    ``execute_code`` (programmatic tool calling) iterations are refunded via
    :meth:`refund` so they don't eat into the budget.
    """

    def __init__(self, max_total: IterationCap):
        self.max_total = max_total
        self._unlimited = is_unlimited(max_total)
        self._used = 0
        self._lock = threading.Lock()

    @property
    def unlimited(self) -> bool:
        """True when this budget has no ceiling at all."""
        return self._unlimited

    def consume(self) -> bool:
        """Try to consume one iteration.  Returns True if allowed."""
        with self._lock:
            if not self._unlimited and self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """Give back one iteration (e.g. for execute_code turns)."""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> IterationCap:
        if self._unlimited:
            return UNLIMITED_ITERATIONS
        with self._lock:
            return max(0, self.max_total - self._used)


def cap_reached(agent, api_call_count: int) -> bool:
    """True when *agent* has hit its ``max_iterations`` ceiling.

    Always False for an unlimited cap.  Bounded caps keep the historical
    ``api_call_count >= agent.max_iterations`` semantics exactly.
    """
    if is_unlimited(getattr(agent, "max_iterations", None)):
        return False
    return api_call_count >= agent.max_iterations


def iterations_available(agent, api_call_count: int) -> bool:
    """True while the agent may make another API call this turn."""
    return not cap_reached(agent, api_call_count) and agent.iteration_budget.remaining > 0


def iterations_exhausted(agent, api_call_count: int) -> bool:
    """True once the ceiling *or* the shared budget is spent."""
    return cap_reached(agent, api_call_count) or agent.iteration_budget.remaining <= 0


__all__ = [
    "UNLIMITED_ITERATIONS",
    "IterationBudget",
    "IterationCap",
    "cap_reached",
    "is_unlimited",
    "iteration_cap_repr",
    "iterations_available",
    "iterations_exhausted",
]
