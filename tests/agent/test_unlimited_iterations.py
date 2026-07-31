"""True-unlimited iteration support for AIAgent / IterationBudget.

Unlimited must be an explicit sentinel — never ``0`` and never a huge
integer — so the conversation loop and the turn finalizer can distinguish
"no cap" from "a very large cap" without arithmetic guesswork.

Pure in-process tests: no provider, no network, no real AIAgent spawn.
Written as ``unittest.TestCase`` classes so pytest collects them normally
while the file also stays runnable standalone.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

# Ensure the repo root is importable when this file runs standalone.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001 — probing an optional runtime dep
        return False


# ``run_agent`` / ``agent.agent_init`` pull in the provider runtime
# (dotenv/httpx/openai). CI installs it; a stripped sandbox may not.
_HAS_RUNTIME_DEPS = _importable("run_agent") and _importable("agent.agent_init")


# ---------------------------------------------------------------------------
# Sentinel identity
# ---------------------------------------------------------------------------


class TestUnlimitedSentinel(unittest.TestCase):
    def test_sentinel_is_not_zero_and_not_an_int(self):
        from agent.iteration_budget import UNLIMITED_ITERATIONS

        self.assertIsNotNone(UNLIMITED_ITERATIONS)
        self.assertNotIsInstance(UNLIMITED_ITERATIONS, int)
        self.assertNotIsInstance(UNLIMITED_ITERATIONS, float)
        self.assertNotEqual(UNLIMITED_ITERATIONS, 0)
        self.assertNotEqual(UNLIMITED_ITERATIONS, 500)

    def test_sentinel_is_greater_than_any_iteration_count(self):
        from agent.iteration_budget import UNLIMITED_ITERATIONS

        for count in (0, 1, 500, 10**6, 10**18):
            with self.subTest(count=count):
                self.assertLess(count, UNLIMITED_ITERATIONS)
                self.assertFalse(count >= UNLIMITED_ITERATIONS)
                self.assertGreater(UNLIMITED_ITERATIONS, count)
                self.assertFalse(UNLIMITED_ITERATIONS <= count)

    def test_sentinel_is_identity_checkable(self):
        from agent.iteration_budget import UNLIMITED_ITERATIONS, is_unlimited

        self.assertIs(is_unlimited(UNLIMITED_ITERATIONS), True)
        for bounded in (0, 1, 500, 10**9, None, "unlimited"):
            with self.subTest(bounded=bounded):
                self.assertIs(is_unlimited(bounded), False)

    def test_sentinel_arithmetic_stays_unlimited(self):
        """``max_iterations - 1`` style guards must not crash or go finite."""
        from agent.iteration_budget import UNLIMITED_ITERATIONS, is_unlimited

        self.assertIs(is_unlimited(UNLIMITED_ITERATIONS - 1), True)
        self.assertIs(is_unlimited(UNLIMITED_ITERATIONS + 1), True)

    def test_sentinel_has_json_safe_representation(self):
        from agent.iteration_budget import UNLIMITED_ITERATIONS, iteration_cap_repr

        self.assertEqual(iteration_cap_repr(UNLIMITED_ITERATIONS), "unlimited")
        self.assertEqual(iteration_cap_repr(500), 500)
        # Must survive persistence (session model_config is JSON encoded).
        json.dumps({"max_iterations": iteration_cap_repr(UNLIMITED_ITERATIONS)})

    @unittest.skipUnless(
        _HAS_RUNTIME_DEPS, "requires the full provider runtime (dotenv/httpx/openai)"
    )
    def test_re_exported_from_run_agent(self):
        import run_agent
        from agent.iteration_budget import UNLIMITED_ITERATIONS

        self.assertIs(run_agent.UNLIMITED_ITERATIONS, UNLIMITED_ITERATIONS)


# ---------------------------------------------------------------------------
# IterationBudget
# ---------------------------------------------------------------------------


class TestUnlimitedIterationBudget(unittest.TestCase):
    def test_unlimited_budget_never_refuses_consume(self):
        from agent.iteration_budget import UNLIMITED_ITERATIONS, IterationBudget

        budget = IterationBudget(UNLIMITED_ITERATIONS)
        self.assertIs(budget.unlimited, True)
        for _ in range(5000):
            self.assertIs(budget.consume(), True)
        self.assertEqual(budget.used, 5000)

    def test_unlimited_budget_remaining_never_reaches_zero(self):
        from agent.iteration_budget import (
            UNLIMITED_ITERATIONS,
            IterationBudget,
            is_unlimited,
        )

        budget = IterationBudget(UNLIMITED_ITERATIONS)
        for _ in range(1000):
            budget.consume()
        self.assertIs(is_unlimited(budget.remaining), True)
        self.assertGreater(budget.remaining, 0)
        self.assertFalse(budget.remaining <= 0)

    def test_bounded_budget_semantics_unchanged(self):
        from agent.iteration_budget import IterationBudget

        budget = IterationBudget(2)
        self.assertIs(budget.unlimited, False)
        self.assertIs(budget.consume(), True)
        self.assertIs(budget.consume(), True)
        self.assertIs(budget.consume(), False)
        self.assertEqual(budget.remaining, 0)
        budget.refund()
        self.assertEqual(budget.remaining, 1)
        self.assertIs(budget.consume(), True)

    def test_unlimited_budget_refund_does_not_underflow(self):
        from agent.iteration_budget import UNLIMITED_ITERATIONS, IterationBudget

        budget = IterationBudget(UNLIMITED_ITERATIONS)
        budget.consume()
        budget.refund()
        budget.refund()
        self.assertEqual(budget.used, 0)
        self.assertIs(budget.consume(), True)


# ---------------------------------------------------------------------------
# Loop / finalizer decision points
# ---------------------------------------------------------------------------


def _fake_agent(max_iterations):
    from agent.iteration_budget import IterationBudget

    return SimpleNamespace(
        max_iterations=max_iterations,
        iteration_budget=IterationBudget(max_iterations),
    )


class TestLoopAndFinalizerPredicates(unittest.TestCase):
    def test_unlimited_agent_always_has_iterations_available(self):
        from agent.iteration_budget import UNLIMITED_ITERATIONS, iterations_available

        agent = _fake_agent(UNLIMITED_ITERATIONS)
        for count in (0, 500, 10**6):
            with self.subTest(count=count):
                self.assertIs(iterations_available(agent, count), True)

    def test_unlimited_agent_is_never_exhausted(self):
        from agent.iteration_budget import UNLIMITED_ITERATIONS, iterations_exhausted

        agent = _fake_agent(UNLIMITED_ITERATIONS)
        for count in (0, 500, 10**6):
            with self.subTest(count=count):
                self.assertIs(iterations_exhausted(agent, count), False)

    def test_unlimited_agent_never_reports_cap_reached(self):
        from agent.iteration_budget import UNLIMITED_ITERATIONS, cap_reached

        agent = _fake_agent(UNLIMITED_ITERATIONS)
        for count in (0, 500, 10**9):
            with self.subTest(count=count):
                self.assertIs(cap_reached(agent, count), False)

    def test_bounded_agent_predicates_unchanged(self):
        from agent.iteration_budget import (
            cap_reached,
            iterations_available,
            iterations_exhausted,
        )

        agent = _fake_agent(3)
        self.assertIs(iterations_available(agent, 0), True)
        self.assertIs(cap_reached(agent, 2), False)
        self.assertIs(cap_reached(agent, 3), True)
        self.assertIs(iterations_available(agent, 3), False)
        self.assertIs(iterations_exhausted(agent, 3), True)

    def test_bounded_agent_exhausted_when_budget_drained(self):
        from agent.iteration_budget import iterations_available, iterations_exhausted

        agent = _fake_agent(5)
        for _ in range(5):
            agent.iteration_budget.consume()
        self.assertIs(iterations_available(agent, 0), False)
        self.assertIs(iterations_exhausted(agent, 0), True)


class TestProductionWiring(unittest.TestCase):
    """The loop/finalizer/init must route through the unlimited-aware helpers."""

    def test_conversation_loop_uses_iterations_available(self):
        from agent import conversation_loop

        src = inspect.getsource(conversation_loop)
        self.assertTrue(
            "iterations_available(" in src,
            "conversation loop must use the explicit unlimited-aware predicate",
        )

    def test_turn_finalizer_uses_unlimited_aware_predicates(self):
        from agent import turn_finalizer

        src = inspect.getsource(turn_finalizer)
        self.assertTrue(
            "iterations_exhausted(" in src,
            "turn finalizer must use the unlimited-aware exhaustion predicate",
        )
        self.assertTrue(
            "cap_reached(" in src,
            "turn finalizer must use the unlimited-aware cap predicate",
        )

    def test_turn_finalizer_normalizes_unlimited_caps_for_logging(self):
        """Logging must not pass the sentinel to an integer formatter."""
        from agent import turn_finalizer

        src = inspect.getsource(turn_finalizer)
        self.assertIn("iteration_cap_repr(agent.max_iterations)", src)
        self.assertIn("iteration_cap_repr(_budget_max)", src)
        self.assertNotIn("api_calls=%d/%d", src)
        self.assertNotIn("budget=%d/%d", src)

    @unittest.skipUnless(
        _HAS_RUNTIME_DEPS, "requires the full provider runtime (dotenv/httpx/openai)"
    )
    def test_session_model_config_is_json_safe_when_unlimited(self):
        """Persisted session model_config must never carry a raw sentinel."""
        from agent import agent_init

        src = inspect.getsource(agent_init)
        self.assertTrue(
            "iteration_cap_repr(agent.max_iterations)" in src,
            "agent_init must JSON-normalize max_iterations for session persistence",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
