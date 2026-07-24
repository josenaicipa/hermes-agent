"""Test-only helpers for LLM-cron admission policy compatibility.

Production ``create_job`` has no env-controlled bypass. Legacy unit tests
that predate ``category`` / ``material_result_criterion`` get defaults via
the autouse fixture in ``tests/conftest.py``, which wraps ``create_job``
only under pytest. The focused admission suite
(``tests/cron/test_llm_admission_policy.py``) opts out so fail-closed
behavior is exercised for real.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

# Stable defaults for legacy tests that never declared admission fields.
LEGACY_TEST_LLM_ADMISSION: Dict[str, str] = {
    "category": "necessary_as_is",
    "material_result_criterion": "test material result criterion",
}


def wrap_create_job_with_test_admission_defaults(
    real_create: Callable[..., Any],
) -> Callable[..., Any]:
    """Return a create_job wrapper that injects admission defaults for LLM jobs.

    Defaults are applied only when both category and material_result_criterion
    are absent/None and no_agent is false. Explicit admission args are kept.
    Script-only no_agent jobs are left untouched (production exempt path).
    """

    def create_job_with_test_admission_defaults(*args: Any, **kwargs: Any) -> Any:
        no_agent = bool(kwargs.get("no_agent", False))
        if not no_agent:
            cat = kwargs.get("category", None)
            crit = kwargs.get("material_result_criterion", None)
            if cat is None and crit is None:
                kwargs = {
                    **kwargs,
                    "category": LEGACY_TEST_LLM_ADMISSION["category"],
                    "material_result_criterion": LEGACY_TEST_LLM_ADMISSION[
                        "material_result_criterion"
                    ],
                }
        return real_create(*args, **kwargs)

    create_job_with_test_admission_defaults._is_cron_admission_test_wrapper = True  # type: ignore[attr-defined]
    create_job_with_test_admission_defaults.__wrapped__ = real_create  # type: ignore[attr-defined]
    return create_job_with_test_admission_defaults
