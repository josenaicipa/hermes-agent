"""Test-only helpers for LLM-cron admission policy compatibility.

Production ``create_job`` / ``save_jobs`` have no env-controlled bypass.
Legacy unit tests that predate ``category`` / ``material_result_criterion``
get defaults via the autouse fixture in ``tests/conftest.py``, which wraps
both ``create_job`` and ``save_jobs`` only under pytest. The focused
admission suite (``tests/cron/test_llm_admission_policy.py``) opts out so
fail-closed behavior is exercised for real.
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


def wrap_save_jobs_with_test_admission_defaults(
    real_save: Callable[..., Any],
) -> Callable[..., Any]:
    """Return a save_jobs wrapper that injects admission defaults for legacy
    tests that persist raw job dicts directly (bypassing create_job).

    Many pre-admission-gate tests build a job dict by hand (fire_claim /
    run_claim / ownership / workdir fixtures) and call ``cron.jobs.save_jobs``
    with it directly. Applies the same narrow default-injection rule as
    ``wrap_create_job_with_test_admission_defaults``: only when a given job
    is a dict, not ``no_agent``, and declares NEITHER category nor
    material_result_criterion. Jobs that already declare either field (even
    partially/invalidly) are passed through untouched so real validation
    failures for those cases still surface.
    """

    def save_jobs_with_test_admission_defaults(jobs: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(jobs, list):
            patched = []
            for job in jobs:
                if isinstance(job, dict) and not job.get("no_agent"):
                    cat = job.get("category", None)
                    crit = job.get("material_result_criterion", None)
                    if cat is None and crit is None:
                        job = {
                            **job,
                            "category": LEGACY_TEST_LLM_ADMISSION["category"],
                            "material_result_criterion": LEGACY_TEST_LLM_ADMISSION[
                                "material_result_criterion"
                            ],
                        }
                patched.append(job)
            jobs = patched
        return real_save(jobs, *args, **kwargs)

    save_jobs_with_test_admission_defaults._is_cron_admission_test_wrapper = True  # type: ignore[attr-defined]
    save_jobs_with_test_admission_defaults.__wrapped__ = real_save  # type: ignore[attr-defined]
    return save_jobs_with_test_admission_defaults
