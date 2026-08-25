"""Asynchronous dispatch broker for the independent agent network.

Dispatch resolves an alias, requires a Linear issue, requests brokered
credentials, records a job, and starts the child without waiting. Job
files and receipts never contain secret values.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from hermes_cli.independent_network.credentials import (
    CredentialBroker,
    CredentialReceipt,
    assert_no_secret_values,
)
from hermes_cli.independent_network.linear import require_linear_issue
from hermes_cli.independent_network.routing import resolve_agent
from hermes_cli.independent_network.store import jobs_dir, read_json, write_json


Runner = Callable[["Job", Dict[str, str]], Optional[int]]


class DispatchError(ValueError):
    """Raised when a job cannot be accepted."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _job_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Job:
    """Durable record of an asynchronous network dispatch."""

    id: str
    target: str
    profile: str
    lane: str
    alias: str
    model: str
    provider: str
    linear: Dict[str, str]
    goal: str
    status: str
    created_at: str
    pid: Optional[int] = None
    credential_names: List[str] = field(default_factory=list)
    credential_receipts: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target": self.target,
            "profile": self.profile,
            "lane": self.lane,
            "alias": self.alias,
            "model": self.model,
            "provider": self.provider,
            "linear": dict(self.linear),
            "goal": self.goal,
            "status": self.status,
            "created_at": self.created_at,
            "pid": self.pid,
            "credential_names": list(self.credential_names),
            "credential_receipts": list(self.credential_receipts),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        return cls(
            id=str(data["id"]),
            target=str(data.get("target") or ""),
            profile=str(data.get("profile") or ""),
            lane=str(data.get("lane") or ""),
            alias=str(data.get("alias") or ""),
            model=str(data.get("model") or ""),
            provider=str(data.get("provider") or ""),
            linear=dict(data.get("linear") or {}),
            goal=str(data.get("goal") or ""),
            status=str(data.get("status") or "queued"),
            created_at=str(data.get("created_at") or ""),
            pid=data.get("pid"),
            credential_names=list(data.get("credential_names") or []),
            credential_receipts=list(data.get("credential_receipts") or []),
            error=data.get("error"),
        )

    def prompt(self) -> str:
        """User-message payload for the child agent. Contains no secrets."""
        identifier = self.linear.get("identifier", "")
        url = self.linear.get("url", "")
        return (
            f"Independent-agent dispatch for {self.alias} ({self.lane}).\n"
            f"Linear: {identifier} {url}\n"
            f"Pinned model: {self.model}\n\n"
            f"{self.goal.strip()}\n"
        )


class DispatchBroker:
    """Queue + spawn broker. ``dispatch()`` returns without waiting."""

    def __init__(
        self,
        *,
        home: Optional[Path] = None,
        credentials: Optional[CredentialBroker] = None,
        runner: Optional[Runner] = None,
    ) -> None:
        self.home = home
        self.credentials = credentials or CredentialBroker(home=home)
        self.runner = runner

    def dispatch(
        self,
        target: str,
        goal: str,
        linear: str,
        *,
        inject_credentials: bool = True,
    ) -> Job:
        """Accept a job, persist it, and start the child asynchronously."""
        if not (goal or "").strip():
            raise DispatchError("goal is required")
        agent = resolve_agent(target)
        link = require_linear_issue(linear)
        job = Job(
            id=_job_id(),
            target=target.strip(),
            profile=agent.profile,
            lane=agent.lane,
            alias=agent.alias,
            model=agent.model,
            provider=agent.provider,
            linear=link.to_dict(),
            goal=goal.strip(),
            status="queued",
            created_at=_utcnow(),
        )

        child_env: Dict[str, str] = {}
        receipts: List[CredentialReceipt] = []
        secret_values: List[str] = []
        if inject_credentials:
            child_env, receipts = self.credentials.collect_for_profile(agent.profile)
            secret_values = [v for v in child_env.values() if v]
            job.credential_names = sorted(child_env)
            job.credential_receipts = [r.to_dict() for r in receipts]

        payload = job.to_dict()
        assert_no_secret_values(payload, secret_values)
        assert_no_secret_values(job.prompt(), secret_values)
        self._save(job)

        try:
            pid = self._start(job, child_env)
        except Exception as exc:
            job.status = "failed"
            job.error = type(exc).__name__
            self._save(job)
            raise DispatchError(f"failed to start job {job.id}") from exc

        job.pid = pid
        job.status = "running" if pid else "queued"
        self._save(job)
        return job

    def get(self, job_id: str) -> Job:
        path = jobs_dir(self.home) / f"{job_id}.json"
        if not path.exists():
            raise DispatchError(f"unknown job {job_id}")
        return Job.from_dict(read_json(path))

    def list_jobs(self) -> List[Job]:
        jobs = []
        for path in sorted(jobs_dir(self.home).glob("*.json")):
            try:
                jobs.append(Job.from_dict(read_json(path)))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs

    def _save(self, job: Job) -> None:
        write_json(jobs_dir(self.home) / f"{job.id}.json", job.to_dict())

    def _start(self, job: Job, child_env: Dict[str, str]) -> Optional[int]:
        runner = self.runner if self.runner is not None else default_runner
        return runner(job, child_env)


def default_runner(job: Job, child_env: Mapping[str, str]) -> Optional[int]:
    """Spawn ``hermes -p <profile> --oneshot`` detached. Does not wait."""
    env = os.environ.copy()
    env.update(child_env)
    prompt = job.prompt()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "-p",
            job.profile,
            "--oneshot",
            prompt,
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return proc.pid
