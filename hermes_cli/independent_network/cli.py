"""``hermes network`` operator CLI."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from hermes_cli.independent_network.broker import DispatchBroker, DispatchError
from hermes_cli.independent_network.credentials import CredentialBroker
from hermes_cli.independent_network.linear import LinearLinkError
from hermes_cli.independent_network.provision import provision_roster, read_pinned_model
from hermes_cli.independent_network.roster import list_roster
from hermes_cli.independent_network.routing import UnknownAgentError, resolve_agent


def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Attach the ``network`` subcommand tree."""
    parser = parent_subparsers.add_parser(
        "network",
        help="Independent hybrid agent network (roster, dispatch, credentials)",
        description=(
            "Canonical roster of isolated agent profiles, deterministic "
            "alias routing, asynchronous dispatch bound to a Linear issue, "
            "and brokered 1Password credential access."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Machine-readable JSON on stdout",
    )
    subs = parser.add_subparsers(dest="network_command")

    subs.add_parser("roster", help="List the canonical agent roster")

    provision = subs.add_parser(
        "provision",
        help="Create isolated profiles with pinned models",
    )
    provision.add_argument(
        "--core",
        action="store_true",
        help="Provision only the six core specialists",
    )
    provision.add_argument(
        "--seed-skills",
        action="store_true",
        help="Copy bundled skills into each new profile (slow)",
    )
    provision.add_argument(
        "names",
        nargs="*",
        help="Optional alias/lane subset (default: full canonical roster)",
    )

    route = subs.add_parser(
        "route",
        help="Resolve an alias, lane, or handle to a roster agent",
    )
    route.add_argument("name", help="Alias, lane, profile, or lane/Alias")

    dispatch = subs.add_parser(
        "dispatch",
        help="Queue an asynchronous job (Linear issue required)",
    )
    dispatch.add_argument("target", help="Alias, lane, profile, or lane/Alias")
    dispatch.add_argument("--linear", required=True, help="Linear issue id or URL")
    dispatch.add_argument("--goal", required=True, help="Task for the target agent")
    dispatch.add_argument(
        "--no-credentials",
        action="store_true",
        help="Skip 1Password grant injection (job still records the request surface)",
    )

    jobs = subs.add_parser("jobs", help="List dispatched jobs")
    jobs.add_argument("--id", dest="job_id", help="Show a single job")

    creds = subs.add_parser(
        "credentials",
        help="Brokered 1Password credential requests (values never printed)",
    )
    cred_subs = creds.add_subparsers(dest="credentials_command")
    cred_subs.add_parser("catalog", help="Show env-name → op:// reference catalog")
    request = cred_subs.add_parser(
        "request",
        help="Request a named secret for a profile (receipt only, no value)",
    )
    request.add_argument("--profile", required=True, help="Roster profile or alias")
    request.add_argument("--name", required=True, help="Env var name, e.g. OPENAI_API_KEY")

    subs.add_parser("status", help="Roster + recent jobs summary")
    return parser


def network_command(args: argparse.Namespace) -> int:
    action = getattr(args, "network_command", None)
    as_json = bool(getattr(args, "json", False))
    try:
        if action == "roster":
            return _cmd_roster(as_json)
        if action == "provision":
            return _cmd_provision(args, as_json)
        if action == "route":
            return _cmd_route(args.name, as_json)
        if action == "dispatch":
            return _cmd_dispatch(args, as_json)
        if action == "jobs":
            return _cmd_jobs(args, as_json)
        if action == "credentials":
            return _cmd_credentials(args, as_json)
        if action == "status":
            return _cmd_status(as_json)
        print(
            "Usage: hermes network {roster|provision|route|dispatch|jobs|credentials|status}",
            file=sys.stderr,
        )
        return 2
    except (UnknownAgentError, LinearLinkError, DispatchError) as exc:
        _emit({"error": str(exc)}, as_json, error=True)
        return 1


def _emit(payload: Any, as_json: bool, *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    if as_json:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        return
    if isinstance(payload, dict) and "error" in payload and len(payload) == 1:
        print(f"error: {payload['error']}", file=stream)
        return
    if isinstance(payload, list):
        for item in payload:
            print(_fmt_line(item))
        return
    print(_fmt_line(payload))


def _fmt_line(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)
    if "handle" in item and "model" in item:
        core = " core" if item.get("core") else ""
        created = ""
        if "created" in item:
            created = " created" if item["created"] else " exists"
        return (
            f"{item['handle']:22s}  {item.get('profile', ''):12s}  "
            f"{item['model']:18s}{core}{created}"
        )
    if "identifier" in item and "goal" in item:
        return (
            f"{item.get('id', '')}  {item.get('status', ''):8s}  "
            f"{item.get('alias', '')}  {item.get('linear', {}).get('identifier', '')}"
        )
    if item.get("granted") is not None and "secret_name" in item:
        state = "granted" if item["granted"] else "denied"
        err = f" ({item['error']})" if item.get("error") else ""
        return f"{item['profile']} {item['secret_name']} {state}{err}"
    if "error" in item:
        return f"error: {item['error']}"
    return json.dumps(item, sort_keys=True)


def _cmd_roster(as_json: bool) -> int:
    rows = [
        {
            "lane": agent.lane,
            "alias": agent.alias,
            "profile": agent.profile,
            "handle": agent.handle,
            "model": agent.model,
            "provider": agent.provider,
            "role": agent.role,
            "core": agent.core,
        }
        for agent in list_roster()
    ]
    _emit(rows, as_json)
    return 0


def _cmd_provision(args: argparse.Namespace, as_json: bool) -> int:
    results = provision_roster(
        core_only=bool(args.core),
        names=list(args.names) or None,
        no_skills=not bool(args.seed_skills),
    )
    _emit([row.to_dict() for row in results], as_json)
    return 0


def _cmd_route(name: str, as_json: bool) -> int:
    agent = resolve_agent(name)
    _emit(
        {
            "query": name,
            "lane": agent.lane,
            "alias": agent.alias,
            "profile": agent.profile,
            "handle": agent.handle,
            "model": agent.model,
            "provider": agent.provider,
            "core": agent.core,
        },
        as_json,
    )
    return 0


def _cmd_dispatch(args: argparse.Namespace, as_json: bool) -> int:
    broker = DispatchBroker()
    job = broker.dispatch(
        args.target,
        args.goal,
        args.linear,
        inject_credentials=not bool(args.no_credentials),
    )
    _emit(job.to_dict(), as_json)
    return 0


def _cmd_jobs(args: argparse.Namespace, as_json: bool) -> int:
    broker = DispatchBroker()
    if args.job_id:
        _emit(broker.get(args.job_id).to_dict(), as_json)
        return 0
    _emit([job.to_dict() for job in broker.list_jobs()], as_json)
    return 0


def _cmd_credentials(args: argparse.Namespace, as_json: bool) -> int:
    action = getattr(args, "credentials_command", None)
    broker = CredentialBroker()
    if action == "catalog":
        catalog = [
            {"name": name, "reference": ref}
            for name, ref in broker.catalog().items()
        ]
        _emit(catalog, as_json)
        return 0
    if action == "request":
        agent = resolve_agent(args.profile)
        receipt = broker.request(agent.profile, args.name)
        _emit(receipt.to_dict(), as_json)
        return 0 if receipt.granted else 1
    print("Usage: hermes network credentials {catalog|request}", file=sys.stderr)
    return 2


def _cmd_status(as_json: bool) -> int:
    from hermes_cli.profiles import get_profile_dir

    roster_rows = []
    for agent in list_roster():
        profile_dir = get_profile_dir(agent.profile)
        provider, model = read_pinned_model(profile_dir) if profile_dir.exists() else ("", "")
        roster_rows.append(
            {
                "handle": agent.handle,
                "profile": agent.profile,
                "expected_model": agent.model,
                "pinned_model": model,
                "pinned_provider": provider,
                "provisioned": profile_dir.exists(),
                "core": agent.core,
            }
        )
    jobs = [job.to_dict() for job in DispatchBroker().list_jobs()[:10]]
    payload = {"roster": roster_rows, "recent_jobs": jobs}
    _emit(payload, as_json)
    return 0
