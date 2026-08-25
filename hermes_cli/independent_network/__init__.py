"""Independent hybrid agent network (NAI-68).

Canonical roster, isolated profiles, deterministic alias routing, an
asynchronous dispatch broker, mandatory Linear issue linking, and
brokered 1Password credential access that never puts secret values in
prompts, memory, receipts, or audit logs.
"""

from hermes_cli.independent_network.broker import (
    DispatchBroker,
    DispatchError,
    Job,
)
from hermes_cli.independent_network.credentials import (
    CredentialBroker,
    CredentialReceipt,
    SecretRevealedError,
)
from hermes_cli.independent_network.linear import (
    LinearLink,
    LinearLinkError,
    parse_linear_issue,
    require_linear_issue,
)
from hermes_cli.independent_network.provision import (
    ProvisionResult,
    provision_roster,
)
from hermes_cli.independent_network.roster import (
    AgentSpec,
    CANONICAL_ROSTER,
    get_agent,
    list_roster,
    provider_for_model,
)
from hermes_cli.independent_network.routing import (
    UnknownAgentError,
    resolve_agent,
)

__all__ = [
    "AgentSpec",
    "CANONICAL_ROSTER",
    "CredentialBroker",
    "CredentialReceipt",
    "DispatchBroker",
    "DispatchError",
    "Job",
    "LinearLink",
    "LinearLinkError",
    "ProvisionResult",
    "SecretRevealedError",
    "UnknownAgentError",
    "get_agent",
    "list_roster",
    "parse_linear_issue",
    "provider_for_model",
    "provision_roster",
    "require_linear_issue",
    "resolve_agent",
]
