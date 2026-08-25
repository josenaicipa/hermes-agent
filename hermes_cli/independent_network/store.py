"""On-disk state for the independent agent network.

Lives under the default Hermes root so every isolated profile shares the
same broker store. Secret *values* are never written here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


NETWORK_DIRNAME = "independent-agent-network"


def network_root(home: Optional[Path] = None) -> Path:
    """Return the network state directory, creating it if needed."""
    if home is None:
        from hermes_constants import get_default_hermes_root

        home = get_default_hermes_root()
    root = Path(home) / NETWORK_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def jobs_dir(home: Optional[Path] = None) -> Path:
    path = network_root(home) / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def audit_dir(home: Optional[Path] = None) -> Path:
    path = network_root(home) / "audit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write(path: Path, text: str, *, mode: int = 0o600) -> None:
    """Write ``text`` via a temp file, then chmod owner-only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def write_json(path: Path, payload: Any, *, mode: int = 0o600) -> None:
    atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n", mode=mode)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: Any, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
    try:
        os.chmod(path, mode)
    except OSError:
        pass
