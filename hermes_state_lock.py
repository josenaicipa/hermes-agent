"""Cross-process write admission for Hermes state databases.

SQLite WAL allows many readers and exactly one writer.  Its busy handler is
not a queue: blocked writers re-probe, and whoever probes at the right
instant wins.  Gateway, Dashboard and ACP are separate processes against the
same file, so that unfairness is visible as ``database is locked`` on
``create_session`` / ``append_message`` while a sibling's schema init (or
another writer) holds the lock — the 2026-08-19 vpsclone incident.

An in-process gate cannot fix that class.  This module is the cross-process
half: one ``flock``/``msvcrt.locking`` admission token, acquired around
schema init and around each ``BEGIN IMMEDIATE``.

Why ``flock`` (and not a pidfile):

* The kernel drops the lock when the holding process dies, so a crash cannot
  leave an orphan that wedges every later writer.
* Readers never take it, so WAL concurrent reads stay intact.

One fixed sidecar per OS account, not one per database (Fase C7):

Earlier revisions named the sidecar after the *database* — a stable hash of
``canonical_db_key(db_path)``, which resolved symlinks and, where the
underlying filesystem actually folded case and/or Unicode normalization,
folded those too, so that ``state.db`` and a symlink or differently-spelled
alias of it would converge on one lock file. That kept growing to cover one
more alias class (symlink, then case, then Unicode normalization, then
compatibility ligatures) because path-based identity is unstable in ways
none of those probes could ever fully enumerate — and the case that broke it
outright was simpler than any of them: an arbitrary *hardlink* to the same
inode, created under any name in any directory the attacker can write to,
resolves through ``realpath`` to the identical canonical path the original
name does. There is no alias-detection scheme to add here; the identity a
hardlink presents to ``os.path.realpath`` is not an imitation of the real
database's identity, it *is* the real database's identity, by construction.

The fix is not a better per-database identity check; it is not needing one.
Every write to every Hermes state database under this OS account now
contends for exactly one sidecar (``global.write.lock``) inside the private
per-user lock root described below. ``db_path`` is still accepted by
:func:`acquire_state_write_lock` — every caller in this codebase already
passes one, and it remains useful for logging/diagnostics — but it is pure
context now: nothing about the lock's identity or location is derived from
it, so no path transform (symlink, hardlink, case fold, Unicode fold,
something this module never anticipated) can ever cause two databases, or
two aliases of one database, to land on different lock files. They cannot,
because there is only one.

The accepted cost: two processes writing to genuinely *different* Hermes
state databases under the same OS account now serialize against each other
too, not just against writers on the same database. That is intentional —
Fase C7 chooses correctness and simplicity over that throughput, and Hermes's
actual deployment shape (Gateway, Dashboard, ACP, one profile) does not run
enough distinct state databases per account for that to be a measurable
regression in practice.

Why the lock file does NOT live next to any ``state.db`` (private lock root):

A sidecar colocated with the database (``state.db.write.lock``, sibling to
the file) lives in a directory a co-tenant of that directory can write to.
Such a co-tenant can ``unlink`` the sidecar and drop in a brand-new, ordinary
regular file under the same name — nothing in that fresh file's metadata
differs from a sidecar this module would have created itself, so no
``fstat``-only check can tell them apart. A holder that kept the old inode
locked and a second process that locks the new one would both believe they
hold admission — the exact mutual-exclusion break this module exists to
prevent.

The fix is not a better detector; it is removing the shared directory. The
sidecar lives in a **private, per-user lock root** that a co-tenant of any
database's directory cannot write into at all, so there is no directory
entry for it to unlink or replace in the first place:

* POSIX: ``<passwd home>/.hermes-state-locks-<uid>`` — the home directory
  from the **passwd database** (``pwd.getpwuid``), never from ``$HOME`` or
  any other environment variable (see the "Fase C8" section below for why
  the root must not depend on per-process environment). When the account
  has no usable passwd entry (no entry at all, or a non-absolute
  ``pw_dir``), the root falls back to the static path
  ``/tmp/hermes-state-locks-<uid>`` — a literal ``/tmp``, not
  ``tempfile.gettempdir()``, for the same reason. Either way the root is
  created with mode ``0700``
  and re-verified on every acquire — owned by the current effective user,
  not a symlink, and exactly ``0700`` (loosened permissions are tightened
  back with ``chmod`` when we own the directory; corrected can't when we
  don't). Any of those checks failing means the invariant this design
  depends on cannot be guaranteed, so the acquire **fails closed** (yields
  ``False``, the same signal ordinary lock contention produces) rather than
  silently falling back to an unprotected acquire — that fallback is exactly
  the co-tenant-writable-directory condition being removed. This is
  distinct from simply being unable to create the root at all (parent
  missing, read-only filesystem, ``ENOSPC``): that is an operational
  condition, not evidence of tampering, and degrades to admitted the same
  way an unopenable lock file always has.
* Windows: ``%LOCALAPPDATA%\\hermes\\state-locks``. ``%LOCALAPPDATA%`` is
  already restricted to the owning user profile by the OS's own ACLs, so
  there is no POSIX-style owner/mode dance to perform there; only creation
  is attempted, and a failure degrades to admitted (same operational-vs-
  tampering distinction as POSIX).

The old symlink-swap (``O_NOFOLLOW``) and hardlink-swap (``fstat``/``lstat``
identity) defenses on the sidecar itself are kept as defense-in-depth: they
now protect against same-account accidents (a rotation script or bug
recreating the lock file while a hold is live), not against a hostile
co-tenant, since a co-tenant can no longer reach the directory at all. They
are also not a defense against a hostile *same-account* actor — see the
threat model below for why that is a different, unclosable case.

Threat model — what this module protects and what it explicitly does not:

* **In scope, fully handled:** cooperative Hermes processes of the same OS
  account racing each other (Gateway, Dashboard, ACP, a CLI invocation) —
  handled by ``flock`` plus the admission bookkeeping below — and a
  *different* OS account's co-tenant that can write into any database's own
  directory but not into this account's private lock root — handled by the
  0700 ownership check on that root.
* **Explicitly out of scope: an adversarial process running as the *same*
  OS account.** That account owns every filesystem object this module
  touches — the private lock root, the sidecar, every database file itself —
  so it can delete and recreate any of them at will, including reproducing a
  sidecar this module would have created itself, with no distinguishing
  metadata left behind. No filesystem-based anchor (inode identity, link
  count, ownership, permission mode) can tell a same-account attacker's
  replacement apart from a legitimate one, because the attacker holds
  exactly the same rights over that location that this module does. This
  module does not claim, and must not be read as claiming, resistance to
  that actor. Where that actor is a real concern, the fix is process
  isolation — a dedicated service account per tenant — not a cleverer check
  in this file; no check in this file can draw that boundary.
* ``_check_sidecar_continuity`` (below) narrows the *accidental* slice of
  that same-account case — a cleanup script, a rotation bug, a
  ``tmpfiles.d``-style sweep of ``/tmp`` recreating the sidecar while this
  process is still running — by remembering, in this process's own memory,
  the identity it last trusted *per lock root* (Fase C9), and refusing
  admission the moment a later acquire in this same process, under that
  same root, disagrees with that memory. It is
  deliberately not a security boundary: a same-account attacker who never
  lets this process observe a consistent baseline (a swap staged before this
  process's first acquire, or one that waits for a restart, which wipes the
  in-memory baseline) is not caught by it — that residual gap is the same
  one the paragraph above describes, not a new one this check introduces.

The lock is **not** a substitute for SQLite's own locking.  Callers still
``BEGIN IMMEDIATE`` and still retry on ``SQLITE_BUSY`` from holders that do
not go through this module (``sqlite3`` CLI, mixed-version processes).
Admission is released before those retries so we never hold the token while
waiting on an ungated writer.

Per-thread re-entrant: a nested write (reconnect during a failed write, a
helper that writes again — to the same database or, now, to any database
under this account) must not block on its own flock. Other threads and
other processes wait.

If the lock file cannot be opened (private root verified, but e.g. exhausted
fds) the acquire degrades to admitted — the in-process ``SessionDB._lock``
plus SQLite busy handling remain, which is what shipped before this module
existed.

Fase C8 — the lock root is deterministic per UID, never per environment
(Nemo gate 20260820T144423Z):

Fase C7's root was ``<tempfile.gettempdir()>/hermes-state-locks-<uid>``,
and ``gettempdir()`` reads ``TMPDIR``/``TEMP``/``TMP`` — *per-process*
state. Two processes of the same account writing the same ``state.db``
with different ``TMPDIR`` (a systemd unit with ``PrivateTmp=``/a
per-service ``TMPDIR=``, a cron job, a login shell — exactly the
Gateway/Dashboard/ACP shape) would derive *different* roots, therefore
different sidecars, and both would be admitted concurrently: the global
lock silently stops being global. C7 documented that as an unenforceable
deployment-consistency caveat; C8 removes the dependency instead of
documenting it.

The root is now a pure function of the **uid**, resolved through
system-wide state only:

* Primary: the account's home directory from the passwd database
  (``pwd.getpwuid(uid).pw_dir``). The passwd database is host state,
  identical for every process of the account no matter what its
  environment says — unlike ``$HOME``, which subprocess spawners in this
  codebase deliberately repoint (``HERMES_HOME``-scoped profiles). A
  process that can write any Hermes state database can, in Hermes's
  deployment shape, also write this account's home, so the root does not
  demand any access the writer did not already have.
* Fallback (account has no passwd entry, or its ``pw_dir`` is not an
  absolute path): the static literal ``/tmp/hermes-state-locks-<uid>``.
  Whether an account has a usable passwd entry is also host state, so
  every process of the account takes the same branch.

``HERMES_STATE_LOCK_ROOT`` remains an explicit override (tests, unusual
layouts). It is an environment variable, so it reintroduces the
per-process divergence risk *by explicit operator choice*; setting it
inconsistently across Hermes processes is unsupported.

What no root choice can survive: mount-namespace isolation that presents
different filesystems to different processes (``ProtectHome=``, chroots,
containers bind-mounting the same ``state.db`` but not the same home /
``/tmp``). No path resolves identically across namespaces that disagree
about what is mounted where; that boundary needs deployment configuration,
not a cleverer path. Similarly, a home directory on a filesystem without
working ``flock`` semantics (NFSv3 without ``lockd``) degrades to SQLite's
own busy handling, as any unopenable/unlockable sidecar always has.

Rolling upgrades: a pre-C8 process (root under ``gettempdir()``) and a
post-C8 process (root under the passwd home) use different sidecars for
the window in which both are running. That window falls back to SQLite
busy handling — the documented behavior for mixed-version writers since
this module was introduced. Restart all Hermes processes together, as the
prior C-phase lock-identity changes already required.

Fase C9 — sidecar continuity is remembered per lock root, never once per
process (Nemo gate 20260820T152324Z):

The C5 continuity baseline was a single process-global ``(st_dev,
st_ino)``. That conflated two very different events: "the sidecar under
the root I am using was replaced" (the accident the check exists to
catch) and "this process is now resolving a different root" (a legitimate
``HERMES_STATE_LOCK_ROOT`` change mid-run, or simply two valid roots
observed by one process). The first acquire under any second valid root
was refused as a phantom swap, permanently — cooperative writers wedged
with no tampering anywhere. The baseline is now a mapping from the lock
root's canonical path (:func:`_lock_root_identity` — ``realpath`` +
``normcase`` of the *root* this module itself resolved, never of
``db_path``; per-database identity stays gone per Fase C7) to the
identity last trusted inside that root. Continuity is compared and
updated only within one root; a witnessed swap still poisons that root
for the life of the process (the same fail-closed rule as before, now
exactly as wide as the evidence), and every other root is unaffected.
"""

from __future__ import annotations

import contextlib
import logging
import os
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("hermes_state")

_IS_WINDOWS = sys.platform == "win32"
_POLL_S = 0.05
_LOCK_ROOT_PREFIX = "hermes-state-locks"
_ROOT_MODE = 0o700
# Fixed name: one sidecar per OS account, shared by every Hermes state
# database that account writes to (see the module docstring's "Fase C7"
# section for why this replaced a per-database hash).
_WRITE_LOCK_FILENAME = "global.write.lock"

# Per-thread hold for the single global lock. Depth tracks nested
# re-entrant acquires on the same thread (a nested write during a failed
# write, a helper that writes again) so they do not open a second fd and
# self-deadlock (Linux flock is per open-file-description; two fds of the
# same path block each other).
_tls = threading.local()

# Per lock root (keyed by the root's canonical path — see
# _lock_root_identity): the (st_dev, st_ino) this *process* (not just this
# thread) last trusted for the sidecar inside that root. Deliberately
# process-wide, not thread-local: the gap this closes is a second thread in
# the same process opening a sidecar that was unlinked-and-recreated (no
# hardlink, so _sidecar_identity_is_trustworthy's nlink check sees nothing
# wrong) while a first thread's hold is still live — that only shows up by
# comparing against what this process itself remembers, not against the
# current acquire attempt's own fd. Keyed per root (Fase C9) rather than
# held as one process-global baseline: a baseline recorded under one root
# says nothing about the sidecar inside a different root, so two valid
# roots in one process (a legitimate ``HERMES_STATE_LOCK_ROOT`` change, a
# test harness) each bootstrap and keep their own continuity instead of
# the second one being refused as a phantom swap of the first. See
# _check_sidecar_continuity and the module docstring's threat model.
_sidecar_identity_lock = threading.Lock()
_known_sidecar_identity: dict[str, tuple[int, int]] = {}


class _Hold:
    __slots__ = ("handle", "depth")

    def __init__(self, handle) -> None:
        self.handle = handle
        self.depth = 1


class _LockRootUnsafe(Exception):
    """The private lock root exists but fails a POSIX safety invariant.

    Raised (never returned as a bool) so ``acquire_state_write_lock`` cannot
    accidentally treat "exists but tampered/misowned" the same as "could not
    be created at all" — the two must fail differently (closed vs. degrade).
    """


class _SidecarIdentitySwapped(Exception):
    """This process has direct evidence its trusted sidecar was replaced.

    Raised (never returned as a bool) when a freshly-opened, otherwise
    "trustworthy" sidecar's ``(st_dev, st_ino)`` disagrees with the identity
    this same process previously recorded *for the same lock root* — see
    :func:`_check_sidecar_continuity`.

    Once raised, this process never admits again under that lock root: the
    recorded baseline is
    intentionally never overwritten with the new, disagreeing identity, so
    a retry cannot quietly converge on a still-live same-account swap.
    Recovering requires a process restart, which starts the in-memory
    baseline over from an unpoisoned bootstrap. That is a deliberate
    fail-closed choice, not an oversight — silently re-trusting whatever now
    has the name is exactly the residual bypass this check exists to close.
    It is not, and is not meant to be, resistance to a determined
    same-account attacker; see the module docstring's threat model.
    """


# Fallback base when the account has no usable passwd entry. A literal
# ``/tmp``, never ``tempfile.gettempdir()``: gettempdir() reads
# TMPDIR/TEMP/TMP, which is per-process state — the exact divergence Fase C8
# removes (see the module docstring).
_POSIX_FALLBACK_TMP = Path("/tmp")


def _posix_passwd_home() -> Optional[Path]:
    """Current account's home from the passwd database, or ``None``.

    Deliberately never consults ``$HOME`` (or any other environment
    variable): subprocess spawners in this codebase repoint ``HOME`` at
    ``HERMES_HOME``-scoped profiles, and a per-process source would split
    the lock root across processes — the Fase C8 bug class. ``None`` means
    "no entry for this uid" or "entry without an absolute ``pw_dir``";
    both are host-wide conditions, so every process of the account
    resolves the same answer.
    """
    try:
        import pwd

        entry = pwd.getpwuid(os.getuid())
    except (ImportError, AttributeError, KeyError, OSError):
        return None
    home = (entry.pw_dir or "").strip()
    if not home or not os.path.isabs(home):
        return None
    return Path(home)


def _posix_lock_root() -> Path:
    """Deterministic per-uid root — a pure function of host state (Fase C8).

    ``<passwd home>/.hermes-state-locks-<uid>`` normally;
    ``/tmp/hermes-state-locks-<uid>`` when the account has no usable passwd
    entry. Nothing here reads the process environment, so two processes of
    the same account always derive the same root regardless of how their
    ``TMPDIR``/``TEMP``/``TMP``/``HOME`` differ.
    """
    try:
        uid = os.getuid()
    except AttributeError:  # pragma: no cover - no POSIX getuid, shouldn't happen
        uid = "unknown"
    home = _posix_passwd_home()
    if home is not None:
        return home / f".{_LOCK_ROOT_PREFIX}-{uid}"
    return _POSIX_FALLBACK_TMP / f"{_LOCK_ROOT_PREFIX}-{uid}"


def _windows_lock_root() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
    return base / "hermes" / "state-locks"


def _lock_root(is_windows: bool = _IS_WINDOWS) -> Path:
    """Return the private per-user lock root (no I/O — pure path math).

    Deliberately independent of ``db_path``/``HERMES_HOME``: the root's
    safety comes from being an OS-account-private location this module
    fully controls, not from wherever a particular database happens to
    live.

    ``HERMES_STATE_LOCK_ROOT`` overrides the platform default when set.
    This exists so tests never touch the real per-account lock root shared
    with any live Hermes process on the same machine, and so an operator
    with an unusual home/profile layout can relocate it. Being an
    environment variable, it is per-process state: the operator owns
    keeping it identical across every Hermes process of the account
    (Fase C8 made the *default* independent of the environment precisely
    so nothing requires this variable in normal deployments). The override
    is still subject to the exact same :func:`_ensure_private_lock_root`
    safety checks, so pointing it at an unsafe directory fails closed rather
    than silently reopening the co-tenant bypass.
    """
    override = os.environ.get("HERMES_STATE_LOCK_ROOT", "").strip()
    if override:
        return Path(override)
    return _windows_lock_root() if is_windows else _posix_lock_root()


def _ensure_private_lock_root(root: Path, is_windows: bool = _IS_WINDOWS) -> bool:
    """Create *root* if needed and verify it is safe to hold lock sidecars.

    Returns ``True`` once *root* is confirmed private to this account.
    Returns ``False`` when *root* could not even be created/stat'd (missing
    parent, read-only filesystem, exhausted resources) — an operational
    condition the caller degrades from, same as an unopenable lock file.

    Raises :class:`_LockRootUnsafe` when *root* exists but fails a safety
    invariant (symlink, wrong owner, or a permission mode we cannot correct)
    — the caller must fail closed for this case, not degrade, because an
    unverified root is precisely the co-tenant-writable-directory condition
    this design removes.
    """
    if is_windows:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        # %LOCALAPPDATA% is already restricted to the owning user profile by
        # NTFS ACLs; there is no POSIX-style owner/mode check to perform.
        return True

    try:
        os.mkdir(root, _ROOT_MODE)
    except FileExistsError:
        pass
    except OSError:
        return False

    try:
        st = os.lstat(root)
    except OSError:
        return False

    if stat.S_ISLNK(st.st_mode):
        # Never follow it, never replace it ourselves — just refuse.
        raise _LockRootUnsafe(f"{root} is a symlink, refusing to use it")
    if not stat.S_ISDIR(st.st_mode):
        raise _LockRootUnsafe(f"{root} is not a directory")

    try:
        current_uid = os.getuid()
    except AttributeError:  # pragma: no cover - no POSIX getuid, shouldn't happen
        raise _LockRootUnsafe("cannot determine current uid to verify ownership")
    if st.st_uid != current_uid:
        raise _LockRootUnsafe(f"{root} is owned by uid {st.st_uid}, not {current_uid}")

    if stat.S_IMODE(st.st_mode) != _ROOT_MODE:
        try:
            os.chmod(root, _ROOT_MODE)
            st = os.lstat(root)
        except OSError as exc:
            raise _LockRootUnsafe(f"could not tighten {root} to 0700: {exc}") from exc
        if stat.S_ISLNK(st.st_mode) or stat.S_IMODE(st.st_mode) != _ROOT_MODE:
            raise _LockRootUnsafe(f"{root} did not converge to 0700")

    return True


def write_lock_path() -> Path:
    """Return the single, per-user sidecar lock path inside the private root.

    Fixed name (``global.write.lock``), not derived from any database path —
    every Hermes state database under this OS account shares this one
    sidecar. See the module docstring's "Fase C7" section for why a
    per-database identity (symlink-resolved, case/Unicode-folded) was
    replaced with this: an arbitrary hardlink to a database's inode
    resolves through ``realpath`` to that database's own canonical path, so
    no per-database identity scheme built on path canonicalization can ever
    fully close that alias class.
    """
    return _lock_root() / _WRITE_LOCK_FILENAME


def _lock_open_flags(is_windows: bool = _IS_WINDOWS) -> int:
    """Flags for opening the sidecar lock file.

    POSIX adds ``O_NOFOLLOW`` so a sidecar swapped for a symlink between
    processes is refused (``ELOOP``) instead of silently followed to
    whatever it points at. This is defense-in-depth against a same-account
    race or bug now that the containing directory is private — a co-tenant
    can no longer reach this path to swap it at all. Windows has no
    equivalent open flag, so the platform split is explicit here rather than
    inside the ``try``. ``O_NOFOLLOW`` is POSIX-only *and* not guaranteed to
    exist on every POSIX ``os`` module (e.g. some minimal/embedded builds) —
    a bare ``os.O_NOFOLLOW`` reference would raise ``AttributeError`` outside
    the caller's ``except OSError``, so it is looked up with ``getattr`` and
    simply omitted where absent rather than crashing the acquire.
    """
    flags = os.O_RDWR | os.O_CREAT
    if not is_windows:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _hold() -> Optional[_Hold]:
    return getattr(_tls, "hold", None)


def _set_hold(hold: Optional[_Hold]) -> None:
    _tls.hold = hold


def _try_exclusive(handle) -> bool:
    """Non-blocking exclusive lock.  True if this handle now holds it."""
    try:
        if _IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _unlock(handle) -> None:
    try:
        if _IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _open_sidecar(lock_path: Path):
    """Open (creating if absent) a fresh handle on *lock_path*."""
    fd = os.open(str(lock_path), _lock_open_flags(), 0o644)
    return os.fdopen(fd, "a+b")


def _sidecar_identity_is_trustworthy(handle, lock_path: Path) -> bool:
    """True if *handle*'s fd is still the sole-linked, live entry at
    *lock_path* — i.e. flock on it means something.

    Defense-in-depth against a same-account race (not a hostile co-tenant,
    who can no longer reach the private lock root at all): a sidecar this
    module creates is always exactly one directory entry pointing at one
    inode. If the name has been unlinked and a hardlink dropped onto it
    since we opened it, the fd we hold and the name now on disk have
    diverged: either the inode gained a second name (``st_nlink != 1``) or
    the name currently points elsewhere entirely (``(st_dev, st_ino)``
    mismatch).
    """
    try:
        fd_stat = os.fstat(handle.fileno())
        path_stat = os.lstat(str(lock_path))
    except OSError:
        return False
    if fd_stat.st_nlink != 1:
        return False
    return (fd_stat.st_dev, fd_stat.st_ino) == (path_stat.st_dev, path_stat.st_ino)


def _lock_root_identity(root: Path) -> str:
    """Stable per-process key for *root* in ``_known_sidecar_identity``.

    ``os.path.realpath`` + ``os.path.normcase`` so different spellings of
    the same physical root (a symlinked *parent* directory — the root
    itself is refused if it is a symlink — or Windows case variance) share
    one continuity baseline instead of splitting into two that could each
    miss a swap the other witnessed. This canonicalizes the *lock root* —
    a directory this module itself resolved and re-verifies on every
    acquire — never ``db_path``; per-database identity stays gone
    (Fase C7), and nothing here reads the database path at all.
    """
    return os.path.normcase(os.path.realpath(os.fspath(root)))


def _check_sidecar_continuity(root_key: str, identity: tuple[int, int]) -> None:
    """Compare *identity* against what this process itself last trusted for
    the sidecar inside the lock root identified by *root_key*, recording a
    first-time baseline for that root rather than rejecting it.

    ``_sidecar_identity_is_trustworthy`` only compares a fd against the
    *current* on-disk name at the instant of one acquire attempt — a plain
    ``unlink`` followed by a brand-new regular file (no hardlink, so
    ``st_nlink`` stays ``1``) is entirely self-consistent from that single
    attempt's point of view; there is nothing in that one stat call to
    disagree with. This closes that gap using memory that check does not
    have: what *this process* itself has already trusted.

    The memory is per lock root (Fase C9, Nemo gate 20260820T152324Z), not
    one process-global value: a baseline recorded under one root is an
    observation about *that root's* sidecar and nothing else. Holding it
    globally turned every second valid root in the same process — a
    legitimate ``HERMES_STATE_LOCK_ROOT`` change mid-run, a harness giving
    each test its own root — into a phantom "swap" that permanently wedged
    admission. Comparing only within one root keeps the real detection
    (below) and removes the false one.

    Bootstrap — no prior identity recorded in this process for this root —
    is not a failure. There is nothing yet to compare against, and refusing
    here would break the very first acquire under any root, which is
    exactly the case that must keep working.

    Once a baseline is recorded for a root, a *different* identity turning
    up on a later acquire under that same root is not inferred, it is
    witnessed: this process previously held that root's sidecar open, and
    something removed and replaced it since. The mismatch is reported by
    raising rather than by updating the baseline to the new value, so a
    same-account swap that is still live (a second thread's hold not yet
    released) cannot be quietly re-trusted on the very next attempt — see
    ``_SidecarIdentitySwapped``. Other roots are unaffected: the poisoning
    is exactly as wide as the evidence.
    """
    with _sidecar_identity_lock:
        previous = _known_sidecar_identity.setdefault(root_key, identity)
    if previous != identity:
        raise _SidecarIdentitySwapped(
            f"sidecar identity under lock root {root_key} changed from "
            f"{previous} to {identity} since this process last trusted it"
        )


def _reset_after_fork() -> None:
    """Drop an inherited hold in a forked child without unlocking the parent.

    ``fork`` duplicates fds onto the same open-file-description, so
    ``LOCK_UN`` here would release the parent's admission.  Closing the
    inherited fd is enough: the parent still holds the description.
    """
    hold = _hold()
    _set_hold(None)
    if hold is not None:
        try:
            hold.handle.close()
        except Exception:
            pass


if hasattr(os, "register_at_fork"):  # pragma: no branch - POSIX only
    os.register_at_fork(after_in_child=_reset_after_fork)


@contextlib.contextmanager
def acquire_state_write_lock(
    db_path: os.PathLike | str,
    *,
    timeout_s: float = 1.0,
) -> Iterator[bool]:
    """Admit this thread as the Hermes writer.

    *db_path* is accepted for call-site compatibility and diagnostics only
    — see the module docstring's "Fase C7" section — and never determines
    the lock's identity or location. Every Hermes state database under this
    OS account shares one admission token.

    Yields ``True`` when this thread holds admission (including a nested
    re-acquire, and including the degrade-open path).  Yields ``False`` when
    the bounded wait expired, OR when the private lock root's safety
    invariant could not be guaranteed, OR when this process's own memory of
    the sidecar's identity disagrees with what it just opened (fail-closed
    in all three cases — see the module docstring's threat model and
    :func:`_check_sidecar_continuity`); the caller has not touched SQLite
    and must treat all three the same way: as contention.

    A non-positive *timeout_s* still performs one non-blocking attempt so a
    caller whose budget is already spent can make one last honest try.
    """
    del db_path  # context/logging only — see docstring; not the lock key.

    existing = _hold()
    if existing is not None:
        existing.depth += 1
        try:
            yield True
        finally:
            existing.depth -= 1
        return

    root = _lock_root()
    try:
        root_ready = _ensure_private_lock_root(root)
    except _LockRootUnsafe as exc:
        logger.error(
            "State write-lock root %s failed a safety check (%s) — "
            "refusing admission rather than proceeding unprotected.",
            root,
            exc,
        )
        yield False
        return
    if not root_ready:
        logger.warning(
            "Could not create state write-lock root %s — proceeding with "
            "SQLite busy handling only.",
            root,
        )
        yield True
        return

    lock_path = write_lock_path()
    try:
        handle = _open_sidecar(lock_path)
    except OSError as exc:
        logger.warning(
            "Could not open state.db write lock %s (%s) — proceeding with "
            "SQLite busy handling only.",
            lock_path,
            exc,
        )
        yield True
        return

    deadline = time.monotonic() + max(0.0, timeout_s)
    acquired = False
    try:
        while True:
            if _try_exclusive(handle):
                if _sidecar_identity_is_trustworthy(handle, lock_path):
                    acquired = True
                    break
                # We won the flock, but the fd is no longer the live,
                # sole-linked entry at lock_path — a same-account race
                # (hardlink, or unlink+recreate racing our own open)
                # swapped it after we opened it. Drop this fd — never the
                # path itself, unlinking it ourselves would just be the
                # same attack performed by trusted code — and fall through
                # to reopen a fresh candidate against whatever is live now.
                _unlock(handle)
                try:
                    handle.close()
                except OSError:
                    pass
                handle = None

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_POLL_S, remaining))

            if handle is None:
                try:
                    handle = _open_sidecar(lock_path)
                except OSError:
                    break

        if not acquired or handle is None:
            yield False
            return

        try:
            fd_stat = os.fstat(handle.fileno())
            identity: Optional[tuple[int, int]] = (fd_stat.st_dev, fd_stat.st_ino)
        except OSError:
            identity = None
        if identity is not None:
            try:
                _check_sidecar_continuity(_lock_root_identity(root), identity)
            except _SidecarIdentitySwapped as exc:
                logger.error(
                    "State write-lock sidecar %s: %s — refusing admission "
                    "rather than trusting a same-account replacement this "
                    "process has not itself verified.",
                    lock_path,
                    exc,
                )
                _unlock(handle)
                try:
                    handle.close()
                except OSError:
                    pass
                handle = None
                yield False
                return

        hold = _Hold(handle)
        _set_hold(hold)
        released = False
        try:
            yield True
        finally:
            current = _hold()
            if current is hold:
                hold.depth -= 1
                if hold.depth <= 0:
                    _set_hold(None)
                    _unlock(handle)
                    try:
                        handle.close()
                    except OSError:
                        pass
                    released = True
        if released:
            handle = None
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
