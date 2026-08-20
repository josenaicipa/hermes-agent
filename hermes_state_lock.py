"""Cross-process write admission for the shared ``state.db``.

SQLite WAL allows many readers and exactly one writer.  Its busy handler is
not a queue: blocked writers re-probe, and whoever probes at the right
instant wins.  Gateway, Dashboard and ACP are separate processes against the
same file, so that unfairness is visible as ``database is locked`` on
``create_session`` / ``append_message`` while a sibling's schema init (or
another writer) holds the lock — the 2026-08-19 vpsclone incident.

An in-process gate cannot fix that class.  This module is the cross-process
half: one ``flock``/``msvcrt.locking`` admission token per canonical database
path, acquired around schema init and around each ``BEGIN IMMEDIATE``.

Why ``flock`` (and not a pidfile):

* The kernel drops the lock when the holding process dies, so a crash cannot
  leave an orphan that wedges every later writer.
* Readers never take it, so WAL concurrent reads stay intact.

Why the lock file does NOT live next to ``state.db`` (private lock root):

Earlier revisions of this module kept the admission token as a sibling
sidecar (``state.db.write.lock``) next to the database, hardened with
``O_NOFOLLOW`` against a symlink swap and an ``fstat``/``lstat`` re-check
(``st_nlink`` + ``(st_dev, st_ino)``) against a hardlink swap. Both defenses
assume the attacker substitutes something *distinguishable* — a symlink, or
a second name pointing at a still-open inode. Neither survives the residual
case Nemo found: a co-tenant with write access to the database's directory
can ``unlink`` the sidecar and drop in a brand-new, ordinary regular file
under the same name. Nothing in that fresh file's metadata differs from a
sidecar this module would have created itself — no ``fstat``-only check can
tell them apart — so a holder that kept the old inode locked and a second
process that locks the new one both believe they hold admission. That is
the exact mutual-exclusion break this module exists to prevent.

The fix is not a better detector; it is removing the shared directory. The
sidecar now lives in a **private, per-user lock root** that a co-tenant of
the database's directory cannot write into at all, so there is no directory
entry for it to unlink or replace in the first place:

* POSIX: ``<tempdir>/hermes-state-locks-<uid>``, created with mode ``0700``
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

The sidecar's *name* inside that root is a stable hash of
``canonical_db_key()`` (which resolves symlinks and, where the underlying
filesystem actually folds case and/or Unicode normalization, folds those
too — see the note on ``canonical_db_key`` below), not the raw ``db_path``,
so ``state.db`` and any alias pointing at it — a symlink, a different
relative spelling, a differently-cased spelling on a case-insensitive
volume, or a differently Unicode-normalized (NFC vs NFD) spelling on a
normalization-insensitive volume — share one lock file instead of
splitting admission across two.

The old symlink-swap (``O_NOFOLLOW``) and hardlink-swap (``fstat``/``lstat``
identity) defenses are kept as defense-in-depth: they now protect against
same-account accidents (a rotation script or bug recreating the lock file
while a hold is live), not against a hostile co-tenant, since a co-tenant
can no longer reach the directory at all. They are no longer the primary
defense — the private lock root's directory permissions are. They are also
not a defense against a hostile *same-account* actor — see the threat model
below for why that is a different, unclosable case.

This closes the case Nemo flagged for a co-tenant of a *different* OS
account: it cannot unlink or hardlink anything inside a ``0700`` directory
it does not own, so "unlink + drop in an indistinguishable fresh regular
file" has no directory entry for it to act on.

Threat model — what this module protects and what it explicitly does not:

* **In scope, fully handled:** cooperative Hermes processes of the same OS
  account racing each other (Gateway, Dashboard, ACP, a CLI invocation) —
  handled by ``flock`` plus the admission bookkeeping below — and a
  *different* OS account's co-tenant that can write into the database's own
  directory but not into this account's private lock root — handled by the
  0700 ownership check on that root.
* **Explicitly out of scope: an adversarial process running as the *same*
  OS account.** That account owns every filesystem object this module
  touches — the private lock root, the sidecar, the database file itself —
  so it can delete and recreate any of them at will, including reproducing
  a sidecar this module would have created itself, with no distinguishing
  metadata left behind. No filesystem-based anchor (inode identity, link
  count, ownership, permission mode) can tell a same-account attacker's
  replacement apart from a legitimate one, because the attacker holds
  exactly the same rights over that location that this module does. This
  module does not claim, and must not be read as claiming, resistance to
  that actor. Where that actor is a real concern, the fix is process
  isolation — a dedicated service account per tenant — not a cleverer
  check in this file; no check in this file can draw that boundary.
* ``_check_sidecar_continuity`` (below) narrows the *accidental* slice of
  that same-account case — a cleanup script, a rotation bug, a
  ``tmpfiles.d``-style sweep of ``/tmp`` recreating the sidecar while this
  process is still running — by remembering, in this process's own memory,
  the identity it last trusted for a given database, and refusing admission
  the moment a later acquire in this same process disagrees with that
  memory. It is deliberately not a security boundary: a same-account
  attacker who never lets this process observe a consistent baseline (a
  swap staged before this process's first acquire, or one that waits for a
  restart, which wipes the in-memory baseline) is not caught by it — that
  residual gap is the same one the paragraph above describes, not a new one
  this check introduces.

The lock is **not** a substitute for SQLite's own locking.  Callers still
``BEGIN IMMEDIATE`` and still retry on ``SQLITE_BUSY`` from holders that do
not go through this module (``sqlite3`` CLI, mixed-version processes).
Admission is released before those retries so we never hold the token while
waiting on an ungated writer.

Per-thread re-entrant: a nested write on the same path (reconnect during a
failed write, a helper that writes again) must not block on its own flock.
Other threads and other processes wait.

If the lock file cannot be opened (private root verified, but e.g. exhausted
fds) the acquire degrades to admitted — the in-process ``SessionDB._lock``
plus SQLite busy handling remain, which is what shipped before this module
existed.

A caveat this module cannot enforce: every Hermes process of the same OS
account must see the same ``tempfile.gettempdir()`` (i.e. a consistent
``TMPDIR``/``TEMP``/``TMP``) for the private root to be the *same* root
across processes. That is a deployment consistency requirement, not
something checkable at call time — the same category as requiring
``HERMES_HOME`` to be propagated consistently to subprocess spawners
elsewhere in this codebase.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import secrets
import stat
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("hermes_state")

_IS_WINDOWS = sys.platform == "win32"
_POLL_S = 0.05
_WRITE_LOCK_SUFFIX = ".write.lock"
_LOCK_ROOT_PREFIX = "hermes-state-locks"
_ROOT_MODE = 0o700
_CASE_PROBE_PREFIX = "HermesCaseProbe-"
_NORM_PROBE_PREFIX = "HermesNormProbe-"
# LATIN SMALL LETTER E WITH ACUTE — has a real canonical decomposition
# ("e" + U+0301 COMBINING ACUTE ACCENT), so appending it gives every probe
# name a genuine NFC-vs-NFD distinction to test, the normalization
# counterpart of _CASE_PROBE_PREFIX needing cased characters to swap.
_NORM_PROBE_NFC_MARKER = "é"
_PROBE_ATTEMPTS = 4

# thread ident -> {canonical key: _Hold}.  Depth is per-thread so a nested
# acquire on the same path does not open a second fd and self-deadlock
# (Linux flock is per open-file-description; two fds of the same path block).
_tls = threading.local()

# canonical key -> (st_dev, st_ino) this *process* (not just this thread)
# last trusted for that key. Deliberately process-wide, not thread-local:
# the gap this closes is a second thread in the same process opening a
# sidecar that was unlinked-and-recreated (no hardlink, so
# _sidecar_identity_is_trustworthy's nlink check sees nothing wrong) while
# a first thread's hold is still live — that only shows up by comparing
# against what this process itself remembers, not against the current
# acquire attempt's own fd. See _check_sidecar_continuity and the module
# docstring's threat model.
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
    this same process previously recorded for the same canonical key — see
    :func:`_check_sidecar_continuity`.

    Once raised for a key, this process never admits under that key again:
    the recorded baseline is intentionally never overwritten with the new,
    disagreeing identity, so a retry cannot quietly converge on a still-live
    same-account swap. Recovering requires a process restart, which starts
    the in-memory baseline over from an unpoisoned bootstrap. That is a
    deliberate fail-closed choice, not an oversight — silently re-trusting
    whatever now has the name is exactly the residual bypass this check
    exists to close. It is not, and is not meant to be, resistance to a
    determined same-account attacker; see the module docstring's threat
    model.
    """


def canonical_db_key(db_path: os.PathLike | str) -> str:
    """Canonicalize a database path so aliases share one lock.

    ``realpath`` collapses symlinks and relative spellings. Case is trickier
    than ``os.path.normcase`` alone can handle: ``normcase`` only folds case
    on Windows (``ntpath``) — on every POSIX platform, including macOS,
    ``posixpath.normcase`` is the identity function, regardless of whether
    the actual filesystem underneath is case-insensitive. macOS's default
    APFS/HFS+ volumes *are* case-insensitive (case-preserving), so
    ``State.db`` and ``state.db`` can be the same on-disk file even though
    ``normcase`` alone would hash them to two different keys and split
    admission across two sidecars — the exact bypass this function exists to
    prevent for symlink aliases.

    Unicode adds two more axes ``normcase`` cannot fold on any platform:
    ``str.lower()`` is not Unicode caseless matching (``'ß'.lower() == 'ß'``,
    unchanged, while ``'ß'.casefold() == 'ss'``), and normalization form
    (NFC vs NFD — e.g. a precomposed "é" vs "e" + a combining accent) is an
    entirely separate distinction ``.lower()``/``.casefold()`` never touch.
    APFS's default mode folds both. See :func:`_fold_alias_spelling` for how
    those two axes are handled without extending the same guess-a-transform
    approach case-folding started with — that approach only ever proves
    convergence for the specific transforms it happens to construct
    (``swapcase()``, NFC-vs-NFD), and real Unicode aliasing is not limited to
    those (a compatibility ligature like "ﬁ" case-folds to "fi" but is
    neither a case-swap nor a canonical decomposition of it).

    There is no static "is this platform case/normalization-insensitive"
    answer (a volume can be formatted case-sensitive on macOS, and a
    case-sensitive network share can be mounted on any OS), so folding is
    decided per resolved path rather than by switching on ``sys.platform``.
    """
    text = str(db_path)
    try:
        real = os.path.realpath(text)
    except OSError:
        return os.path.normcase(text)
    if _IS_WINDOWS:
        # ntpath.normcase already folds case; Unicode normalization is not
        # folded by NTFS, so there is nothing else to do here.
        return os.path.normcase(real)
    return os.path.normcase(_fold_alias_spelling(real))


def _fold_alias_spelling(real_path: str) -> str:
    """POSIX alias folding for an already symlink-resolved *real_path*.

    For a target that already exists, this prefers the resolved object's own
    stable identity — ``(st_dev, st_ino)`` — over guessing which Unicode
    transform an alias might use. It builds the single, fully-folded
    candidate spelling (``unicodedata.normalize("NFC", ...)`` then
    ``str.casefold()``) and asks the filesystem itself, via ``os.stat``,
    whether that candidate names the same object as *real_path*. That is one
    comprehensive, non-guessing check: unlike probing individual
    hand-picked transforms (a case swap, or NFC vs NFD), it also converges
    aliases those specific probes never construct — a compatibility ligature
    that case-folds to plain letters, Turkish dotted/dotless I, German
    ß-vs-ss, anything Unicode's real case/normalization tables cover that
    this module's synthetic probes do not enumerate.

    Deliberately NOT ``f"{st_dev}:{st_ino}"`` as the returned key: the
    schema-init race this module exists to close (two sibling processes
    racing to *create* ``state.db``) happens precisely during the window
    where the file transitions from not-existing to existing. If the
    returned key's *format* changed at that transition (text before, raw
    identity after), a process that computed the pre-existence key and is
    still holding its sidecar could be joined by a sibling that computes the
    identity-based key moments later — once the creator's ``sqlite3.connect``
    has touched the file into existence but before schema init finishes and
    the pre-existence key's sidecar is released — landing on a *different*
    sidecar and getting admitted concurrently. That is the exact bug this
    module exists to prevent, reintroduced by the key format itself. Staying
    string-valued and derived only from *real_path* sidesteps it: the
    fold-or-not decision this function makes is answering the same
    time-invariant question (does an alias of this spelling resolve to the
    same object) regardless of whether it is answered via direct identity
    comparison (object exists) or via :func:`_fold_by_probed_capability`
    (object does not exist yet, so the containing directory's own folding
    behavior is probed instead) — both branches necessarily agree for the
    same location, because neither is testing something that changes from
    one moment to the next, only whether *this* filesystem folds at all. See
    ``TestUnicodeAliasConvergesViaStableIdentity`` and
    ``TestNewDatabaseNormalizationInsensitiveAliasConverges`` in the test
    suite for both sides of that invariant pinned down directly.

    A fully-folded candidate identical to *real_path* (the common case: a
    plain ASCII name with no cased or decomposable characters) short-circuits
    before touching the filesystem at all — folding a no-op string can never
    change the answer, existing or not.
    """
    candidate = unicodedata.normalize("NFC", real_path).casefold()
    if candidate == real_path:
        return real_path
    try:
        real_stat = os.stat(real_path)
    except OSError:
        return _fold_by_probed_capability(real_path)
    try:
        candidate_stat = os.stat(candidate)
    except OSError:
        return real_path
    if real_stat.st_ino == 0 or candidate_stat.st_ino == 0:
        # Some filesystems (certain FUSE/virtual mounts) never populate a
        # usable inode; "stat succeeded" there carries no identity
        # guarantee, so fall back to the probe-based decision rather than
        # trusting a zero into a false merge.
        return _fold_by_probed_capability(real_path)
    if (real_stat.st_dev, real_stat.st_ino) == (
        candidate_stat.st_dev,
        candidate_stat.st_ino,
    ):
        return candidate
    return real_path


def _fold_by_probed_capability(real_path: str) -> str:
    """Fallback folding for a *real_path* whose identity could not be used —
    most commonly because it does not exist yet (the schema-init race: two
    sibling processes racing to create a fresh database through differently
    -spelled paths before either has). There is no object to stat, so this
    probes the containing directory's own folding behavior instead, on both
    axes, and only folds what was actually proven — never assumed from
    ``sys.platform`` — exactly as :func:`_is_case_insensitive_fs` already did
    for case; :func:`_is_normalization_insensitive_fs` extends the same
    approach to Unicode normalization (NFC vs NFD).

    Both probes run against the untouched *real_path*, not against a
    partially-folded intermediate, so neither probe's own behavior depends
    on whether the other axis already applied — each answers a single,
    independent question about the directory.
    """
    folded = real_path
    if _is_normalization_insensitive_fs(real_path):
        folded = unicodedata.normalize("NFC", folded)
    if _is_case_insensitive_fs(real_path):
        folded = folded.casefold()
    return folded


def _is_case_insensitive_fs(real_path: str) -> bool:
    """Probe whether *real_path*'s filesystem folds case on lookup.

    ``os.path.normcase`` cannot answer this on POSIX (see
    :func:`canonical_db_key`), so this stats the same directory entry twice:
    once by its real spelling, once by a case-swapped spelling of just the
    basename. If both stats land on the same ``(st_dev, st_ino)``, the
    filesystem folded the case difference away on lookup — case-insensitive.
    A mismatch, a missing swapped entry, or any non-``ENOENT`` ``OSError``
    means "cannot prove case-insensitive", so callers keep the case as-is
    rather than fold it — failing toward the existing (already correct)
    symlink-alias behavior, never toward silently merging two genuinely
    distinct files.

    ``swapcase`` rather than ``upper``/``lower`` alone so a basename that
    happens to already be all-lowercase (or all-uppercase) still produces a
    differently-spelled candidate to probe with. A basename with no cased
    characters at all (e.g. ``"12345.db"``) has nothing to swap — folding
    case would be a no-op on it anyway, so returning ``False`` there costs
    nothing.

    ``real_path`` not existing yet (``FileNotFoundError``) is *not* treated
    as "cannot prove" here — a brand-new ``state.db`` is exactly the case
    that matters most: two sibling processes racing schema init through
    differently-cased spellings of a database that neither has created yet.
    There is no directory entry to stat two ways in that case, so this falls
    back to :func:`_probe_directory_case_folds`, which answers the same
    question — does this directory fold case on lookup? — against the
    parent directory instead, which does exist. Any other ``OSError``
    (permission denied, a symlink loop, ...) still degrades to ``False``:
    those are not "not created yet", and probing the parent would not
    resolve them either.
    """
    directory, name = os.path.split(real_path)
    swapped = name.swapcase()
    if swapped == name:
        return False
    try:
        original_stat = os.stat(real_path)
    except FileNotFoundError:
        return _probe_directory_case_folds(directory or ".")
    except OSError:
        return False
    try:
        swapped_stat = os.stat(os.path.join(directory, swapped))
    except OSError:
        return False
    return (original_stat.st_dev, original_stat.st_ino) == (
        swapped_stat.st_dev,
        swapped_stat.st_ino,
    )


def _probe_directory_case_folds(directory: str) -> bool:
    """Probe whether *directory* (which must already exist) folds case.

    Used when the target database file itself does not exist yet, so there
    is no real directory entry for :func:`_is_case_insensitive_fs` to stat
    two ways. Creates a throwaway file with a randomized, collision-resistant
    name and checks whether a case-swapped spelling of that same name
    resolves to the identical ``(st_dev, st_ino)`` — the same test
    :func:`_is_case_insensitive_fs` runs, just against a probe entry this
    function controls instead of the caller's real (absent) target.

    Safety properties, each load-bearing:

    * **Randomized name, bounded retries** — ``secrets.token_hex`` makes the
      probe name unpredictable, so two Hermes processes (or two calls racing
      in the same process) never contend over the *same* probe entry the way
      a fixed name would. ``O_EXCL`` still guards the astronomically
      unlikely collision by raising instead of clobbering a pre-existing
      file; that retries with a fresh name up to ``_CASE_PROBE_ATTEMPTS``
      times before giving up and returning ``False`` (fail toward "cannot
      prove", never toward crashing or hanging the caller).
    * **``O_CREAT | O_EXCL`` (+ ``O_NOFOLLOW`` on POSIX)** — the probe file
      is never opened if something already exists under that exact name, so
      this can never overwrite or follow a symlink onto a co-tenant's file.
    * **Identity from the open fd, not a second path lookup** — ``fstat`` on
      the fd this call itself just created is authoritative regardless of
      what happens to the directory afterward; there is no re-``stat``-by-
      path of the original name that a TOCTOU swap could race.
    * **The swapped name is only ever ``lstat``-ed, never opened or
      unlinked** — on a case-sensitive filesystem that name was never
      created by this probe and, in the rare case it happens to already
      exist for unrelated reasons, this must not read through it, create it,
      or delete it. A stat mismatch (or the name being altogether absent) is
      treated as "cannot prove case-insensitive", same as any other
      ambiguous result.
    * **Always cleans up** — the probe file is unlinked in a ``finally``
      regardless of which branch returns or raises. Only the exact name this
      call created is ever removed; on a genuinely case-insensitive
      directory that single directory entry *is* the swapped name too, so
      nothing is left behind either way.
    """
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    if not _IS_WINDOWS:
        flags |= getattr(os, "O_NOFOLLOW", 0)

    for _ in range(_PROBE_ATTEMPTS):
        name = f"{_CASE_PROBE_PREFIX}{secrets.token_hex(16)}"
        swapped = name.swapcase()
        probe_path = os.path.join(directory, name)

        try:
            fd = os.open(probe_path, flags, 0o600)
        except FileExistsError:
            continue
        except OSError:
            return False

        try:
            try:
                probe_stat = os.fstat(fd)
            finally:
                os.close(fd)
            try:
                swapped_stat = os.lstat(os.path.join(directory, swapped))
            except OSError:
                return False
            return (probe_stat.st_dev, probe_stat.st_ino) == (
                swapped_stat.st_dev,
                swapped_stat.st_ino,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(probe_path)

    return False


def _is_normalization_insensitive_fs(real_path: str) -> bool:
    """Probe whether *real_path*'s filesystem folds Unicode normalization
    forms on lookup — the normalization counterpart of
    :func:`_is_case_insensitive_fs`.

    Most POSIX filesystems (ext4 included) are normalization-*sensitive*: a
    name stored with a precomposed character (NFC, e.g. U+00E9 "é") and the
    same name spelled with a combining accent (NFD, "e" + U+0301) are two
    different byte sequences and, on such a filesystem, two different
    files. HFS+/APFS's default case-insensitive, case-preserving mode
    additionally folds normalization on lookup, so a database opened
    through an NFC-spelled path and an NFD-spelled alias of the identical
    visible name can be the very same on-disk file there — the same class
    of problem :func:`_is_case_insensitive_fs` exists for, on a different
    Unicode axis, and just as unprovable from ``sys.platform`` alone (a
    volume can be reformatted case/normalization-sensitive on any OS).

    A basename with nothing decomposable at all (no precomposed/accented
    characters) short-circuits to ``False`` — folding would be a no-op on
    it regardless of what the filesystem does, mirroring
    :func:`_is_case_insensitive_fs`'s no-cased-characters short circuit.

    *real_path* not existing yet is handled the same way
    :func:`_is_case_insensitive_fs` handles it: there is no directory entry
    to stat two ways, so this falls back to
    :func:`_probe_directory_normalization_folds` against the parent
    directory. Any other ``OSError`` degrades to ``False`` — cannot prove,
    never merge.
    """
    directory, name = os.path.split(real_path)
    nfc_name = unicodedata.normalize("NFC", name)
    nfd_name = unicodedata.normalize("NFD", name)
    if nfc_name == nfd_name:
        return False
    try:
        nfc_stat = os.stat(os.path.join(directory, nfc_name))
    except FileNotFoundError:
        return _probe_directory_normalization_folds(directory or ".")
    except OSError:
        return False
    try:
        nfd_stat = os.stat(os.path.join(directory, nfd_name))
    except OSError:
        return False
    return (nfc_stat.st_dev, nfc_stat.st_ino) == (nfd_stat.st_dev, nfd_stat.st_ino)


def _probe_directory_normalization_folds(directory: str) -> bool:
    """Probe whether *directory* (which must already exist) folds Unicode
    normalization forms — the normalization counterpart of
    :func:`_probe_directory_case_folds`, used for the same reason: the
    target database does not exist yet, so there is no real directory entry
    for :func:`_is_normalization_insensitive_fs` to stat two ways.

    Mirrors :func:`_probe_directory_case_folds`'s safety properties exactly
    — randomized name (``secrets.token_hex``) with bounded retries,
    ``O_CREAT | O_EXCL`` (+ ``O_NOFOLLOW`` on POSIX) so it never overwrites
    or follows a symlink onto a co-tenant's file, identity taken from the
    fd this call itself created rather than a second path lookup, the
    alternate spelling only ever ``lstat``-ed (never opened or unlinked),
    and cleanup of the exact created name guaranteed in a ``finally`` — the
    only difference is which two spellings are compared: the NFC- and
    NFD-normalized forms of one probe name (which differ because the name
    embeds a character with a real canonical decomposition), not an
    original/case-swapped pair.
    """
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    if not _IS_WINDOWS:
        flags |= getattr(os, "O_NOFOLLOW", 0)

    for _ in range(_PROBE_ATTEMPTS):
        core = f"{_NORM_PROBE_PREFIX}{secrets.token_hex(16)}{_NORM_PROBE_NFC_MARKER}"
        nfc_name = unicodedata.normalize("NFC", core)
        nfd_name = unicodedata.normalize("NFD", core)
        probe_path = os.path.join(directory, nfc_name)

        try:
            fd = os.open(probe_path, flags, 0o600)
        except FileExistsError:
            continue
        except OSError:
            return False

        try:
            try:
                probe_stat = os.fstat(fd)
            finally:
                os.close(fd)
            try:
                alt_stat = os.lstat(os.path.join(directory, nfd_name))
            except OSError:
                return False
            return (probe_stat.st_dev, probe_stat.st_ino) == (
                alt_stat.st_dev,
                alt_stat.st_ino,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(probe_path)

    return False


def _posix_lock_root() -> Path:
    try:
        uid = os.getuid()
    except AttributeError:  # pragma: no cover - no POSIX getuid, shouldn't happen
        uid = "unknown"
    return Path(tempfile.gettempdir()) / f"{_LOCK_ROOT_PREFIX}-{uid}"


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
    with an unusual ``TMPDIR``/profile layout can relocate it; the override
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


def write_lock_path(db_path: os.PathLike | str) -> Path:
    """Return the sidecar lock path for *db_path* inside the private root.

    Named by a stable hash of :func:`canonical_db_key` (symlink-resolved,
    case-folded), not colocated with the database file. Two consequences:

    * Aliases converge — ``state.db`` and a symlink alias pointing at it
      (``alias.db -> state.db``) hash to the same key and share one lock
      file, so two processes can never hold admission on the same
      underlying SQLite file through different names.
    * The lock never lives in a directory a co-tenant of the database's
      directory can write to — see the module docstring for why that
      colocation was the residual bypass this module used to have.
    """
    key = canonical_db_key(db_path)
    digest = hashlib.sha256(key.encode("utf-8", "surrogateescape")).hexdigest()
    return _lock_root() / f"{digest}{_WRITE_LOCK_SUFFIX}"


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


def _holds() -> dict:
    holds = getattr(_tls, "holds", None)
    if holds is None:
        holds = {}
        _tls.holds = holds
    return holds


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


def _check_sidecar_continuity(key: str, identity: tuple[int, int]) -> None:
    """Compare *identity* against what this process itself last trusted for
    *key*, recording a first-time baseline rather than rejecting it.

    ``_sidecar_identity_is_trustworthy`` only compares a fd against the
    *current* on-disk name at the instant of one acquire attempt — a plain
    ``unlink`` followed by a brand-new regular file (no hardlink, so
    ``st_nlink`` stays ``1``) is entirely self-consistent from that single
    attempt's point of view; there is nothing in that one stat call to
    disagree with. This closes that gap using memory that check does not
    have: what *this process* itself has already trusted for *key*.

    Bootstrap — no prior identity recorded for *key* in this process — is
    not a failure. There is nothing yet to compare against, and refusing
    here would break the very first acquire on a fresh database, which is
    exactly the case that must keep working. ``setdefault`` records this
    identity as the trusted baseline atomically with the read, so two
    threads racing a brand-new key's first acquire cannot each observe "no
    baseline" and both proceed on two different answers.

    Once a baseline is recorded, a *different* identity turning up on a
    later acquire is not inferred, it is witnessed: this process previously
    held that sidecar open, and something removed and replaced it since.
    The mismatch is reported by raising rather than by updating the
    baseline to the new value, so a same-account swap that is still live
    (a second thread's hold not yet released) cannot be quietly re-trusted
    on the very next attempt — see ``_SidecarIdentitySwapped``.
    """
    with _sidecar_identity_lock:
        previous = _known_sidecar_identity.setdefault(key, identity)
    if previous != identity:
        raise _SidecarIdentitySwapped(
            f"sidecar identity for {key!r} changed from {previous} to "
            f"{identity} since this process last trusted it"
        )


def _reset_after_fork() -> None:
    """Drop inherited holds in a forked child without unlocking the parent.

    ``fork`` duplicates fds onto the same open-file-description, so
    ``LOCK_UN`` here would release the parent's admission.  Closing the
    inherited fd is enough: the parent still holds the description.
    """
    holds = getattr(_tls, "holds", None) or {}
    _tls.holds = {}
    for hold in holds.values():
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
    """Admit this thread as the Hermes writer for *db_path*.

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
    key = canonical_db_key(db_path)
    holds = _holds()
    existing: Optional[_Hold] = holds.get(key)
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

    lock_path = write_lock_path(db_path)
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
                _check_sidecar_continuity(key, identity)
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

        holds[key] = _Hold(handle)
        released = False
        try:
            yield True
        finally:
            hold = holds.get(key)
            if hold is not None:
                hold.depth -= 1
                if hold.depth <= 0:
                    holds.pop(key, None)
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
