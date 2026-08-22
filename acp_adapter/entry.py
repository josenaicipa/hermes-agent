"""CLI entry point for the hermes-agent ACP adapter.

Loads environment variables from ``~/.hermes/.env``, configures logging
to write to stderr (so stdout is reserved for ACP JSON-RPC transport),
and starts the ACP agent server.

Usage::

    python -m acp_adapter.entry
    # or
    hermes acp
    # or
    hermes-acp
"""

# IMPORTANT: hermes_bootstrap must be the very first import — UTF-8 stdio
# on Windows.  No-op on POSIX.  See hermes_bootstrap.py for full rationale.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Graceful fallback when hermes_bootstrap isn't registered in the venv
    # yet — happens during partial ``hermes update`` where git-reset landed
    # new code but ``uv pip install -e .`` didn't finish.  Missing bootstrap
    # means UTF-8 stdio setup is skipped on Windows; POSIX is unaffected.
    pass
else:
    # Stop a ``utils/``/``proxy/``/``ui/`` package in the launch directory from
    # shadowing Hermes's own modules — ``hermes acp`` can be started from any
    # cwd, including a project that has same-named packages on its path.
    hermes_bootstrap.harden_import_path()

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from hermes_constants import get_hermes_home


# Methods clients send as periodic liveness probes. They are not part of the
# ACP schema, so the acp router correctly returns JSON-RPC -32601 to the
# caller — but the supervisor task that dispatches the request then surfaces
# the raised RequestError via ``logging.exception("Background task failed")``,
# which dumps a traceback to stderr every probe interval. Clients like
# acp-bridge already treat the -32601 response as "agent alive", so the
# traceback is pure noise. We keep the protocol response intact and only
# silence the stderr noise for this specific benign case.
_BENIGN_PROBE_METHODS = frozenset({"ping", "health", "healthcheck"})


class _BenignProbeMethodFilter(logging.Filter):
    """Suppress acp 'Background task failed' tracebacks caused by unknown
    liveness-probe methods (e.g. ``ping``) while leaving every other
    background-task error — including method_not_found for any non-probe
    method — visible in stderr.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage() != "Background task failed":
            return True
        exc_info = record.exc_info
        if not exc_info:
            return True
        exc = exc_info[1]
        # Imported lazily so this module stays importable when the optional
        # ``agent-client-protocol`` dependency is not installed.
        try:
            from acp.exceptions import RequestError
        except ImportError:
            return True
        if not isinstance(exc, RequestError):
            return True
        if getattr(exc, "code", None) != -32601:
            return True
        data = getattr(exc, "data", None)
        method = data.get("method") if isinstance(data, dict) else None
        return method not in _BENIGN_PROBE_METHODS


def _setup_logging() -> None:
    """Route all logging to stderr so stdout stays clean for ACP stdio."""
    from agent.redact import RedactingFormatter

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        RedactingFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(_BenignProbeMethodFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _load_env() -> None:
    """Load .env from HERMES_HOME (default ``~/.hermes``)."""
    from hermes_cli.env_loader import load_hermes_dotenv

    hermes_home = get_hermes_home()
    loaded = load_hermes_dotenv(hermes_home=hermes_home)
    if loaded:
        for env_file in loaded:
            logging.getLogger(__name__).info("Loaded env from %s", env_file)
    else:
        logging.getLogger(__name__).info(
            "No .env found at %s, using system env", hermes_home / ".env"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hermes-acp",
        description="Run Hermes Agent as an ACP stdio server.",
    )
    parser.add_argument("--version", action="store_true", help="Print Hermes version and exit")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify ACP dependencies and adapter imports, then exit",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run interactive Hermes provider/model setup for ACP terminal auth",
    )
    parser.add_argument(
        "--setup-browser",
        action="store_true",
        help="Install agent-browser + Playwright Chromium into ~/.hermes/node/ "
             "for browser tool support. Idempotent.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        dest="assume_yes",
        help="Accept all prompts (currently used by --setup-browser to skip the "
             "~400 MB Chromium download confirmation).",
    )
    return parser.parse_args(argv)


def _print_version() -> None:
    from hermes_cli import __version__ as hermes_version

    print(hermes_version)


def _run_check() -> None:
    import acp  # noqa: F401
    from acp_adapter.server import HermesACPAgent  # noqa: F401

    print("Hermes ACP check OK")


def _run_setup() -> None:
    from hermes_cli.main import main as hermes_main

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0] if old_argv else "hermes", "model"]
        hermes_main()
    finally:
        sys.argv = old_argv

    # Offer browser-tools install as a follow-up. The terminal auth method
    # is the one supported first-run UX for registry installs, so this is
    # the natural moment to ask. Skip silently if stdin isn't a TTY (the
    # answer can't be collected anyway).
    if not sys.stdin.isatty():
        return
    try:
        reply = input(
            "\nInstall browser tools? Downloads agent-browser (npm) and "
            "optionally Playwright Chromium (~400 MB). [y/N] "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if reply in {"y", "yes"}:
        _run_setup_browser(assume_yes=False)


def _run_setup_browser(assume_yes: bool = False) -> int:
    """Bootstrap agent-browser + Chromium.

    Routes through dep_ensure -> install.{sh,ps1} --ensure, sharing code
    with the runtime lazy installer.

    Returns 0 on success, 1 on failure.
    """
    from hermes_cli.dep_ensure import ensure_dependency

    try:
        node_ok = ensure_dependency("node", interactive=not assume_yes)
        if not node_ok:
            print("Node.js installation failed — cannot proceed with browser tools.",
                  file=sys.stderr)
            return 1

        browser_ok = ensure_dependency("browser", interactive=not assume_yes)
        if not browser_ok:
            print("Browser tools installation failed.", file=sys.stderr)
            return 1

        return 0
    except OSError as exc:
        print(f"Browser bootstrap failed: {exc}", file=sys.stderr)
        return 1


# Techos de espera larga, SOLO en el proceso ACP (Jose 2026-08-21).
#
# Por qué existe. En una sesión ACP el agente NO puede ser despertado: el aviso
# de un proceso de fondo se encola en ``process_registry.completion_queue`` y
# solo lo drenan el gateway, el CLI y el TUI. Y no se puede suplir empujando
# ``session/update`` fuera de turno: ACP Pro deduce "estoy trabajando" del
# último evento del log y, en ese estado, el Enter del operador NO se envía
# (se queda en la cola del webview esperando un fin de turno que nunca llega).
#
# Así que en ACP la única forma honesta de no perder una misión es mantener
# VIVO el turno que la despachó: el agente espera al vigía en primer plano
# (``cursor-wait-run.sh``, un ``sleep`` de shell: cero tokens mientras la
# misión corre) y reporta en ese mismo turno. Para eso hay que levantar dos
# techos que están calibrados para turnos cortos:
#
#   - ``TERMINAL_MAX_FOREGROUND_TIMEOUT`` (600 s): el terminal rechaza un
#     ``timeout`` mayor en primer plano y empuja a segundo plano — que es
#     justo lo que no sirve aquí. Se lee al IMPORTAR ``tools.terminal_tool``,
#     por eso esto corre antes de importar el servidor.
#   - ``HERMES_CONCURRENT_TOOL_TIMEOUT_S`` (420 s): plazo del ejecutor para
#     UNA llamada de herramienta. Sin esto la espera vuelve como
#     ``tool_timeout`` a los 7 minutos aunque la misión siga viva.
#
# Ambos por encima del techo real de una misión (``CURSOR_TIMEOUT_MS`` = 2 h y
# el plazo por defecto del vigía, 7500 s), con margen.
#
# Qué se pierde a cambio: en ACP el plazo del ejecutor deja de ser una red
# corta para CUALQUIER herramienta. Es aceptable justamente aquí y no en el
# gateway: (a) el proceso ACP es el del editor, con un humano delante y un
# botón Stop que SÍ interrumpe la espera (el bucle de espera sondea la
# interrupción y mata el grupo de procesos), y (b) la primera red sigue siendo
# el ``timeout`` que la propia llamada declara (180 s por defecto), no este
# techo. El gateway corre en otro proceso y no se ve afectado.
#
# ``setdefault``: si Jose fija estas variables en la config del editor o en un
# ``.env``, manda su valor.
_ACP_FOREGROUND_CEILING_S = "7800"
_ACP_TOOL_CALL_CEILING_S = "8100"


def _apply_acp_long_wait_ceilings() -> None:
    """Permite esperar una misión completa dentro de un turno ACP vivo."""
    os.environ.setdefault("TERMINAL_MAX_FOREGROUND_TIMEOUT", _ACP_FOREGROUND_CEILING_S)
    os.environ.setdefault("HERMES_CONCURRENT_TOOL_TIMEOUT_S", _ACP_TOOL_CALL_CEILING_S)


def main(argv: list[str] | None = None) -> None:
    """Entry point: load env, configure logging, run the ACP agent."""
    args = _parse_args(argv)
    if args.version:
        _print_version()
        return
    if args.check:
        _run_check()
        return
    if args.setup:
        _run_setup()
        return
    if args.setup_browser:
        rc = _run_setup_browser(assume_yes=args.assume_yes)
        if rc != 0:
            sys.exit(rc)
        return

    _setup_logging()
    _load_env()

    _apply_acp_long_wait_ceilings()

    logger = logging.getLogger(__name__)
    logger.info("Starting hermes-agent ACP adapter")

    # Ensure the project root is on sys.path so ``from run_agent import AIAgent`` works
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import acp
    from .server import HermesACPAgent

    # MCP tool discovery from config.yaml — fire-and-forget in a
    # background daemon thread so the ACP server becomes responsive
    # immediately while MCP servers connect.  Previously this blocked
    # asyncio.run() for 2-5 s.  (ACP also registers per-session MCP
    # servers dynamically via asyncio.to_thread inside the event loop;
    # that path is unaffected.)  Moved from model_tools.py module scope
    # to avoid freezing the gateway's loop on lazy import (#16856).
    # Metadata-only hosts can opt out of unrelated global MCP startup.
    if os.environ.get("HERMES_ACP_SKIP_CONFIGURED_MCP", "").strip() != "1":
        try:
            from hermes_cli.mcp_startup import start_background_mcp_discovery

            start_background_mcp_discovery(
                logger=logger,
                thread_name="acp-mcp-discovery",
            )
        except Exception:
            logger.debug("MCP tool discovery failed at ACP startup", exc_info=True)

    agent = HermesACPAgent()
    try:
        asyncio.run(acp.run_agent(agent, use_unstable_protocol=True))
    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt)")
    except Exception:
        logger.exception("ACP agent crashed")
        sys.exit(1)


if __name__ == "__main__":
    main()
