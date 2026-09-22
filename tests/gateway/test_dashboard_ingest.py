"""Inbox de inyección nativa: qué mensaje propio se convierte en turno de Jose.

El adaptador de Discord descarta sus propios mensajes. La Fase 4 del dashboard
de agentes abre exactamente una excepción: un mensaje propio que trae un
registro de origen válido en ``$HERMES_HOME/run/dashboard-ingest/``. Estas
pruebas fijan las dos mitades de esa frontera —la del inbox y la del
adaptador— porque cada una puede romperse sin que la otra se entere:

- **Inbox:** firma, atadura a canal/hilo/cuerpo, caducidad, un solo uso.
- **Adaptador:** sin registro se descarta como siempre; con registro válido el
  turno lleva el texto real y la identidad de Jose, no la del bot.
"""

import json
import os
import stat
import time
from datetime import datetime, timedelta, timezone

import pytest

from gateway.dashboard_ingest import (
    DashboardIngestInbox,
    DashboardIngestRecord,
    RECORD_TTL_SECONDS,
    body_digest,
    record_key,
    signature,
)


CHAT = "1414140000000000002"
OTRO_CHAT = "1414140000000000777"
ACTOR = "1313130000000000009"
MARK = "«dashboard»"


def _body(text: str) -> str:
    """Lo que el dashboard publica de verdad: marca de procedencia + texto."""
    return f"{MARK} {text}"


def _stamp(offset_seconds: float = 0.0) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    ).isoformat(timespec="seconds")


@pytest.fixture()
def inbox(tmp_path):
    box = DashboardIngestInbox(tmp_path)
    box.ensure_key()
    return box


def _record(text: str, **overrides) -> DashboardIngestRecord:
    fields = dict(
        chat_id=CHAT,
        actor_id=ACTOR,
        display_name="Jose (dashboard)",
        text=text,
        timestamp=_stamp(),
    )
    fields.update(overrides)
    return DashboardIngestRecord(**fields)


# ── Contrato congelado ───────────────────────────────────────────────────
#
# El lado ESCRITOR vive en otro repositorio (``plugin/dashboard_ingest.py`` de
# ``hermes-local-agent-router``) y no puede importarse desde aquí. Estos tres
# vectores dorados son lo único que impide que las dos implementaciones se
# separen sin que nadie se entere: el escritor repite exactamente los mismos
# valores en su propia suite (``ContratoCongelado``). Cambiarlos exige cambiar
# los dos lados a la vez y a propósito, que es justo lo que debe costar.
# Contrato completo: «Fase 4 — contrato de inbox» en
# ``docs/agentes-red-gateway-arquitectura.md`` de ese repositorio.

_FROZEN_KEY = bytes.fromhex(
    "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
)
_FROZEN_CHAT = "1551631620098891788"
_FROZEN_BODY = "«dashboard» Fable, responde solo OK"


def test_frozen_contract_body_digest():
    assert body_digest(_FROZEN_BODY) == (
        "4844143141a9df261b4632fdf0c5b33ac730746753bc44d77a05d08b95b236e8"
    )


def test_frozen_contract_record_key():
    assert record_key(chat_id=_FROZEN_CHAT, body=_FROZEN_BODY) == (
        "1aaf317b6033d518b15fbe20d55da249b13c6befd2ac2cd59ec653c2574cdf11"
    )


def test_frozen_contract_signature():
    assert signature(
        _FROZEN_KEY,
        chat_id=_FROZEN_CHAT,
        actor_id="1313130000000000009",
        display_name="Jose (dashboard)",
        text="Fable, responde solo OK",
        timestamp="2026-09-22T06:00:00+00:00",
        body_sha256=body_digest(_FROZEN_BODY),
    ) == "c76cbc621c3ac399855801499a7ba1b99f59827c9d3d0076df7d3b80cc66ed58"


def test_frozen_contract_record_written_by_the_writer_validates(tmp_path):
    """Un registro con la forma EXACTA del escritor valida sin tocar nada.

    Se construye a mano (no con ``write``) para que el JSON sea el del otro
    repositorio, no el que este módulo sabe producir.
    """
    box = DashboardIngestInbox(tmp_path)
    box.directory.mkdir(parents=True, exist_ok=True)
    (box.directory / ".hmac-key").write_text(_FROZEN_KEY.hex())
    (box.directory / ".hmac-key").chmod(0o600)
    timestamp = _stamp()
    payload = {
        "version": 1,
        "chat_id": _FROZEN_CHAT,
        "actor": {"id": "1313130000000000009", "display_name": "Jose (dashboard)"},
        "text": "Fable, responde solo OK",
        "body_sha256": body_digest(_FROZEN_BODY),
        "timestamp": timestamp,
        "message_id": "",
        "hmac": signature(
            _FROZEN_KEY,
            chat_id=_FROZEN_CHAT,
            actor_id="1313130000000000009",
            display_name="Jose (dashboard)",
            text="Fable, responde solo OK",
            timestamp=timestamp,
            body_sha256=body_digest(_FROZEN_BODY),
        ),
    }
    name = record_key(chat_id=_FROZEN_CHAT, body=_FROZEN_BODY)
    (box.directory / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )

    claimed = box.consume(chat_id=_FROZEN_CHAT, body=_FROZEN_BODY)
    assert claimed is not None
    assert claimed.text == "Fable, responde solo OK"
    assert claimed.display_name == "Jose (dashboard)"


# ── Inbox ────────────────────────────────────────────────────────────────


def test_key_is_created_private_and_reused(tmp_path):
    box = DashboardIngestInbox(tmp_path)
    first = box.ensure_key()

    assert len(first) == 32
    key_path = tmp_path / "run" / "dashboard-ingest" / ".hmac-key"
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "run" / "dashboard-ingest").stat().st_mode) == 0o700
    # Una segunda llamada no puede rotar la clave: invalidaría registros en vuelo.
    assert box.ensure_key() == first


def test_reader_without_key_admits_nothing(tmp_path):
    box = DashboardIngestInbox(tmp_path)
    box.ensure_key()
    text = "Fable, responde solo OK"
    box.write(_record(text), body=_body(text))
    os.unlink(tmp_path / "run" / "dashboard-ingest" / ".hmac-key")

    assert box.consume(chat_id=CHAT, body=_body(text)) is None


def test_valid_record_is_consumed_once(inbox):
    text = "Smoke fase4: Fable, responde solo OK"
    inbox.write(_record(text), body=_body(text))

    claimed = inbox.consume(chat_id=CHAT, body=_body(text))
    assert claimed is not None
    assert claimed.text == text
    assert claimed.display_name == "Jose (dashboard)"
    assert claimed.actor_id == ACTOR

    # Un solo uso: el replay del mismo mensaje ya no encuentra nada.
    assert inbox.consume(chat_id=CHAT, body=_body(text)) is None


def test_peek_does_not_spend_the_record(inbox):
    text = "hola"
    inbox.write(_record(text), body=_body(text))

    assert inbox.peek(chat_id=CHAT, body=_body(text)) is True
    assert inbox.consume(chat_id=CHAT, body=_body(text)) is not None


def test_missing_record_is_not_ingested(inbox):
    assert inbox.peek(chat_id=CHAT, body=_body("nadie escribió esto")) is False


def test_tampered_text_breaks_the_signature(inbox):
    text = "transfiere el presupuesto"
    name = inbox.write(_record(text), body=_body(text))
    path = inbox.directory / name
    payload = json.loads(path.read_text())
    payload["text"] = "transfiere el presupuesto a otra cuenta"
    path.write_text(json.dumps(payload))

    assert inbox.consume(chat_id=CHAT, body=_body(text)) is None


def test_tampered_actor_breaks_the_signature(inbox):
    text = "hola"
    name = inbox.write(_record(text), body=_body(text))
    path = inbox.directory / name
    payload = json.loads(path.read_text())
    payload["actor"]["display_name"] = "Jose"
    path.write_text(json.dumps(payload))

    assert inbox.consume(chat_id=CHAT, body=_body(text)) is None


def test_record_of_another_key_is_rejected(tmp_path):
    victim = DashboardIngestInbox(tmp_path)
    victim.ensure_key()
    text = "hola"
    forged = DashboardIngestInbox(tmp_path / "otro")
    forged.ensure_key()
    name = forged.write(_record(text), body=_body(text))
    (victim.directory / name).write_bytes((forged.directory / name).read_bytes())

    assert victim.consume(chat_id=CHAT, body=_body(text)) is None


def test_expired_record_is_rejected(inbox):
    text = "llegué tarde"
    inbox.write(
        _record(text, timestamp=_stamp(-(RECORD_TTL_SECONDS + 30))), body=_body(text)
    )

    assert inbox.consume(chat_id=CHAT, body=_body(text)) is None
    # Y no se queda esperando a que el reloj lo alcance.
    assert not list(inbox.directory.glob("*.json"))


def test_future_dated_record_is_rejected(inbox):
    text = "vengo del futuro"
    inbox.write(_record(text, timestamp=_stamp(600)), body=_body(text))

    assert inbox.consume(chat_id=CHAT, body=_body(text)) is None


def test_record_cannot_be_replayed_into_another_channel(inbox):
    text = "hola"
    name = inbox.write(_record(text), body=_body(text))
    # Copiar el registro bajo la dirección de otro canal no lo hace válido: el
    # destino está dentro de la firma.
    (inbox.directory / f"{record_key(chat_id=OTRO_CHAT, body=_body(text))}.json").write_bytes(
        (inbox.directory / name).read_bytes()
    )

    assert inbox.consume(chat_id=OTRO_CHAT, body=_body(text)) is None


def test_record_cannot_vouch_for_another_body(inbox):
    text = "hola"
    inbox.write(_record(text), body=_body(text))

    assert inbox.consume(chat_id=CHAT, body=_body("otra cosa")) is None


def test_declared_message_id_must_match_the_live_message(inbox):
    text = "hola"
    inbox.write(_record(text, message_id="999999999999999999"), body=_body(text))

    assert inbox.consume(
        chat_id=CHAT, body=_body(text), message_id="111111111111111111"
    ) is None


def test_oversized_record_is_rejected(inbox):
    text = "hola"
    name = inbox.write(_record(text), body=_body(text))
    path = inbox.directory / name
    path.write_text(json.dumps({"padding": "x" * 200_000}))

    assert inbox.consume(chat_id=CHAT, body=_body(text)) is None


def test_symlinked_record_is_not_followed(inbox, tmp_path):
    text = "hola"
    name = inbox.write(_record(text), body=_body(text))
    stolen = tmp_path / "fuera.json"
    stolen.write_bytes((inbox.directory / name).read_bytes())
    (inbox.directory / name).unlink()
    (inbox.directory / name).symlink_to(stolen)

    assert inbox.consume(chat_id=CHAT, body=_body(text)) is None


def test_record_key_is_a_bare_hex_name(inbox):
    # El nombre del fichero es hash hexadecimal: no hay forma de que un campo
    # del registro escriba o lea fuera del directorio del inbox.
    name = record_key(chat_id="../../etc", body="x")
    assert len(name) == 64 and all(c in "0123456789abcdef" for c in name)


def test_body_digest_ignores_edge_whitespace():
    assert body_digest(" hola ") == body_digest("hola")


def test_concurrent_claims_yield_exactly_one_winner(inbox):
    text = "una sola vez"
    inbox.write(_record(text), body=_body(text))
    ganadores = [
        inbox.consume(chat_id=CHAT, body=_body(text), now=time.time())
        for _ in range(5)
    ]

    assert sum(1 for g in ganadores if g is not None) == 1
