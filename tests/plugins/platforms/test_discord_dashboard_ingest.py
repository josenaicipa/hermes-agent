"""El adaptador de Discord y el único mensaje propio que sí es un turno.

``_discord_message_admission`` descarta todo lo que el bot escribe: sin eso, cada
respuesta del agente reentraría como pregunta. La Fase 4 del dashboard de
agentes abre una grieta estrecha y verificable: un mensaje propio que trae un
registro de origen firmado en ``$HERMES_HOME/run/dashboard-ingest/`` es un turno
que Jose escribió en el panel, y se entrega como tal.

Lo que estas pruebas fijan:

- sin registro, el descarte de siempre (y no se toca ni el reloj ni el disco);
- con registro válido, el turno lleva el TEXTO real —no el cuerpo publicado con
  su marca de procedencia— y la identidad de Jose, no la del bot;
- caducado o con firma rota, se descarta;
- el registro se gasta: el mismo mensaje no puede entregarse dos veces.
"""

import datetime as dt
from types import SimpleNamespace

import discord
import pytest

import hermes_constants
from gateway.config import Platform, PlatformConfig
from gateway.dashboard_ingest import (
    DashboardIngestInbox,
    DashboardIngestRecord,
    RECORD_TTL_SECONDS,
)
from plugins.platforms.discord.adapter import DiscordAdapter


BOT_ID = 900900900900900900
JOSE_ID = "1313130000000000009"
PARENT_ID = 1414140000000000001
THREAD_ID = 1414140000000000002
MESSAGE_ID = 1515150000000000003
MARK = "«dashboard»"


def _published(text: str) -> str:
    """El cuerpo EXACTO que ``hermes send`` deja en el canal."""
    return f"{MARK} {text}"


def _stamp(offset_seconds: float = 0.0) -> str:
    return (
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=offset_seconds)
    ).isoformat(timespec="seconds")


class _Threads(set):
    def mark(self, thread_id):
        self.add(str(thread_id))


class _Dedup:
    def __init__(self):
        self.seen = set()

    def is_duplicate(self, message_id):
        if message_id in self.seen:
            return True
        self.seen.add(message_id)
        return False

    def contains(self, message_id):
        return message_id in self.seen


@pytest.fixture()
def inbox(tmp_path, monkeypatch):
    home = tmp_path / "profile-home"
    home.mkdir()
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda *a, **k: home)
    box = DashboardIngestInbox(home)
    box.ensure_key()
    return box


@pytest.fixture()
def adapter():
    ad = object.__new__(DiscordAdapter)
    # ``name`` es propiedad de solo lectura: se deriva de la plataforma.
    ad.platform = Platform.DISCORD
    ad.config = PlatformConfig(enabled=True, token="x", extra={})
    ad.gateway_runner = None
    ad._owner_profile = None
    ad._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, display_name="Hermes"))
    ad._threads = _Threads()
    ad._dedup = _Dedup()
    ad._voice_text_channels = {}
    ad._text_batch_delay_seconds = 0
    ad.delivered = []

    async def _handle(event):
        ad.delivered.append(event)

    async def _no_media(_attachments):
        return [], [], None

    ad.handle_message = _handle
    ad._collect_attachment_media = _no_media
    ad._get_allowed_channels = lambda: set()
    ad._get_ignored_channels = lambda: set()
    ad._discord_free_response_channels = lambda: set()
    ad._get_no_thread_channels = lambda: set()
    # Canal con mención obligatoria e hilo NO adoptado: las dos puertas que un
    # mensaje sin @mención tendría que cruzar si fuera un mensaje normal.
    ad._discord_require_mention = lambda: True
    ad._discord_thread_require_mention = lambda: True
    ad._discord_history_backfill = lambda: False
    ad._resolve_channel_skills = lambda *a, **k: None
    ad._resolve_channel_prompt = lambda *a, **k: None
    ad._get_allow_bots = lambda: "none"
    return ad


def _thread_channel():
    channel = object.__new__(discord.Thread)
    channel.id = THREAD_ID
    channel.parent_id = PARENT_ID
    channel.name = "Dahsboard Agentes"
    channel.guild = SimpleNamespace(id=777, name="Hermes Agentes Test")
    return channel


def _self_message(adapter, text: str, *, message_id: int = MESSAGE_ID):
    """Mensaje tal y como vuelve del websocket: autor = el propio bot."""
    return SimpleNamespace(
        id=message_id,
        content=text,
        author=adapter._client.user,
        channel=_thread_channel(),
        guild=SimpleNamespace(id=777, name="Hermes Agentes Test"),
        type=discord.MessageType.default,
        mentions=[],
        attachments=[],
        reference=None,
        created_at=dt.datetime.now(dt.timezone.utc),
        message_snapshots=[],
        thread=None,
    )


def _record(text: str, **overrides) -> DashboardIngestRecord:
    fields = dict(
        # El destino es el HILO: el id que el adaptador saca de
        # ``message.channel.id`` y el mismo que el panel resuelve como destino.
        chat_id=str(THREAD_ID),
        actor_id=JOSE_ID,
        display_name="Jose (dashboard)",
        text=text,
        timestamp=_stamp(),
    )
    fields.update(overrides)
    return DashboardIngestRecord(**fields)


# ── Admisión ─────────────────────────────────────────────────────────────


def test_own_message_without_record_is_discarded(adapter, inbox):
    message = _self_message(adapter, _published("respuesta del agente"))

    assert adapter._discord_message_admission(message, claim=True) == (False, False)


def test_own_message_with_valid_record_is_admitted(adapter, inbox):
    text = "Smoke fase4: Fable, responde solo OK"
    inbox.write(_record(text), body=_published(text))
    message = _self_message(adapter, _published(text))

    assert adapter._discord_message_admission(message, claim=True) == (True, False)


def test_expired_record_does_not_admit(adapter, inbox):
    text = "llegué tarde"
    inbox.write(
        _record(text, timestamp=_stamp(-(RECORD_TTL_SECONDS + 60))),
        body=_published(text),
    )
    message = _self_message(adapter, _published(text))

    assert adapter._discord_message_admission(message, claim=True) == (False, False)


def test_forged_record_does_not_admit(adapter, inbox, tmp_path):
    text = "hola"
    otro = DashboardIngestInbox(tmp_path / "otro-perfil")
    otro.ensure_key()
    name = otro.write(_record(text), body=_published(text))
    (inbox.directory / name).write_bytes((otro.directory / name).read_bytes())
    message = _self_message(adapter, _published(text))

    assert adapter._discord_message_admission(message, claim=True) == (False, False)


def test_admission_does_not_spend_the_record(adapter, inbox):
    text = "hola"
    inbox.write(_record(text), body=_published(text))
    message = _self_message(adapter, _published(text))

    assert adapter._discord_message_admission(message, claim=True)[0] is True
    # El registro sigue ahí para que ``_handle_message`` lo gaste una sola vez.
    assert inbox.peek(chat_id=str(THREAD_ID), body=_published(text)) is True


# ── Construcción del turno ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dashboard_turn_carries_jose_identity_and_real_text(adapter, inbox):
    text = "Fable, responde solo OK"
    inbox.write(_record(text), body=_published(text))
    message = _self_message(adapter, _published(text))

    assert await adapter._handle_message(message) is True

    event = adapter.delivered[-1]
    # El turno lleva lo que Jose escribió, no el cuerpo publicado con su marca.
    assert event.text == text
    assert MARK not in event.text
    assert event.source.user_id == JOSE_ID
    assert event.source.user_name == "Jose (dashboard)"
    assert event.source.is_bot is False
    assert event.source.chat_id == str(THREAD_ID)
    assert event.source.thread_id == str(THREAD_ID)
    assert event.source.parent_chat_id == str(PARENT_ID)
    assert event.message_id == str(MESSAGE_ID)


@pytest.mark.asyncio
async def test_own_message_without_record_never_reaches_a_session(adapter, inbox):
    message = _self_message(adapter, _published("respuesta del agente"))

    assert await adapter._handle_message(message) is False
    assert adapter.delivered == []


@pytest.mark.asyncio
async def test_record_is_spent_so_the_same_message_cannot_replay(adapter, inbox):
    text = "una sola vez"
    inbox.write(_record(text), body=_published(text))
    message = _self_message(adapter, _published(text))

    assert await adapter._handle_message(message) is True
    assert await adapter._handle_message(message) is False
    assert len(adapter.delivered) == 1


@pytest.mark.asyncio
async def test_record_without_actor_id_still_names_the_dashboard(adapter, inbox):
    text = "hola"
    inbox.write(_record(text, actor_id=""), body=_published(text))
    message = _self_message(adapter, _published(text))

    assert await adapter._handle_message(message) is True
    event = adapter.delivered[-1]
    assert event.source.user_name == "Jose (dashboard)"
    # Sin id de Discord conocido queda el del emisor real, nunca uno inventado.
    assert event.source.user_id == str(BOT_ID)


@pytest.mark.asyncio
async def test_a_plain_channel_message_is_ingested_with_the_channel_id(adapter, inbox):
    """Sin hilo, el destino es el canal — y es el MISMO id que escribe el panel.

    El directorio que el gateway publica no siempre distingue un hilo de su
    padre (hay entradas con la forma ``<X>:<X>``), así que el contrato no
    depende del padre: el único id que los dos lados calculan igual es el del
    canal donde el mensaje vive.
    """
    # Un canal cualquiera que no es ni DM ni hilo: es todo lo que el adaptador
    # comprueba. No se instancia ``discord.TextChannel`` porque otro módulo de
    # esta carpeta lo sustituye por un mock y el orden de ejecución decidiría
    # si este test pasa.
    channel = SimpleNamespace(
        id=PARENT_ID,
        name="Dahsboard Agentes",
        guild=SimpleNamespace(id=777, name="Hermes Agentes Test"),
        topic=None,
        parent_id=None,
    )
    # Canal sin auto-hilo: el mensaje se atiende en el canal, como lo haría uno
    # humano en un canal de respuesta libre.
    adapter._get_no_thread_channels = lambda: {str(PARENT_ID)}

    text = "hola canal"
    inbox.write(_record(text, chat_id=str(PARENT_ID)), body=_published(text))
    message = _self_message(adapter, _published(text))
    message.channel = channel

    assert adapter._discord_message_admission(message, claim=True) == (True, False)
    assert await adapter._handle_message(message) is True

    event = adapter.delivered[-1]
    assert event.text == text
    assert event.source.chat_id == str(PARENT_ID)
    assert event.source.thread_id is None
    assert event.source.user_name == "Jose (dashboard)"


@pytest.mark.asyncio
async def test_a_dashboard_turn_is_never_coalesced_into_another_author(adapter, inbox):
    """El lote de texto fusiona por sesión y conserva el ``source`` del primero.

    Un mensaje humano llegado en la misma ventana se llevaría la autoría de un
    turno firmado. La atribución es justo lo que este camino garantiza, así que
    un turno del dashboard se entrega solo.
    """
    adapter._text_batch_delay_seconds = 0.6
    adapter.batched = []
    adapter._enqueue_text_event = lambda event: adapter.batched.append(event)

    text = "no me mezcles"
    inbox.write(_record(text), body=_published(text))
    message = _self_message(adapter, _published(text))

    assert await adapter._handle_message(message) is True
    assert adapter.batched == []
    assert adapter.delivered[-1].source.user_name == "Jose (dashboard)"
