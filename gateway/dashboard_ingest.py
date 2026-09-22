"""Inbox local de inyección nativa: mensajes escritos desde el dashboard de agentes.

El dashboard del router (``local-agent-router``) publica en un canal real de
Discord con la identidad del propio bot, por ``hermes send``. El adaptador de
Discord descarta sus propios mensajes (``message.author == self._client.user``),
así que hasta ahora esos mensajes no llegaban a ninguna sesión: se perdían sin
rastro.

Este módulo es la ÚNICA puerta por la que un mensaje del propio bot puede dejar
de descartarse. No la abre el mensaje: la abre un **registro de origen** que el
dashboard escribe en el home del perfil ANTES de publicar, firmado con un
secreto local del propio perfil. Sin registro válido el mensaje se descarta
exactamente igual que antes.

Por qué el registro se escribe ANTES de publicar, y por qué NO se indexa por
``message_id``: el id de Discord no existe hasta que el mensaje ya está
publicado, y para entonces el websocket del gateway ya entregó el evento. Un
registro indexado por id llegaría siempre tarde. El índice es, por eso, la
**dirección de contenido** del mensaje —el canal de destino y el hash del
cuerpo publicado—, que el escritor conoce de antemano y el lector recalcula
del mensaje vivo. El ``message_id`` sigue siendo un campo del registro: es
opcional, y si viene, DEBE coincidir con el mensaje real.

El destino es UN solo id, el ``chat_id``: el canal o el hilo donde el mensaje
vive de verdad (``message.channel.id`` aquí, ``thread_id or channel_id`` del
otro lado). No entra el canal padre, y es a propósito: el directorio que el
gateway publica NO siempre distingue un hilo de su padre —hay entradas con la
forma ``<X>:<X>``, donde el "padre" es el propio hilo—, así que atar el
registro a un padre derivado sería atarlo a un dato que los dos lados no
calculan igual. El ``chat_id`` sí: es el mismo id que identifica la sesión.

Propiedades que este contrato sostiene, todas fail-closed:

- **Autenticidad.** HMAC-SHA256 con una clave de 32 bytes que vive en
  ``run/dashboard-ingest/.hmac-key`` con modo 0600. Sin clave, sin registro o
  con firma que no valide en tiempo constante, no hay inyección.
- **Atadura al mensaje real.** El registro fija el canal de destino y el hash
  del cuerpo publicado. Un registro no puede prestarle su firma a otro mensaje
  ni a otro canal.
- **Caducidad.** 5 minutos. Un registro viejo no revive.
- **Un solo uso.** El consumo es un ``rename`` atómico dentro del directorio:
  quien gana la carrera se lleva el registro; el resto ve un registro ausente.
  No hay replay.
- **Sin bucles.** Las respuestas del agente no llevan registro, así que vuelven
  a caer en el descarte de siempre.
- **Sin symlinks ni traversal.** Todo se abre relativo a un descriptor del
  propio directorio, con ``O_NOFOLLOW``, y el nombre del fichero es un hash
  hexadecimal: no hay nombre que pueda salirse del directorio.

El contrato completo (campos, canonicalización y lado escritor) está en
``docs/agentes-red-gateway-arquitectura.md``, sección «Fase 4 — contrato de
inbox», del repositorio ``hermes-local-agent-router``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Subdirectorio del home del perfil donde vive el inbox.
INBOX_SUBDIR = ("run", "dashboard-ingest")
#: Clave HMAC local. Nunca se imprime, nunca sale del disco del perfil.
KEY_FILENAME = ".hmac-key"
#: Etiqueta de dominio: ata cada hash/firma a ESTE contrato y a esta versión.
DOMAIN = "hermes-dashboard-ingest/v1"
RECORD_VERSION = 1
#: Separador canónico. No puede aparecer en ids, timestamps ni nombres visibles.
_SEP = "\x1f"

#: Un registro caduca a los 5 minutos.
RECORD_TTL_SECONDS = 300.0
#: Tolerancia de reloj hacia el futuro: un registro fechado más allá no vale.
FUTURE_SKEW_SECONDS = 60.0
#: Tope duro de lectura. Un registro honesto ronda el kilobyte.
MAX_RECORD_BYTES = 64 * 1024
MAX_TEXT_CHARS = 4000
MAX_DISPLAY_NAME_CHARS = 80
_KEY_BYTES = 32

_SNOWFLAKE_RE = re.compile(r"^[0-9]{1,32}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class DashboardIngestError(Exception):
    """Registro presente pero inválido. Nunca se propaga al camino del mensaje."""


@dataclass(frozen=True)
class DashboardIngestRecord:
    """Registro de origen ya validado: el turno que hay que construir."""

    chat_id: str
    actor_id: str
    display_name: str
    text: str
    timestamp: str
    message_id: str = ""


def body_digest(body: str) -> str:
    """Hash del cuerpo EXACTO que se publica en el canal (sin espacios al borde)."""
    return hashlib.sha256(str(body or "").strip().encode("utf-8")).hexdigest()


def record_key(*, chat_id: str, body: str) -> str:
    """Nombre (sin extensión) del registro: dirección de contenido del mensaje.

    Lo calculan por igual el escritor —antes de publicar— y el lector —al ver
    su propio mensaje—, sin que ninguno necesite el ``message_id``.
    """
    canonical = _SEP.join([DOMAIN, str(chat_id or ""), body_digest(body)])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def signature(
    key: bytes,
    *,
    chat_id: str,
    actor_id: str,
    display_name: str,
    text: str,
    timestamp: str,
    body_sha256: str,
    message_id: str = "",
) -> str:
    """HMAC-SHA256 sobre la serialización canónica del registro.

    El texto entra por su hash, no en claro: así ningún salto de línea del
    mensaje puede fabricar una separación de campos que no existía.
    """
    fields = [
        DOMAIN,
        str(chat_id or ""),
        str(actor_id or ""),
        str(display_name or ""),
        str(timestamp or ""),
        str(body_sha256 or ""),
        str(message_id or ""),
        hashlib.sha256(str(text or "").encode("utf-8")).hexdigest(),
    ]
    for field in fields:
        if _SEP in field or "\n" in field:
            raise DashboardIngestError("campo con separador ilegal")
    return hmac.new(key, _SEP.join(fields).encode("utf-8"), hashlib.sha256).hexdigest()


def _parse_timestamp(raw: Any) -> datetime:
    text = str(raw or "").strip()
    if not text:
        raise DashboardIngestError("timestamp vacío")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DashboardIngestError("timestamp no es ISO-8601") from exc
    if parsed.tzinfo is None:
        raise DashboardIngestError("timestamp sin zona horaria")
    return parsed.astimezone(timezone.utc)


def _closed_id(raw: Any, *, allow_empty: bool) -> str:
    text = str(raw or "").strip()
    if not text:
        if allow_empty:
            return ""
        raise DashboardIngestError("id vacío")
    if not _SNOWFLAKE_RE.fullmatch(text):
        raise DashboardIngestError("id con forma inesperada")
    return text


def _closed_display_name(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text or len(text) > MAX_DISPLAY_NAME_CHARS:
        raise DashboardIngestError("display_name fuera de rango")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text):
        raise DashboardIngestError("display_name con caracteres de control")
    return text


class DashboardIngestInbox:
    """Lector/escritor del inbox de un perfil concreto."""

    def __init__(self, home: Any) -> None:
        self.directory = Path(home).joinpath(*INBOX_SUBDIR)

    # -- clave ------------------------------------------------------------

    def _open_dir_fd(self) -> Optional[int]:
        """Descriptor del directorio del inbox, o ``None`` si no existe.

        Todo lo demás se abre relativo a este descriptor: así ni un symlink
        intermedio ni un rename del directorio pueden mover el objetivo entre
        la comprobación y el uso.
        """
        try:
            return os.open(
                str(self.directory), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
        except OSError:
            return None

    @staticmethod
    def _read_key_fd(dir_fd: int) -> bytes:
        fd = os.open(KEY_FILENAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise DashboardIngestError("la clave no es un fichero regular")
            if info.st_mode & 0o077:
                raise DashboardIngestError("la clave es legible por otros")
            raw = os.read(fd, 4096).decode("ascii", "strict").strip()
        finally:
            os.close(fd)
        try:
            key = bytes.fromhex(raw)
        except ValueError as exc:
            raise DashboardIngestError("la clave no es hexadecimal") from exc
        if len(key) != _KEY_BYTES:
            raise DashboardIngestError("la clave no mide 32 bytes")
        return key

    def load_key(self) -> Optional[bytes]:
        """Clave del perfil, o ``None`` si no hay inbox o la clave no sirve."""
        dir_fd = self._open_dir_fd()
        if dir_fd is None:
            return None
        try:
            return self._read_key_fd(dir_fd)
        except (OSError, DashboardIngestError, UnicodeDecodeError):
            return None
        finally:
            os.close(dir_fd)

    def ensure_key(self) -> bytes:
        """Clave existente, o una nueva de 32 bytes creada con modo 0600.

        Solo el ESCRITOR debería llamar a esto. El adaptador nunca crea la
        clave: si no está, no hay inyección y punto.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        dir_fd = self._open_dir_fd()
        if dir_fd is None:
            raise DashboardIngestError("el inbox no existe o no es un directorio real")
        try:
            # fchmod sobre el descriptor ya abierto con ``O_NOFOLLOW``: un
            # symlink colocado donde va el inbox no puede desviar el permiso.
            os.fchmod(dir_fd, 0o700)
            try:
                return self._read_key_fd(dir_fd)
            except FileNotFoundError:
                pass  # primera vez: se crea abajo
            # Cualquier otro fallo (permisos flojos, clave corrupta) se propaga:
            # sobrescribir una clave existente invalidaría registros en vuelo.
            tmp_name = f".hmac-key.{uuid.uuid4().hex}.tmp"
            fd = os.open(
                tmp_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=dir_fd,
            )
            try:
                key = secrets.token_bytes(_KEY_BYTES)
                os.write(fd, key.hex().encode("ascii"))
                os.fsync(fd)
            finally:
                os.close(fd)
            try:
                # ``link`` y no ``rename``: si otro proceso ganó la carrera y ya
                # dejó una clave, esto falla con EEXIST en vez de pisarla.
                os.link(tmp_name, KEY_FILENAME, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            except FileExistsError:
                key = self._read_key_fd(dir_fd)
            finally:
                try:
                    os.unlink(tmp_name, dir_fd=dir_fd)
                except OSError:
                    logger.debug("dashboard ingest: temporal de clave residual", exc_info=True)
            return key
        finally:
            os.close(dir_fd)

    # -- escritura (referencia; el dashboard implementa el mismo contrato) --

    def write(
        self,
        record: DashboardIngestRecord,
        *,
        body: str,
        key: Optional[bytes] = None,
    ) -> str:
        """Escribe el registro de ``body`` y devuelve su nombre de fichero.

        Atómico: se escribe en un temporal 0600 y se renombra encima. El
        escritor llama a esto ANTES de publicar en el canal.
        """
        signing_key = key if key is not None else self.ensure_key()
        digest = body_digest(body)
        payload = {
            "version": RECORD_VERSION,
            "chat_id": record.chat_id,
            "actor": {"id": record.actor_id, "display_name": record.display_name},
            "text": record.text,
            "body_sha256": digest,
            "timestamp": record.timestamp,
            "message_id": record.message_id,
            "hmac": signature(
                signing_key,
                chat_id=record.chat_id,
                actor_id=record.actor_id,
                display_name=record.display_name,
                text=record.text,
                timestamp=record.timestamp,
                body_sha256=digest,
                message_id=record.message_id,
            ),
        }
        name = record_key(chat_id=record.chat_id, body=body)
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        dir_fd = self._open_dir_fd()
        if dir_fd is None:
            raise DashboardIngestError("el inbox no existe")
        try:
            tmp_name = f".{name}.{uuid.uuid4().hex}.tmp"
            fd = os.open(
                tmp_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=dir_fd,
            )
            try:
                os.write(fd, blob)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rename(
                tmp_name, f"{name}.json", src_dir_fd=dir_fd, dst_dir_fd=dir_fd
            )
        finally:
            os.close(dir_fd)
        return f"{name}.json"

    # -- lectura ----------------------------------------------------------

    def _validate(
        self,
        payload: Any,
        key: bytes,
        *,
        chat_id: str,
        body: str,
        message_id: str,
        now: float,
    ) -> DashboardIngestRecord:
        if not isinstance(payload, dict):
            raise DashboardIngestError("el registro no es un objeto")
        if payload.get("version") != RECORD_VERSION:
            raise DashboardIngestError("versión de registro desconocida")
        actor = payload.get("actor")
        if not isinstance(actor, dict):
            raise DashboardIngestError("actor ausente")
        record_chat = _closed_id(payload.get("chat_id"), allow_empty=False)
        if record_chat != chat_id:
            raise DashboardIngestError("el registro apunta a otro canal")
        digest = str(payload.get("body_sha256") or "").strip().lower()
        if not _HEX64_RE.fullmatch(digest) or digest != body_digest(body):
            raise DashboardIngestError("el registro no describe este mensaje")
        record_message_id = _closed_id(payload.get("message_id"), allow_empty=True)
        if record_message_id and record_message_id != str(message_id or ""):
            raise DashboardIngestError("el registro nombra otro message_id")
        actor_id = _closed_id(actor.get("id"), allow_empty=True)
        display_name = _closed_display_name(actor.get("display_name"))
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise DashboardIngestError("texto vacío")
        if len(text) > MAX_TEXT_CHARS:
            raise DashboardIngestError("texto fuera de rango")
        timestamp = str(payload.get("timestamp") or "").strip()
        stamped = _parse_timestamp(timestamp)
        age = now - stamped.timestamp()
        if age > RECORD_TTL_SECONDS:
            raise DashboardIngestError("registro caducado")
        if age < -FUTURE_SKEW_SECONDS:
            raise DashboardIngestError("registro fechado en el futuro")
        expected = signature(
            key,
            chat_id=record_chat,
            actor_id=actor_id,
            display_name=display_name,
            text=text,
            timestamp=timestamp,
            body_sha256=digest,
            message_id=record_message_id,
        )
        provided = str(payload.get("hmac") or "").strip().lower()
        if not _HEX64_RE.fullmatch(provided) or not hmac.compare_digest(
            expected, provided
        ):
            raise DashboardIngestError("firma inválida")
        return DashboardIngestRecord(
            chat_id=record_chat,
            actor_id=actor_id,
            display_name=display_name,
            text=text,
            timestamp=timestamp,
            message_id=record_message_id,
        )

    @staticmethod
    def _read_json_fd(dir_fd: int, name: str) -> Any:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise DashboardIngestError("el registro no es un fichero regular")
            if info.st_size > MAX_RECORD_BYTES:
                raise DashboardIngestError("registro demasiado grande")
            raw = os.read(fd, MAX_RECORD_BYTES)
        finally:
            os.close(fd)
        return json.loads(raw.decode("utf-8"))

    def peek(
        self,
        *,
        chat_id: str,
        body: str,
        message_id: str = "",
        now: Optional[float] = None,
    ) -> bool:
        """¿Hay un registro válido para este mensaje? No consume nada.

        Es la comprobación barata de la puerta de admisión: decide si el
        mensaje propio merece seguir, sin gastar el registro por si el camino
        posterior lo descarta por otra regla.
        """
        return self._lookup(
            chat_id=chat_id,
            body=body,
            message_id=message_id,
            now=now,
            claim=False,
        ) is not None

    def consume(
        self,
        *,
        chat_id: str,
        body: str,
        message_id: str = "",
        now: Optional[float] = None,
    ) -> Optional[DashboardIngestRecord]:
        """Reclama el registro de forma atómica y lo devuelve validado.

        El ``rename`` es la reclamación: dos lectores concurrentes no pueden
        ganarlo los dos. Un registro reclamado nunca vuelve al inbox, valide o
        no, porque un registro que no valida tampoco debe quedarse esperando.
        """
        return self._lookup(
            chat_id=chat_id,
            body=body,
            message_id=message_id,
            now=now,
            claim=True,
        )

    def _lookup(
        self,
        *,
        chat_id: str,
        body: str,
        message_id: str,
        now: Optional[float],
        claim: bool,
    ) -> Optional[DashboardIngestRecord]:
        try:
            closed_chat = _closed_id(chat_id, allow_empty=False)
        except DashboardIngestError:
            return None
        text = str(body or "").strip()
        if not text:
            return None
        dir_fd = self._open_dir_fd()
        if dir_fd is None:
            return None
        stamp = datetime.now(timezone.utc).timestamp() if now is None else float(now)
        name = f"{record_key(chat_id=closed_chat, body=text)}.json"
        target = name
        try:
            key = self._read_key_fd(dir_fd)
            if claim:
                target = f".claim-{uuid.uuid4().hex}.tmp"
                os.rename(name, target, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            try:
                payload = self._read_json_fd(dir_fd, target)
                return self._validate(
                    payload,
                    key,
                    chat_id=closed_chat,
                    body=text,
                    message_id=str(message_id or ""),
                    now=stamp,
                )
            finally:
                if claim:
                    try:
                        os.unlink(target, dir_fd=dir_fd)
                    except OSError:
                        logger.debug(
                            "dashboard ingest: no se pudo borrar el registro reclamado",
                            exc_info=True,
                        )
        except FileNotFoundError:
            return None
        except (OSError, ValueError, DashboardIngestError, UnicodeDecodeError):
            logger.debug("dashboard ingest: registro descartado", exc_info=True)
            return None
        finally:
            os.close(dir_fd)
