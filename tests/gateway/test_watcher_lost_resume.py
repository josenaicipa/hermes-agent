"""Una sesion que espera un FABLE_WAKE se reanuda tras un reinicio.

Jose, 2026-09-21. El caso que faltaba: la sesion despacha trabajo, instala su
vigia en background y CIERRA su turno limpio. No esta en ``_running_agents``,
asi que el apagado no la marcaba -- y el reinicio mataba el vigia. Nadie
volvia a despertar a nadie: la mision seguia viva y muda hasta que un humano
escribia algo. Medido ese dia: 20 scopes de mision vivos y CERO vigias.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _mixin():
    import gateway.run_shutdown as RS

    for obj in vars(RS).values():
        if isinstance(obj, type) and hasattr(obj, "_sessions_awaiting_control_wake"):
            return obj
    raise AssertionError("no se encontro el mixin de apagado")


class _Sesion:
    def __init__(self, **campos):
        base = {
            "exit_code": None,
            "notify_on_failure": True,
            "watch_patterns": ["FABLE_WAKE"],
            "session_key": "discord:ada",
        }
        base.update(campos)
        self.__dict__.update(base)


def _detectar(monkeypatch, sesiones):
    import tools.process_registry as PR

    monkeypatch.setattr(
        PR.process_registry, "list_sessions", lambda **k: list(sesiones), raising=False
    )
    inst = _mixin().__new__(_mixin())
    return inst._sessions_awaiting_control_wake()


def test_un_vigia_vivo_marca_su_sesion(monkeypatch):
    assert _detectar(monkeypatch, [_Sesion(session_key="discord:ada")]) == ["discord:ada"]


@pytest.mark.parametrize(
    "campos,motivo",
    [
        ({"exit_code": 0}, "ya termino: su notificacion sale por el camino normal"),
        ({"notify_on_failure": False}, "nadie espera un despertar"),
        ({"watch_patterns": ["OTRA"]}, "no es el patron de control reliable"),
        ({"watch_patterns": ["FABLE_WAKE", "X"]}, "patron mezclado, no es el contrato"),
        ({"session_key": "   "}, "sin sesion a la que dar turno"),
        ({"watch_patterns": []}, "sin patron"),
    ],
)
def test_un_proceso_de_fondo_cualquiera_no_genera_turnos(monkeypatch, campos, motivo):
    assert _detectar(monkeypatch, [_Sesion(**campos)]) == [], motivo


def test_no_duplica_la_misma_sesion(monkeypatch):
    sesiones = [_Sesion(session_key="X"), _Sesion(session_key="X"), _Sesion(session_key="Y")]
    assert sorted(_detectar(monkeypatch, sesiones)) == ["X", "Y"]


def test_un_registro_que_estalla_no_tumba_el_apagado(monkeypatch):
    """El apagado nunca puede fallar por esto: devuelve vacio y sigue."""
    import tools.process_registry as PR

    def _boom(**k):
        raise RuntimeError("registro roto")

    monkeypatch.setattr(PR.process_registry, "list_sessions", _boom, raising=False)
    inst = _mixin().__new__(_mixin())
    assert inst._sessions_awaiting_control_wake() == []


def test_el_motivo_nuevo_esta_en_la_lista_blanca_del_auto_resume():
    """Sin esto, marcar la sesion no serviria de nada: se descartaria."""
    fuente = (REPO / "gateway" / "run.py").read_text(encoding="utf-8")
    bloque = re.search(r"_AUTO_RESUME_REASONS = frozenset\(\s*\{([^}]*)\}", fuente)
    assert bloque is not None
    assert "watcher_lost" in bloque.group(1)


def test_el_turno_reanudado_sabe_que_no_se_cayo():
    """Si leyera "interruption" podria rehacer trabajo ya entregado."""
    fuente = (REPO / "gateway" / "run.py").read_text(encoding="utf-8")
    assert "watcher_lost" in fuente
    assert "the work itself was not interrupted" in fuente
