import time

from app.domain.entities.engine_spec import EngineSpec
from app.domain.errors import EngineError
from app.infrastructure.repositories.engine_registry import EngineRegistry


def wait_until_loaded(reg, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end and any(e["state"] == "loading" for e in reg.describe()):
        time.sleep(0.01)


def boom():
    raise RuntimeError("no weights")


def test_ready_and_failed_engines_are_reported_independently():
    reg = EngineRegistry([EngineSpec("ok", "OK", "", lambda: object()), EngineSpec("bad", "Bad", "", boom)], "ok")
    reg.start()
    wait_until_loaded(reg)
    states = {e["id"]: e for e in reg.describe()}
    assert states["ok"]["state"] == "ready" and states["bad"]["state"] == "error"
    assert reg.get(None) is not None
    try:
        reg.get("bad")
    except EngineError as exc:
        assert exc.status == 503 and "no weights" in str(exc)


def test_unknown_engine_is_a_400_and_loading_is_a_503():
    reg = EngineRegistry([EngineSpec("slow", "Slow", "", lambda: object())], "slow")
    for engine_id, status in (("nope", 400), ("slow", 503)):  # not started yet -> still "loading"
        try:
            reg.get(engine_id)
        except EngineError as exc:
            assert exc.status == status
        else:
            raise AssertionError("expected EngineError")


def test_default_falls_back_to_first_spec():
    reg = EngineRegistry([EngineSpec("a", "A", "", lambda: 1)], "missing")
    assert reg.default_id == "a"
