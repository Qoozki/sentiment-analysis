from app.domain.entities.prediction import Prediction
from app.domain.errors import TranslationError, TranslationUnavailable
from app.infrastructure.ml.translate_engine import TranslateEngine, needs_translation


class Classifier:
    def predict(self, texts, progress=None):
        out = [Prediction("Positive" if "good" in t else "Negative", 0.8 if "good" in t else -0.8, 0.9) for t in texts]
        if progress:
            progress(len(texts), len(texts))
        return out


class Client:
    workers = 2

    def __init__(self, fail=False, down=False):
        self.fail, self.down, self.calls = fail, down, 0

    def list_models(self):
        if self.down:
            raise TranslationUnavailable("Cannot reach Ollama")
        return ["qwen2.5:7b"]

    def translate(self, text, model):
        self.calls += 1
        if self.fail:
            raise TranslationError("boom")
        return "this is good"


def test_needs_translation():
    assert needs_translation("المنتج رائع جدا")
    assert not needs_translation("I love it")
    assert not needs_translation("   ")
    assert not needs_translation("12345")


def test_only_non_english_rows_are_translated_and_cached():
    client = Client()
    eng = TranslateEngine(Classifier(), client, "qwen2.5:7b")
    preds = eng.predict(["المنتج رائع", "so good", "", "المنتج رائع"])
    assert [p.extras["translation_status"] for p in preds] == ["translated", "english", "empty", "translated"]
    assert preds[0].extras["english_text"] == "this is good" and preds[1].extras["english_text"] is None
    assert client.calls == 1  # the repeated sentence is translated once


def test_failed_rows_are_flagged_not_hidden():
    eng = TranslateEngine(Classifier(), Client(fail=True), "qwen2.5:7b")
    try:
        eng.predict(["المنتج رائع"])
    except TranslationError as exc:
        assert "could not translate any text" in str(exc)
    else:
        raise AssertionError("expected TranslationError")


def test_unreachable_service_stops_the_whole_run():
    eng = TranslateEngine(Classifier(), Client(down=True), "qwen2.5:7b")
    try:
        eng.predict(["المنتج رائع"])
    except TranslationUnavailable:
        return
    raise AssertionError("expected TranslationUnavailable")


def test_bind_keeps_request_choice_separate():
    base = TranslateEngine(Classifier(), Client(), "qwen2.5:7b")
    bound = base.bind("llama3")
    assert bound.requested == "llama3" and base.requested is None and bound.lock is base.lock
