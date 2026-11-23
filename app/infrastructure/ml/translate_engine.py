"""Engine: translate non-English text to English with Ollama, then classify with an English model."""
from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.domain.entities.prediction import Prediction
from app.domain.errors import TranslationError, TranslationUnavailable
from app.infrastructure.config.settings import ENGLISH_MODEL
from app.infrastructure.ml.ollama_client import OllamaClient

try:  # optional but recommended: tells Latin-script languages (French, Spanish...) apart from English
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0
except ImportError:  # pragma: no cover
    detect_langs = None

MIN_DETECT_LETTERS = 15
MAX_TRANSLATE_CHARS = 2000  # the classifiers only read ~512 tokens anyway
CACHE_MAX = 20000


# --------------------------------------------------------------------------
# Language check: does this text need translating?
# --------------------------------------------------------------------------
def needs_translation(text: str) -> bool:
    """True when the text is probably not English. When unsure, assume English (no translation)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if sum(ord(c) > 127 for c in letters) / len(letters) >= 0.3:
        return True  # Arabic, Cyrillic, CJK, ... (any mostly non-Latin text)
    if detect_langs is None:
        return False
    if len(letters) < MIN_DETECT_LETTERS and not any(ord(c) > 127 for c in letters):
        return False  # detectors are unreliable on short Latin-script text ("ok", "I love it"): assume English
    try:
        top = detect_langs(text)[0]
    except Exception:  # too short or no usable letters
        return False
    return top.lang != "en" and top.prob >= 0.7


# --------------------------------------------------------------------------
# Engine: translate to English with Ollama, then classify with an English model
# --------------------------------------------------------------------------
class TranslateEngine:
    def __init__(self, classifier, client: OllamaClient, default_model: str | None = None, classifier_name: str = ENGLISH_MODEL) -> None:
        self.classifier = classifier
        self.classifier_name = classifier_name
        self.client = client
        self.default_model = default_model
        self.requested: str | None = None  # the model the user picked for this request (None = automatic)
        self.lock = threading.Lock()
        self.label = f"Ollama translate + {classifier_name}"
        self._cache: dict[tuple[str, str], str] = {}
        self._cache_lock = threading.Lock()
        self._used: str | None = None

    def bind(self, ollama_model: str | None = None) -> "TranslateEngine":
        """A per-request copy that remembers which Ollama model to use. Lock, classifier and cache are shared."""
        b = copy.copy(self)
        b.requested = (ollama_model or "").strip() or None
        b._used = None
        return b

    def describe(self) -> str:
        return f"Ollama translate ({self._used or self.requested or self.default_model or 'auto'}) + {self.classifier_name}"

    def _pick_model(self) -> str:
        installed = self.client.list_models()  # raises TranslationUnavailable when Ollama is off

        def has(m: str) -> bool:
            return m in installed or f"{m}:latest" in installed

        if self.requested:
            if not has(self.requested):
                raise TranslationError(f"The Ollama model '{self.requested}' is not installed. Install it with: ollama pull {self.requested}")
            return self.requested
        if self.default_model and has(self.default_model):
            return self.default_model
        if installed:
            return installed[0]
        raise TranslationError("Ollama has no models installed. Install one with: ollama pull qwen2.5:7b")

    def _translate_one(self, text: str, model: str) -> str:
        key = (model, text)
        with self._cache_lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        last: TranslationError | None = None
        for _ in range(2):  # one retry for a hiccup
            try:
                out = self.client.translate(text, model)
            except TranslationUnavailable:
                raise
            except TranslationError as exc:
                last = exc
                continue
            with self._cache_lock:
                if len(self._cache) >= CACHE_MAX:
                    self._cache.clear()
                self._cache[key] = out
            return out
        raise last  # type: ignore[misc]

    def predict(self, texts: list[str], progress=None) -> list[tuple]:
        n = len(texts)

        def report(frac: float) -> None:
            if progress:
                progress(min(n, int(frac * n)), n)

        status = ["empty" if not t.strip() else "english" for t in texts]
        todo = [i for i, t in enumerate(texts) if status[i] == "english" and needs_translation(t)]
        keys = {i: texts[i].strip()[:MAX_TRANSLATE_CHARS] for i in todo}
        unique = list(dict.fromkeys(keys.values()))

        translated: dict[str, str] = {}
        if unique:
            model = self._pick_model()
            self._used = model
            last_error: TranslationError | None = None
            with ThreadPoolExecutor(max_workers=max(1, self.client.workers)) as ex:
                futs = {ex.submit(self._translate_one, t, model): t for t in unique}
                try:
                    for k, fut in enumerate(as_completed(futs), 1):
                        try:
                            translated[futs[fut]] = fut.result()
                        except TranslationUnavailable:
                            raise
                        except TranslationError as exc:  # this row only: it will be scored untranslated
                            last_error = exc
                        report(0.8 * k / len(unique))
                except TranslationUnavailable:
                    for f in futs:
                        f.cancel()
                    raise
            if not translated:
                raise TranslationError(f"Ollama could not translate any text with '{model}': {last_error}")

        english: list[str | None] = [None] * n
        for i in todo:
            if keys[i] in translated:
                english[i], status[i] = translated[keys[i]], "translated"
            else:
                status[i] = "failed"
        base = 0.8 if unique else 0.0
        preds = self.classifier.predict(
            [english[i] or texts[i] for i in range(n)],
            progress=lambda d, t: report(base + (1 - base) * (d / t if t else 1)),
        )
        report(1.0)
        return [Prediction(*p[:3], {"english_text": english[i], "translation_status": status[i]}) for i, p in enumerate(preds)]
