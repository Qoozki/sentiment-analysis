"""Engine backed by a pre-trained Hugging Face text-classification model (torch + transformers)."""
from __future__ import annotations

import sys
import threading

from app.domain.entities.prediction import Prediction
from app.domain.value_objects.sentiment_scale import LEVELS3, LEVELS5, WEIGHTS3, WEIGHTS5
from app.infrastructure.config.settings import DEFAULT_MODEL

_LOAD_LOCK = threading.Lock()  # torch/transformers must not be imported by several threads at once


class SentimentModel:
    """Thin wrapper around a pre-trained Hugging Face text-classification model."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self.label = model_name  # shown in History
        self.lock = threading.Lock()  # one call at a time per model; pipelines are not thread-safe
        # The web app loads several models in parallel threads. Importing torch/transformers from two
        # threads at the same moment fails with confusing import errors, so loading is done one at a time.
        with _LOAD_LOCK:
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    f"Could not import torch/transformers ({exc}). If they are not installed, run: pip install transformers torch"
                ) from exc
            print(f"Loading model {model_name} (the first run downloads it)...", file=sys.stderr)
            self.pipe = pipeline(
                "text-classification",
                model=model_name,
                top_k=None,
                truncation=True,
                max_length=512,
                device=0 if torch.cuda.is_available() else -1,
            )
        n = getattr(self.pipe.model.config, "num_labels", 5)
        if n == 5:
            self.levels, self.weights = LEVELS5, WEIGHTS5
        elif n == 3:
            self.levels, self.weights = LEVELS3, WEIGHTS3
        else:
            raise RuntimeError(f"Model has {n} labels; this script supports 3- or 5-class sentiment models.")

    def _index(self, label: str) -> int:
        norm = label.strip().lower().replace("_", " ")
        if norm.startswith("label "):  # generic names: LABEL_0, LABEL_1, ...
            return int(norm.split()[1])
        names = [lv.lower() for lv in self.levels]
        if norm in names:
            return names.index(norm)
        raise ValueError(f"Unexpected label from model: {label!r}")

    def predict(self, texts: list[str], progress=None) -> list[Prediction]:
        """Return (detail_level, score, confidence) for each text.

        progress, if given, is called as progress(done, total) after each chunk.
        """
        results = []
        safe = [t if t.strip() else "." for t in texts]  # blanks are overwritten later
        for done in range(0, len(safe), 256):
            chunk = safe[done : done + 256]
            for probs in self.pipe(chunk, batch_size=self.batch_size):
                vec = [0.0] * len(self.levels)
                for r in probs:
                    vec[self._index(r["label"])] = r["score"]
                best = max(range(len(vec)), key=vec.__getitem__)
                score = sum(p * w for p, w in zip(vec, self.weights))
                results.append(Prediction(self.levels[best], score, vec[best]))
            end = min(done + 256, len(safe))
            if progress:
                progress(end, len(safe))
            else:
                print(f"  {end}/{len(safe)} rows", file=sys.stderr)
        return results

