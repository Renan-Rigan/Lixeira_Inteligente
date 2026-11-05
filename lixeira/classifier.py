"""Classificação do resíduo com YOLO (modo classify, da Ultralytics).

Por que classificação e não detecção? A lixeira recebe UM objeto por vez numa área
de captura controlada. Classificar a imagem inteira é mais rápido no Raspberry Pi,
mais preciso e o dataset é só "uma pasta por classe" — sem desenhar caixas.
Se no futuro houver vários objetos ao mesmo tempo, troque por um modelo detect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


@dataclass
class Prediction:
    label: str               # classe final (ou reject_label)
    confidence: float        # probabilidade média da classe vencedora
    raw_label: str           # classe vencedora antes do limiar de confiança
    probs: dict[str, float] = field(default_factory=dict)
    infer_ms: float = 0.0    # tempo total de inferência dos frames votados


class Classifier(Protocol):
    names: list[str]
    def predict_probs(self, image: np.ndarray) -> np.ndarray: ...


class YoloClassifier:
    """Wrapper do ultralytics.YOLO. Aceita .pt, .onnx ou pasta *_ncnn_model."""

    def __init__(self, weights: str, imgsz: int = 224, device: str = "cpu", **_ignored):
        from ultralytics import YOLO  # import tardio: os testes rodam sem ultralytics

        self.model = YOLO(weights, task="classify")
        self.imgsz, self.device = imgsz, device
        names = self.model.names
        self.names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    def predict_probs(self, image: np.ndarray) -> np.ndarray:
        r = self.model.predict(image, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return r.probs.data.cpu().numpy().astype(np.float32)


class FakeClassifier:
    """Classificador de mentira para testar o resto do sistema sem modelo treinado.
    Olha a cor média dos pixels que fogem do fundo:
    azulado -> metal, avermelhado -> plastico, esverdeado -> resto (orgânico)."""

    def __init__(self, names: Sequence[str] = ("metal", "plastico", "resto"), **_ignored):
        self.names = list(names)

    def predict_probs(self, image: np.ndarray) -> np.ndarray:
        px = image.reshape(-1, 3).astype(np.float32)
        far = np.abs(px - np.median(px, axis=0)).max(axis=1) > 30
        b, g, r = (px[far] if far.any() else px).mean(axis=0)
        scores = {"metal": b - r, "plastico": r - b, "resto": g - (b + r) / 2}
        v = np.array([scores.get(n, 0.0) for n in self.names], dtype=np.float32) / 20
        e = np.exp(v - v.max())
        return e / e.sum()


def decide(classifier: Classifier, images: Sequence[np.ndarray],
           conf_threshold: float, reject_label: str) -> Prediction:
    """Soma as probabilidades de vários frames (votação suave) e aplica o limiar.
    Votar em 3 frames reduz erros causados por um frame borrado ou com reflexo."""
    import time

    t0 = time.perf_counter()
    total = np.zeros(len(classifier.names), dtype=np.float32)
    for img in images:
        total += classifier.predict_probs(img)
    infer_ms = (time.perf_counter() - t0) * 1000
    probs = total / max(len(images), 1)
    return from_probs(probs, classifier.names, conf_threshold, reject_label, infer_ms)


def from_probs(probs: np.ndarray, names: Sequence[str], conf_threshold: float,
               reject_label: str, infer_ms: float = 0.0) -> Prediction:
    """Classe vencedora; abaixo do limiar de confiança vira reject_label."""
    best = int(np.argmax(probs))
    conf = float(probs[best])
    raw = names[best]
    label = raw if conf >= conf_threshold else reject_label
    return Prediction(label, conf, raw, {n: float(p) for n, p in zip(names, probs)}, infer_ms)


class ProbSmoother:
    """Média móvel exponencial das probabilidades, para o modo ao vivo
    (scripts/live.py): a resposta não pisca de um frame para o outro."""

    def __init__(self, n_classes: int, alpha: float = 0.3):
        self.alpha = alpha
        self.probs = np.full(n_classes, 1.0 / n_classes, dtype=np.float32)
        self._first = True

    def update(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float32)
        if self._first:
            self.probs, self._first = probs.copy(), False
        else:
            self.probs = (1 - self.alpha) * self.probs + self.alpha * probs
        return self.probs


def build_classifier(cfg: dict) -> Classifier:
    c = cfg["classifier"]
    if c["backend"] == "fake":
        return FakeClassifier(cfg["classes"])
    if c["backend"] == "yolo":
        clf = YoloClassifier(**c)
        unknown = set(clf.names) - set(cfg["actuators"]["bin_angles"])
        if unknown:
            raise ValueError(f"O modelo tem classes sem compartimento em bin_angles: {unknown}")
        return clf
    raise ValueError(f"classifier.backend desconhecido: {c['backend']}")
