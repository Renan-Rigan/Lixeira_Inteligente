"""Máquina de estados que integra captura -> presença -> classificação -> servos.

    VAZIO --objeto aparece--> ESTABILIZANDO --parado N frames--> CLASSIFICANDO
      ^                                                              |
      |                                             (vota em K frames, decide)
      |                                                              v
    AGUARDANDO_LIBERAR <------------------- descarte (gira, abre, fecha)

A lógica é independente de hardware: recebe frames e chama um Sorter, então
dá para testar com imagens sintéticas e servos simulados.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .actuators import Sorter
from .classifier import Classifier, Prediction, decide
from .presence import PresenceDetector, PresenceReading


class State(str, Enum):
    VAZIO = "vazio"
    ESTABILIZANDO = "estabilizando"
    CLASSIFICANDO = "classificando"
    AGUARDANDO_LIBERAR = "aguardando_liberar"


@dataclass
class DisposalEvent:
    timestamp: float
    prediction: Prediction
    detect_to_decision_ms: float   # do objeto aparecer até a decisão
    actuation_ms: float            # tempo do mecanismo
    total_ms: float
    image: np.ndarray = field(repr=False, default=None)


class SortingPipeline:
    def __init__(self, detector: PresenceDetector, classifier: Classifier,
                 sorter: Sorter, cfg: dict, clock=time.perf_counter):
        self.det, self.clf, self.sorter = detector, classifier, sorter
        p, c = cfg["presence"], cfg["classifier"]
        self.stable_needed = p["stable_frames"]
        self.clear_needed = p["clear_frames"]
        self.votes = c["votes"]
        self.conf_threshold = c["conf_threshold"]
        self.reject_label = cfg["reject_label"]
        self.clock = clock
        self.state = State.VAZIO
        self.last_reading: PresenceReading | None = None
        self._reset()

    def _reset(self) -> None:
        self._stable = 0
        self._clear = 0
        self._vote_frames: list[np.ndarray] = []
        self._t_detect: float | None = None

    def step(self, frame: np.ndarray) -> DisposalEvent | None:
        r = self.det.update(frame)
        self.last_reading = r

        if self.state == State.VAZIO:
            if r.present:
                self._t_detect = self.clock()
                self.state = State.ESTABILIZANDO
            return None

        if self.state == State.ESTABILIZANDO:
            if not r.present:            # foi embora antes de assentar
                self.state = State.VAZIO
                self._reset()
            elif r.still:
                self._stable += 1
                if self._stable >= self.stable_needed:
                    self.state = State.CLASSIFICANDO
            else:
                self._stable = 0         # ainda mexendo (mão, rolando)
            return None

        if self.state == State.CLASSIFICANDO:
            if not r.present:
                self.state = State.VAZIO
                self._reset()
                return None
            if not r.still:              # mexeu de novo: volta a esperar
                self.state = State.ESTABILIZANDO
                self._stable = 0
                self._vote_frames.clear()
                return None
            self._vote_frames.append(frame.copy())
            if len(self._vote_frames) < self.votes:
                return None
            return self._classify_and_dispose()

        if self.state == State.AGUARDANDO_LIBERAR:
            self._clear = self._clear + 1 if not r.present else 0
            if self._clear >= self.clear_needed:
                self.state = State.VAZIO
                self._reset()
            return None
        return None

    def _classify_and_dispose(self) -> DisposalEvent:
        pred = decide(self.clf, self._vote_frames, self.conf_threshold, self.reject_label)
        t_dec = self.clock()
        self.sorter.dispose(pred.label)
        t_end = self.clock()
        t0 = self._t_detect if self._t_detect is not None else t_dec
        event = DisposalEvent(time.time(), pred, (t_dec - t0) * 1000,
                              (t_end - t_dec) * 1000, (t_end - t0) * 1000,
                              self._vote_frames[len(self._vote_frames) // 2])
        self.state = State.AGUARDANDO_LIBERAR
        self._reset()
        return event
