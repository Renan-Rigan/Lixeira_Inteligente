"""Detecção de presença com OpenCV: há um objeto parado na plataforma?

Ideia: aprender a imagem da plataforma vazia (fundo) e comparar cada frame com ela.
- objeto presente  -> fração de pixels diferentes do fundo > min_area_frac
- objeto parado    -> fração de pixels diferentes do frame anterior < motion_frac
Só classificamos quando o objeto está presente E parado por stable_frames frames,
o que evita classificar a mão do usuário ou um objeto ainda caindo/rolando.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PresenceReading:
    present: bool
    still: bool
    area_frac: float     # quanto da ROI difere do fundo
    motion_frac: float   # quanto mudou desde o frame anterior
    bbox: tuple[int, int, int, int] | None  # caixa do objeto (x, y, w, h)


class PresenceDetector:
    def __init__(self, blur_ksize=7, diff_threshold=30, min_area_frac=0.03,
                 motion_frac=0.004, background_lr=0.02, **_ignored):
        self.k = blur_ksize | 1  # garante ímpar
        self.diff_threshold = diff_threshold
        self.min_area_frac = min_area_frac
        self.motion_frac = motion_frac
        self.background_lr = background_lr
        self.background: np.ndarray | None = None
        self._prev: np.ndarray | None = None

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        # Mantém as 3 cores: em escala de cinza, um objeto vermelho ou azul sobre
        # fundo cinza pode ter o mesmo brilho do fundo e ficar "invisível".
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return cv2.GaussianBlur(frame, (self.k, self.k), 0).astype(np.float32)

    def learn_background(self, frames: list[np.ndarray]) -> None:
        """Mediana de vários frames da plataforma vazia (robusta a ruído)."""
        stack = np.stack([self._prep(f) for f in frames])
        self.background = np.median(stack, axis=0).astype(np.float32)
        self._prev = None

    def _mask(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(a, b).max(axis=2)  # maior diferença entre os canais B, G, R
        mask = (diff > self.diff_threshold).astype(np.uint8) * 255
        # abertura remove pontinhos de ruído; fechamento une partes do objeto
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def update(self, frame: np.ndarray) -> PresenceReading:
        if self.background is None:
            raise RuntimeError("Chame learn_background() antes de update()")
        cur = self._prep(frame)
        fg = self._mask(cur, self.background)
        area = float(np.count_nonzero(fg)) / fg.size

        motion = 1.0 if self._prev is None else \
            float(np.count_nonzero(self._mask(cur, self._prev))) / fg.size
        self._prev = cur

        present = area > self.min_area_frac
        bbox = None
        if present:
            pts = cv2.findNonZero(fg)
            if pts is not None:
                bbox = tuple(int(v) for v in cv2.boundingRect(pts))
        else:
            # plataforma vazia: acompanha devagar mudanças de iluminação
            cv2.accumulateWeighted(cur, self.background, self.background_lr)

        return PresenceReading(present, motion < self.motion_frac, area, motion, bbox)
