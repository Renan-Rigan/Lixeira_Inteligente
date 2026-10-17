"""Captura de imagens: webcam via OpenCV, celular via Wi-Fi ou câmera CSI do
Raspberry Pi via Picamera2.

Todas as fontes devolvem frames BGR (padrão do OpenCV) já recortados na ROI.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    def read(self) -> np.ndarray: ...
    def close(self) -> None: ...


def crop_roi(frame: np.ndarray, roi) -> np.ndarray:
    if not roi:
        return frame
    x, y, w, h = (int(v) for v in roi)
    return frame[y:y + h, x:x + w]


class OpenCVCamera:
    """Webcam USB ou câmera do notebook."""

    def __init__(self, index=0, width=640, height=480, warmup_frames=15, roi=None):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não abriu a câmera de índice {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.roi = roi
        for _ in range(warmup_frames):
            self.cap.read()

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Falha ao ler frame da câmera")
        return crop_roi(frame, self.roi)

    def close(self) -> None:
        self.cap.release()


class NetworkCamera:
    """Celular como câmera pelo Wi-Fi (apps IP Webcam ou DroidCam).

    Uma thread lê o stream sem parar e guarda só o frame mais recente: sem isso o
    OpenCV entrega frames atrasados do buffer e o sistema "vê o passado". Se o
    Wi-Fi cair, reconecta sozinho. Os frames são redimensionados para
    width x height, para a ROI não mudar se o app mudar a resolução."""

    def __init__(self, url: str, width=640, height=480, warmup_frames=15, roi=None,
                 timeout_s: float = 5.0):
        self.url, self.size, self.roi, self.timeout_s = url, (width, height), roi, timeout_s
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não conectou em {url}. O celular e o PC estão na mesma "
                               "rede Wi-Fi? O app está transmitindo?")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame: np.ndarray | None = None
        self._seq = self._last_seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        for _ in range(warmup_frames):
            self.read()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:  # stream caiu: espera um pouco e reconecta
                self.cap.release()
                if self._stop.wait(1.0):
                    break
                self.cap = cv2.VideoCapture(self.url)
                continue
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            with self._cond:
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def read(self) -> np.ndarray:
        # Espera um frame NOVO: o detector de presença compara frames seguidos,
        # e devolver o mesmo frame duas vezes faria tudo parecer "parado".
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._last_seq, self.timeout_s):
                raise RuntimeError(f"Sem imagem do celular há {self.timeout_s:.0f} s ({self.url})")
            self._last_seq = self._seq
            return crop_roi(self._frame, self.roi)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
        self.cap.release()


class PiCamera:
    """Câmera CSI no Raspberry Pi OS (Bookworm+). O cv2.VideoCapture não acessa
    a pilha libcamera, por isso usamos Picamera2 (vem instalado no Raspberry Pi OS)."""

    def __init__(self, width=640, height=480, warmup_frames=15, roi=None):
        from picamera2 import Picamera2  # import tardio: só existe no Pi

        self.cam = Picamera2()
        cfg = self.cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"})
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(1.0)
        # Trava exposição e balanço de branco depois de estabilizar: com LEDs
        # fixos isso deixa as imagens consistentes entre um descarte e outro.
        meta = self.cam.capture_metadata()
        self.cam.set_controls({
            "AeEnable": False, "AwbEnable": False,
            "ExposureTime": meta.get("ExposureTime", 10000),
            "AnalogueGain": meta.get("AnalogueGain", 1.0),
            "ColourGains": meta.get("ColourGains", (1.5, 1.5)),
        })
        self.roi = roi
        for _ in range(warmup_frames):
            self.cam.capture_array()

    def read(self) -> np.ndarray:
        # "RGB888" na Picamera2 entrega os bytes em ordem BGR, que é o que o OpenCV espera.
        return crop_roi(self.cam.capture_array(), self.roi)

    def close(self) -> None:
        self.cam.stop()


class FolderSource:
    """Lê imagens de uma pasta em sequência (útil para depurar sem câmera)."""

    def __init__(self, folder: str | Path, loop: bool = True):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)
        if not self.files:
            raise RuntimeError(f"Nenhuma imagem em {folder}")
        self.loop, self.i = loop, 0

    def read(self) -> np.ndarray:
        if self.i >= len(self.files):
            if not self.loop:
                raise StopIteration
            self.i = 0
        img = cv2.imread(str(self.files[self.i]))
        self.i += 1
        return img

    def close(self) -> None:
        pass


def open_camera(cfg: dict) -> FrameSource:
    c = cfg["camera"]
    if c["backend"] == "picamera2":
        return PiCamera(c["width"], c["height"], c["warmup_frames"], c["roi"])
    if c["backend"] == "opencv" and c.get("url"):
        return NetworkCamera(c["url"], c["width"], c["height"], c["warmup_frames"], c["roi"])
    if c["backend"] == "opencv":
        return OpenCVCamera(c["index"], c["width"], c["height"], c["warmup_frames"], c["roi"])
    raise ValueError(f"camera.backend desconhecido: {c['backend']}")


def frames(source: FrameSource) -> Iterator[np.ndarray]:
    while True:
        try:
            yield source.read()
        except StopIteration:
            return
