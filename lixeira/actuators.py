"""Controle dos dois servos: base rotatória e escotilha.

Sequência de descarte (seção 5.3 do projeto):
  1. escotilha fechada  2. gira a base até o compartimento  3. espera estabilizar
  4. abre a escotilha   5. espera o resíduo cair            6. fecha  7. volta ao início
A escotilha NUNCA abre durante a rotação — isso é garantido aqui no código.
"""
from __future__ import annotations

import time
from typing import Callable, Protocol


class Servo(Protocol):
    def set_angle(self, angle: float) -> None: ...
    def release(self) -> None: ...


class MockServo:
    """Servo simulado: registra os ângulos (usado em testes e no PC)."""

    def __init__(self, name: str, verbose: bool = True):
        self.name, self.verbose = name, verbose
        self.angle: float | None = None
        self.history: list[float] = []

    def set_angle(self, angle: float) -> None:
        self.angle = angle
        self.history.append(angle)
        if self.verbose:
            print(f"  [servo {self.name}] -> {angle:.0f}°")

    def release(self) -> None:
        pass


class GpioServo:
    """Servo real via gpiozero. No Pi 5 use pin_factory 'lgpio'; no Pi 4,
    'pigpio' (com o daemon pigpiod rodando) gera PWM por hardware e tira o tremor.
    Se tremer muito ou precisar de mais servos, considere uma placa PCA9685."""

    def __init__(self, pin: int, min_angle: float, max_angle: float,
                 min_pulse_ms: float, max_pulse_ms: float, pin_factory: str | None = None):
        from gpiozero import AngularServo, Device

        if pin_factory:
            if pin_factory == "pigpio":
                from gpiozero.pins.pigpio import PiGPIOFactory as F
            elif pin_factory == "lgpio":
                from gpiozero.pins.lgpio import LGPIOFactory as F
            else:
                raise ValueError(f"pin_factory desconhecido: {pin_factory}")
            Device.pin_factory = F()
        self.servo = AngularServo(pin, min_angle=min_angle, max_angle=max_angle,
                                  min_pulse_width=min_pulse_ms / 1000,
                                  max_pulse_width=max_pulse_ms / 1000)

    def set_angle(self, angle: float) -> None:
        self.servo.angle = angle

    def release(self) -> None:
        self.servo.detach()  # para de mandar pulso (evita tremor/aquecimento parado)


class Sorter:
    def __init__(self, rotation: Servo, hatch: Servo, cfg: dict,
                 sleep: Callable[[float], None] = time.sleep):
        a = cfg["actuators"]
        self.rot, self.hatch = rotation, hatch
        self.rot_cfg, self.hatch_cfg = a["rotation"], a["hatch"]
        self.bin_angles = a["bin_angles"]
        self.home_angle = a["home_angle"]
        self.return_home = a["return_home"]
        self.sleep = sleep
        self.hatch_open = False

    def home(self) -> None:
        self._close_hatch()
        self.rot.set_angle(self.home_angle)
        self.sleep(self.rot_cfg["settle_s"])

    def _close_hatch(self) -> None:
        self.hatch.set_angle(self.hatch_cfg["closed_angle"])
        self.sleep(self.hatch_cfg["settle_s"])
        self.hatch_open = False

    def _rotate(self, angle: float) -> None:
        if self.hatch_open:  # trava de segurança
            raise RuntimeError("Tentativa de girar com a escotilha aberta")
        self.rot.set_angle(angle)
        self.sleep(self.rot_cfg["settle_s"])

    def dispose(self, label: str) -> None:
        if label not in self.bin_angles:
            raise KeyError(f"Sem compartimento para '{label}'")
        self._close_hatch()
        self._rotate(self.bin_angles[label])
        self.hatch.set_angle(self.hatch_cfg["open_angle"])
        self.hatch_open = True
        self.sleep(self.hatch_cfg["open_s"])
        self._close_hatch()
        if self.return_home:
            self._rotate(self.home_angle)

    def close(self) -> None:
        try:
            self.home()
        finally:
            self.rot.release()
            self.hatch.release()


def build_sorter(cfg: dict, verbose: bool = True) -> Sorter:
    a = cfg["actuators"]
    if a["backend"] == "mock":
        return Sorter(MockServo("base", verbose), MockServo("escotilha", verbose), cfg,
                      sleep=lambda s: None)
    if a["backend"] == "gpiozero":
        r, h = a["rotation"], a["hatch"]
        rot = GpioServo(r["pin"], r["min_angle"], r["max_angle"],
                        r["min_pulse_ms"], r["max_pulse_ms"], a["pin_factory"])
        hatch = GpioServo(h["pin"], h["min_angle"], h["max_angle"],
                          h["min_pulse_ms"], h["max_pulse_ms"], None)
        return Sorter(rot, hatch, cfg)
    raise ValueError(f"actuators.backend desconhecido: {a['backend']}")
