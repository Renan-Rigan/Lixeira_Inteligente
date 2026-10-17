"""Carregamento da configuração (config.yaml) com valores padrão."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "classes": ["metal", "papel", "plastico"],
    "reject_label": "indefinido",
    "camera": {
        "backend": "opencv", "index": 0, "url": None, "width": 640, "height": 480,
        "warmup_frames": 15, "roi": None,
    },
    "presence": {
        "background_frames": 20, "blur_ksize": 7, "diff_threshold": 30,
        "min_area_frac": 0.03, "motion_frac": 0.004, "stable_frames": 8,
        "clear_frames": 10, "background_lr": 0.02,
    },
    "classifier": {
        "backend": "yolo", "weights": "models/best.pt", "imgsz": 224,
        "device": "cpu", "conf_threshold": 0.70, "votes": 3,
    },
    "actuators": {
        "backend": "mock", "pin_factory": None,
        "rotation": {"pin": 18, "min_angle": 0, "max_angle": 180,
                     "min_pulse_ms": 0.5, "max_pulse_ms": 2.5, "settle_s": 0.8},
        "hatch": {"pin": 13, "min_angle": 0, "max_angle": 180,
                  "closed_angle": 0, "open_angle": 90,
                  "min_pulse_ms": 0.5, "max_pulse_ms": 2.5,
                  "open_s": 1.0, "settle_s": 0.4},
        "home_angle": 90, "return_home": True,
        "bin_angles": {"metal": 0, "papel": 60, "plastico": 120, "indefinido": 180},
    },
    "logging": {"dir": "logs", "save_images": True, "images_dir": "data/inbox"},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_with_extends(path: Path, depth: int = 0) -> dict:
    """Lê um YAML; se tiver `extends: outro.yaml`, herda dele e sobrescreve."""
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    parent = data.pop("extends", None)
    if parent and depth < 5:
        return _merge(_read_with_extends(path.parent / parent, depth + 1), data)
    return data


def load_config(path: str | Path | None = "config.yaml") -> dict:
    """Lê o YAML e completa com os padrões. Valida o mapeamento classe -> ângulo."""
    user = _read_with_extends(Path(path)) if path else {}
    cfg = _merge(DEFAULTS, user)

    labels = list(cfg["classes"]) + [cfg["reject_label"]]
    missing = [c for c in labels if c not in cfg["actuators"]["bin_angles"]]
    if missing:
        raise ValueError(f"Sem ângulo em actuators.bin_angles para: {missing}")
    rot = cfg["actuators"]["rotation"]
    for label, ang in cfg["actuators"]["bin_angles"].items():
        if not rot["min_angle"] <= ang <= rot["max_angle"]:
            raise ValueError(f"Ângulo {ang} de '{label}' fora de "
                             f"[{rot['min_angle']}, {rot['max_angle']}]")
    return cfg
