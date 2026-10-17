"""Registro de cada descarte em CSV + imagem.

O CSV alimenta a análise quantitativa da seção 5.5 (acurácia, tempo de resposta).
As imagens salvas em data/inbox/<classe_prevista>/ são o "volante" do projeto:
você revisa, move as que o modelo errou para a pasta certa, e elas entram no
próximo treino (scripts/prepare_dataset.py --source data/inbox).
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import cv2

from .pipeline import DisposalEvent

FIELDS = ["timestamp", "label", "raw_label", "confidence", "probs",
          "infer_ms", "detect_to_decision_ms", "actuation_ms", "total_ms",
          "image", "true_label"]


class EventLogger:
    def __init__(self, dir="logs", save_images=True, images_dir="data/inbox"):  # noqa: A002
        self.csv_path = Path(dir) / "events.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_images = save_images
        self.images_dir = Path(images_dir)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, FIELDS).writeheader()

    def log(self, ev: DisposalEvent, true_label: str = "") -> dict:
        stamp = datetime.fromtimestamp(ev.timestamp).strftime("%Y%m%d_%H%M%S_%f")
        img_path = ""
        if self.save_images and ev.image is not None:
            folder = self.images_dir / ev.prediction.raw_label
            folder.mkdir(parents=True, exist_ok=True)
            img_path = str(folder / f"{stamp}.jpg")
            cv2.imwrite(img_path, ev.image)
        p = ev.prediction
        row = {
            "timestamp": datetime.fromtimestamp(ev.timestamp).isoformat(timespec="milliseconds"),
            "label": p.label, "raw_label": p.raw_label,
            "confidence": f"{p.confidence:.4f}",
            "probs": ";".join(f"{k}={v:.3f}" for k, v in p.probs.items()),
            "infer_ms": f"{p.infer_ms:.1f}",
            "detect_to_decision_ms": f"{ev.detect_to_decision_ms:.1f}",
            "actuation_ms": f"{ev.actuation_ms:.1f}",
            "total_ms": f"{ev.total_ms:.1f}",
            "image": img_path, "true_label": true_label,
        }
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, FIELDS).writerow(row)
        return row
