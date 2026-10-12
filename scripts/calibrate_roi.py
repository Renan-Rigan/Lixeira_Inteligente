"""Escolhe a região de interesse (só a área de captura) arrastando um retângulo.

  python scripts/calibrate_roi.py --config config.yaml

Imprime a linha `roi: [x, y, w, h]` para colar em camera.roi.
Recortar a ROI tira do quadro bordas, paredes e a mão do usuário fora da
plataforma — melhora tanto a detecção de presença quanto a classificação.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lixeira.camera import open_camera  # noqa: E402
from lixeira.config import load_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cfg["camera"]["roi"] = None
    cam = open_camera(cfg)
    frame = cam.read()
    cam.close()
    x, y, w, h = cv2.selectROI("Arraste sobre a area de captura e aperte ENTER", frame, False)
    cv2.destroyAllWindows()
    if w and h:
        print(f"\nroi: [{x}, {y}, {w}, {h}]")
    else:
        print("Nenhuma região selecionada.")


if __name__ == "__main__":
    main()
