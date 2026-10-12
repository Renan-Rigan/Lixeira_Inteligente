"""Exporta o modelo para rodar rápido no Raspberry Pi (seção 6.2: otimização).

  python scripts/export.py --weights models/best.pt               # NCNN (recomendado no Pi)
  python scripts/export.py --weights models/best.pt --format onnx
  python scripts/export.py --weights models/best.pt --int8        # quantização 8 bits

NCNN é o formato que a Ultralytics recomenda para Raspberry Pi (CPU ARM).
Copie a pasta gerada (models/best_ncnn_model/) para o Pi e aponte
classifier.weights para ela no config.pi.yaml. Sempre rode evaluate.py com o
modelo exportado: quantização pode custar alguns pontos de acurácia.
"""
from __future__ import annotations

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--format", default="ncnn", choices=["ncnn", "onnx", "tflite", "openvino"])
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--int8", action="store_true", help="quantização INT8 (precisa de --data para calibrar)")
    ap.add_argument("--half", action="store_true", help="FP16")
    ap.add_argument("--data", default="data/dataset", help="dados de calibração para INT8")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    kw = dict(format=args.format, imgsz=args.imgsz, half=args.half)
    if args.int8:
        kw.update(int8=True, data=args.data)
    path = YOLO(args.weights).export(**kw)
    print(f"\nExportado: {path}")


if __name__ == "__main__":
    main()
