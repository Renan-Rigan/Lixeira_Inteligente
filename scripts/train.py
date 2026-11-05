"""Treina o classificador YOLO com transfer learning (seção 5.2 do projeto).

O YOLO-cls é uma rede neural convolucional (CNN) pré-treinada no ImageNet; aqui
só as classes mudam: metal, plastico (PET) e resto. Monte o dataset antes com
scripts/prepare_dataset.py (fotos próprias e/ou banco de imagens).

Rode no PC/notebook com GPU ou no Google Colab — NÃO no Raspberry Pi.

  python scripts/train.py                              # yolo11n-cls, 50 épocas
  python scripts/train.py --model yolo11s-cls.pt --epochs 80
  python scripts/train.py --resume runs/classify/lixeira/weights/last.pt

O melhor peso é copiado para models/best.pt e as métricas de validação vão para
runs/classify/<nome>/. Depois: scripts/evaluate.py no split de teste.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dataset")
    # Modelo "n" (nano) pré-treinado no ImageNet: o mais leve, ideal para o Pi.
    # Versões mais novas da Ultralytics podem ter outros nomes (ex.: yolo26n-cls.pt).
    ap.add_argument("--model", default="yolo11n-cls.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=15, help="early stopping")
    ap.add_argument("--device", default=None, help="0 para GPU, cpu, mps (Mac)")
    ap.add_argument("--name", default="lixeira")
    ap.add_argument("--resume", help="continuar de um last.pt")
    ap.add_argument("--out", default="models/best.pt")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(
            data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            patience=args.patience, device=args.device,
            project="runs/classify", name=args.name, exist_ok=False,
            # Aumento de dados pensado na lixeira: o objeto cai em qualquer
            # orientação (flips + RandAugment, que inclui rotações), a luz dos
            # LEDs varia pouco (cor moderada), e partes podem estar amassadas
            # ou cobertas (erasing).
            fliplr=0.5, flipud=0.5, scale=0.5,
            hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, erasing=0.3,
            auto_augment="randaugment",
            seed=42, deterministic=True, plots=True,
        )

    best = Path(results.save_dir) / "weights" / "best.pt"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.out)
    print(f"\nMelhor modelo: {best}\nCopiado para: {args.out}")
    print("Próximo passo: python scripts/evaluate.py --weights", args.out)


if __name__ == "__main__":
    main()
