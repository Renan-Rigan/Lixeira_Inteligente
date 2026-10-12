"""Coleta de imagens para o dataset, usando a MESMA câmera, ROI e iluminação da lixeira.

Isso é o que mais pesa na acurácia: datasets públicos (TrashNet etc.) têm outro
fundo e outra luz. Fotos tiradas na sua plataforma ensinam o modelo o cenário real.

Uso:
  python scripts/collect.py                      # config.yaml (webcam)
  python scripts/collect.py --config config.pi.yaml
  (no Pi sem monitor: rode com VNC, ou use --headless e digite as teclas no terminal)

Teclas:
  1..N   escolhe a classe atual (mostrada na tela)
  n      novo objeto (incrementa o id — IMPORTANTE para separar treino/teste certo)
  espaço salva 1 foto do objeto atual
  r      rajada: 8 fotos em ~2 s (gire/vire o objeto enquanto isso)
  q      sai

Arquivos: data/raw/<classe>/<obj_id>__<n>.jpg. O prefixo obj_id agrupa fotos do
mesmo objeto; o prepare_dataset.py nunca põe o mesmo objeto em treino E teste.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lixeira.camera import open_camera  # noqa: E402
from lixeira.config import load_config  # noqa: E402


def next_obj_id(root: Path) -> int:
    ids = [int(p.stem.split("__")[0].removeprefix("obj"))
           for p in root.rglob("obj*__*.jpg") if p.stem.split("__")[0][3:].isdigit()]
    return max(ids, default=0) + 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--burst", type=int, default=8)
    ap.add_argument("--headless", action="store_true", help="sem janela; comandos pelo terminal")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    classes = cfg["classes"]
    root = Path(args.out)
    cam = open_camera(cfg)
    cls_i, obj = 0, next_obj_id(root)
    shot = 0

    def save(frame):
        nonlocal shot
        folder = root / classes[cls_i]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"obj{obj:04d}__{shot:03d}.jpg"
        cv2.imwrite(str(path), frame)
        shot += 1
        return path

    def handle(key: str, frame) -> bool:
        nonlocal cls_i, obj, shot
        if key == "q":
            return False
        if key.isdigit() and 1 <= int(key) <= len(classes):
            cls_i = int(key) - 1
        elif key == "n":
            obj += 1
            shot = 0
        elif key == " ":
            print("salvo", save(frame))
        elif key == "r":
            for _ in range(args.burst):
                save(cam.read())
                time.sleep(0.25)
            print(f"rajada de {args.burst} fotos: {classes[cls_i]} obj{obj:04d}")
        return True

    help_txt = " ".join(f"{i+1}={c}" for i, c in enumerate(classes))
    print(f"Classes: {help_txt} | n=novo objeto, espaço=foto, r=rajada, q=sair")
    try:
        if args.headless:
            while True:
                cmd = input(f"[{classes[cls_i]} obj{obj:04d}] > ") or " "
                if not handle(cmd[0], cam.read()):
                    break
        else:
            while True:
                frame = cam.read()
                view = frame.copy()
                n_cls = len(list((root / classes[cls_i]).glob("*.jpg"))) if (root / classes[cls_i]).exists() else 0
                txt = f"{classes[cls_i]}  obj{obj:04d}  ({n_cls} fotos na classe)"
                cv2.putText(view, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("coleta", view)
                k = cv2.waitKey(1) & 0xFF
                if k != 255 and not handle(chr(k), frame):
                    break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
