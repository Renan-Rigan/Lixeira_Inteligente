"""Identificação em tempo real pela câmera do celular (ou webcam), sem plataforma.

Classifica o quadro continuamente e mostra a classe, as probabilidades e se a
imagem está nítida e bem iluminada. Serve para testar o modelo treinado com o
banco de imagens apontando o celular para latas, garrafas PET e outros resíduos,
antes de ter o protótipo. Diferente do run.py, não precisa de fundo fixo nem de
servos: dá para segurar o celular na mão.

  python scripts/live.py --config config.celular.yaml               # modelo em models/best.pt
  python scripts/live.py --config config.celular.yaml --fake-model  # sem modelo ainda
  python scripts/live.py --weights runs/classify/lixeira/weights/best.pt

O stream é reduzido para camera.width x camera.height (640x480): é a mesma
resolução que o Raspberry Pi entrega, então o celular simula o que o Pi vai ver.
O quadrado no centro é o que o modelo recebe (o YOLO-cls corta o centro).

Teclas: q sai | espaço pausa | s salva o quadro em data/live/<classe prevista>/
(revise e use como teste: prepare_dataset.py --source data/live)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lixeira.camera import open_camera  # noqa: E402
from lixeira.classifier import ProbSmoother, build_classifier, from_probs  # noqa: E402
from lixeira.config import load_config  # noqa: E402
from lixeira.quality import measure, quality_params  # noqa: E402

COLORS = {"metal": (200, 160, 60), "plastico": (60, 60, 230), "resto": (60, 170, 60)}  # BGR


def center_square(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    x, y = (w - s) // 2, (h - s) // 2
    return x, y, s


def draw(frame, pred, names, quality_msgs, fps, infer_ms, paused):
    out = frame.copy()
    x, y, s = center_square(out)
    color = COLORS.get(pred.label, (255, 255, 255))
    cv2.rectangle(out, (x, y), (x + s - 1, y + s - 1), color, 2)

    def text(t, row, col=(255, 255, 255), scale=0.6):
        org = (8, 24 + 24 * row)
        cv2.putText(out, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
        cv2.putText(out, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1)

    head = pred.label.upper()
    if pred.label != pred.raw_label:
        head += f" (incerto: {pred.raw_label}?)"
    text(head, 0, color, 0.8)
    for i, n in enumerate(names):
        p = pred.probs.get(n, 0.0)
        y0 = 24 + 24 * (i + 1) - 13
        cv2.rectangle(out, (140, y0), (140 + int(150 * p), y0 + 14), COLORS.get(n, (200, 200, 200)), -1)
        text(f"{n} {p:.0%}", i + 1)
    text(f"{fps:.1f} fps  modelo {infer_ms:.0f} ms" + ("  [PAUSADO]" if paused else ""), len(names) + 1)
    for j, m in enumerate(quality_msgs):
        text(m, len(names) + 2 + j, (0, 220, 255))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--weights", help="sobrescreve classifier.weights")
    ap.add_argument("--fake-model", action="store_true", help="classificador falso (sem YOLO)")
    ap.add_argument("--smooth", type=float, default=0.3, help="0..1; menor = resposta mais estável e mais lenta")
    ap.add_argument("--save-dir", default="data/live")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.fake_model:
        cfg["classifier"]["backend"] = "fake"
    if args.weights:
        cfg["classifier"]["weights"] = args.weights
    clf = build_classifier(cfg)
    thr, reject = cfg["classifier"]["conf_threshold"], cfg["reject_label"]
    qp = quality_params(cfg)
    smoother = ProbSmoother(len(clf.names), args.smooth)
    cam = open_camera(cfg)
    print(f"Classes do modelo: {clf.names}  |  q sai, espaço pausa, s salva")

    paused, frame, fps, t_last = False, None, 0.0, time.perf_counter()
    try:
        while True:
            if not paused or frame is None:
                frame = cam.read()
                t0 = time.perf_counter()
                probs = smoother.update(clf.predict_probs(frame))
                infer_ms = (time.perf_counter() - t0) * 1000
                pred = from_probs(probs, clf.names, thr, reject)
                sharp, bright, _ = measure(frame, qp.ref_size)
                msgs = []
                if sharp < qp.blur_threshold:
                    msgs.append(f"imagem borrada ({sharp:.0f})")
                if bright < qp.dark_threshold:
                    msgs.append(f"imagem escura ({bright:.0f})")
                elif bright > qp.bright_threshold:
                    msgs.append(f"imagem estourada ({bright:.0f})")
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6) if fps else 1 / max(now - t_last, 1e-6)
                t_last = now
            cv2.imshow("lixeira - ao vivo", draw(frame, pred, clf.names, msgs, fps, infer_ms, paused))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                paused = not paused
            if k == ord("s"):
                folder = Path(args.save_dir) / pred.raw_label
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"salvo {path} ({pred.raw_label} {pred.confidence:.2f})")
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
