"""Executa a lixeira: câmera -> presença -> YOLO -> servos.

Exemplos:
  python run.py --show                           # PC com webcam, servos simulados
  python run.py --fake-model --show              # sem modelo treinado ainda
  python run.py --config config.pi.yaml          # no Raspberry Pi
  python run.py --show --true-label plastico     # sessão de teste: registra o rótulo
                                                 # verdadeiro para medir acurácia
Teclas (com --show): q sai, b reaprende o fundo vazio.
"""
from __future__ import annotations

import argparse
import sys

import cv2

from lixeira.actuators import build_sorter
from lixeira.camera import FolderSource, open_camera
from lixeira.classifier import build_classifier
from lixeira.config import load_config
from lixeira.event_log import EventLogger
from lixeira.pipeline import SortingPipeline
from lixeira.presence import PresenceDetector


def learn_background(cam, det, n):
    print(f"Aprendendo o fundo com {n} frames — deixe a plataforma VAZIA...")
    det.learn_background([cam.read() for _ in range(n)])
    print("Fundo aprendido.")


def draw_overlay(frame, pipe, last):
    out = frame.copy()
    r = pipe.last_reading
    if r and r.bbox:
        x, y, w, h = r.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    lines = [f"estado: {pipe.state.value}"]
    if r:
        lines.append(f"area {r.area_frac:.3f}  mov {r.motion_frac:.4f}")
    if last:
        p = last.prediction
        lines.append(f"ultimo: {p.label} ({p.raw_label} {p.confidence:.2f})")
    for i, t in enumerate(lines):
        cv2.putText(out, t, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(out, t, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fake-model", action="store_true", help="classificador falso (sem YOLO)")
    ap.add_argument("--mock-servos", action="store_true", help="força servos simulados")
    ap.add_argument("--weights", help="sobrescreve classifier.weights")
    ap.add_argument("--source", help="pasta de imagens no lugar da câmera (depuração)")
    ap.add_argument("--show", action="store_true", help="janela com a imagem e o estado")
    ap.add_argument("--true-label", default="", help="rótulo real dos itens desta sessão de teste")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.fake_model:
        cfg["classifier"]["backend"] = "fake"
    if args.mock_servos:
        cfg["actuators"]["backend"] = "mock"
    if args.weights:
        cfg["classifier"]["weights"] = args.weights

    cam = FolderSource(args.source, loop=False) if args.source else open_camera(cfg)
    det = PresenceDetector(**cfg["presence"])
    clf = build_classifier(cfg)
    sorter = build_sorter(cfg)
    log = EventLogger(**cfg["logging"])
    pipe = SortingPipeline(det, clf, sorter, cfg)
    print(f"Classes do modelo: {clf.names}")

    sorter.home()
    learn_background(cam, det, cfg["presence"]["background_frames"])
    last = None
    try:
        while True:
            frame = cam.read()
            ev = pipe.step(frame)
            if ev:
                last = ev
                row = log.log(ev, args.true_label)
                ok = ""
                if args.true_label:
                    ok = "  ACERTOU" if ev.prediction.label == args.true_label else "  ERROU"
                print(f"-> {row['label']:<10} conf={row['confidence']} "
                      f"inferência={row['infer_ms']}ms total={row['total_ms']}ms{ok}")
            if args.show:
                cv2.imshow("lixeira", draw_overlay(frame, pipe, last))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    learn_background(cam, det, cfg["presence"]["background_frames"])
    except (KeyboardInterrupt, StopIteration):
        pass
    finally:
        sorter.close()
        cam.close()
        if args.show:
            cv2.destroyAllWindows()
        print(f"Eventos registrados em {log.csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
