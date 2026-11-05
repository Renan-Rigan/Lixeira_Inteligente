"""Calibração dos servos (seção 6.4). Rode no Raspberry Pi com o mecanismo montado.

  python scripts/servo_test.py --config config.pi.yaml

Comandos:
  b <ângulo>     move a base           (ex.: b 60)
  h <ângulo>     move a escotilha      (ex.: h 90)
  d <classe>     ciclo completo de descarte (ex.: d plastico)
  ciclo <n>      repete n descartes por todas as classes (ensaio repetitivo)
  q              sai
Anote os ângulos que alinham cada compartimento e coloque em actuators.bin_angles.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lixeira.actuators import build_sorter  # noqa: E402
from lixeira.config import load_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.pi.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    s = build_sorter(cfg)
    labels = list(cfg["actuators"]["bin_angles"])
    print(__doc__.split("Comandos:")[1])
    s.home()
    try:
        while True:
            parts = input("> ").split()
            if not parts:
                continue
            cmd = parts[0]
            if cmd == "q":
                break
            try:
                if cmd == "b":
                    s._rotate(float(parts[1]))
                elif cmd == "h":
                    s.hatch.set_angle(float(parts[1]))
                    s.hatch_open = float(parts[1]) != cfg["actuators"]["hatch"]["closed_angle"]
                elif cmd == "d":
                    t = time.perf_counter()
                    s.dispose(parts[1])
                    print(f"ciclo em {(time.perf_counter() - t) * 1000:.0f} ms")
                elif cmd == "ciclo":
                    for i in range(int(parts[1])):
                        for lab in labels:
                            print(f"[{i + 1}] {lab}")
                            s.dispose(lab)
                else:
                    print("comando desconhecido")
            except (IndexError, ValueError, KeyError, RuntimeError) as e:
                print("erro:", e)
    finally:
        s.close()


if __name__ == "__main__":
    main()
