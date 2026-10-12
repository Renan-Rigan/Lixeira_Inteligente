"""Analisa os testes experimentais com o protótipo (seção 5.5): lê logs/events.csv
e calcula acurácia de classificação e tempos de resposta das sessões em que você
rodou `run.py --true-label <classe>`.

  python scripts/analyze_events.py
  python scripts/analyze_events.py --since 2026-11-01

A "taxa de acerto no direcionamento" (o resíduo caiu no compartimento certo?) é
mecânica e o software não enxerga: anote à mão e preencha a coluna true_label
com o compartimento onde ele REALMENTE caiu, se quiser medir as duas coisas.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lixeira.config import load_config  # noqa: E402
from lixeira.metrics import latency_stats, summarize, to_markdown, wilson_interval  # noqa: E402


def analyze(rows: list[dict], classes, reject_label) -> dict:
    rows = [r for r in rows if r.get("true_label")]
    y_true = [r["true_label"] for r in rows]
    y_pred = [r["label"] for r in rows]
    s = summarize(y_true, y_pred, classes, reject_label)
    s["accuracy_ci95"] = wilson_interval(round(s["accuracy"] * s["n"]), s["n"])
    s["latency"] = latency_stats([float(r["total_ms"]) for r in rows])
    s["inference"] = latency_stats([float(r["infer_ms"]) for r in rows])
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--events", default="logs/events.csv")
    ap.add_argument("--since", default="", help="AAAA-MM-DD")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    with open(args.events, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["timestamp"] >= args.since]
    s = analyze(rows, cfg["classes"], cfg["reject_label"])
    if not s["n"]:
        sys.exit("Nenhum evento com true_label. Rode run.py com --true-label <classe>.")
    print(to_markdown(s, "Testes experimentais do protótipo"))
    I = s["inference"]
    print(f"Inferência (só o modelo, {cfg['classifier']['votes']} frames): média {I['mean_ms']:.1f} ms")


if __name__ == "__main__":
    main()
