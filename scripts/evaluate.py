"""Avalia o modelo no split de TESTE (objetos que ele nunca viu).

  python scripts/evaluate.py --weights models/best.pt
  python scripts/evaluate.py --weights models/best_ncnn_model     # no Pi: mede latência real
  python scripts/evaluate.py --threshold 0.6                      # testa outro limiar

Gera reports/eval_<data>.md (tabela + matriz de confusão) e .json.
Também mostra como a acurácia e a rejeição mudam com o limiar de confiança —
use isso para escolher classifier.conf_threshold no config.yaml.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lixeira.classifier import YoloClassifier  # noqa: E402
from lixeira.config import load_config  # noqa: E402
from lixeira.dataset import EXTS  # noqa: E402
from lixeira.metrics import latency_stats, summarize, to_markdown, wilson_interval  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--weights")
    ap.add_argument("--split", default="data/dataset/test")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    c = cfg["classifier"]
    weights = args.weights or c["weights"]
    thr = c["conf_threshold"] if args.threshold is None else args.threshold
    clf = YoloClassifier(weights, c["imgsz"], c["device"])

    files = [(p, p.parent.name) for p in sorted(Path(args.split).rglob("*")) if p.suffix.lower() in EXTS]
    if not files:
        sys.exit(f"Nenhuma imagem em {args.split}")

    clf.predict_probs(cv2.imread(str(files[0][0])))  # aquecimento (1ª inferência é lenta)
    y_true, probs, lat = [], [], []
    for path, label in files:
        img = cv2.imread(str(path))
        t0 = time.perf_counter()
        probs.append(clf.predict_probs(img))
        lat.append((time.perf_counter() - t0) * 1000)
        y_true.append(label)
    probs = np.array(probs)
    best = probs.argmax(1)
    conf = probs.max(1)

    def preds_at(t):
        return [clf.names[b] if cf >= t else cfg["reject_label"] for b, cf in zip(best, conf)]

    s = summarize(y_true, preds_at(thr), clf.names, cfg["reject_label"])
    correct = round(s["accuracy"] * s["n"])
    s["accuracy_ci95"] = wilson_interval(correct, s["n"])
    s["latency"] = latency_stats(lat)
    s["threshold"] = thr
    s["weights"] = str(weights)
    s["threshold_sweep"] = []
    for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        st = summarize(y_true, preds_at(t), clf.names, cfg["reject_label"])
        s["threshold_sweep"].append({"threshold": t, "accuracy": st["accuracy"],
                                     "accuracy_accepted": st["accuracy_accepted"],
                                     "reject_rate": st["reject_rate"]})
    # erros mais confiantes: os primeiros que você deve olhar
    errors = [(float(cf), str(p), t, clf.names[b]) for (p, t), b, cf in zip(files, best, conf)
              if clf.names[b] != t]
    s["top_errors"] = sorted(errors, reverse=True)[:15]

    md = to_markdown(s, f"Avaliação — {Path(weights).name} (limiar {thr})")
    md += "\n## Limiar de confiança\n\n| limiar | acurácia | acurácia aceitos | rejeição |\n|---|---|---|---|\n"
    for r in s["threshold_sweep"]:
        md += f"| {r['threshold']:.1f} | {r['accuracy']:.1%} | {r['accuracy_accepted']:.1%} | {r['reject_rate']:.1%} |\n"
    md += "\n## Erros mais confiantes (revise essas imagens)\n\n"
    md += "\n".join(f"- {cf:.2f} previu **{pr}**, era **{t}** — `{p}`" for cf, p, t, pr in s["top_errors"]) or "- nenhum"

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    (out / f"eval_{stamp}.md").write_text(md + "\n", encoding="utf-8")
    (out / f"eval_{stamp}.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)
    print(f"\nRelatório salvo em {out}/eval_{stamp}.md")


if __name__ == "__main__":
    main()
