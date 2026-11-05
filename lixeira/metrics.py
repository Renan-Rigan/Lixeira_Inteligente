"""Métricas de avaliação (só numpy — usadas por evaluate.py e analyze_events.py)."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str],
                     labels: Sequence[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def summarize(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str],
              reject_label: str | None = None,
              rejected: Sequence[bool] | None = None) -> dict:
    """Acurácia geral, taxa de rejeição e precisão/recall/F1 por classe.

    - accuracy: acertos / total (visão do usuário: o resíduo foi para o lugar certo?)
    - accuracy_accepted: acertos / itens que passaram do limiar de confiança
    - reject_rate: fração mandada para reject_label por falta de confiança

    `rejected` diz, item a item, se a decisão veio do limiar de confiança. É
    necessário quando reject_label também é uma classe (ex.: "resto"): aí não dá
    para saber pela previsão se foi "resto" com confiança ou por rejeição.
    Sem ele, conta como rejeitado tudo o que foi previsto como reject_label.
    """
    y_true, y_pred = list(y_true), list(y_pred)
    n = len(y_true)
    labels = list(classes) + ([reject_label] if reject_label and reject_label not in classes else [])
    cm = confusion_matrix(y_true, y_pred, labels)
    correct_flags = [t == p for t, p in zip(y_true, y_pred)]
    correct = sum(correct_flags)
    if rejected is None:
        rejected = [bool(reject_label) and p == reject_label for p in y_pred]
    rejected = list(rejected)
    n_rejected = sum(rejected)
    accepted = n - n_rejected
    correct_accepted = sum(c for c, r in zip(correct_flags, rejected) if not r)
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i, :].sum())}
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "accuracy_accepted": correct_accepted / accepted if accepted else 0.0,
        "reject_rate": n_rejected / n if n else 0.0,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def latency_stats(ms: Sequence[float]) -> dict:
    a = np.asarray(ms, dtype=float)
    if a.size == 0:
        return {}
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confiança 95% para uma proporção (acurácia). Com poucas
    amostras de teste isso mostra o quanto o '90%' é incerto — vale pôr no relatório."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def to_markdown(s: dict, title: str = "Avaliação") -> str:
    lines = [f"# {title}", "",
             f"- Amostras: **{s['n']}**",
             f"- Acurácia: **{s['accuracy']:.1%}**"]
    if "accuracy_ci95" in s:
        lo, hi = s["accuracy_ci95"]
        lines.append(f"- IC 95% da acurácia: {lo:.1%} – {hi:.1%}")
    lines += [f"- Acurácia entre os não rejeitados: {s['accuracy_accepted']:.1%}",
              f"- Taxa de rejeição (abaixo do limiar de confiança): {s['reject_rate']:.1%}", ""]
    if s.get("latency"):
        L = s["latency"]
        lines += [f"- Latência: média {L['mean_ms']:.1f} ms · p95 {L['p95_ms']:.1f} ms", ""]
    lines += ["## Por classe", "", "| classe | precisão | recall | F1 | n |", "|---|---|---|---|---|"]
    for c, m in s["per_class"].items():
        lines.append(f"| {c} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.1%} | {m['support']} |")
    labels = s["labels"]
    lines += ["", "## Matriz de confusão (linha = real, coluna = previsto)", "",
              "| real \\ previsto | " + " | ".join(labels) + " |",
              "|---" * (len(labels) + 1) + "|"]
    for lab, row in zip(labels, s["confusion_matrix"]):
        lines.append(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"
