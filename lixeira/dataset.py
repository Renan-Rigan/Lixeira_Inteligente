"""Montagem do dataset no formato do YOLO classify:

    data/dataset/
      train/<classe>/*.jpg
      val/<classe>/*.jpg
      test/<classe>/*.jpg

Regras importantes para a acurácia medida ser honesta:
1. Divisão por OBJETO, não por foto: fotos do mesmo objeto (prefixo antes de "__")
   ficam todas no mesmo split. Senão o teste vê "quase a mesma foto" do treino e a
   acurácia sai inflada — e cai quando o protótipo encontra objetos novos.
2. Datasets públicos (TrashNet, etc.) entram só no TREINO (--train-only): o teste
   deve refletir a sua plataforma, câmera e iluminação.
"""
from __future__ import annotations

import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Item:
    path: Path
    label: str
    group: str        # id do objeto físico
    train_only: bool


def group_of(path: Path, source_tag: str) -> str:
    stem = path.stem
    base = stem.split("__")[0] if "__" in stem else stem
    return f"{source_tag}:{path.parent.name}:{base}"


def scan_source(root: Path, mapping: dict[str, str] | None, train_only: bool,
                tag: str) -> list[Item]:
    """Uma pasta por classe. `mapping` traduz nomes (ex.: plastic -> plastico);
    pastas sem mapeamento (quando há mapping) são ignoradas."""
    items = []
    for cls_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        name = cls_dir.name
        label = mapping.get(name) if mapping else name
        if not label:
            continue
        for f in sorted(cls_dir.rglob("*")):
            if f.suffix.lower() in EXTS:
                items.append(Item(f, label, group_of(f, tag), train_only))
    return items


def split_items(items: list[Item], val: float = 0.15, test: float = 0.15,
                seed: int = 42) -> dict[str, list[Item]]:
    """Divide por grupo e estratifica por classe."""
    rng = random.Random(seed)
    out: dict[str, list[Item]] = {"train": [], "val": [], "test": []}
    by_class: dict[str, dict[str, list[Item]]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        if it.train_only:
            out["train"].append(it)
        else:
            by_class[it.label][it.group].append(it)
    for label in sorted(by_class):
        groups = sorted(by_class[label])
        rng.shuffle(groups)
        n = len(groups)
        n_test = round(n * test) if n >= 3 else 0
        n_val = round(n * val) if n >= 3 else 0
        if n >= 3:
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        for i, g in enumerate(groups):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            out[split].extend(by_class[label][g])
    return out


def materialize(splits: dict[str, list[Item]], out_dir: Path, clean: bool = True) -> dict:
    out_dir = Path(out_dir)
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    counts: dict[str, dict[str, int]] = {}
    for split, items in splits.items():
        for it in items:
            dst_dir = out_dir / split / it.label
            dst_dir.mkdir(parents=True, exist_ok=True)
            tag = it.group.split(":")[0]
            dst = dst_dir / f"{tag}__{it.path.parent.name}__{it.path.name}"
            shutil.copy2(it.path, dst)
            counts.setdefault(split, {}).setdefault(it.label, 0)
            counts[split][it.label] += 1
    return counts
