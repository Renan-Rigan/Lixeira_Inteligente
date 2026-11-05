"""Montagem do dataset no formato do YOLO classify:

    data/dataset/
      train/<classe>/*.jpg
      val/<classe>/*.jpg
      test/<classe>/*.jpg

As fontes podem ser fotos suas (collect.py) ou um banco de imagens pronto
(TrashNet, RealWaste, Kaggle...). Pastas com outros nomes são traduzidas por um
mapeamento (ex.: "PET" -> plastico, "Food Organics" -> resto, "*" -> resto).

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


def resolve_label(name: str, mapping: dict[str, str] | None,
                  classes=None) -> str | None:
    """Traduz o nome de uma pasta para uma classe do projeto (None = ignorar).

    Ordem: mapeamento explícito (exato, depois sem diferenciar maiúsculas) ->
    o nome já é uma classe do projeto (sem diferenciar maiúsculas) -> curinga
    "*" -> ignora. Um mapeamento
    vazio (ex.: "glass=") descarta a pasta. Sem mapping nem classes, o nome da
    pasta é a classe."""
    if not mapping and classes is None:
        return name
    mapping = mapping or {}
    lower = {k.lower(): v for k, v in mapping.items()}
    if name in mapping:
        return mapping[name] or None
    if name.lower() in lower:
        return lower[name.lower()] or None
    if classes is not None:
        by_lower = {c.lower(): c for c in classes}
        if name.lower() in by_lower:          # "Metal" -> metal
            return by_lower[name.lower()]
    return mapping.get("*") or None


def scan_source(root: Path, mapping: dict[str, str] | None, train_only: bool,
                tag: str, classes=None) -> list[Item]:
    """Uma pasta por classe. `mapping` traduz nomes (ex.: plastic -> plastico);
    veja resolve_label. Pastas que não viram nenhuma classe são ignoradas."""
    items = []
    for cls_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        name = cls_dir.name
        label = resolve_label(name, mapping, classes)
        if not label:
            continue
        for f in sorted(cls_dir.rglob("*")):
            if f.suffix.lower() in EXTS:
                items.append(Item(f, label, group_of(f, tag), train_only))
    return items


def cap_per_class(items: list[Item], max_per_class: int, seed: int = 42) -> list[Item]:
    """Limita cada classe a ~max_per_class imagens, sorteando OBJETOS inteiros.

    Útil quando várias pastas de um banco viram "resto" e ele fica muito maior
    que metal e plastico: o modelo aprenderia a chutar "resto". Fotos suas
    (não train_only) têm prioridade: só imagens de bancos públicos são cortadas."""
    if not max_per_class:
        return items
    rng = random.Random(seed)
    by_label: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_label[it.label].append(it)
    out: list[Item] = []
    for label in sorted(by_label):
        own = [it for it in by_label[label] if not it.train_only]
        pub = [it for it in by_label[label] if it.train_only]
        out += own
        room = max_per_class - len(own)
        if room <= 0 or not pub:
            continue
        groups: dict[str, list[Item]] = defaultdict(list)
        for it in pub:
            groups[it.group].append(it)
        keys = sorted(groups)
        rng.shuffle(keys)
        for g in keys:
            if room <= 0:
                break
            out += groups[g]
            room -= len(groups[g])
    return out


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


def _copy(src: Path, dst: Path, max_side: int) -> None:
    """Copia; com max_side, reduz imagens grandes (fotos de 12 MP de celular
    viram ~640 px: o treino fica mais rápido e o modelo só vê imgsz mesmo)."""
    if max_side:
        import cv2

        img = cv2.imread(str(src))
        if img is not None and max(img.shape[:2]) > max_side:
            f = max_side / max(img.shape[:2])
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst.with_suffix(".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return
    shutil.copy2(src, dst)


def materialize(splits: dict[str, list[Item]], out_dir: Path, clean: bool = True,
                max_side: int = 0) -> dict:
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
            _copy(it.path, dst, max_side)
            counts.setdefault(split, {}).setdefault(it.label, 0)
            counts[split][it.label] += 1
    return counts
