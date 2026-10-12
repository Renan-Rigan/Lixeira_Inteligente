"""Monta data/dataset/{train,val,test} a partir das suas fotos e de datasets públicos.

Exemplos:
  # só as fotos tiradas com collect.py
  python scripts/prepare_dataset.py --source data/raw

  # + TrashNet (pastas em inglês) só no treino, traduzindo os nomes
  python scripts/prepare_dataset.py --source data/raw \\
      --train-only data/public/trashnet \\
      --map plastic=plastico --map paper=papel --map cardboard=papel --map metal=metal

  # + imagens revisadas que a lixeira capturou em uso (data/inbox, após corrigir)
  python scripts/prepare_dataset.py --source data/raw --source data/inbox
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lixeira.dataset import materialize, scan_source, split_items  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", default=[], help="pasta com uma subpasta por classe (vai para treino/val/teste)")
    ap.add_argument("--train-only", action="append", default=[], help="pasta que entra só no treino (datasets públicos)")
    ap.add_argument("--map", action="append", default=[], help="nome_original=classe (vale para --train-only)")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    mapping = dict(m.split("=", 1) for m in args.map) or None
    items = []
    for i, src in enumerate(args.source):
        items += scan_source(Path(src), None, False, f"s{i}")
    for i, src in enumerate(args.train_only):
        items += scan_source(Path(src), mapping, True, f"pub{i}")
    if not items:
        sys.exit("Nenhuma imagem encontrada.")

    counts = materialize(split_items(items, args.val, args.test, args.seed), Path(args.out))
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    for split in ("val", "test"):
        if not counts.get(split):
            print(f"AVISO: '{split}' vazio — fotografe pelo menos 3 objetos diferentes por classe.")
    print(f"Dataset pronto em {args.out}")


if __name__ == "__main__":
    main()
