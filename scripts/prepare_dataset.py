"""Monta data/dataset/{train,val,test} a partir das suas fotos e/ou de um banco de imagens.

Fontes:
  --source DIR       uma subpasta por classe; é dividida em treino/val/teste
  --train-only DIR   entra só no treino (banco público quando você TEM fotos próprias
                     para testar: o teste fica com a sua câmera e a sua luz)
  --map NOME=CLASSE  traduz pastas de um banco para as classes do projeto (vale
                     para todas as fontes). "*=resto" manda as pastas que sobrarem
                     para "resto"; "glass=" descarta a pasta. Pastas que já têm o
                     nome de uma classe (metal, Metal, resto...) não precisam de --map.

Exemplos:
  # só fotos tiradas com collect.py
  python scripts/prepare_dataset.py --source data/raw

  # só bancos de imagens (ainda sem fotos próprias): eles são divididos em
  # treino/val/teste. A acurácia sai otimista — confirme com o celular (live.py).
  python scripts/prepare_dataset.py \\
      --source data/bancos/drinking_waste --source data/bancos/realwaste \\
      --map PET=plastico --map AluCans=metal --map HDPEM= --map Plastic= \\
      --map "*=resto" --max-per-class 1500 --max-side 640 --check-quality

  # banco no treino + fotos suas para val/teste (o recomendado)
  python scripts/prepare_dataset.py --source data/raw \\
      --train-only data/bancos/realwaste \\
      --map Plastic= --map "*=resto" --max-per-class 1500 --check-quality

  # "plastico" aqui é garrafa PET. Pastas de plástico genérico (sacolas, potes)
  # foram descartadas (Plastic=) para não ensinar o modelo que PET é "resto".

  # + imagens revisadas que a lixeira capturou em uso (data/inbox, após corrigir)
  python scripts/prepare_dataset.py --source data/raw --source data/inbox
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lixeira.config import load_config  # noqa: E402
from lixeira.dataset import cap_per_class, materialize, scan_source, split_items  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml", help="de onde vêm as classes e os limiares de qualidade")
    ap.add_argument("--source", action="append", default=[], help="pasta com uma subpasta por classe (vai para treino/val/teste)")
    ap.add_argument("--train-only", action="append", default=[], help="pasta que entra só no treino (banco público)")
    ap.add_argument("--map", action="append", default=[], help="nome_da_pasta=classe (\"*=resto\" para o que sobrar)")
    ap.add_argument("--max-per-class", type=int, default=0, help="limita imagens de banco por classe (equilibra o 'resto')")
    ap.add_argument("--max-side", type=int, default=0, help="reduz imagens maiores que isso (ex.: 640)")
    ap.add_argument("--check-quality", action="store_true", help="deixa de fora imagens reprovadas no check_images")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    classes = cfg["classes"]
    mapping = dict(m.split("=", 1) for m in args.map) or None
    items = []
    for i, src in enumerate(args.source):
        items += scan_source(Path(src), mapping, False, f"s{i}", classes)
    for i, src in enumerate(args.train_only):
        items += scan_source(Path(src), mapping, True, f"pub{i}", classes)
    if not items:
        sys.exit("Nenhuma imagem encontrada. As pastas têm nomes de classes ou um --map?")

    unknown = sorted({it.label for it in items} - set(classes))
    if unknown:
        sys.exit(f"Classes que não estão em {args.config}: {unknown}. Corrija o --map.")

    if args.check_quality:
        from lixeira.quality import check_image, find_duplicates, quality_params

        params = quality_params(cfg)
        bad = set()
        for it in items:
            if check_image(it.path, params).status == "erro":
                bad.add(it.path)
        dups = find_duplicates([(it.path, it.label) for it in items])
        for group in dups:
            if len({lab for _, lab in group}) > 1:      # mesma imagem, classes diferentes
                bad.update(p for p, _ in group)
            else:                                        # cópia repetida: fica só uma
                bad.update(p for p, _ in group[1:])
        before = len(items)
        items = [it for it in items if it.path not in bad]
        print(f"Qualidade: {before - len(items)} de {before} imagens deixadas de fora "
              "(detalhes: python scripts/check_images.py)")

    items = cap_per_class(items, args.max_per_class, args.seed)
    totals = Counter(it.label for it in items)
    print("Imagens por classe:", dict(sorted(totals.items())))
    if totals and max(totals.values()) > 3 * min(totals.values()):
        print("AVISO: classes muito desequilibradas — considere --max-per-class.")

    splits = split_items(items, args.val, args.test, args.seed)
    counts = materialize(splits, Path(args.out), max_side=args.max_side)
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    for split in ("val", "test"):
        if not counts.get(split):
            print(f"AVISO: '{split}' vazio — use --source com pelo menos 3 objetos diferentes por classe.")
    if args.train_only and not args.source:
        print("AVISO: sem --source, val/teste ficam vazios. Para treinar só com o banco, use-o em --source.")
    print(f"Dataset pronto em {args.out}")


if __name__ == "__main__":
    main()
