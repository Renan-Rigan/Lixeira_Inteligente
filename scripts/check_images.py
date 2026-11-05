"""Verifica a qualidade das imagens de treino, tendo a câmera do Raspberry Pi como régua.

  python scripts/check_images.py data/raw
  python scripts/check_images.py data/bancos/realwaste data/bancos/drinking_waste
  python scripts/check_images.py data/dataset                     # já montado (train/val/test)
  python scripts/check_images.py data/raw --quarantine data/reprovadas   # tira as reprovadas

Cada pasta tem uma subpasta por classe (a classe é a pasta onde a imagem está).
Por padrão usa config.pi.yaml: a referência é o frame que o Pi entrega
(camera.width x camera.height, recortado na ROI) reduzido para classifier.imgsz.

Verifica: arquivo que não abre, resolução (menor que imgsz = erro; menor que a
câmera do Pi = aviso), nitidez (borrada), brilho (escura/estourada), contraste,
proporção e imagens repetidas — inclusive a MESMA imagem em classes diferentes
(rótulo conflitante) ou em treino e teste ao mesmo tempo (vazamento).

Gera reports/qualidade_<data>.md (resumo) e .csv (uma linha por imagem).
Os limiares ficam em `quality:` no config.yaml; use as medianas do relatório de
fotos tiradas com a sua câmera (collect.py) para calibrá-los.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lixeira.config import load_config  # noqa: E402
from lixeira.dataset import EXTS  # noqa: E402
from lixeira.quality import AVISO, ERRO, OK, check_image, find_duplicates, quality_params  # noqa: E402

SPLITS = {"train", "val", "test"}


def split_of(path: Path) -> str:
    return next((p for p in path.parts if p in SPLITS), "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="pastas com uma subpasta por classe")
    ap.add_argument("--config", default="config.pi.yaml", help="câmera de referência e limiares")
    ap.add_argument("--quarantine", help="move as imagens com ERRO para esta pasta (<pasta>/<classe>/)")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    params = quality_params(cfg)
    files = [(p, p.parent.name) for f in args.folders
             for p in sorted(Path(f).rglob("*")) if p.suffix.lower() in EXTS]
    if not files:
        sys.exit("Nenhuma imagem encontrada.")
    print(f"Verificando {len(files)} imagens (referência: {params.ref_size[0]}x{params.ref_size[1]} "
          f"do Pi -> modelo {cfg['classifier']['imgsz']}px)...")

    results = {}
    for i, (path, _) in enumerate(files, 1):
        results[path] = check_image(path, params)
        if i % 500 == 0:
            print(f"  {i}/{len(files)}")

    for group in find_duplicates(files):
        labels = {lab for _, lab in group}
        splits = {split_of(p) for p, _ in group} - {""}
        names = ", ".join(str(p) for p, _ in group[:4])
        for j, (path, _) in enumerate(group):
            r = results[path]
            if len(labels) > 1:
                r.add(ERRO, f"mesma imagem em classes diferentes: {names}")
            elif len(splits) > 1:
                r.add(ERRO, f"mesma imagem em {'/'.join(sorted(splits))} (vazamento): {names}")
            elif j > 0:
                r.add(AVISO, f"repetida ({len(group)} cópias): {group[0][0]}")

    # ---- relatório
    by_class: dict[str, list] = defaultdict(list)
    for path, label in files:
        by_class[label].append(results[path])
    problem_counts = Counter(p.split(" (")[0].split(":")[0] for r in results.values() for p in r.problems)

    def pct(vals, q):
        return float(np.percentile(vals, q)) if vals else 0.0

    md = [f"# Qualidade das imagens — {datetime.now():%d/%m/%Y %H:%M}", "",
          f"Referência ({args.config}): o Pi entrega **{params.ref_size[0]}x{params.ref_size[1]}** "
          f"(frame/ROI) e o modelo vê **{cfg['classifier']['imgsz']}x{cfg['classifier']['imgsz']}**. "
          f"Mínimo aceito: {params.min_side} px no lado menor. "
          f"Nitidez medida na escala do Pi (limiar {params.blur_threshold:.0f}).", "",
          "| classe | imagens | ok | aviso | erro | lado menor (mediana) | nitidez (p10 / mediana) | brilho (mediana) |",
          "|---|---|---|---|---|---|---|---|"]
    for label in sorted(by_class):
        rs = by_class[label]
        st = Counter(r.status for r in rs)
        valid = [r for r in rs if r.width]
        sides = [min(r.width, r.height) for r in valid]
        sharp = [r.sharpness for r in valid]
        md.append(f"| {label} | {len(rs)} | {st[OK]} | {st[AVISO]} | {st[ERRO]} | "
                  f"{pct(sides, 50):.0f} px | {pct(sharp, 10):.0f} / {pct(sharp, 50):.0f} | "
                  f"{pct([r.brightness for r in valid], 50):.0f} |")
    counts = [len(v) for v in by_class.values()]
    if len(counts) > 1 and max(counts) > 3 * min(counts):
        md += ["", f"**Classes desequilibradas** (de {min(counts)} a {max(counts)} imagens): "
               "use `--max-per-class` no prepare_dataset.py."]
    md += ["", "## Problemas encontrados", ""]
    md += [f"- {name}: {n}" for name, n in problem_counts.most_common()] or ["- nenhum"]
    bad = [r for r in results.values() if r.status == ERRO]
    md += ["", f"## Imagens com erro ({len(bad)})", ""]
    md += [f"- `{r.path}` — {'; '.join(r.problems)}" for r in bad[:40]] or ["- nenhuma"]
    if len(bad) > 40:
        md.append(f"- ... e mais {len(bad) - 40} (veja o .csv)")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    (out / f"qualidade_{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(out / f"qualidade_{stamp}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["imagem", "classe", "status", "largura", "altura", "nitidez", "brilho", "contraste", "problemas"])
        for path, label in files:
            r = results[path]
            w.writerow([path, label, r.status, r.width, r.height, f"{r.sharpness:.1f}",
                        f"{r.brightness:.1f}", f"{r.contrast:.1f}", "; ".join(r.problems)])
    print("\n".join(md))
    print(f"\nRelatório: {out}/qualidade_{stamp}.md e .csv")

    if args.quarantine and bad:
        q = Path(args.quarantine)
        label_of = dict(files)
        for r in bad:
            dst = q / label_of[r.path] / r.path.name
            n = 1
            while dst.exists():
                dst = dst.with_name(f"{r.path.stem}_{n}{r.path.suffix}")
                n += 1
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(r.path), dst)
        print(f"{len(bad)} imagens movidas para {q}/ (revise antes de apagar)")


if __name__ == "__main__":
    main()
