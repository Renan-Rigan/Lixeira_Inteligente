"""Verificação de qualidade das imagens de treino, tendo a câmera do Raspberry Pi
como referência.

O que o modelo realmente vê no Pi NÃO é a foto de 12 MP do sensor: o sistema
captura o stream em camera.width x camera.height (640x480 por padrão), recorta a
ROI e o YOLO reduz para classifier.imgsz (224x224). Então uma imagem de treino:

- menor que imgsz no lado menor precisa ser AMPLIADA -> fica mais borrada do que
  qualquer coisa que o Pi vai mostrar (erro);
- menor que a ROI do Pi, mas maior que imgsz, ainda serve (aviso);
- maior que a ROI é reduzida no treino e não ganha nada (use --max-side 640 no
  prepare_dataset.py para economizar disco e tempo).

A nitidez (variância do Laplaciano, nos blocos mais nítidos da imagem) é medida
depois de reduzir a imagem para a escala da câmera do Pi: assim uma foto de
celular de 4000 px e um frame de 640 px são comparados na mesma régua.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

OK, AVISO, ERRO = "ok", "aviso", "erro"


@dataclass
class QualityParams:
    min_side: int = 224          # abaixo disso a imagem é ampliada no treino
    ref_size: tuple[int, int] = (640, 480)   # (largura, altura) do que o Pi entrega (ROI)
    blur_threshold: float = 40
    dark_threshold: float = 40
    bright_threshold: float = 215
    contrast_threshold: float = 18
    max_aspect: float = 2.5


@dataclass
class QualityResult:
    path: Path
    width: int = 0
    height: int = 0
    sharpness: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    problems: list[str] = field(default_factory=list)
    status: str = OK

    def add(self, level: str, msg: str) -> None:
        self.problems.append(msg)
        if level == ERRO or self.status == OK:
            self.status = level


def quality_params(cfg: dict) -> QualityParams:
    """Monta os limiares a partir do config (de preferência o config.pi.yaml)."""
    q, cam = cfg["quality"], cfg["camera"]
    roi = cam.get("roi")
    ref = (int(roi[2]), int(roi[3])) if roi else (int(cam["width"]), int(cam["height"]))
    return QualityParams(
        min_side=int(q.get("min_side") or cfg["classifier"]["imgsz"]),
        ref_size=ref,
        blur_threshold=q["blur_threshold"], dark_threshold=q["dark_threshold"],
        bright_threshold=q["bright_threshold"], contrast_threshold=q["contrast_threshold"],
        max_aspect=q["max_aspect"],
    )


def to_reference_scale(img: np.ndarray, ref_size: tuple[int, int]) -> np.ndarray:
    """Reduz (nunca amplia) para o lado menor bater com o da câmera do Pi."""
    h, w = img.shape[:2]
    f = min(ref_size) / min(h, w)
    if f >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))),
                      interpolation=cv2.INTER_AREA)


def sharpness(gray: np.ndarray, grid: int = 8, q: float = 90) -> float:
    """Variância do Laplaciano medida em blocos (grid x grid), percentil q.

    Na imagem inteira, um fundo liso (TrashNet, bancada branca) puxa o valor
    para baixo mesmo com o objeto em foco. Olhando os blocos mais nítidos, a
    medida reflete o foco do OBJETO; uma foto tremida ou desfocada continua
    baixa em todos os blocos."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = gray.shape
    v = [lap[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid].var()
         for i in range(grid) for j in range(grid)]
    return float(np.percentile(v, q))


def measure(img: np.ndarray, ref_size: tuple[int, int]) -> tuple[float, float, float]:
    """(nitidez, brilho médio, contraste) na escala da câmera do Pi."""
    gray = cv2.cvtColor(to_reference_scale(img, ref_size), cv2.COLOR_BGR2GRAY)
    return sharpness(gray), float(gray.mean()), float(gray.std())


def check_array(img: np.ndarray, p: QualityParams, path: Path | str = "") -> QualityResult:
    r = QualityResult(Path(path))
    r.height, r.width = img.shape[:2]
    short, long_ = min(r.width, r.height), max(r.width, r.height)
    if short < p.min_side:
        r.add(ERRO, f"pequena ({r.width}x{r.height}; mínimo {p.min_side} px no lado menor)")
    elif short < min(p.ref_size):
        r.add(AVISO, f"abaixo da câmera do Pi ({r.width}x{r.height} < {p.ref_size[0]}x{p.ref_size[1]})")
    if long_ / short > p.max_aspect:
        r.add(AVISO, f"proporção {long_ / short:.1f}:1 (o corte central pode cortar o objeto)")
    r.sharpness, r.brightness, r.contrast = measure(img, p.ref_size)
    if r.sharpness < p.blur_threshold:
        r.add(ERRO, f"borrada (nitidez {r.sharpness:.0f} < {p.blur_threshold:.0f})")
    if r.brightness < p.dark_threshold:
        r.add(ERRO, f"escura (brilho {r.brightness:.0f})")
    elif r.brightness > p.bright_threshold:
        r.add(ERRO, f"clara demais (brilho {r.brightness:.0f})")
    if r.contrast < p.contrast_threshold:
        r.add(AVISO, f"pouco contraste ({r.contrast:.0f})")
    return r


def check_image(path: Path | str, p: QualityParams) -> QualityResult:
    img = cv2.imread(str(path))
    if img is None:
        r = QualityResult(Path(path))
        r.add(ERRO, "não abre (arquivo corrompido ou formato não suportado)")
        return r
    return check_array(img, p, path)


def dhash(img: np.ndarray) -> int:
    """Hash de diferença 8x8 (64 bits): quase igual para a mesma foto mesmo
    redimensionada ou recomprimida. Pega cópias repetidas entre pastas e bancos."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):          # numpy >= 2.0
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def find_duplicates(items: list[tuple[Path, str]],
                    max_distance: int = 3) -> list[list[tuple[Path, str]]]:
    """Grupos de imagens repetidas ou quase iguais (dHash a até `max_distance`
    bits de distância, de 64). Cada grupo: [(caminho, classe), ...]."""
    keep, hashes = [], []
    for path, label in items:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h = dhash(img)
        if h == 0 or h == (1 << 64) - 1:   # imagem lisa: hash não diz nada
            continue
        keep.append((path, label))
        hashes.append(h)
    if not hashes:
        return []
    arr = np.array(hashes, dtype=np.uint64)
    parent = list(range(len(arr)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(arr) - 1):
        close = np.nonzero(_popcount(arr[i + 1:] ^ arr[i]) <= max_distance)[0]
        for j in close + i + 1:
            parent[root(int(j))] = root(i)
    groups: dict[int, list[tuple[Path, str]]] = {}
    for i, it in enumerate(keep):
        groups.setdefault(root(i), []).append(it)
    return [g for g in groups.values() if len(g) > 1]
