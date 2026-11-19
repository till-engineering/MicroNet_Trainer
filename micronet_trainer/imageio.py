"""Bilder und Masken lesen / schreiben – wie Pore_Detection (RGB, uint8 / uint16, Umlaute im Pfad)."""

import os

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 40))   # sehr große Aufnahmen erlauben

from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def read_image(path: Path) -> np.ndarray:
    """RGB (H, W, 3) uint8 / uint16 wie in Pore_Detection (Grau -> 3 Kanäle, Alpha weg)."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        image = tifffile.imread(path, key=0)            # erste Seite; tifffile liefert RGB-Reihenfolge
        if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[2] not in (3, 4):
            image = np.moveaxis(image, 0, -1)          # planar gespeichert (C, H, W)
        bgr = False
    else:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"kann {path.name} nicht lesen")
        bgr = True
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"{path.name}: Pixeltyp {image.dtype} nicht unterstützt (nur 8 und 16 bit)")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] >= 4:
        image = cv2.cvtColor(image[..., :4], cv2.COLOR_BGRA2RGB) if bgr else image[..., :3]
    elif bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(image)


def read_mask(path: Path, threshold: int = 127) -> np.ndarray:
    """Weiß = Pore (wie Pore_Detection metrics.read_mask): Farbmasken hellster Kanal, 16 bit -> 0..255."""
    mask = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"kann {Path(path).name} nicht lesen")
    if mask.ndim == 3:
        mask = mask[..., :3].max(axis=2)
    if mask.dtype == np.uint16:
        mask = mask // 257
    return mask > threshold


def write_png(path: Path, array: np.ndarray) -> None:
    """Verlustfrei, Bittiefe bleibt (8 / 16 bit), auch bei Umlauten im Pfad. RGB erwartet."""
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, data = cv2.imencode(".png", array)
    if not ok:
        raise ValueError(f"kann {Path(path).name} nicht schreiben")
    data.tofile(str(path))
