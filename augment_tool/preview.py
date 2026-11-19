"""Vorschau der Augmentierung: zeigt, was das Training aus den Bildern macht.

    python augment_tool/preview.py                        Einstellungen: augment_tool/augment_config.yaml
    python augment_tool/preview.py --config andere.yaml

Aus jedem Bild in input/ wird ein Ausschnitt in Patch-Größe geschnitten, wie im Training (liegt eine
Maske in input/masks/, auf einer Pore, an zufälliger Stelle). Geschrieben werden nach output/:

    <Bild>_einzeln.png      eine Zeile je Augmentierung: Original | min | max | Zufallswerte
                            (min / max = Grenzen des range nach strength) – zum Einstellen jeder einzeln
    <Bild>_kombiniert.png   Patches mit ALLEN eingeschalteten Augmentierungen, wie das Training sie erzeugt
                            (über jedem Patch: was angewendet wurde)
    augment_config_used.yaml

Benutzt micronet_trainer/augment.py – dieselbe Augmentierung wie das Training. Gespeichert als PNG
in voller Größe, damit Rauschen, Unschärfe und JPEG-Artefakte so zu sehen sind, wie das Netz sie sieht.
"""

import argparse
import shutil
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from micronet_trainer import augment, compat, imageio  # noqa: E402

DEFAULT_CONFIG = HERE / "augment_config.yaml"
GAP_PX = 6
LINE_PX = 18
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
MASK_COLOR = (255, 0, 255)
FLIP_VARIANTS = [(1, False), (2, False), (0, True), (3, True), (1, True), (3, False), (2, True)]


# ---------------------------------------------------------------- Einstellungen und Daten

def load_settings(path: Path) -> dict:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("input", "masks", "output"):
        folder = Path(settings["preview"][key])
        settings["preview"][key] = (folder if folder.is_absolute() else path.parent / folder).resolve()
    augment.check(settings["augment"])
    return settings


def find_mask(folder: Path, stem: str) -> Path | None:
    if not folder.is_dir():
        return None
    for path in imageio.find_images(folder):
        if path.stem == stem:
            return path
    return None


def crop(unit: np.ndarray, mask: np.ndarray, size: int, rng: np.random.Generator):
    """Ausschnitt wie im Training: mit Pore an zufälliger Stelle, sonst irgendwo; zu klein = gespiegelt."""
    height, width = mask.shape
    pores = np.argwhere(mask)
    if len(pores):
        y, x = pores[rng.integers(len(pores))]
        y0 = int(np.clip(y - rng.integers(size), 0, max(height - size, 0)))
        x0 = int(np.clip(x - rng.integers(size), 0, max(width - size, 0)))
    else:
        y0, x0 = int(rng.integers(max(height - size, 0) + 1)), int(rng.integers(max(width - size, 0) + 1))
    image, patch_mask = unit[y0:y0 + size, x0:x0 + size], mask[y0:y0 + size, x0:x0 + size]
    if image.shape[:2] != (size, size):
        extra_y, extra_x = size - image.shape[0], size - image.shape[1]
        image = compat.pad(image, 0, extra_y, 0, extra_x)
        patch_mask = compat.pad(patch_mask, 0, extra_y, 0, extra_x)
    return np.ascontiguousarray(image), np.ascontiguousarray(patch_mask)


# ---------------------------------------------------------------- Bilder zusammensetzen

def to_rgb8(image: np.ndarray, mask: np.ndarray, show_mask: bool) -> np.ndarray:
    rgb = (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)
    if show_mask and mask.any():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, MASK_COLOR, 1)
    return rgb


def tile(rgb: np.ndarray, text: str, lines: int) -> np.ndarray:
    width = rgb.shape[1]
    chars = max(10, int(width / (19 * FONT_SCALE)))
    wrapped = textwrap.wrap(text, chars)[:lines] or [""]
    header = np.full((lines * LINE_PX + 8, width, 3), 250, np.uint8)
    for i, line in enumerate(wrapped):
        cv2.putText(header, line, (4, 16 + i * LINE_PX), FONT, FONT_SCALE, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([header, rgb])


def sheet(rows: list[list[np.ndarray]]) -> np.ndarray:
    height = max(t.shape[0] for row in rows for t in row)
    width = max(t.shape[1] for row in rows for t in row)
    columns = max(len(row) for row in rows)
    out = np.full((len(rows) * (height + GAP_PX) + GAP_PX, columns * (width + GAP_PX) + GAP_PX, 3), 215, np.uint8)
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            y, x = GAP_PX + r * (height + GAP_PX), GAP_PX + c * (width + GAP_PX)
            out[y:y + t.shape[0], x:x + t.shape[1]] = t
    return out


# ---------------------------------------------------------------- die zwei Vorschauen

def single(image, mask, cfg, preview, seed) -> np.ndarray:
    """Eine Zeile je Augmentierung: Original, min, max und Zufallswerte – jede für sich allein."""
    strength = cfg["strength"]
    show = preview["show_mask"]
    count = 2 + preview["random_values"]
    rows = []
    for index, name in enumerate(augment.ORDER):
        entry = cfg.get(name)
        if entry is None or (preview["only_enabled"] and not entry["enabled"]):
            continue
        state = "an" if entry["enabled"] else "AUS"
        p = "" if name == "flip_rotate" else f", p {entry['probability']}"
        head = f"{name} ({state}{p})"
        cells = [tile(to_rgb8(image, mask, show), f"{head}: Original", 2)]
        rng = np.random.default_rng([seed, index])
        if name == "flip_rotate":
            values = [(v, "") for v in FLIP_VARIANTS[:count]]
        elif name == "grayscale":
            values = [(True, "")]
        elif name == "tint":
            lo, hi = augment.value_range(name, cfg, strength)
            colors = [[float(c) for c in color] for color in entry["colors"]]
            values = [((hi, color), f"max Farbe {i}") for i, color in enumerate(colors, 1)]
            values += [((lo, colors[0]), "min")]
            values += [(augment.sample(name, cfg, rng, strength), "zufällig") for _ in range(preview["random_values"])]
        else:
            lo, hi = augment.value_range(name, cfg, strength)
            values = [(lo, "min"), (hi, "max")]
            values += [(augment.sample(name, cfg, rng, strength), "zufällig") for _ in range(preview["random_values"])]
        for value, label in values:
            out_image, out_mask = augment.apply_one(name, image.copy(), mask.copy(), value, rng)
            text = f"{name} {label} {augment.describe(name, value)}".replace("  ", " ")
            cells.append(tile(to_rgb8(out_image, out_mask, show), text, 2))
        rows.append(cells)
    return sheet(rows)


def combined(unit, full_mask, cfg, preview, seed) -> np.ndarray:
    """Neue Ausschnitte mit allen eingeschalteten Augmentierungen, wie im Training."""
    rng = np.random.default_rng([seed, 1000])
    cells = []
    for _ in range(preview["combined"]):
        image, mask = crop(unit, full_mask, preview["patch_px"], rng)
        log = []
        image, mask = augment.apply(image, mask, cfg, rng, log=log)
        text = ", ".join(f"{n} {augment.describe(n, v)}" for n, v in log) or "keine"
        cells.append(tile(to_rgb8(image, mask, preview["show_mask"]), text, 4))
    columns = max(1, preview["columns"])
    return sheet([cells[i:i + columns] for i in range(0, len(cells), columns)])


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Einstellungen")
    args = parser.parse_args()
    config_path = args.config.resolve()
    try:
        settings = load_settings(config_path)
    except (ValueError, KeyError) as error:
        sys.exit(f"Fehler in {config_path.name}: {error}")
    preview, cfg = settings["preview"], settings["augment"]
    if not preview["input"].is_dir():
        sys.exit(f"Ordner fehlt: {preview['input']}")
    files = imageio.find_images(preview["input"])[:preview["max_images"]]
    if not files:
        sys.exit(f"keine Bilder in {preview['input']}")
    out = preview["output"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, out / "augment_config_used.yaml")

    for number, path in enumerate(files):
        seed = preview["seed"] * 100_003 + number
        original = imageio.read_image(path)
        mask_path = find_mask(preview["masks"], path.stem)
        full_mask = imageio.read_mask(mask_path) if mask_path else np.zeros(original.shape[:2], bool)
        if full_mask.shape != original.shape[:2]:
            print(f"  {path.name}: Maske hat eine andere Größe – ohne Maske")
            full_mask = np.zeros(original.shape[:2], bool)
        unit = compat.to_unit(original)
        image, mask = crop(unit, full_mask, preview["patch_px"], np.random.default_rng([seed, 0]))
        imageio.write_png(out / f"{path.stem}_einzeln.png", single(image, mask, cfg, preview, seed))
        imageio.write_png(out / f"{path.stem}_kombiniert.png", combined(unit, full_mask, cfg, preview, seed))
        print(f"  {path.name}{'' if mask_path else ' (ohne Maske)'} -> {path.stem}_einzeln.png, "
              f"{path.stem}_kombiniert.png")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
