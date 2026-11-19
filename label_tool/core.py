"""Logik des Label-Tools ohne GUI: Aufnahmen laden, Modell rechnen (mit Cache), Vorschläge, Speichern.

Die Vorhersage nutzt micronet_trainer/compat.py = dieselbe Eingabe wie Pore_Detection.
Wahrscheinlichkeiten werden als uint8 (0..255) in cache/<Aufnahme>_<Kennung>.npy gespeichert und
per Memory-Map gelesen, damit auch sehr große Aufnahmen in den Speicher passen.
"""

import os

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 40))   # sehr große Aufnahmen erlauben

import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR.parent))

from micronet_trainer import compat, imageio  # noqa: E402

EXTENSIONS = imageio.EXTENSIONS
COARSE_MAX_PX = 3000          # Vorschläge werden auf einer Karte mit höchstens dieser Kantenlänge gesucht
MAX_COMPONENTS = 20_000       # mehr Porengruppen: Zufallsauswahl
CSV_FIELDS = ["file", "source", "y", "x", "height", "width", "pores", "pore_px", "negative",
              "edited_px", "kind", "threshold", "model", "saved"]


# ---------------------------------------------------------------- Einstellungen

def load_settings(path: Path) -> dict:
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key, folder in settings["paths"].items():
        folder = Path(folder)
        settings["paths"][key] = (folder if folder.is_absolute() else Path(path).parent / folder).resolve()
    for key in ("input", "model", "output", "cache"):
        settings["paths"][key].mkdir(parents=True, exist_ok=True)
    (settings["paths"]["output"] / "images").mkdir(exist_ok=True)
    (settings["paths"]["output"] / "masks").mkdir(exist_ok=True)
    return settings


def find_inputs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


# ---------------------------------------------------------------- Bilder

read_image = imageio.read_image     # RGB uint8 / uint16 wie in Pore_Detection


def to_display(rgb: np.ndarray, auto_contrast: bool) -> np.ndarray:
    """Nur zur Anzeige: RGB uint8, optional Kontrast gestreckt (0,5 / 99,5 % Perzentil)."""
    data = rgb.astype(np.float32)
    if auto_contrast:
        lo, hi = np.percentile(data[::4, ::4], (0.5, 99.5))
        if hi > lo:
            data = (data - lo) * (255.0 / (hi - lo))
    elif rgb.dtype == np.uint16:
        data = data / 257.0
    return np.ascontiguousarray(np.clip(data, 0, 255).astype(np.uint8))


def overview(image: np.ndarray, long_px: int, auto_contrast: bool) -> tuple[np.ndarray, float]:
    """Verkleinerte Übersicht (RGB uint8) und Faktor volle Auflösung / Übersicht."""
    height, width = image.shape[:2]
    factor = max(1.0, max(height, width) / long_px)
    size = (max(1, round(width / factor)), max(1, round(height / factor)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if factor > 1 else image
    return to_display(small, auto_contrast), factor


write_png = imageio.write_png       # verlustfrei, Bittiefe bleibt


# ---------------------------------------------------------------- Modell

class Model:
    """Altes Modell aus paths.model; ohne model.yaml / weights.pt arbeitet das Tool ohne Modell."""

    def __init__(self, folder: Path, settings: dict):
        self.folder = folder
        self.settings = settings
        self.available = (folder / compat.MODEL_FILE).is_file() and (folder / compat.WEIGHTS_FILE).is_file()
        self.recipe = compat.read_recipe(folder) if self.available else None
        self.device = None
        self._net = None

    @property
    def name(self) -> str:
        if not self.available:
            return "ohne Modell"
        trained = self.recipe.get("trained", "")
        return f"{self.folder.name} ({self.recipe['encoder']}, {trained})".replace(" ()", "")

    @property
    def threshold(self) -> float:
        value = self.settings["model"]["threshold"]
        if value == "auto":
            return float(self.recipe["threshold"]) if self.available else 0.5
        return float(value)

    def fingerprint(self) -> str:
        if not self.available:
            return "none"
        stat = (self.folder / compat.WEIGHTS_FILE).stat()
        return f"{stat.st_size}-{stat.st_mtime_ns}"

    def net(self):
        if self._net is None:
            torch, _ = compat._libraries()
            wanted = self.settings["model"]["device"]
            self.device = "cuda" if wanted == "cuda" or (wanted == "auto" and torch.cuda.is_available()) else "cpu"
            _, self._net = compat.load(self.folder, self.device)
        return self._net


def cache_path(settings: dict, source: Path, model: Model) -> Path:
    stat = source.stat()
    key = f"{source.name}|{stat.st_size}|{stat.st_mtime_ns}|{model.fingerprint()}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return settings["paths"]["cache"] / f"{source.stem}_{digest}.npy"


def predict_to_cache(model: Model, image: np.ndarray, path: Path, progress) -> None:
    """Ganze Aufnahme in Kacheln rechnen (je Kachel overlap_px echter Rand als Kontext)."""
    net = model.net()
    recipe, cfg = model.recipe, model.settings["model"]
    height, width = image.shape[:2]
    tile, margin = cfg["tile_px"], recipe["overlap_px"]
    tiles = [(y, x) for y in range(0, height, tile) for x in range(0, width, tile)]
    tmp = path.with_name(path.stem + "_tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(height, width))
    try:
        for index, (y, x) in enumerate(tiles):
            progress(index, len(tiles), f"Modell rechnet ({model.device}) – Kachel {index + 1}/{len(tiles)}")
            y1, x1 = min(y + tile, height), min(x + tile, width)
            ry, rx = max(y - margin, 0), max(x - margin, 0)
            region = image[ry:min(y1 + margin, height), rx:min(x1 + margin, width)]
            prob = compat.predict(net, recipe, region, model.device, cfg["batch_size"])
            out[y:y1, x:x1] = np.round(prob[y - ry:y1 - ry, x - rx:x1 - rx] * 255).astype(np.uint8)
        out.flush()
    finally:
        del out
    os.replace(tmp, path)


# ---------------------------------------------------------------- Fenster und Vorschläge

@dataclass
class Window:
    y: int
    x: int
    height: int
    width: int
    kind: str                   # Poren | leer | zufällig | manuell | gelabelt
    pore_px: int = 0            # erkannte Porenpixel (grob, für die Sortierung)
    uncertain_px: int = 0       # Pixel mit unsicherer Wahrscheinlichkeit (grob)

    def overlap(self, other: "Window") -> float:
        """Überlappung als Anteil der Fläche dieses Fensters."""
        dy = min(self.y + self.height, other.y + other.height) - max(self.y, other.y)
        dx = min(self.x + self.width, other.x + other.width) - max(self.x, other.x)
        return max(dy, 0) * max(dx, 0) / (self.height * self.width)


def _integral(array: np.ndarray) -> np.ndarray:
    total = np.zeros((array.shape[0] + 1, array.shape[1] + 1), np.int64)
    total[1:, 1:] = array.cumsum(0).cumsum(1)
    return total


def coarse_maps(prob: np.ndarray, threshold8: int, lo8: int, hi8: int, block: int):
    """Je Block x Block Pixel: Anzahl Porenpixel und unsicherer Pixel (streifenweise, wenig Speicher)."""
    height, width = prob.shape
    hb, wb = -(-height // block), -(-width // block)
    pores = np.zeros((hb, wb), np.int32)
    uncertain = np.zeros((hb, wb), np.int32)
    rows = block * max(1, 2048 // block)
    for r0 in range(0, height, rows):
        strip = np.asarray(prob[r0:r0 + rows])
        padded = np.zeros((-(-strip.shape[0] // block) * block, wb * block), np.uint8)   # 0 = keine Pore
        padded[:strip.shape[0], :width] = strip
        n = padded.shape[0] // block
        shape = (n, block, wb, block)
        pores[r0 // block:r0 // block + n] = (padded >= threshold8).reshape(shape).sum(axis=(1, 3))
        uncertain[r0 // block:r0 // block + n] = ((padded >= lo8) & (padded <= hi8)).reshape(shape).sum(axis=(1, 3))
    return pores, uncertain


def _accept(window: Window, taken: list[Window], max_overlap: float) -> bool:
    return all(window.overlap(other) <= max_overlap and other.overlap(window) <= max_overlap for other in taken)


def proposals(prob: np.ndarray | None, shape: tuple[int, int], threshold: float, settings: dict,
              labeled: list[Window]) -> list[Window]:
    """Fenster mit erkannten Poren (unsichere zuerst), die Pore an zufälliger Stelle im Fenster,
    dazu einige Fenster ohne erkannte Pore. Ohne Modell: zufällige Fenster."""
    cfg = settings["proposals"]
    rng = np.random.default_rng(cfg["seed"])
    height, width = shape
    win_h, win_w = min(settings["window_px"], height), min(settings["window_px"], width)
    taken = list(labeled)
    result: list[Window] = []

    def random_windows(count: int, kind: str, allowed=None):
        for _ in range(200 * count):
            if count <= 0:
                break
            y, x = int(rng.integers(height - win_h + 1)), int(rng.integers(width - win_w + 1))
            if allowed is not None and not allowed(y, x):
                continue
            window = Window(y, x, win_h, win_w, kind)
            if _accept(window, taken, cfg["max_overlap"]):
                taken.append(window)
                result.append(window)
                count -= 1

    if prob is None:
        random_windows(cfg["max"], "zufällig")
        return result

    block = max(1, math.ceil(max(height, width) / COARSE_MAX_PX))
    lo, hi = cfg["uncertain"]
    pores, uncertain = coarse_maps(prob, round(threshold * 255), max(1, round(lo * 255)), round(hi * 255), block)
    sum_pores, sum_uncertain = _integral(pores), _integral(uncertain)

    def window_sum(total: np.ndarray, y: int, x: int) -> int:
        y0, x0 = y // block, x // block
        y1, x1 = -(-(y + win_h) // block), -(-(x + win_w) // block)
        return int(total[y1, x1] - total[y0, x1] - total[y1, x0] + total[y0, x0])

    margin_y = min(cfg["edge_margin_px"], max(win_h // 2 - 1, 0))
    margin_x = min(cfg["edge_margin_px"], max(win_w // 2 - 1, 0))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((pores > 0).astype(np.uint8), connectivity=8)
    ids = np.arange(1, count)
    if len(ids) > MAX_COMPONENTS:
        ids = rng.choice(ids, MAX_COMPONENTS, replace=False)
    candidates = []
    for i in ids:
        left, top, w, h = stats[i, :4]
        cells = np.flatnonzero(labels[top:top + h, left:left + w] == i)
        cell = int(cells[rng.integers(len(cells))])
        cy = (top + cell // w) * block + int(rng.integers(block))      # ein Punkt der Porengruppe
        cx = (left + cell % w) * block + int(rng.integers(block))
        # die Pore liegt an zufälliger Stelle im Fenster, nicht immer in der Mitte
        y = int(np.clip(cy - rng.integers(margin_y, win_h - margin_y + 1), 0, height - win_h))
        x = int(np.clip(cx - rng.integers(margin_x, win_w - margin_x + 1), 0, width - win_w))
        candidates.append(Window(y, x, win_h, win_w, "Poren", window_sum(sum_pores, y, x),
                                 window_sum(sum_uncertain, y, x)))
    candidates.sort(key=lambda c: c.uncertain_px + 0.1 * c.pore_px, reverse=True)
    for window in candidates:
        if len(result) >= cfg["max"]:
            break
        if _accept(window, taken, cfg["max_overlap"]):
            taken.append(window)
            result.append(window)
    random_windows(cfg["empty_windows"], "leer", lambda y, x: window_sum(sum_pores, y, x) == 0)
    return result


def manual_window(cy: int, cx: int, shape: tuple[int, int], settings: dict, rng: np.random.Generator) -> Window:
    height, width = shape
    win_h, win_w = min(settings["window_px"], height), min(settings["window_px"], width)
    jitter = settings["proposals"]["manual_jitter_px"]
    cy += int(rng.integers(-jitter, jitter + 1)) if jitter else 0
    cx += int(rng.integers(-jitter, jitter + 1)) if jitter else 0
    y = int(np.clip(cy - win_h // 2, 0, height - win_h))
    x = int(np.clip(cx - win_w // 2, 0, width - win_w))
    return Window(y, x, win_h, win_w, "manuell")


# ---------------------------------------------------------------- Speichern

def label_name(source: Path, window: Window) -> str:
    """<Aufnahme>_y<Zeile>_x<Spalte>: im Training gehören alle Ausschnitte einer Aufnahme zusammen
    (group_pattern: ^.+(?=_y\\d{6}_x\\d{6}$))."""
    return f"{source.stem}_y{window.y:06d}_x{window.x:06d}"


def read_log(settings: dict) -> list[dict]:
    path = settings["paths"]["output"] / "labels.csv"
    if not path.is_file():
        return []
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=";"))


def labeled_windows(settings: dict, source: Path) -> list[Window]:
    return [Window(int(r["y"]), int(r["x"]), int(r["height"]), int(r["width"]), "gelabelt")
            for r in read_log(settings) if r["source"] == source.name]


def exists(settings: dict, source: Path, window: Window) -> bool:
    return (settings["paths"]["output"] / "masks" / f"{label_name(source, window)}.png").is_file()


def save(settings: dict, source: Path, window: Window, crop: np.ndarray, mask: np.ndarray,
         initial: np.ndarray, threshold: float, model_name: str) -> str:
    """Ausschnitt (Original-Bittiefe) + Maske (Weiß = Pore) als PNG, Zeile in labels.csv."""
    out = settings["paths"]["output"]
    name = label_name(source, window)
    write_png(out / "images" / f"{name}.png", crop)
    write_png(out / "masks" / f"{name}.png", (mask > 0).astype(np.uint8) * 255)
    pores = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)[0] - 1
    row = {
        "file": f"{name}.png", "source": source.name, "y": window.y, "x": window.x,
        "height": window.height, "width": window.width, "pores": pores,
        "pore_px": int(np.count_nonzero(mask)), "negative": "ja" if pores == 0 else "nein",
        "edited_px": int(np.count_nonzero((mask > 0) != (initial > 0))), "kind": window.kind,
        "threshold": round(threshold, 3), "model": model_name, "saved": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
    }
    rows = [r for r in read_log(settings) if r["file"] != row["file"]] + [row]
    tmp = out / "labels.csv.tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as file:    # utf-8-sig + ; : Excel liest es direkt
        writer = csv.DictWriter(file, CSV_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, out / "labels.csv")
    return name


# ---------------------------------------------------------------- Sitzung

@dataclass
class Opened:
    source: Path
    image: np.ndarray               # RGB uint8 / uint16, volle Auflösung
    prob: np.ndarray | None         # (H, W) uint8 0..255 (Memory-Map) oder None ohne Modell
    overview: np.ndarray            # RGB uint8
    factor: float                   # volle Auflösung / Übersicht
    proposals: list[Window]
    labeled: list[Window]


def open_source(settings: dict, model: Model, source: Path, threshold: float, progress) -> Opened:
    progress(0, 1, f"lade {source.name} …")
    image = read_image(source)
    prob = None
    if model.available:
        path = cache_path(settings, source, model)
        if not path.is_file():
            predict_to_cache(model, image, path, progress)
        prob = np.load(path, mmap_mode="r")
    progress(0, 1, "suche Vorschläge …")
    labeled = labeled_windows(settings, source)
    found = proposals(prob, image.shape[:2], threshold, settings, labeled)
    small, factor = overview(image, settings["display"]["overview_px"], settings["display"]["auto_contrast"])
    progress(1, 1, "fertig")
    return Opened(source, image, prob, small, factor, found, labeled)
