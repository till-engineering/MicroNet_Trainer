"""Augmentierung der Trainings-Patches – dieselbe für das Training und für augment_tool/preview.py.

Arbeitet auf einem Patch: Bild (H, W, 3) float32 RGB 0..1 (vor der Normierung) und Maske (H, W) bool.
Geometrische Änderungen wirken auf beide, alle anderen nur auf das Bild.

Jede Augmentierung ist einzeln einstellbar (Abschnitt augment: der Config):
    enabled       an / aus
    probability   Wahrscheinlichkeit je Patch (flip_rotate: immer, wenn an)
    range         [min, max] des zufälligen Werts (gleichverteilt; gamma und scale logarithmisch)

    name          Wert                                            neutral
    flip_rotate   Drehung 0/90/180/270° + Spiegeln (8 Varianten)  –
    scale         Vergrößerungsfaktor (Bild + Maske)             1
    hue           Verschiebung des Farbtons, Anteil am Farbkreis  0      (0.03 = ±11°)
    saturation    Faktor der Sättigung                            1
    grayscale     Bild wird grau (kein range)                     –
    brightness    Faktor der Helligkeit                           1
    contrast      Faktor des Kontrasts um den Patch-Mittelwert    1
    gamma         Exponent (andere Kamerakurve)                   1
    gradient      Helligkeitsunterschied über den Patch           0      (0.3 = ±15 %, Richtung zufällig)
    vignette      Abdunklung zum Rand der Optik                   0      (0.3 = bis -30 %, Mitte zufällig)
    blur          Gauß-Unschärfe sigma in px                      0
    noise         Standardabweichung Gauß-Rauschen (0..1)         0
    jpeg          JPEG-Qualität                                   100

Reihenfolge: wie in der Tabelle (Geometrie, Farbe, Belichtung, Optik, Sensor, Speicherung).

strength (global, fürs Tuning): skaliert bei allen Augmentierungen außer flip_rotate, scale und grayscale
den Abstand der Bereichsgrenzen zum neutralen Wert (1 = wie eingestellt, 0 = alles neutral,
2 = doppelt so stark). gamma: im Logarithmus.
"""

import math

import cv2
import numpy as np

# name -> (neutral, logarithmisch, von strength betroffen)
SPEC = {
    "flip_rotate": (None, False, False),
    "scale": (1.0, True, False),
    "hue": (0.0, False, True),
    "saturation": (1.0, False, True),
    "grayscale": (None, False, False),
    "brightness": (1.0, False, True),
    "contrast": (1.0, False, True),
    "gamma": (1.0, True, True),
    "gradient": (0.0, False, True),
    "vignette": (0.0, False, True),
    "blur": (0.0, False, True),
    "noise": (0.0, False, True),
    "jpeg": (100.0, False, True),
}
ORDER = list(SPEC)


# ---------------------------------------------------------------- Werte

def value_range(name: str, cfg: dict, strength: float = 1.0) -> tuple[float, float] | None:
    """[min, max] nach strength; None für Augmentierungen ohne Bereich."""
    neutral, log, scaled = SPEC[name]
    if neutral is None:
        return None
    lo, hi = (float(v) for v in cfg[name]["range"])
    if scaled:
        if log:
            lo, hi = (math.exp(strength * math.log(v)) for v in (lo, hi))
        else:
            lo, hi = (neutral + strength * (v - neutral) for v in (lo, hi))
    return lo, hi


def sample(name: str, cfg: dict, rng: np.random.Generator, strength: float = 1.0):
    if name == "flip_rotate":
        return int(rng.integers(4)), bool(rng.random() < 0.5)
    if name == "grayscale":
        return True
    lo, hi = value_range(name, cfg, strength)
    if SPEC[name][1]:
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return float(rng.uniform(lo, hi))


def describe(name: str, value) -> str:
    if name == "flip_rotate":
        turns, flip = value
        return f"{90 * turns}°{' + Spiegel' if flip else ''}"
    if name == "grayscale":
        return "grau"
    if name == "jpeg":
        return f"Q {value:.0f}"
    if name == "hue":
        return f"{value * 360:+.1f}°"
    return f"{value:.3g}"


# ---------------------------------------------------------------- einzelne Augmentierungen

def _flip_rotate(image, mask, value, rng):
    turns, flip = value
    image, mask = np.rot90(image, turns), np.rot90(mask, turns)
    if flip:
        image, mask = image[:, ::-1], mask[:, ::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def _scale(image, mask, value, rng):
    """Bild + Maske skalieren, danach wieder auf die Patch-Größe (Mitte ausschneiden / Rand spiegeln)."""
    height, width = mask.shape
    size = (max(1, round(width * value)), max(1, round(height * value)))
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA if value < 1 else cv2.INTER_LINEAR)
    mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)

    def fit(array):
        h, w = array.shape[:2]
        if h < height or w < width:
            dy, dx = max(height - h, 0), max(width - w, 0)
            source = array.view(np.uint8) if array.dtype == bool else array
            source = cv2.copyMakeBorder(source, dy // 2, dy - dy // 2, dx // 2, dx - dx // 2, cv2.BORDER_REFLECT_101)
            array = source.view(bool) if array.dtype == bool else source
            h, w = array.shape[:2]
        y0, x0 = (h - height) // 2, (w - width) // 2
        return np.ascontiguousarray(array[y0:y0 + height, x0:x0 + width])

    return fit(image), fit(mask)


def _hsv(image, hue_shift=0.0, saturation=1.0):
    hsv = cv2.cvtColor(np.clip(image, 0, 1), cv2.COLOR_RGB2HSV)   # float: H 0..360, S und V 0..1
    hsv[..., 0] = (hsv[..., 0] + hue_shift * 360) % 360
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _hue(image, mask, value, rng):
    return _hsv(image, hue_shift=value), mask


def _saturation(image, mask, value, rng):
    return _hsv(image, saturation=value), mask


def _grayscale(image, mask, value, rng):
    gray = image @ np.array([0.299, 0.587, 0.114], np.float32)
    return np.repeat(gray[..., None], 3, axis=2), mask


def _brightness(image, mask, value, rng):
    return image * np.float32(value), mask


def _contrast(image, mask, value, rng):
    mean = image.mean()
    return (image - mean) * np.float32(value) + mean, mask


def _gamma(image, mask, value, rng):
    return np.power(np.clip(image, 0, 1), np.float32(value)), mask


def _gradient(image, mask, value, rng):
    """Linearer Helligkeitsverlauf in zufälliger Richtung: 1 - value/2 ... 1 + value/2."""
    height, width = mask.shape
    angle = rng.uniform(0, 2 * np.pi)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = (x - width / 2) * np.cos(angle) + (y - height / 2) * np.sin(angle)
    ramp /= max(np.abs(ramp).max(), 1)
    return image * (1 + ramp * np.float32(value / 2))[..., None], mask


def _vignette(image, mask, value, rng):
    """Radiale Abdunklung; die Mitte der Optik liegt zufällig (der Patch ist nur ein Teil des Bilds)."""
    height, width = mask.shape
    cy, cx = rng.uniform(-0.5, 1.5) * height, rng.uniform(-0.5, 1.5) * width
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) / math.hypot(height, width)
    return image * (1 - np.float32(value) * np.clip(r, 0, 1) ** 2)[..., None], mask


def _blur(image, mask, value, rng):
    if value < 0.1:
        return image, mask
    return cv2.GaussianBlur(image, (0, 0), value, borderType=cv2.BORDER_REFLECT_101), mask


def _noise(image, mask, value, rng):
    return image + rng.normal(0, value, image.shape).astype(np.float32), mask


def _jpeg(image, mask, value, rng):
    data = cv2.cvtColor((np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", data, [cv2.IMWRITE_JPEG_QUALITY, int(round(value))])
    decoded = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return decoded.astype(np.float32) / 255, mask


FUNCTIONS = {
    "flip_rotate": _flip_rotate, "scale": _scale, "hue": _hue, "saturation": _saturation,
    "grayscale": _grayscale, "brightness": _brightness, "contrast": _contrast, "gamma": _gamma,
    "gradient": _gradient, "vignette": _vignette, "blur": _blur, "noise": _noise, "jpeg": _jpeg,
}


# ---------------------------------------------------------------- anwenden

def check(cfg: dict) -> None:
    """Fehlende / unbekannte Einträge früh melden."""
    unknown = set(cfg) - set(ORDER) - {"strength"}
    if unknown:
        raise ValueError(f"augment: unbekannte Einträge {', '.join(sorted(unknown))}")
    for name in ORDER:
        if name not in cfg:
            raise ValueError(f"augment: {name} fehlt")
        needed = {"enabled"} | ({"probability"} if name != "flip_rotate" else set()) | \
                 ({"range"} if SPEC[name][0] is not None else set())
        missing = needed - set(cfg[name])
        if missing:
            raise ValueError(f"augment.{name}: {', '.join(sorted(missing))} fehlt")


def apply_one(name: str, image: np.ndarray, mask: np.ndarray, value, rng: np.random.Generator):
    image, mask = FUNCTIONS[name](image, mask, value, rng)
    return np.clip(image, 0, 1).astype(np.float32, copy=False), mask


def apply(image: np.ndarray, mask: np.ndarray, cfg: dict, rng: np.random.Generator,
          strength: float | None = None, log: list | None = None):
    """Alle eingeschalteten Augmentierungen, jede mit ihrer Wahrscheinlichkeit. log: (name, Wert) je Schritt."""
    strength = cfg.get("strength", 1.0) if strength is None else strength
    for name in ORDER:
        entry = cfg[name]
        if not entry["enabled"]:
            continue
        if name != "flip_rotate" and rng.random() >= entry["probability"]:
            continue
        value = sample(name, cfg, rng, strength)
        image, mask = FUNCTIONS[name](image, mask, value, rng)
        if log is not None:
            log.append((name, value))
    return np.clip(image, 0, 1).astype(np.float32, copy=False), mask
