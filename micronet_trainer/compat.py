"""Kopie der U-Net-Ein-/Ausgabe aus Pore_Detection (pore_detection/core/unet_model.py, Stand 2026-11).

Diese Funktionen MÜSSEN identisch zu Pore_Detection bleiben, sonst sieht das Modell im Training
bzw. im Label-Tool etwas anderes als im Pipeline-Lauf:

    original (H, W, 3) RGB, uint8 / uint16, unverändert (kein Filter, keine Skalierung)
      -> to_unit():   0..1 (8 bit / 255, 16 bit / 65535)
      -> normalize(): (x - mean) / std je Kanal (Statistik des Encoder-Vortrainings)
      -> Fenster patch_px x patch_px, Überlappung overlap_px; nur die Mitte jedes Fensters zählt,
         der Bildrand wird gespiegelt (BORDER_REFLECT_101)
      -> Netz -> Sigmoid -> Porenwahrscheinlichkeit 0..1 je Pixel

Ein Modellordner enthält weights.pt (state dict) und model.yaml (Rezept: encoder, mean, std,
patch_px, overlap_px, threshold, ...). torch und segmentation_models_pytorch werden erst bei
Bedarf importiert.
"""

from pathlib import Path

import cv2
import numpy as np
import yaml

MODEL_FILE = "model.yaml"
WEIGHTS_FILE = "weights.pt"
MULTIPLE_PX = 32            # 5 Halbierungen im Encoder: Eingangsgrößen sind Vielfache von 32
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# MicroNet-Gewichte der NASA (Stuckner et al. 2022): ResNet50-Encoder, vortrainiert auf ~110 000
# Schliffbildern, Eingang RGB 0..1 mit ImageNet-Statistik normalisiert
MICRONET_URLS = {
    "resnet50": "https://huggingface.co/jstuckner/microscopy-resnet50-micronet/resolve/main/"
                "resnet50_micronet_weights.pth",
}


# ---------------------------------------------------------------- Eingang

def to_unit(original: np.ndarray) -> np.ndarray:
    """RGB uint8 / uint16 -> float32 0..1 (über den vollen Wertebereich des Datentyps)."""
    scale = 255.0 if original.dtype == np.uint8 else 65535.0
    return original.astype(np.float32) * np.float32(1.0 / scale)


def normalize(unit: np.ndarray, recipe: dict) -> np.ndarray:
    """(H, W, 3) float32 0..1 -> (3, H, W) float32, normalisiert mit mean / std des Rezepts."""
    mean = np.asarray(recipe["mean"], dtype=np.float32)
    std = np.asarray(recipe["std"], dtype=np.float32)
    return ((unit - mean) / std).transpose(2, 0, 1).copy()


def pad(array: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """Rand spiegeln (BORDER_REFLECT_101), auch über die Bildgröße hinaus."""
    while top or bottom or left or right:
        height, width = array.shape[:2]
        step = [min(v, n - 1) for v, n in ((top, height), (bottom, height), (left, width), (right, width))]
        if array.dtype == bool:
            array = cv2.copyMakeBorder(array.view(np.uint8), *step, cv2.BORDER_REFLECT_101).view(bool)
        else:
            array = cv2.copyMakeBorder(array, *step, cv2.BORDER_REFLECT_101)
        top, bottom, left, right = (v - s for v, s in zip((top, bottom, left, right), step))
    return array


# ---------------------------------------------------------------- Modell

def _libraries():
    try:
        import segmentation_models_pytorch as smp
        import torch
    except ImportError as error:
        raise ImportError(f"das U-Net braucht PyTorch und segmentation_models_pytorch ({error}); "
                          f"siehe requirements.txt") from None
    return torch, smp


def build(recipe: dict, pretrained: str | None = None):
    """Netz eines Rezepts. pretrained: None (Gewichte kommen aus weights.pt), imagenet oder micronet."""
    torch, smp = _libraries()
    if recipe["architecture"] != "unet":
        raise ValueError(f"unbekannte Architektur {recipe['architecture']!r} (nur unet)")
    encoder = recipe["encoder"]
    model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained == "imagenet" else None,
                     in_channels=3, classes=1, activation=None)
    if pretrained == "micronet":
        if encoder not in MICRONET_URLS:
            raise ValueError(f"MicroNet-Gewichte nur für {', '.join(MICRONET_URLS)}, nicht {encoder}")
        state = torch.hub.load_state_dict_from_url(MICRONET_URLS[encoder], map_location="cpu", progress=True)
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}  # Klassifikationskopf
        model.encoder.load_state_dict(state)
    elif pretrained not in (None, "imagenet"):
        raise ValueError(f"unbekannte Vortrainings-Gewichte {pretrained!r} (imagenet | micronet)")
    return model


def read_recipe(model_dir: Path) -> dict:
    path = Path(model_dir) / MODEL_FILE
    if not path.is_file():
        raise FileNotFoundError(f"kein U-Net-Modell in {model_dir} ({MODEL_FILE} fehlt)")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_recipe(model_dir: Path, recipe: dict) -> None:
    text = yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True)
    (Path(model_dir) / MODEL_FILE).write_text(text, encoding="utf-8")


def save(model_dir: Path, model, recipe: dict) -> None:
    torch, _ = _libraries()
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / WEIGHTS_FILE)
    write_recipe(model_dir, recipe)


def load(model_dir: Path, device: str):
    """(Rezept, Netz im eval-Modus auf `device`)."""
    torch, _ = _libraries()
    recipe = read_recipe(model_dir)
    model = build(recipe)
    state = torch.load(Path(model_dir) / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return recipe, model.to(device).eval()


# ---------------------------------------------------------------- Vorhersage

def windows(height: int, width: int, recipe: dict) -> tuple[list[tuple[int, int]], int, int]:
    """Fensteranfänge im gepolsterten Bild und die Polsterung (unten, rechts) über overlap_px hinaus."""
    patch, overlap = recipe["patch_px"], recipe["overlap_px"]
    step = patch - 2 * overlap
    rows, cols = -(-height // step), -(-width // step)
    starts = [(y * step, x * step) for y in range(rows) for x in range(cols)]
    return starts, rows * step - height, cols * step - width


def predict(model, recipe: dict, original: np.ndarray, device: str, batch_size: int = 8) -> np.ndarray:
    """Porenwahrscheinlichkeit (H, W) float32 0..1 eines RGB-Originals. GPU: float16 (autocast);
    GPU-Speicher voll -> Batch wird halbiert."""
    torch, _ = _libraries()
    height, width = original.shape[:2]
    patch, overlap = recipe["patch_px"], recipe["overlap_px"]
    step = patch - 2 * overlap
    starts, extra_y, extra_x = windows(height, width, recipe)
    data = normalize(to_unit(pad(original, overlap, overlap + extra_y, overlap, overlap + extra_x)), recipe)
    result = np.empty((height + extra_y, width + extra_x), dtype=np.float32)
    cuda = device == "cuda"
    done = 0
    with torch.inference_mode():
        while done < len(starts):
            batch = starts[done:done + batch_size]
            stack = np.stack([data[:, y:y + patch, x:x + patch] for y, x in batch])
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                    logits = model(torch.from_numpy(stack).to(device))
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                batch_size //= 2
                continue
            prob = torch.sigmoid(logits.float())[:, 0, overlap:overlap + step, overlap:overlap + step]
            for (y, x), tile in zip(batch, prob.cpu().numpy()):
                result[y:y + step, x:x + step] = tile
            done += len(batch)
    return result[:height, :width]
