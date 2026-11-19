"""Einstellungen lesen (configs/study.yaml), Pfade auflösen, prüfen; Testlauf (--smoke)."""

from pathlib import Path

import yaml

from . import augment


def _resolve(base: Path, value) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def load(path: Path) -> dict:
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent
    settings["config_path"] = path
    settings["output"] = _resolve(base, settings["output"])
    for part in ("train", "test"):
        for key in ("images", "masks"):
            settings["data"][part][key] = _resolve(base, settings["data"][part][key])
    settings["data"]["cache"] = _resolve(base, settings["data"]["cache"])
    reference = settings["evaluation"]["reference_model"]
    settings["evaluation"]["reference_model"] = _resolve(base, reference) if reference else None

    augment.check(settings["augment"])
    for name, spec in settings["search"].items():
        if name not in settings["training"]:
            raise ValueError(f"search.{name}: kein Parameter unter training:")
        for key in ("low", "high", "log"):
            if key not in spec:
                raise ValueError(f"search.{name}: {key} fehlt")
        if not isinstance(spec["low"], (int, float)) or not isinstance(spec["high"], (int, float)):
            raise ValueError(f"search.{name}: low / high sind keine Zahlen (Exponent mit Punkt: 1.0e-4)")
    for name, value in settings["training"].items():
        if isinstance(value, str) and name != "image_weighting":
            raise ValueError(f"training.{name} = {value!r} ist keine Zahl (Exponent mit Punkt: 1.0e-4)")
    if settings["training"]["image_weighting"] not in ("area", "sqrt_area", "equal"):
        raise ValueError("training.image_weighting: area | sqrt_area | equal")
    if settings["objective"]["aggregate"] not in ("mean", "p10"):
        raise ValueError("objective.aggregate: mean | p10")
    lambdas = settings["objective"]["phantom_lambdas"]
    if not lambdas:
        raise ValueError("objective.phantom_lambdas: mindestens ein Wert")
    settings["objective"]["phantom_lambdas"] = [float(v) for v in lambdas]
    m = settings["model"]
    if m["patch_px"] % 32 or not 0 <= 2 * m["overlap_px"] < m["patch_px"]:
        raise ValueError("model: patch_px muss ein Vielfaches von 32 und größer als 2 x overlap_px sein")
    settings["smoke"] = None
    return settings


def apply_smoke(settings: dict) -> None:
    """Testlauf: wenige Bilder, Mini-Epochen, 2 Trials, 1 Bestätigung – prüft alle Schritte in ~10 min."""
    settings["training"].update(steps_per_epoch=10, max_epochs=2, val_every=1, early_stopping=10, warmup_epochs=0)
    settings["study"].update(random_trials=1, pruner_startup_trials=5, max_trials=2, confirm_top=1, confirm_epochs=1)
    settings["budget"].update(max_hours=100, confirm_hours=1, final_hours=1)
    settings["objective"]["final_thresholds"] = {"start": 0.1, "stop": 0.9, "step": 0.1}
    settings["evaluation"]["bootstrap"] = 50
    settings["evaluation"]["worst_images"] = 2
    settings["smoke"] = {"train_images": 12, "test_images": 6}
