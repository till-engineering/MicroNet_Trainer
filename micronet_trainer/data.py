"""Daten: Paare finden, in den Cache legen, prüfen, Split (nach Gruppe, je Art geschichtet), Patch-Strom.

Cache (data.cache): je Paar <key>_img.npy (RGB uint8/uint16), <key>_mask.npy (uint8 0/1),
<key>_pores.npz (je Pore: Fläche + bis zu 64 Pixel zum Ziehen), <key>.json (Größe, Porenpixel; zuletzt
geschrieben = Eintrag vollständig). Die Worker lesen per Memory-Map: die Daten liegen nur einmal im RAM.
"""

import hashlib
import json
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from . import augment, compat, imageio

MAX_PIXELS_PER_PORE = 64


@dataclass
class Record:
    name: str               # relativer Pfad ohne Endung, "/" getrennt (eindeutig je Teil)
    image: str
    mask: str
    subset: str             # erster Unterordner ("" = keiner) -> Auswertungsgruppe im Test
    group: str = ""         # Split-Gruppe
    kind: str = ""          # Art (Regel, Poren / negativ) -> Schichtung des Splits
    height: int = 0
    width: int = 0
    pore_px: int = 0
    n_pores: int = 0
    cache: str = ""         # Präfix der Cache-Dateien

    @property
    def negative(self) -> bool:
        return self.pore_px == 0

    @property
    def stem(self) -> str:
        return self.name.split("/")[-1]


# ---------------------------------------------------------------- Paare

def _files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in imageio.EXTENSIONS)


def pair(images_dir: Path, masks_dir: Path) -> tuple[list[Record], list[dict]]:
    """Bild + Maske mit gleichem relativen Pfad ohne Endung."""
    for folder in (images_dir, masks_dir):
        if not folder.is_dir():
            raise SystemExit(f"Ordner fehlt: {folder}")
    skipped, masks = [], {}
    for path in _files(masks_dir):
        key = path.relative_to(masks_dir).with_suffix("").as_posix()
        if key in masks:
            skipped.append({"file": str(path), "reason": "zweite Maske mit gleichem Namen"})
            continue
        masks[key] = path
    records, seen = [], set()
    for path in _files(images_dir):
        key = path.relative_to(images_dir).with_suffix("").as_posix()
        if key in seen:
            skipped.append({"file": str(path), "reason": "zweites Bild mit gleichem Namen"})
            continue
        seen.add(key)
        if key not in masks:
            skipped.append({"file": str(path), "reason": "keine Maske"})
            continue
        parts = key.split("/")
        records.append(Record(key, str(path), str(masks[key]), parts[0] if len(parts) > 1 else ""))
    skipped += [{"file": str(m), "reason": "kein Bild"} for key, m in masks.items() if key not in seen]
    return records, skipped


# ---------------------------------------------------------------- Cache

def _file_id(path: str) -> str:
    stat = os.stat(path)
    return f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}"


def pore_index(mask: np.ndarray, seed: int):
    """Je Pore (8er-Nachbarschaft): Fläche und bis zu MAX_PIXELS_PER_PORE Pixel (y, x).
    Pore gleichverteilt ziehen = jede Pore gleich oft; nach Fläche ziehen = wie ein zufälliges Porenpixel."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    area = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    if n <= 1:
        return area, np.zeros(1, np.int64), np.zeros((0, 2), np.int32)
    ys, xs = np.nonzero(labels)
    labs = labels[ys, xs]
    order = np.argsort(labs, kind="stable")
    ys, xs, labs = ys[order], xs[order], labs[order]
    starts = np.searchsorted(labs, np.arange(1, n))
    take = np.minimum(area, MAX_PIXELS_PER_PORE)
    rng = np.random.default_rng(seed)
    index = np.concatenate([starts[i] + rng.integers(area[i], size=take[i]) for i in range(n - 1)])
    offsets = np.concatenate([[0], np.cumsum(take)]).astype(np.int64)
    return area, offsets, np.stack([ys[index], xs[index]], axis=1).astype(np.int32)


def _cache_one(record: Record, cache_dir: Path, threshold: int) -> str | None:
    """Legt den Cache an (falls nötig) und füllt Größe / Porenpixel. Rückgabe: Grund zum Überspringen."""
    key = hashlib.sha1(f"{_file_id(record.image)}|{_file_id(record.mask)}|{threshold}".encode()).hexdigest()[:20]
    prefix = cache_dir / key
    meta_path = prefix.with_suffix(".json")
    if not meta_path.is_file():
        image = imageio.read_image(record.image)
        mask = imageio.read_mask(record.mask, threshold)
        if mask.shape != image.shape[:2]:
            return f"Maske {mask.shape[1]} x {mask.shape[0]} px, Bild {image.shape[1]} x {image.shape[0]} px"
        for suffix, array in (("_img", image), ("_mask", mask.astype(np.uint8))):
            tmp = Path(f"{prefix}{suffix}.tmp.npy")
            np.save(tmp, array)
            os.replace(tmp, f"{prefix}{suffix}.npy")
        area, offsets, pixels = pore_index(mask, int(key[:8], 16))
        tmp = Path(f"{prefix}_pores.tmp.npz")
        np.savez(tmp, area=area, offsets=offsets, pixels=pixels)
        os.replace(tmp, f"{prefix}_pores.npz")
        meta = {"height": int(mask.shape[0]), "width": int(mask.shape[1]),
                "pore_px": int(np.count_nonzero(mask)), "n_pores": int(len(area))}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    record.height, record.width = meta["height"], meta["width"]
    record.pore_px, record.n_pores = meta["pore_px"], meta["n_pores"]
    record.cache = str(prefix)
    return None


def build_cache(records: list[Record], cache_dir: Path, threshold: int, console, label: str):
    cache_dir.mkdir(parents=True, exist_ok=True)
    ok, skipped = [], []
    console.trial_start(f"Cache {label}", len(records))
    with ThreadPoolExecutor(4) as pool:
        for i, (record, reason) in enumerate(zip(records, pool.map(
                lambda r: _safe_cache(r, cache_dir, threshold), records)), 1):
            if reason:
                skipped.append({"file": record.image, "reason": reason})
            else:
                ok.append(record)
            console.trial_update(i, f"{i}/{len(records)} Bilder")
    return ok, skipped


def _safe_cache(record, cache_dir, threshold):
    try:
        return _cache_one(record, cache_dir, threshold)
    except Exception as error:       # noqa: BLE001 – kaputte Datei: überspringen, nicht abbrechen
        return f"{type(error).__name__}: {error}"


def load_image(record: Record) -> np.ndarray:
    return np.load(record.cache + "_img.npy", mmap_mode="r")


def load_mask(record: Record) -> np.ndarray:
    return np.load(record.cache + "_mask.npy") > 0


# ---------------------------------------------------------------- Gruppen, Prüfung, Split

def assign_groups(records: list[Record], rules: list[str]) -> None:
    compiled = [re.compile(rule) for rule in rules]
    for record in records:
        for i, rule in enumerate(compiled, 1):
            match = rule.search(record.stem)
            if match:
                record.group = f"Regel {i}: {match.group(1) if match.groups() else match.group(0)}"
                origin = f"Regel {i}"
                break
        else:
            record.group = f"Bild: {record.name}"
            origin = "ohne Regel"
        record.kind = f"{origin}, {'negativ' if record.negative else 'Poren'}"


def pore_stats(records: list[Record]) -> str:
    shares = np.array([r.pore_px / (r.height * r.width) for r in records if not r.negative])
    negatives = sum(r.negative for r in records)
    if not len(shares):
        return f"{len(records)} Bilder, alle ohne Pore"
    return (f"{len(records)} Bilder ({len(shares)} mit Poren, {negatives} negativ); Porenanteil der Porenbilder: "
            f"Median {100 * np.median(shares):.2f} %, min {100 * shares.min():.3f} %, max {100 * shares.max():.1f} %, "
            f"{int(np.sum(shares < 0.001))} Bilder unter 0,1 %")


def kinds(records: list[Record]) -> str:
    counts = defaultdict(int)
    for r in records:
        counts[r.kind] += 1
    return ", ".join(f"{k}: {n}" for k, n in sorted(counts.items()))


def check(train: list[Record], test: list[Record]) -> list[str]:
    """Warnungen (Lecks, fehlende Arten). Fehler, die ein Training unmöglich machen: SystemExit."""
    warnings = []
    if sum(not r.negative for r in train) < 2:
        raise SystemExit("train: mindestens 2 Bilder mit Poren nötig")
    rule_groups = {r.group for r in train if r.group.startswith("Regel")}
    leaks = sorted(rule_groups & {r.group for r in test})
    if leaks:
        warnings.append(f"LECK: {len(leaks)} Aufnahme(n) liegen in train UND test, z. B. {', '.join(leaks[:5])} "
                        f"– Ausschnitte einer Aufnahme gehören zusammen in train oder test")
    same = sorted({r.stem for r in train} & {r.stem for r in test})
    if same:
        warnings.append(f"gleiche Dateinamen in train und test: {', '.join(same[:5])}"
                        f"{' …' if len(same) > 5 else ''} ({len(same)}) – dasselbe Bild?")
    if not any(r.negative for r in train):
        warnings.append("train: keine Negativbilder – Phantome werden auf val nicht gemessen (lambda wirkungslos)")
    if not test:
        warnings.append("test: keine Bilder – keine Abschluss-Evaluation")
    elif not any(not r.negative for r in test):
        warnings.append("test: keine Bilder mit Poren")
    return warnings


def split(records: list[Record], val_share: float, seed: int) -> tuple[list[Record], list[Record]]:
    """Ganze Gruppen nach val, je Art ~val_share der Bilder; Art einer Gruppe = Regel + 'Poren', wenn
    irgendein Bild der Gruppe Poren hat."""
    rng = random.Random(seed)
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        groups[r.group].append(r)
    by_kind: dict[str, list[str]] = defaultdict(list)
    for name, members in groups.items():
        origin = members[0].kind.split(",")[0]
        by_kind[f"{origin}, {'Poren' if any(not m.negative for m in members) else 'negativ'}"].append(name)
    val_groups = set()
    for kind in sorted(by_kind):
        names = sorted(by_kind[kind])
        if len(names) < 2:
            continue
        rng.shuffle(names)
        target = val_share * sum(len(groups[n]) for n in names)
        count = 0
        for name in names[:-1]:                     # mindestens eine Gruppe bleibt in train
            if count >= target and count > 0:
                break
            val_groups.add(name)
            count += len(groups[name])
    return [r for r in records if r.group not in val_groups], [r for r in records if r.group in val_groups]


def save_split(path: Path, train: list[Record], val: list[Record], test: list[Record]) -> None:
    def names(rs):
        return sorted(r.name for r in rs)
    path.write_text(yaml.safe_dump({"train": names(train), "val": names(val), "test": names(test)},
                                   sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_split(path: Path, records: list[Record]) -> tuple[list[Record], list[Record]]:
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    by_name = {r.name: r for r in records}
    wanted = set(saved["train"]) | set(saved["val"])
    if wanted != set(by_name):
        missing, new = len(wanted - set(by_name)), len(set(by_name) - wanted)
        raise SystemExit(f"die Trainingsdaten haben sich seit dem Start dieses Laufs geändert ({missing} fehlen, "
                         f"{new} neu) – neuen Lauf starten: python run.py --new")
    return [by_name[n] for n in saved["train"]], [by_name[n] for n in saved["val"]]


def subset(records: list[Record], count: int, seed: int) -> list[Record]:
    """Kleine Auswahl, die jede Art enthält (Testlauf)."""
    rng = random.Random(seed)
    by_kind = defaultdict(list)
    for r in records:
        by_kind[r.kind].append(r)
    for members in by_kind.values():
        rng.shuffle(members)
    chosen, lists = [], [by_kind[k] for k in sorted(by_kind)]
    while len(chosen) < count and any(lists):
        for members in lists:
            if members and len(chosen) < count:
                chosen.append(members.pop())
    return chosen


def to_dicts(records):
    return [asdict(r) for r in records]


# ---------------------------------------------------------------- Patch-Strom

class PatchStream(torch.utils.data.IterableDataset):
    """Endloser Strom augmentierter, normierter Trainings-Patches (Bild (3, P, P), Maske (1, P, P)).

    Bild nach image_weighting; mit pore_share auf eine Pore (davon pore_object_share: Pore gleichverteilt,
    sonst nach Fläche = zufälliges Porenpixel), die Pore an zufälliger Stelle im Patch; sonst irgendwo.
    Negativbilder liefern nur zufällige Patches. Picklebar (Windows-Worker starten per spawn).
    """

    def __init__(self, records: list[Record], params: dict, augment_cfg: dict, recipe: dict, patch_px: int, seed):
        self.items = [(r.cache, r.n_pores) for r in records]
        area = np.array([r.height * r.width for r in records], dtype=np.float64)
        weights = {"area": area, "sqrt_area": np.sqrt(area), "equal": np.ones_like(area)}[params["image_weighting"]]
        self.cdf = np.cumsum(weights / weights.sum())
        self.pore_share = float(params["pore_share"])
        self.object_share = float(params["pore_object_share"])
        self.strength = float(params["augment_strength"])
        self.augment_cfg = augment_cfg
        self.norm = {"mean": recipe["mean"], "std": recipe["std"]}
        self.size = patch_px
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([*np.atleast_1d(self.seed).tolist(), info.id if info else 0])
        opened: dict = {}
        while True:
            yield self.patch(rng, opened)

    def _open(self, index: int, opened: dict):
        if index not in opened:
            prefix, n_pores = self.items[index]
            image = np.load(prefix + "_img.npy", mmap_mode="r")
            mask = np.load(prefix + "_mask.npy", mmap_mode="r")
            pores = None
            if n_pores:
                with np.load(prefix + "_pores.npz") as z:
                    area = z["area"].astype(np.float64)
                    pores = (np.cumsum(area / area.sum()), z["offsets"], z["pixels"])
            opened[index] = (image, mask, pores)
        return opened[index]

    def patch(self, rng: np.random.Generator, opened: dict):
        index = min(int(np.searchsorted(self.cdf, rng.random(), side="right")), len(self.items) - 1)
        image, mask, pores = self._open(index, opened)
        height, width = mask.shape
        size = self.size
        if pores is not None and rng.random() < self.pore_share:
            area_cdf, offsets, pixels = pores
            n = len(area_cdf)
            if rng.random() < self.object_share:
                c = int(rng.integers(n))
            else:
                c = min(int(np.searchsorted(area_cdf, rng.random(), side="right")), n - 1)
            y, x = pixels[rng.integers(offsets[c], offsets[c + 1])]
            y0 = int(np.clip(y - rng.integers(size), 0, max(height - size, 0)))
            x0 = int(np.clip(x - rng.integers(size), 0, max(width - size, 0)))
        else:
            y0, x0 = int(rng.integers(max(height - size, 0) + 1)), int(rng.integers(max(width - size, 0) + 1))
        crop = np.array(image[y0:y0 + size, x0:x0 + size])
        crop_mask = np.array(mask[y0:y0 + size, x0:x0 + size]) > 0
        if crop.shape[:2] != (size, size):                  # Bild kleiner als ein Patch: gespiegelt
            extra_y, extra_x = size - crop.shape[0], size - crop.shape[1]
            crop = compat.pad(crop, 0, extra_y, 0, extra_x)
            crop_mask = compat.pad(crop_mask, 0, extra_y, 0, extra_x)
        unit, crop_mask = augment.apply(compat.to_unit(crop), crop_mask, self.augment_cfg, rng, strength=self.strength)
        return (torch.from_numpy(compat.normalize(unit, self.norm)),
                torch.from_numpy(crop_mask[None].astype(np.float32)))


def make_loader(stream: PatchStream, batch_size: int, workers: int, cuda: bool):
    return torch.utils.data.DataLoader(stream, batch_size=batch_size, num_workers=workers, pin_memory=cuda,
                                       prefetch_factor=4 if workers else None, persistent_workers=bool(workers))
