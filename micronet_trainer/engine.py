"""Training eines Trials: Modell, Optimierer, Schleife, Validierung (ganze Bilder wie die Pipeline),
beste Gewichte je lambda, Checkpoints (fortsetzbar), Early Stopping, Pruning-Rückmeldung."""

import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import compat, data, losses, metrics, report

warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")


def lambda_tag(lam: float) -> str:
    return f"lambda{lam:g}"


def make_recipe(settings: dict) -> dict:
    m = settings["model"]
    return {
        "architecture": "unet",
        "encoder": m["encoder"],
        "pretrained": m["pretrained"],
        "input": "RGB original, unchanged (no filter, no scaling)",
        "value_range": "0..1 (8 bit / 255, 16 bit / 65535)",
        "mean": compat.IMAGENET_MEAN,
        "std": compat.IMAGENET_STD,
        "patch_px": m["patch_px"],
        "overlap_px": m["overlap_px"],
        "threshold": 0.5,
    }


@dataclass
class Run:
    """Alles, was ein Lauf braucht."""
    settings: dict
    folder: Path
    console: object
    clock: object
    device: str
    batch_size: int
    workers: int
    recipe: dict
    train: list = field(default_factory=list)
    val: list = field(default_factory=list)
    test: list = field(default_factory=list)
    gpu: object = None
    status: str = ""

    @property
    def cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def lambdas(self) -> list[float]:
        return self.settings["objective"]["phantom_lambdas"]

    @property
    def val_batch(self) -> int:
        return self.batch_size * 2 if self.cuda else 4

    def tick(self, force: bool = False) -> None:
        self.console.overall(self.clock.used(), self.status, force)


# ---------------------------------------------------------------- Bausteine

def make_optimizer(model, params: dict):
    encoder = list(model.encoder.parameters())
    ids = {id(p) for p in encoder}
    rest = [p for p in model.parameters() if id(p) not in ids]
    lr = float(params["learning_rate"])
    return torch.optim.AdamW([{"params": encoder, "lr": lr * float(params["encoder_lr_factor"])},
                              {"params": rest, "lr": lr}], weight_decay=float(params["weight_decay"]))


def lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min((step - warmup) / max(1, total - warmup), 1.0)
    return 0.5 * (1 + math.cos(math.pi * progress))


def validate(run: Run, model, records: list, thresholds, label: str = "Validierung") -> dict:
    """Zählwerte je Schwelle und Bild (metrics.quick), Vorhersage mit compat.predict wie im Pipeline-Lauf."""
    cfg = run.settings["evaluation"]
    rows = {float(t): [] for t in thresholds}
    model.eval()
    try:
        for i, record in enumerate(records, 1):
            image = np.array(data.load_image(record))
            gt = data.load_mask(record)
            prob = compat.predict(model, run.recipe, image, run.device, run.val_batch)
            info = metrics.components(gt, cfg["connectivity"])
            for t in rows:
                rows[t].append(metrics.quick(prob >= t, info, cfg))
            run.console.trial_update(run.console.trial.n if run.console.trial else 0, f"{label} {i}/{len(records)}")
            run.tick()
    finally:
        model.train()
    return rows


def scores(rows: dict, lambdas: list[float], aggregate: str) -> dict:
    """Je lambda: Zielmaß bei der besten Schwelle (+ Schwelle, Pixel-F1, Objekt-F1, Phantome/Mpx)."""
    result = {}
    for lam in lambdas:
        best = None
        for t, image_rows in rows.items():
            value = metrics.objective(image_rows, lam, aggregate)
            if best is None or value["score"] > best["score"]:
                best = {**value, "threshold": t}
        result[lam] = best
    return result


def _save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- ein Trial

def train_trial(run: Run, params: dict, folder: Path, max_epochs: int, seed: int, deadline: float,
                report_fn=None, label: str = "Trial") -> dict:
    """Trainiert bis max_epochs / Early Stopping / Pruning / deadline. Setzt aus folder/checkpoint.pt fort.

    Rückgabe: status (complete | early_stop | pruned | time), epochs, minutes, best {tag: {...}}, history.
    best[tag] enthält score, threshold, epoch, pixel_f1, object_f1, phantoms_per_mpx; die Gewichte liegen in
    folder/best_<tag>.pt.
    """
    s, t = run.settings, run.settings["training"]
    steps = int(t["steps_per_epoch"])
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if run.cuda:
        torch.backends.cudnn.benchmark = True

    model = compat.build(run.recipe, s["model"]["pretrained"]).to(run.device)
    optimizer = make_optimizer(model, params)
    total, warmup = max_epochs * steps, int(t["warmup_epochs"] * steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))
    scaler = torch.amp.GradScaler("cuda", enabled=run.cuda)
    state = {"epoch": 0, "history": [], "best": {}, "since_best": 0, "seconds": 0.0}

    checkpoint = folder / "checkpoint.pt"
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        run.console.log(f"{label}: setze nach Epoche {state['epoch']} fort")
        if report_fn:
            for row in state["history"]:
                key = f"score_{lambda_tag(run.lambdas[0])}"
                if key in row:
                    report_fn(row["epoch"], row[key], replay=True)

    main = lambda_tag(run.lambdas[0])
    thresholds = s["objective"]["thresholds"]
    status = "complete"
    started = time.time() - state["seconds"]
    loader = iterator = None
    try:
        if state["epoch"] < max_epochs:
            stream = data.PatchStream(run.train, params, s["augment"], run.recipe, s["model"]["patch_px"],
                                      [seed, state["epoch"]])
            loader = data.make_loader(stream, run.batch_size, run.workers, run.cuda)
            iterator = iter(loader)
        for epoch in range(state["epoch"] + 1, max_epochs + 1):
            if time.time() > deadline:
                status = "time"
                break
            model.train()
            losses_, pore, utils = [], [], []
            wait, bad = 0.0, 0
            epoch_start = time.perf_counter()
            run.console.trial_start(label, steps)
            best_text = f"bester {state['best'][main]['score']:.3f}" if main in state["best"] else ""
            finished_epoch = True
            for step in range(steps):
                t0 = time.perf_counter()
                images, masks = next(iterator)
                wait += time.perf_counter() - t0
                images = images.to(run.device, non_blocking=True)
                masks = masks.to(run.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=run.cuda):
                    logits = model(images)
                loss = losses.bce_dice(logits.float(), masks, float(params["bce_weight"]))
                if not torch.isfinite(loss):
                    bad += 1
                    if bad > 20:
                        raise FloatingPointError("Loss nicht endlich (NaN/inf) – Lernrate zu hoch?")
                    continue
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                losses_.append(loss.item())
                pore.append(masks.mean().item())
                if step % 10 == 0 or step == steps - 1:
                    util, memory = run.gpu.sample() if run.gpu else (None, None)
                    if util is not None:
                        utils.append(util)
                    elapsed = max(time.perf_counter() - epoch_start, 1e-6)
                    gpu = f"GPU {util:.0f} % {memory} GB" if util is not None else "GPU –"
                    run.console.trial_update(step + 1, f"E{epoch}/{max_epochs}  Loss {np.mean(losses_):.4f}  {gpu}  "
                                                       f"Daten-Warten {100 * wait / elapsed:.0f} %  {best_text}")
                    run.tick()
                    if time.time() > deadline:
                        status, finished_epoch = "time", step == steps - 1
                        break
            if not finished_epoch or not losses_:
                break
            seconds = time.perf_counter() - epoch_start
            row = {"epoch": epoch, "train_loss": float(np.mean(losses_)), "pore_share_in_patches": float(np.mean(pore)),
                   "lr": optimizer.param_groups[1]["lr"], "data_wait_percent": 100 * wait / seconds,
                   "gpu_util_percent": float(np.mean(utils)) if utils else None, "epoch_minutes": seconds / 60}
            if row["data_wait_percent"] > 30 and epoch == state["epoch"] + 1:
                run.console.log(f"{label}: Hinweis – die GPU wartet {row['data_wait_percent']:.0f} % der Zeit auf "
                                f"Patches (mehr num_workers? Daten auf SSD?)")

            validated = epoch % int(t["val_every"]) == 0 or epoch == max_epochs or status == "time"
            improved = False
            if validated:
                result = scores(validate(run, model, run.val, thresholds), run.lambdas, s["objective"]["aggregate"])
                for lam, value in result.items():
                    tag = lambda_tag(lam)
                    for key in ("score", "threshold", "pixel_f1", "object_f1", "phantoms_per_mpx"):
                        row[f"{key}_{tag}"] = value[key]
                    if tag not in state["best"] or value["score"] > state["best"][tag]["score"]:
                        _save_atomic(model.state_dict(), folder / f"best_{tag}.pt")
                        state["best"][tag] = {**value, "epoch": epoch}
                        improved |= tag == main
                state["since_best"] = 0 if improved else state["since_best"] + int(t["val_every"])
            state["epoch"] = epoch
            state["seconds"] = time.time() - started
            row["minutes"] = state["seconds"] / 60
            state["history"].append(row)
            report.trial_history(folder, state["history"], run.lambdas)
            if validated:
                b = state["best"][main]
                run.console.log(f"{label} E{epoch}: Loss {row['train_loss']:.4f}, Ziel {row[f'score_{main}']:.4f} "
                                f"(Pixel-F1 {row[f'pixel_f1_{main}']:.3f}, Objekt-F1 {row[f'object_f1_{main}']:.3f}, "
                                f"Phantome {row[f'phantoms_per_mpx_{main}']:.1f}/Mpx, Schwelle "
                                f"{row[f'threshold_{main}']:.2f}){'  * bester' if improved else ''}  "
                                f"[bester {b['score']:.4f} E{b['epoch']}]")
                _save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "state": state},
                             checkpoint)
                if report_fn and report_fn(epoch, row[f"score_{main}"]):
                    status = "pruned"
                    break
            if state["since_best"] >= int(t["early_stopping"]):
                status = "early_stop"
                break
            if status == "time":
                break
    finally:
        iterator = loader = None        # beendet die Worker
        del model, optimizer
        if run.cuda:
            torch.cuda.empty_cache()
    return {"status": status, "epochs": state["epoch"], "minutes": state["seconds"] / 60,
            "best": state["best"], "history": state["history"]}


# ---------------------------------------------------------------- vor dem Start

def find_batch_size(settings: dict, recipe: dict, device: str) -> int:
    """Größte Batch-Größe aus batch_candidates, die mit Reserve in den GPU-Speicher passt."""
    hw = settings["hardware"]
    candidates = sorted((int(c) for c in hw["batch_candidates"]), reverse=True)
    if hw["batch_size"] != "auto":
        return int(hw["batch_size"])
    if device != "cuda":
        return candidates[-1]
    size = settings["model"]["patch_px"]
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    for batch in candidates:
        model = optimizer = None
        ok = False
        try:
            model = compat.build(recipe).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                images = torch.randn(batch, 3, size, size, device=device)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(images).float().mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            ok = torch.cuda.max_memory_reserved() <= total * (1 - float(hw["memory_reserve"]))
        except torch.cuda.OutOfMemoryError:
            ok = False
        finally:
            del model, optimizer
            torch.cuda.empty_cache()
        if ok:
            return batch
    return candidates[-1]


def estimate(run: Run) -> dict:
    """Misst Sekunden je Trainingsschritt und je Validierung -> erwartete Dauer eines Trials."""
    s, t = run.settings, run.settings["training"]
    model = compat.build(run.recipe, s["model"]["pretrained"]).to(run.device)
    optimizer = make_optimizer(model, t)
    scaler = torch.amp.GradScaler("cuda", enabled=run.cuda)
    stream = data.PatchStream(run.train, t, s["augment"], run.recipe, s["model"]["patch_px"], [12345])
    loader = data.make_loader(stream, run.batch_size, run.workers, run.cuda)
    iterator = iter(loader)
    measured, total_steps = 0.0, 25
    run.console.trial_start("Messung", total_steps)
    try:
        for step in range(total_steps):
            if step == 5:
                if run.cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
            images, masks = next(iterator)
            images, masks = images.to(run.device), masks.to(run.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=run.cuda):
                logits = model(images)
            loss = losses.bce_dice(logits.float(), masks, float(t["bce_weight"]))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss.item()
            run.console.trial_update(step + 1, "Trainingsschritte")
        measured = (time.perf_counter() - start) / (total_steps - 5)
        sample = run.val[:5]
        start = time.perf_counter()
        validate(run, model, sample, s["objective"]["thresholds"], "Messung Val")
        per_px = (time.perf_counter() - start) / max(sum(r.height * r.width for r in sample), 1)
        val_s = per_px * sum(r.height * r.width for r in run.val)
    finally:
        iterator = loader = None
        del model, optimizer
        if run.cuda:
            torch.cuda.empty_cache()
    epochs, every = int(t["max_epochs"]), int(t["val_every"])
    trial_s = epochs * int(t["steps_per_epoch"]) * measured + math.ceil(epochs / every) * val_s + 30
    return {"step_s": measured, "val_s": val_s, "trial_s": trial_s}
