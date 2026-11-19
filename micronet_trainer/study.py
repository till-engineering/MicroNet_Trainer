"""Ablauf eines Laufs: Studie (Optuna) -> Bestätigung -> Export -> Evaluation, mit Zeitplan und Fortsetzen.

state.yaml im Laufordner hält Phase, verbrauchte Zeit, Batch-Größe, Schätzung, Bestätigungsergebnisse.
Optuna speichert in study.db (SQLite). Abbruch (Strg+C, Absturz, Stromausfall): derselbe Befehl setzt
fort – ein unterbrochener Trial läuft ab seinem letzten Checkpoint weiter.
"""

import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml

from . import compat, engine, evaluation, report
from .hardware import plain
from .engine import lambda_tag

COMPLETE, PRUNED, FAIL, RUNNING = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                                   optuna.trial.TrialState.FAIL, optuna.trial.TrialState.RUNNING)


# ---------------------------------------------------------------- Zeit und Zustand

class Clock:
    """Verbrauchte Laufzeit über alle Sitzungen (Pausen zählen nicht)."""

    def __init__(self, used_s: float):
        self.base = float(used_s)
        self.start = time.time()

    def used(self) -> float:
        return self.base + time.time() - self.start

    def deadline(self, used_limit_s: float) -> float:
        """Uhrzeit (time.time), zu der used() den Wert used_limit_s erreicht."""
        return time.time() + (used_limit_s - self.used())


class State:
    def __init__(self, path: Path):
        self.path = path
        self.data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setdefault(self, key, default):
        return self.data.setdefault(key, default)

    def save(self, clock: Clock | None = None) -> None:
        if clock is not None:
            self.data["used_s"] = round(clock.used(), 1)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(yaml.safe_dump(plain(self.data), sort_keys=False, allow_unicode=True), encoding="utf-8")
        tmp.replace(self.path)


def budget(settings: dict) -> dict:
    b = settings["budget"]
    total = float(b["max_hours"]) * 3600
    final, confirm = float(b["final_hours"]) * 3600, float(b["confirm_hours"]) * 3600
    return {"total": total, "study_end": max(total - final - confirm, 0), "confirm_end": max(total - final, 0)}


# ---------------------------------------------------------------- Modellordner

def export_model(run, destination: Path, weights: Path, threshold: float, meta: dict) -> Path:
    """Modellordner für Pore_Detection (weights.pt + model.yaml), atomar: erst .tmp, dann umbenennen."""
    recipe = {**run.recipe, "threshold": round(float(threshold), 4), "trained": f"{datetime.now():%Y-%m-%d %H:%M}",
              "train_images": len(run.train), "val_images": len(run.val), **meta}
    tmp = destination.with_name(destination.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copyfile(weights, tmp / compat.WEIGHTS_FILE)
    compat.write_recipe(tmp, plain(recipe))
    for name in ("split.yaml", "hardware.yaml"):
        if (run.folder / name).is_file():
            shutil.copyfile(run.folder / name, tmp / name)
    old = destination.with_name(destination.name + ".old")
    try:
        if destination.exists():
            shutil.rmtree(old, ignore_errors=True)
            destination.rename(old)
        tmp.rename(destination)
        shutil.rmtree(old, ignore_errors=True)
    except OSError as error:
        run.console.log(f"WARNUNG: {destination} konnte nicht ersetzt werden ({error}) – das neue Modell liegt in {tmp}")
        return tmp
    return destination


def _meta(lam, trial_number, info, params, source) -> dict:
    return {"lambda": lam, "trial": trial_number, "source": source,
            "val_score": round(float(info["score"]), 4), "val_pixel_f1": round(float(info["pixel_f1"]), 4),
            "val_object_f1": round(float(info["object_f1"]), 4),
            "val_phantoms_per_mpx": round(float(info["phantoms_per_mpx"]), 2), "params": params}


# ---------------------------------------------------------------- Studie

def trial_dir(run, number: int) -> Path:
    return run.folder / "trials" / f"{number:03d}"


def suggest(trial, settings: dict) -> dict:
    params = dict(settings["training"])
    for name, spec in settings["search"].items():
        params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec["log"]))
    return params


def open_study(run):
    st = run.settings["study"]
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    storage = f"sqlite:///{(run.folder / 'study.db').as_posix()}"
    return optuna.create_study(
        study_name="study", storage=storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=int(run.settings["split"]["seed"]),
                                           n_startup_trials=int(st["random_trials"])),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=int(st["pruner_startup_trials"]),
                                           n_warmup_steps=int(st["pruner_warmup_epochs"])))


def _resume_interrupted(run, study, state: State) -> None:
    """Unterbrochene Trials (RUNNING nach Absturz, oder Strg+C) mit ihren Parametern erneut einreihen."""
    done = set(state.setdefault("resumed", []))
    for trial in study.trials:
        if trial.number in done:
            continue
        interrupted = trial.state == RUNNING or (trial.state == FAIL and trial.user_attrs.get("interrupted"))
        if not interrupted:
            continue
        if trial.state == RUNNING:
            study._storage.set_trial_state_values(trial._trial_id, state=FAIL)
        if (trial_dir(run, trial.number) / "checkpoint.pt").is_file():
            study.enqueue_trial(trial.params, user_attrs={"resume_from": trial.number})
            run.console.log(f"Trial {trial.number} war unterbrochen – wird ab dem letzten Checkpoint fortgesetzt")
        done.add(trial.number)
    state["resumed"] = sorted(done)
    state.save(run.clock)


def _objective(run, trial, deadline: float):
    params = suggest(trial, run.settings)
    trial.set_user_attr("params", params)
    folder = trial_dir(run, trial.number)
    source = trial.user_attrs.get("resume_from")
    if source is not None:
        old = trial_dir(run, source)
        if old.is_dir() and not folder.exists():
            old.rename(folder)
    run.status = f"Studie · Trial {trial.number}"

    def report_fn(epoch, score, replay=False):
        trial.report(float(score), int(epoch))
        return (not replay) and trial.should_prune()

    seed = int(run.settings["split"]["seed"]) * 1000 + trial.number
    try:
        result = engine.train_trial(run, params, folder, int(run.settings["training"]["max_epochs"]), seed, deadline,
                                    report_fn, label=f"Trial {trial.number}")
    except KeyboardInterrupt:
        trial.set_user_attr("interrupted", True)
        raise
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        trial.set_user_attr("error", message)
        run.console.log(f"Trial {trial.number} FEHLGESCHLAGEN: {message}")
        if run.cuda:
            torch.cuda.empty_cache()
        raise
    trial.set_user_attr("status", result["status"])
    trial.set_user_attr("epochs", result["epochs"])
    trial.set_user_attr("minutes", round(result["minutes"], 2))
    for tag, info in result["best"].items():
        trial.set_user_attr(tag, {k: float(v) for k, v in info.items()})
    (folder / "checkpoint.pt").unlink(missing_ok=True)
    main = lambda_tag(run.lambdas[0])
    if main not in result["best"]:
        raise optuna.TrialPruned("keine Validierung")
    score = result["best"][main]["score"]
    run.console.log(f"Trial {trial.number} {result['status']}: Ziel {score:.4f} nach {result['epochs']} Epochen, "
                    f"{result['minutes']:.0f} min")
    if result["status"] == "pruned":
        raise optuna.TrialPruned()
    return score


def _ranked(study, tag: str, states=(COMPLETE, PRUNED)):
    trials = [t for t in study.trials if t.state in states and tag in t.user_attrs]
    return sorted(trials, key=lambda t: t.user_attrs[tag]["score"], reverse=True)


def _after_trial(run, study, state: State) -> None:
    keep = set()
    for lam in run.lambdas:
        tag = lambda_tag(lam)
        ranked = [t for t in _ranked(study, tag) if (trial_dir(run, t.number) / f"best_{tag}.pt").is_file()]
        keep |= {t.number for t in ranked[:int(run.settings["study"]["keep_weights"])]}
        if ranked:
            best = ranked[0]
            best_so_far = state.setdefault("best_so_far", {})
            if best_so_far.get(tag) != best.number:
                info = best.user_attrs[tag]
                export_model(run, run.folder / "best_so_far" / tag, trial_dir(run, best.number) / f"best_{tag}.pt",
                             info["threshold"], _meta(lam, best.number, info, best.user_attrs.get("params"), "Studie"))
                best_so_far[tag] = best.number
                run.console.log(f"neues bestes Modell λ={lam:g}: Trial {best.number}, Ziel {info['score']:.4f} "
                                f"-> best_so_far/{tag}/")
    for trial in study.trials:
        if trial.state in (COMPLETE, PRUNED) and trial.number not in keep:
            for path in trial_dir(run, trial.number).glob("best_*.pt"):
                path.unlink(missing_ok=True)
    table = report.trials_table(study, run.lambdas)
    sheets = {"Trials": table}
    if sum(t.state == COMPLETE for t in study.trials) >= 3:
        sheets["Importance"] = report.importance(study)
    report.write_excel(run.folder / "trials.xlsx", sheets)
    report.progress_plot(run.folder / "progress.png", table, run.lambdas)
    state.save(run.clock)


def run_study(run, state: State):
    study = open_study(run)
    _resume_interrupted(run, study, state)
    plan = budget(run.settings)
    deadline = run.clock.deadline(plan["study_end"])
    max_trials = run.settings["study"]["max_trials"]
    while True:
        failed = [t for t in study.trials if t.state == FAIL and not t.user_attrs.get("interrupted")]
        finished = [t for t in study.trials if t.state in (COMPLETE, PRUNED)] + failed
        waiting = any(t.state == optuna.trial.TrialState.WAITING for t in study.trials)
        last = [t for t in study.trials if t.state != optuna.trial.TrialState.WAITING][-3:]
        failed_numbers = {t.number for t in failed}
        if len(last) == 3 and all(t.number in failed_numbers for t in last):
            raise SystemExit("3 Trials hintereinander fehlgeschlagen – Ursache in log.txt (Spalte error in trials.xlsx)")
        if max_trials and len(finished) >= int(max_trials) and not waiting:
            break
        durations = [t.user_attrs["minutes"] * 60 for t in study.trials
                     if t.state == COMPLETE and "minutes" in t.user_attrs]
        expected = float(np.median(durations)) if durations else float(state.get("estimate", {}).get("trial_s", 600))
        remaining = deadline - time.time()
        if remaining < 0.5 * expected and not max_trials:
            run.console.log(f"Studie beendet: Restzeit {remaining / 60:.0f} min reicht nicht für einen weiteren Trial "
                            f"(~{expected / 60:.0f} min); {len(finished)} Trials")
            break
        best = _ranked(study, lambda_tag(run.lambdas[0]))
        run.status = (f"Studie · Trial {len(study.trials)}" +
                      (f" · bester {best[0].user_attrs[lambda_tag(run.lambdas[0])]['score']:.3f} (T{best[0].number})"
                       if best else ""))
        study.optimize(lambda trial: _objective(run, trial, deadline), n_trials=1, catch=(Exception,),
                       gc_after_trial=True)
        _after_trial(run, study, state)
    return study


# ---------------------------------------------------------------- Bestätigung

def run_confirm(run, study, state: State) -> None:
    st = run.settings["study"]
    main = lambda_tag(run.lambdas[0])
    complete = [t for t in study.trials if t.state == COMPLETE]
    candidates = [t.number for t in _ranked(study, main, (COMPLETE,))[:int(st["confirm_top"])]]
    for lam in run.lambdas[1:]:
        ranked = _ranked(study, lambda_tag(lam), (COMPLETE,))
        if ranked and ranked[0].number not in candidates:
            candidates.append(ranked[0].number)
    state["confirm_candidates"] = candidates
    results = state.setdefault("confirm", {})
    if not complete:
        run.console.log("Bestätigung übersprungen: kein vollständiger Trial")
        state.save(run.clock)
        return
    deadline = run.clock.deadline(budget(run.settings)["confirm_end"])
    epochs = int(st["confirm_epochs"])
    by_number = {t.number: t for t in study.trials}
    for number in candidates:
        if str(number) in results:
            continue
        trial = by_number[number]
        per_epoch = trial.user_attrs["minutes"] * 60 / max(trial.user_attrs["epochs"], 1)
        expected = per_epoch * epochs
        if deadline - time.time() < 0.5 * expected and not st["max_trials"]:
            run.console.log(f"Bestätigung Trial {number} übersprungen: Restzeit reicht nicht (~{expected / 60:.0f} min)")
            results[str(number)] = {"skipped": True}
            state.save(run.clock)
            continue
        run.status = f"Bestätigung · Trial {number} ({candidates.index(number) + 1}/{len(candidates)})"
        folder = run.folder / "confirm" / f"{number:03d}"
        seed = int(run.settings["split"]["seed"]) * 1000 + 100_000 + number
        result = engine.train_trial(run, trial.user_attrs["params"], folder, epochs, seed, deadline, None,
                                    label=f"Best. {number}")
        (folder / "checkpoint.pt").unlink(missing_ok=True)
        results[str(number)] = {"status": result["status"], "epochs": result["epochs"],
                                "best": {tag: {k: float(v) for k, v in info.items()}
                                         for tag, info in result["best"].items()}}
        state.save(run.clock)
        main_info = result["best"].get(main)
        if main_info:
            run.console.log(f"Bestätigung Trial {number}: Ziel {main_info['score']:.4f} "
                            f"(Studie {trial.user_attrs[main]['score']:.4f})")


def choose_winners(run, study, state: State) -> dict:
    """Je lambda: Konfiguration mit dem besten Mittel aus Studie + Bestätigung; Gewichte des besseren Laufs."""
    by_number = {t.number: t for t in study.trials}
    results = state.get("confirm", {}) or {}
    winners = {}
    for lam in run.lambdas:
        tag = lambda_tag(lam)
        options = []
        for number in state.get("confirm_candidates", []) or []:
            trial = by_number.get(number)
            if trial is None:
                continue
            runs = []
            if tag in trial.user_attrs:
                runs.append((trial.user_attrs[tag], trial_dir(run, number) / f"best_{tag}.pt", "Studie"))
            confirmed = (results.get(str(number)) or {}).get("best", {}).get(tag)
            if confirmed:
                runs.append((confirmed, run.folder / "confirm" / f"{number:03d}" / f"best_{tag}.pt", "Bestätigung"))
            scores = [info["score"] for info, _, _ in runs]
            runs = [r for r in runs if r[1].is_file()]
            if scores and runs:
                options.append((float(np.mean(scores)), number, max(runs, key=lambda r: r[0]["score"]), len(scores)))
        if not options:                     # keine Bestätigung: bester Trial mit Gewichten
            for trial in _ranked(study, tag):
                path = trial_dir(run, trial.number) / f"best_{tag}.pt"
                if path.is_file():
                    options.append((trial.user_attrs[tag]["score"], trial.number,
                                    (trial.user_attrs[tag], path, "Studie"), 1))
                    break
        if not options:
            continue
        mean, number, (info, path, source), count = max(options, key=lambda o: o[0])
        winners[tag] = {"lambda": lam, "trial": number, "mean_score": mean, "runs": count, "info": info,
                        "weights": path, "source": source, "params": by_number[number].user_attrs.get("params")}
    return winners


# ---------------------------------------------------------------- Export + Evaluation

def run_export(run, study, state: State) -> list[tuple[str, Path]]:
    winners = choose_winners(run, study, state)
    if not winners:
        raise SystemExit("kein Trial mit Gewichten – nichts zu exportieren (Fehler in log.txt?)")
    fine = run.settings["objective"]["final_thresholds"]
    thresholds = np.round(np.arange(fine["start"], fine["stop"] + 1e-9, fine["step"]), 4)
    finals = []
    for tag, w in winners.items():
        run.status = f"Export · {tag}"
        model = compat.build(run.recipe).to(run.device)
        model.load_state_dict(torch.load(w["weights"], map_location="cpu", weights_only=True))
        rows = engine.validate(run, model, run.val, thresholds, f"Schwelle {tag}")
        best = engine.scores(rows, [w["lambda"]], run.settings["objective"]["aggregate"])[w["lambda"]]
        del model
        if run.cuda:
            torch.cuda.empty_cache()
        meta = _meta(w["lambda"], w["trial"], best, w["params"], w["source"])
        meta["confirm_mean_score"] = round(w["mean_score"], 4)
        meta["confirm_runs"] = w["runs"]
        folder = export_model(run, run.folder / "final" / tag, w["weights"], best["threshold"], meta)
        finals.append((f"neu {tag}", folder))
        run.console.log(f"finales Modell λ={w['lambda']:g}: Trial {w['trial']} ({w['source']}), Schwelle "
                        f"{best['threshold']:.3f}, val Ziel {best['score']:.4f} -> {folder}")
    state["finals"] = [[label, str(path)] for label, path in finals]
    state.save(run.clock)
    return finals


def run_evaluation(run, models: list[tuple[str, Path]], out: Path) -> None:
    reference = run.settings["evaluation"]["reference_model"]
    models = list(models)
    if reference and (Path(reference) / compat.MODEL_FILE).is_file() and (Path(reference) / compat.WEIGHTS_FILE).is_file():
        models.append(("Referenz", Path(reference)))
    elif reference:
        run.console.log(f"Referenzmodell fehlt ({reference}) – Evaluation ohne Vergleich")
    evaluation.evaluate_models(run, models, out)
    run.console.log(f"Evaluation: {out / 'evaluation.xlsx'}")
