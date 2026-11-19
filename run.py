"""MicroNet_Trainer: Hyperparameter-Studie für das U-Net von Pore_Detection – ein Befehl, ~24 h.

    python run.py                          letzten unfertigen Lauf fortsetzen, sonst neuen starten
    python run.py --new                    immer neuen Lauf starten
    python run.py --smoke                  Testlauf (~10 min, wenige Bilder, alle Schritte) – VOR dem 24-h-Lauf
    python run.py --name versuch1          Lauf mit festem Namen (fortsetzen, falls vorhanden)
    python run.py --evaluate <Modell>      ein Modell (z. B. runs/<name>/best_so_far/lambda0.005) auf test auswerten
    python run.py --config andere.yaml     andere Einstellungen (Standard: configs/study.yaml)

Ablauf: Hardware -> Daten prüfen + Cache -> Split -> Batch-Größe -> Zeitschätzung -> Studie (Optuna) ->
Bestätigung der Besten -> finale Modelle (je lambda) -> Evaluation auf test (mit Referenzmodell).
Strg+C bricht sauber ab; derselbe Befehl setzt fort. Ergebnisse: runs/<name>/ (siehe PLAN.md).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DEFAULT_CONFIG = HERE / "configs" / "study.yaml"


def choose_folder(settings: dict, args) -> Path:
    import yaml
    root = settings["output"]
    root.mkdir(parents=True, exist_ok=True)
    name = args.name or settings["name"]
    if name:
        return root / name
    if args.smoke:
        return root / f"smoke_{datetime.now():%Y-%m-%d_%H%M%S}"
    if not args.new:
        runs = [p for p in root.iterdir() if (p / "state.yaml").is_file() and not p.name.startswith("smoke_")]
        for folder in sorted(runs, key=lambda p: (p / "state.yaml").stat().st_mtime, reverse=True):
            state = yaml.safe_load((folder / "state.yaml").read_text(encoding="utf-8")) or {}
            if state.get("phase") != "done":
                return folder
    return root / f"{datetime.now():%Y-%m-%d_%H%M}"


def prepare_data(settings: dict, console, folder: Path):
    from micronet_trainer import data
    d = settings["data"]
    train, skipped = data.pair(d["train"]["images"], d["train"]["masks"])
    if d["test"]["images"].is_dir() and d["test"]["masks"].is_dir():
        test, skipped_test = data.pair(d["test"]["images"], d["test"]["masks"])
    else:
        test, skipped_test = [], []
        console.log(f"WARNUNG: Testordner fehlt ({d['test']['images']}) – keine Abschluss-Evaluation")
    train, s1 = data.build_cache(train, d["cache"], d["mask_threshold"], console, "train")
    test, s2 = data.build_cache(test, d["cache"], d["mask_threshold"], console, "test")
    data.assign_groups(train + test, settings["split"]["group_rules"])
    skipped = [{**s, "part": "train"} for s in skipped + s1] + [{**s, "part": "test"} for s in skipped_test + s2]
    if skipped:
        import pandas as pd
        pd.DataFrame(skipped).to_csv(folder / "skipped.csv", index=False, encoding="utf-8-sig", sep=";")
        console.log(f"{len(skipped)} Dateien übersprungen (ohne Partner, falsche Größe, unlesbar) -> skipped.csv")
    console.log(f"train: {data.pore_stats(train)}\n  Arten: {data.kinds(train)}")
    if test:
        console.log(f"test:  {data.pore_stats(test)}\n  Unterordner: "
                    f"{', '.join(sorted({r.subset for r in test if r.subset})) or 'keine (alles eine Gruppe)'}")
    for warning in data.check(train, test):
        console.log(f"WARNUNG: {warning}")
    return train, test


def evaluate_only(settings: dict, model_dir: Path) -> int:
    from micronet_trainer import compat, engine, hardware, progress, study
    model_dir = model_dir.resolve()
    if not (model_dir / compat.MODEL_FILE).is_file():
        sys.exit(f"kein Modell in {model_dir} ({compat.MODEL_FILE} fehlt)")
    out = model_dir / f"evaluation_{datetime.now():%Y-%m-%d_%H%M}"
    out.mkdir(parents=True)
    console = progress.Console(out / "log.txt")
    device = hardware.resolve_device(settings["hardware"]["device"])
    clock = study.Clock(0)
    console.open_bars(3600)
    try:
        from micronet_trainer import data
        d = settings["data"]
        test, skipped = data.pair(d["test"]["images"], d["test"]["masks"])
        test, _ = data.build_cache(test, d["cache"], d["mask_threshold"], console, "test")
        data.assign_groups(test, settings["split"]["group_rules"])
        run = engine.Run(settings, out, console, clock, device, 8 if device == "cuda" else 2, 0,
                         engine.make_recipe(settings), test=test)
        study.run_evaluation(run, [(model_dir.name, model_dir)], out)
    finally:
        console.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Einstellungen")
    parser.add_argument("--new", action="store_true", help="neuen Lauf starten statt fortzusetzen")
    parser.add_argument("--smoke", action="store_true", help="kurzer Testlauf aller Schritte")
    parser.add_argument("--name", help="Name des Laufordners")
    parser.add_argument("--evaluate", type=Path, help="nur dieses Modell auf test auswerten")
    args = parser.parse_args()

    import yaml

    from micronet_trainer import config
    try:
        settings = config.load(args.config)
    except (ValueError, KeyError) as error:
        sys.exit(f"Fehler in {args.config}: {error}")
    if args.evaluate:
        return evaluate_only(settings, args.evaluate)
    if args.smoke:
        config.apply_smoke(settings)

    folder = choose_folder(settings, args)
    folder.mkdir(parents=True, exist_ok=True)
    resumed_config = folder / "config_used.yaml"
    if resumed_config.is_file() and not args.smoke:
        settings = config.load(resumed_config)            # ein Lauf behält seine Einstellungen
    else:
        resumed_config.write_text(yaml.safe_dump(json.loads(json.dumps(settings, default=str)), sort_keys=False,
                                                 allow_unicode=True), encoding="utf-8")

    from micronet_trainer import data, engine, hardware, progress, report, study

    console = progress.Console(folder / "log.txt")
    state = study.State(folder / "state.yaml")
    if not state.data:
        state.data = {"created": f"{datetime.now():%Y-%m-%d %H:%M}", "phase": "start", "used_s": 0.0,
                      "smoke": bool(args.smoke)}
        console.log(f"=== neuer Lauf: {folder} ===")
    else:
        console.log(f"=== setze Lauf fort: {folder} (Phase {state['phase']}, {state['used_s'] / 3600:.1f} h verbraucht)"
                    f" – für einen neuen Lauf: python run.py --new ===")
    if state["phase"] == "done":
        console.log("Dieser Lauf ist fertig. Neuer Lauf: python run.py --new")
        return 0
    state.save()

    plan = study.budget(settings)
    clock = study.Clock(state["used_s"])
    with hardware.prevent_sleep(settings["hardware"]["prevent_sleep"]) as awake:
        console.open_bars(plan["total"])
        try:
            device = hardware.resolve_device(settings["hardware"]["device"])
            workers = hardware.auto_workers(settings["hardware"]["num_workers"])
            train_all, test = prepare_data(settings, console, folder)
            if settings["smoke"]:
                train_all = data.subset(train_all, settings["smoke"]["train_images"], 0)
                test = data.subset(test, settings["smoke"]["test_images"], 0)
                console.log(f"Testlauf: {len(train_all)} Trainings-, {len(test)} Testbilder")
            split_path = folder / "split.yaml"
            if split_path.is_file():
                train, val = data.load_split(split_path, train_all)
            else:
                train, val = data.split(train_all, float(settings["split"]["val_share"]), int(settings["split"]["seed"]))
                data.save_split(split_path, train, val, test)
            if not val or not any(not r.negative for r in val):
                raise SystemExit("val enthält keine Bilder mit Poren – zu wenige Gruppen? (split.group_rules)")
            if not any(r.negative for r in val):
                console.log("WARNUNG: val enthält keine Negativbilder – Phantome werden nicht gemessen")
            console.log(f"Split: {len(train)} train, {len(val)} val ({data.kinds(val)}), {len(test)} test")

            recipe = engine.make_recipe(settings)
            if not state.get("batch_size"):
                console.log("teste die Batch-Größe …")
                state["batch_size"] = engine.find_batch_size(settings, recipe, device)
                state.save()
            info = hardware.collect(device, settings["output"], workers, state["batch_size"])
            hardware.save(info, folder / "hardware.yaml")
            console.log(hardware.describe(info) + f"\nStandby-Sperre: {'an' if awake else 'aus'}")
            if device == "cpu":
                console.log("WARNUNG: keine CUDA-GPU gefunden – das Training läuft auf der CPU und ist sehr langsam")
            if info["disk_free_gb"] < 10:
                console.log(f"WARNUNG: nur {info['disk_free_gb']} GB frei im Ausgabeordner (~2-3 GB je Lauf nötig)")

            run = engine.Run(settings, folder, console, clock, device, int(state["batch_size"]), workers, recipe,
                             train, val, test, hardware.GpuMonitor(device))
            if not (folder / "augment_examples.png").is_file():
                stream = data.PatchStream(train, settings["training"], settings["augment"], recipe,
                                          settings["model"]["patch_px"], [0])
                report.augment_examples(folder / "augment_examples.png", stream)
                console.log("Beispiel-Patches (wie das Training sie sieht): augment_examples.png")
            if not state.get("estimate"):
                run.status = "Zeitmessung"
                estimate = engine.estimate(run)
                state["estimate"] = estimate
                state.save(clock)
                study_s = plan["study_end"] - clock.used()
                console.log(f"Messung: {estimate['step_s']:.2f} s je Schritt, {estimate['val_s'] / 60:.1f} min je "
                            f"Validierung -> ein Trial (max. {settings['training']['max_epochs']} Epochen) "
                            f"~{estimate['trial_s'] / 60:.0f} min; Studie {study_s / 3600:.1f} h -> "
                            f"~{max(study_s / estimate['trial_s'], 0):.0f} volle Trials (mit Pruning mehr)")

            if state["phase"] in ("start", "study"):
                state["phase"] = "study"
                state.save(clock)
                optuna_study = study.run_study(run, state)
                state["phase"] = "confirm"
                state.save(clock)
            else:
                optuna_study = study.open_study(run)
            if state["phase"] == "confirm":
                study.run_confirm(run, optuna_study, state)
                state["phase"] = "export"
                state.save(clock)
            if state["phase"] == "export":
                study.run_export(run, optuna_study, state)
                state["phase"] = "evaluation"
                state.save(clock)
            if state["phase"] == "evaluation":
                finals = [(label, Path(path)) for label, path in state["finals"]]
                run.status = "Evaluation"
                study.run_evaluation(run, finals, folder / "final" / "evaluation")
                state["phase"] = "done"
                state.save(clock)
            run.tick(force=True)
            console.log(f"=== fertig nach {clock.used() / 3600:.1f} h: {folder / 'final'} ===")
        except KeyboardInterrupt:
            state.save(clock)
            console.log("\nabgebrochen (Strg+C). Der Stand ist gespeichert; derselbe Befehl setzt fort. "
                        "Zwischenmodelle: best_so_far/")
            return 130
        finally:
            console.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
