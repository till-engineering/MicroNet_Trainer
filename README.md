# MicroNet_Trainer

Ausgelagertes Training des U-Net-Porendetektors von **Pore_Detection**, mit automatischer
Hyperparameter-Optimierung (Optuna). Ergebnis sind Modellordner (`weights.pt` + `model.yaml`), die
Pore_Detection direkt nutzt: `detection: {name: unet, model_dir: <Ordner>, threshold: auto}`.

Details zu allen Entscheidungen: [PLAN.md](PLAN.md).

## Installation (Windows)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
```

## Training

1. In `configs/study.yaml` die Pfade unter `data:` eintragen (train / test mit `images/` und `masks/`).
   Unterordner im Testset (z. B. `eigen/`, `extern/`) ergeben eigene Auswertungsgruppen.
2. Testlauf (~10 min, prüft alle Schritte):
   `.venv\Scripts\python run.py --smoke`
3. Richtiger Lauf (~24 h):
   `.venv\Scripts\python run.py`
   Mit Strg+C abbrechen. Derselbe Befehl setzt den Lauf fort.
4. Ergebnisse liegen in `runs/<name>/`:
   - `best_so_far/lambda0.005/`, `best_so_far/lambda0/`: bestes Modell bisher, laufend aktualisiert
   - `final/lambda0.005/`, `final/lambda0/`: finale Modelle
   - `final/evaluation/evaluation.xlsx`: Auswertung auf test, neben dem Referenzmodell
   - `trials.xlsx`, `progress.png`, `log.txt`: Verlauf der Studie

Ein beliebiges Modell auswerten: `.venv\Scripts\python run.py --evaluate runs/<name>/best_so_far/lambda0.005`

## Werkzeuge

- **Label-Tool** `python label_tool/label.py`: Vorschläge des alten Modells auf großen Aufnahmen korrigieren,
  Ausschnitte 1000×1000 + Masken speichern.
- **Augmentierungs-Vorschau** `python augment_tool/preview.py`: zeigt jede Augmentierung einzeln (min / max)
  und kombiniert. Den Abschnitt `augment:` danach in `configs/study.yaml` übernehmen.
