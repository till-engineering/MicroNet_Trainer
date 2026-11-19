# MicroNet_Trainer – Plan

Ausgelagertes Training des U-Net-Porendetektors von Pore_Detection, mit automatischer
Hyperparameter-Optimierung. Ergebnis ist ein Modellordner, den Pore_Detection direkt nutzt
(`detection: {name: unet, model_dir: <Ordner>, threshold: auto}`).

Stand: 2026-11-19 · Label-Tool, Augmentierungs-Vorschau und Training v1 gebaut (noch nicht auf dem Trainings-Laptop getestet → zuerst `run.py --smoke`).

## Rahmen

- Training läuft auf einem **Windows-Laptop** (16 GB GPU, 32 GB RAM). Die Daten liegen nur dort.
- Die U-Net-Pipeline in Pore_Detection ist **fix**. Ziel ist nur ein besseres Modell (`weights.pt` + `model.yaml`).
- Alle Pakete stehen in `requirements.txt`. Auf dem Entwicklungsrechner wird nichts installiert oder ausgeführt.
- **Budget:** ein Lauf von ~24 h (ein Tag und eine Nacht), Start mit einem Befehl, nach einem Abbruch fortsetzbar.
- Es wird **nur das Modell** bewertet, nicht die ganze Pipeline. Die Bilder haben weder Einbettmittel
  noch Maßstab: Das ganze Bild ist Probe, alle Größen sind in px.

## Schnittstelle zu Pore_Detection (darf sich nicht ändern)

- `weights.pt` ist der state_dict von `smp.Unet(encoder, in_channels=3, classes=1)`.
- `model.yaml` muss diese Schlüssel enthalten: `architecture: unet`, `encoder`, `mean`, `std`, `patch_px`, `overlap_px`, `threshold`.
  Weitere Schlüssel (Metriken, Trial, Herkunft) sind erlaubt, Pore_Detection ignoriert sie.
- Eingabe: RGB-Original → 0..1 → mean/std. Keine Skalierung. Fenster `patch_px`, davon wird
  nur die Mitte ohne `overlap_px` behalten, der Bildrand wird gespiegelt.
- `micronet_trainer/compat.py` ist die Kopie dieser Funktionen aus `pore_detection/core/unet_model.py`.
  Training, Label-Tool und Evaluation verwenden nur diese Kopie. Weil die Pipeline fix ist, gibt es kein Prüfskript.

## Daten

```
<daten>/train/images, train/masks     gleiche Dateinamen (Endung darf abweichen), Weiß = Pore
<daten>/test/images,  test/masks      nur für die Abschluss-Evaluation
```
- Bestand: ~240 externe Bilder mit Poren, teils größer als 1000 px. Dazu ~70 eigene Gefüge-Ausschnitte
  1000×1000 mit leerer Maske (Negativbilder gegen fälschlich als Pore erkannte Gefügebereiche).
  Neu dazu kommen eigene Ausschnitte mit Poren aus dem Label-Tool, ein Teil davon ins Testset.
- Das Testset enthält auch Negativbilder.
- Namen: eigene alte Negative heißen nur nach Nummer (`1.png`, `54.png`), externe Bilder haben lange Namen aus Ziffern und Buchstaben,
  Label-Tool-Ausschnitte heißen `<Aufnahme>_y<Zeile>_x<Spalte>`.
- **val** = 20 % von train, **getrennt nach Gruppe** über `group_rules`. Die erste passende Regex bestimmt die Gruppe:
  `^(.+)_y\d{6}_x\d{6}$` ergibt die Aufnahme. Passt keine Regel, ist jedes Bild seine eigene Gruppe.
- **Geschichtet:** ~20 % jeder Art kommen nach val (Label-Tool mit Poren, Label-Tool negativ, nummerierte Negative, extern).
  Die Art wird nur für den Split verwendet, nicht im Training.
  Der Split wird einmal festgelegt (`split.yaml`) und gilt für alle Trials.
- Unterordner unter `images/` und `masks/` sind erlaubt. Bild und Maske gehören über denselben relativen Pfad und Namen zusammen.
  Im **Testset** ergeben die Unterordner (z. B. `eigen/`, `extern/`) eigene Auswertungsgruppen.
- **Bilder-Cache:** Beim ersten Start werden die Bilder als `.npy` abgelegt. Die Worker lesen sie per Memory-Map,
  dadurch liegen die Daten nur einmal im RAM.
- Im Training werden keine Domänen unterschieden (eigen/extern). Negativbilder werden an der leeren Maske erkannt.
- **Prüfung vor dem Start:** Partner ohne Bild oder Maske, falsche Größen, Gruppen, die sowohl in train als auch in test
  vorkommen (Leck), keine Negativ- oder Porenbilder in val bzw. test → klare Meldung.
- Die Prüfung gibt außerdem den **Porenanteil** der Bilder aus (Median, Min, Max, Anzahl Bilder unter 0,1 %).

## Hardware (vor jedem Start ausgegeben, dazu `hardware.yaml` im Modellordner)

System, Python, CPU (Name, Kerne), RAM gesamt/frei, GPU (Name, VRAM gesamt/frei, Compute),
CUDA/cuDNN, AMP, PyTorch/smp, freier Platz im Ausgabeordner, DataLoader-Worker.
`device: auto` nimmt die GPU, `cuda:0` legt sie fest. Ohne GPU gibt es nur eine **Warnung**, das Training läuft auf der CPU weiter.
Für RAM und CPU wird `psutil` genutzt.

Windows: Die DataLoader-Worker sind `spawn`-sicher (picklebares Dataset, Start unter `__main__`).
`num_workers: auto`. Der Standby wird während des Laufs gesperrt (`SetThreadExecutionState`, Standard an).
Die Batch-Größe wird automatisch getestet (16 → 8 → 4 Patches, ~15 % Reserve), alternativ fest in der Config.

## Patches (Training)

- Die **Epoche besteht aus einer festen Zahl Schritte**, z. B. 250 Batches, damit die Trial-Dauer planbar ist.
- **Bild-Auswahl nach Fläche** (Option: Fläche | Wurzel der Fläche | gleich). Große Bilder werden nicht unterrepräsentiert.
- **Poren-Patches** (`pore_share`) liegen auf einer Pore, an zufälliger Stelle im Patch.
  Es ist eine Mischung (`pore_object_share`): eine Pore als Ganzes ziehen, sodass kleine Poren gleich oft drankommen,
  oder ein Porenpixel. Standard ist 50/50, der Anteil wird im Tuning mitgesucht.
  Die übrigen Patches liegen zufällig. Negativbilder liefern nur zufällige Patches.
- Später (nicht v1): **automatisches Hard Negative Mining**, also häufiger Patches an Stellen mit Phantom-Poren auf train.
- Die Patches werden in DataLoader-Workern parallel zur GPU erzeugt.
- **Klassenungleichgewicht** (Poren sind nur ein kleiner Teil des Bilds): wird gezielt über Poren-Patches und die Pore/Pixel-Mischung
  ausgeglichen, im Loss durch den Dice-Anteil (Dice über den ganzen Batch), in der Bewertung durch F1/Objekt-F1 statt Genauigkeit
  und über die Schwellensuche auf val. Pro Epoche wird der tatsächliche Porenanteil in den Patches protokolliert.

## Augmentierung (nur Training, Basis `augment.py` aus Pore_Detection)

| behalten | neu | neu, optional | nicht |
|---|---|---|---|
| 90°-Drehung, Spiegeln, Helligkeit, Kontrast, Gamma, Farbton, Sättigung, linearer Helligkeitsverlauf, Unschärfe, Rauschen | radiale Vignettierung | JPEG-Artefakte, Graustufen, Copy-Paste von Poren, Skalierung ±5–10 % (Standard aus) | elastische Verzerrung, freie Rotation |

Jede Augmentierung ist **einzeln** einstellbar (`enabled`, `probability`, `range`), Code in
`micronet_trainer/augment.py` ✓. Im Tuning steuert **ein globaler Stärkefaktor** (`strength`) alle Farb- und
Helligkeitsaugmentierungen. Eingestellt wird mit `augment_tool/preview.py` ✓ (gleicher Code): Der Abschnitt
`augment:` aus `augment_tool/augment_config.yaml` wird 1:1 in die Trainings-Config übernommen.
Vor dem Trainingsstart werden zusätzlich Beispiel-Patches gespeichert.

## Training eines Trials

- AdamW, eigene Lernrate für den Encoder (Faktor), Warmup + Cosine passend zur Höchstzahl der Epochen, AMP float16.
- Loss: BCE + Dice (Gewicht gesucht), optional Focal + Dice oder Tversky.
- Validierung auf ganzen Val-Bildern mit `compat.predict`, also wie in der Pipeline, jede 2. Epoche.
- Early Stopping. Checkpoint mit Optimizer, Scheduler und Zufallszustand, damit der Lauf fortsetzbar ist.

## Tuning (Optuna)

- TPE-Sampler, Median-Pruner, Speicher in SQLite. Ein Abbruch ist harmlos, derselbe Befehl macht weiter.
- **Fest:** resnet50 + micronet, `patch_px` 512, `overlap_px` 64, `batch_size` (größte, die passt).
- **Gesucht:** learning_rate (log), weight_decay (log), Encoder-LR-Faktor, Loss-Gewicht,
  `pore_share`, `pore_object_share`, Augmentierungsstärke. Jeder Parameter kann per Config fest gesetzt oder freigegeben werden.
- Trials sind kurz: höchstens ~30 Epochen, Early Stopping nach 8 Epochen, Pruning nach 5–10 Epochen.
- Es bleiben nur die Gewichte der besten ~5 Trials. Alle Trials stehen in der Tabelle.

**Zielmaß (val, kein Loss, nur zur Auswahl):** `0,5 · Pixel-F1 + 0,5 · Objekt-F1` auf Bildern mit Poren
**minus** `λ · Phantom-Poren pro Mpx` auf Negativbildern. Optional p10 statt Mittelwert.
- `phantom_lambdas: [0.005, 0]`. Der **erste Wert steuert die Suche** (Optuna, Pruning). Für jeden Wert werden die beste
  Epoche, die Schwelle und das Modell **getrennt** geführt. Ergebnis sind **zwei finale Modelle**.
- Das λ = 0-Modell ist das beste unter den ausprobierten Trials, aber nicht eigens darauf optimiert.

## Zeitplan aus `max_hours: 24`

```
~19 h  Studie         kein neuer Trial, wenn die Restzeit nicht reicht
 ~4 h  Bestätigung    Top-2 (λ = 0,005) + bester Trial für λ = 0, falls ein anderer, je 1 weiterer Seed
 ~1 h  Export + Evaluation auf test
```
Vor dem Start wird die Dauer geschätzt (einige Batches und ein Val-Durchlauf), mit der zu erwartenden Zahl an Trials.
Die **finalen Modelle** sind die besten bestätigten Läufe, ohne Neu-Training auf train+val.
Die letzte Stunde ist fest reserviert. Auch wenn die Studie vorzeitig endet, gibt es finale Modelle.
Bei Strg+C wird der beste Stand exportiert, ein Neustart mit demselben Befehl setzt fort.
Die Schwelle wird auf val mit demselben Zielmaß gesucht.

Optionaler Modus **Lernkurve**: beste Konfiguration auf 25/50/75/100 % von train. Zeigt, ob mehr gelabelte Bilder noch etwas bringen.

## Evaluation (test, einmal am Ende)

Definitionen wie in `Pore_Detection/dataset_eval/metrics.py` (`match_iou` 0,5, `rim_px` 2,
connectivity 8, `size_classes_px` 0/5/10/20/50/100), ohne ROI, Maßstab und Filter:

- **Bilder mit Poren:** Pixel-Precision, -Recall, -F1, -IoU; FP getrennt nach Porenrand und Matrix, FN nach Rand und Innerem;
  Objekt-Precision, -Recall, -F1; übersehene, Phantom-, zerteilte und verschmolzene Poren;
  Größenklassen (Recall, Precision, Flächenfehler; Recall auch bei IoU ≥ 0,3); Porositätsfehler pro Bild.
- **Negativbilder:** FP-Pixelanteil, Phantom-Poren pro Mpx.
- **Zusätzlich:** PR-Kurve mit AP, F1 über der Schwelle, Bootstrap-Konfidenzintervalle.
- **Gruppen:** gesamt und pro Unterordner von test (z. B. `eigen`, `extern`), jeweils getrennt nach Bildern mit Poren und Negativbildern.
- **Referenzmodell** `reference_model: label_tool/model`: Das alte Modell wird auf demselben Testset mitbewertet und neben das neue gestellt.
  Achtung: Die neuen Testmasken wurden aus seinen Vorschlägen erstellt, der Vergleich ist daher leicht zu seinen Gunsten verzerrt.
- `python run.py --evaluate <Modellordner>` wertet jederzeit ein beliebiges Modell (z. B. `best_so_far/…`) auf test aus.
- **Ausgabe:** `evaluation.xlsx` (Summary, Images, Size classes, Error origin), Plots, Overlays der
  schlechtesten Bilder (Farben wie `report.py`: grau korrekt, orange falsch, blau übersehen).

## Fortschritt und Logging

- **Zwei feste Fortschrittsbalken** (tqdm). Der Gesamtbalken ist **zeitbasiert** und zeigt Phase, Trial, Bestwert und Restzeit.
  Der Trial-Balken zeigt Epoche, Loss, Val-Wert, **GPU-Auslastung** und **Daten-Wartezeit**.
  Meldungen erscheinen oberhalb der Balken (`tqdm.write`). Empfohlen sind Windows Terminal oder PowerShell.
- GPU-Auslastung und Speicher kommen über `nvidia-ml-py`. Bei hoher Daten-Wartezeit gibt das Skript einen Hinweis.
- `log.txt` enthält alle Meldungen, `trials.xlsx` und `progress.png` werden laufend aktualisiert,
  pro Trial gibt es `history.csv` und eine Kurve. Kein TensorBoard.
- **Zwischenmodelle:** Bei jedem neuen Bestwert wird `best_so_far/lambda<λ>/` (weights.pt + model.yaml mit Val-Schwelle)
  atomar geschrieben, also nie halb. Der Ordner ist direkt in Pore_Detection und im Label-Tool nutzbar.

## Ausgabe

Standard ist `runs/<name>/` (Pfad in der Config wählbar). Rechne mit ~2–3 GB pro Lauf, der freie Platz wird vor dem Start geprüft.

```
runs/<studie>/
  study.db                 Optuna (fortsetzbar)
  split.yaml, hardware.yaml, config_used.yaml, augment_examples.png
  trials.xlsx              alle Trials: Parameter, Zielmaß, Dauer, Status
  tuning_report.png        Verlauf, Wichtigkeit der Parameter
  trials/<nr>/             history.csv, Kurven, (Gewichte der besten)
  log.txt, progress.png
  best_so_far/lambda0.005/, best_so_far/lambda0/    laufend aktualisiert
  final/lambda0.005/       ← Modellordner für Pore_Detection: weights.pt, model.yaml, split.yaml, hardware.yaml
  final/lambda0/
  final/evaluation/        evaluation.xlsx (beide Modelle + Referenz), plots/, overlays/
```

## Geplante Struktur

```
requirements.txt
PLAN.md
micronet_trainer/
  compat.py      ✓ Ein-/Ausgabe wie Pore_Detection
  imageio.py     ✓ Bilder/Masken lesen und schreiben
  augment.py     ✓ Augmentierung (jede einzeln einstellbar)
  config.py      ✓ Einstellungen laden/prüfen, Testlauf
  hardware.py    ✓ Hardware-Ausgabe, GPU-Auslastung, Standby-Sperre
  data.py        ✓ Paaren, Cache, Prüfen, Split, PatchStream
  losses.py      ✓ BCE + Dice
  metrics.py     ✓ Pixel-, Objekt-, Negativ- und Porositätsmetriken (wie Pore_Detection)
  engine.py      ✓ Trainingsloop, Validierung, Checkpoints, Batch-Test, Zeitmessung
  study.py       ✓ Optuna, Zeitplan, Fortsetzen, Bestätigung, Export
  evaluation.py  ✓ Abschluss-Evaluation (Gruppen, Referenz, Bootstrap, Plots, Overlays)
  report.py      ✓ trials.xlsx, Verlauf, Beispiel-Patches
  progress.py    ✓ zwei Fortschrittsbalken, log.txt
configs/
  study.yaml     ✓ Daten, Budget, Suchraum, Zielmaß, Augmentierung
run.py           ✓ ein Befehl: Prüfen → Studie → Bestätigung → Export → Evaluation (--smoke, --new, --evaluate)
label_tool/      ✓ Label-Tool (siehe unten)
augment_tool/    ✓ Vorschau der Augmentierung: input/ (+ input/masks/) -> output/<Bild>_einzeln.png, _kombiniert.png
```

## Label-Tool (gebaut)

`python label_tool/label.py` (Einstellungen in `label_tool/label_config.yaml`):

1. Das alte Modell kommt in `label_tool/model/`, die großen Aufnahmen in `label_tool/input/`.
2. Beim Öffnen rechnet das Modell die Aufnahme einmal in Kacheln durch, das Ergebnis liegt in `cache/` (uint8, Memory-Map).
3. **Vorschläge:** 1000×1000-Fenster um erkannte Porengruppen, **die Pore an zufälliger Stelle** im
   Fenster (Abstand zum Rand mindestens `edge_margin_px`). Fenster mit vielen unsicheren Pixeln (0,3–0,7) kommen zuerst.
   Die Fenster überlappen höchstens zu 20 %. Dazu einige Fenster **ohne** erkannte Pore. Ein Klick in die Übersicht wählt ein eigenes Fenster.
4. **Korrigieren:** Pinsel/Radierer, ganze Pore per Klick löschen, Polygon, Schwellen-Regler,
   Wahrscheinlichkeits-Heatmap, Zoom/Pan, Rückgängig/Wiederholen.
5. **Speichern** (Enter): `output/images` und `output/masks` als PNG (Original-Bittiefe), dazu eine Zeile in
   `output/labels.csv`. Gelabelte Fenster werden nicht erneut vorgeschlagen.

Danach ziehst du zufällige Ausschnitte nach train und test. Ausschnitte derselben Aufnahme sollten zusammen
in train oder in test landen (Spalte `source` in `labels.csv`).

## Umfang

**v1:** Hardware, Datenprüfung, Split, Cache, Training (BCE+Dice), Studie mit Pruning und Zeitplan, Bestätigung,
zwei finale Modelle, Evaluation mit Gruppen und Referenz, Fortschritt, Zwischenmodelle und **`--smoke`**
(~10 min durch alle Schritte, vor dem 24-h-Lauf starten).

**Später:** Lernkurve, Hard Negative Mining, Copy-Paste von Poren, Focal/Tversky/pos_weight, k-fold.

## Offen

- λ nach dem ersten Lauf nachjustieren (Start 0,005).
