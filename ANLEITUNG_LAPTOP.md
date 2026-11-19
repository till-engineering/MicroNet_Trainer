# Anleitung für den Laptop: vom Labeln bis zum Training

Reihenfolge: **0 Einrichten → 1 Labeln → 2 Sortieren → 3 Augmentierung einstellen → 4 Config → 5 Testlauf → 6 Training → 7 Ergebnis prüfen**

Zeitbedarf grob: Einrichten 30 min · Labeln 4–6 h (verteilbar) · Sortieren 30 min · Augmentierung 30–60 min ·
Testlauf 15 min · Training 24 h.

---

## 0. Einrichten (einmalig, ~30 min)

```powershell
git clone https://github.com/till-engineering/MicroNet_Trainer.git
cd MicroNet_Trainer
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
```

- Nimm **Python 3.12**. Für sehr neue Versionen (3.14) gibt es PyTorch eventuell noch nicht.
- **CUDA:** Kein CUDA-Toolkit nötig, das PyTorch-Paket bringt CUDA mit. Du brauchst nur einen aktuellen NVIDIA-Treiber.
  `nvidia-smi` zeigt oben rechts die höchste unterstützte CUDA-Version. Die Zahl im Paket (cu126) muss kleiner oder gleich sein.
  **Ausnahme RTX 50er-Serie:** Nimm `cu128` statt `cu126`.
- Die MicroNet-Gewichte (unten) sind nur der vortrainierte **Startpunkt** des ResNet50-Encoders fürs Training.
  Pore_Detection und das Label-Tool brauchen sie nicht.
- GPU prüfen. Es muss `True` und der Name deiner Grafikkarte erscheinen:
  ```powershell
  .venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
  Steht dort `False`, passt die CUDA-Version nicht (Treiber aktualisieren oder `cu121` statt `cu126` probieren).
- MicroNet-Gewichte einmal herunterladen (~90 MB, braucht Internet):
  ```powershell
  .venv\Scripts\python -c "import torch; torch.hub.load_state_dict_from_url('https://huggingface.co/jstuckner/microscopy-resnet50-micronet/resolve/main/resnet50_micronet_weights.pth')"
  ```

---

## 1. Eigene Poren labeln (Label-Tool, ~4–6 h)

### 1.1 Vorbereiten
1. Kopiere das alte Modell nach `label_tool\model\`: `weights.pt` und `model.yaml` aus
   `Pore_Detection\models\unet_2026-10-22_1952\`. Es dient zugleich als **Referenzmodell** in der Evaluation.
2. Wähle **12 große Aufnahmen** von deinem Mikroskop, möglichst von **12 verschiedenen Proben**, und lege sie nach `label_tool\input\`.
   Mische dabei: viele und wenige Poren, große und kleine Poren, Stellen, an denen das alte Modell Gefüge als Pore erkannt hat,
   verschiedene Ätzungen und Helligkeiten.
3. **Lege jetzt schon fest,** welche **4 der 12 Aufnahmen** später ins Testset kommen. Schreib dir die Namen auf.
   Alles aus diesen 4 Aufnahmen kommt nach test, alles aus den anderen 8 nach train.
   Warum vorher? Dann labelst du die Testaufnahmen besonders sorgfältig, und es gibt sicher keine Vermischung.

### 1.2 Wie viel labeln?

| | Aufnahmen | Ausschnitte mit Poren | Ausschnitte ohne Pore (negativ) |
|---|---|---|---|
| **train** | 8 | **4 je Aufnahme = 32** | 0–1 je Aufnahme (optional, du hast schon ~70) |
| **test** | 4 | **4 je Aufnahme = 16** | **1–2 je Aufnahme = 4–8** |
| **Summe** | 12 | **~48** | ~4–16 |

- Das **Minimum**, wenn die Zeit knapp ist: train 20 mit Poren, test 10 mit Poren aus mindestens 3 Aufnahmen.
  Weniger als 10 Testbilder mit Poren schwanken zu stark, um zwei Modelle sicher zu vergleichen.
- Mehr als ~60 bringt im ersten Durchgang wenig. Ob sich mehr lohnt, zeigt der Lauf.
- Rechne mit **5–8 min pro Ausschnitt** mit Vorlabeln. Bei 48 Ausschnitten sind das also 4–6 h, gut auf mehrere Sitzungen verteilbar.
  Gespeicherte Fenster bleiben erhalten, du kannst jederzeit aufhören.

### 1.3 Labeln
```powershell
.venv\Scripts\python label_tool\label.py
```
1. Aufnahme wählen und auf **Aufnahme öffnen** klicken. Das erste Mal rechnet das Modell etwa 1 min.
2. Die Liste links beginnt mit den **unsicheren** Fenstern. Die nimm bevorzugt, daran lernt das Modell am meisten.
3. Pro Aufnahme **4 Fenster mit Poren** speichern. Nimm dazu bei den Testaufnahmen **1–2 Fenster aus „leer“**
   oder ein Fenster mit reinem Gefüge, nur wenn dort wirklich keine Pore ist.
4. **Korrigieren:**
   - **D** + Klick löscht eine falsch erkannte Pore ganz (z. B. Gefüge).
   - **B** Pinsel (links malen, rechts radieren) für Ränder und übersehene Poren. Größe mit Strg+Mausrad.
   - **P** Polygon für große Poren, Doppelklick füllt.
   - **H** blendet die Wahrscheinlichkeit ein. So siehst du, wo das Modell unsicher war.
   - **Enter** speichert und geht zum nächsten Fenster, **N** überspringt.
5. **Qualitätsregeln** (wichtig):
   - **Jede** Pore im Fenster muss in der Maske sein, auch die kleinen. Eine vergessene Pore lernt das Modell als „keine Pore“.
   - Ränder so genau wie möglich, besonders in den Testaufnahmen. Die neuen Testmasken entstehen aus den Vorschlägen des alten Modells.
     Wer nicht korrigiert, bevorzugt beim Vergleich das alte Modell.
   - Fenster, bei denen du dir unsicher bist, was Pore ist: **N** (überspringen), statt zu raten.
   - Achte besonders auf **übersehene** Poren. Das Modell neigt auf deinen Bildern zum Übersehen.

Ergebnis: `label_tool\output\images\`, `label_tool\output\masks\` und `label_tool\output\labels.csv`.

---

## 2. Daten sortieren (~30 min)

Zielstruktur (der Ordner `D:\daten` ist ein Beispiel, der Pfad ist frei wählbar):
```
D:\daten\
  train\images\   train\masks\        externe + deine alten Negative + neue Ausschnitte der 8 Train-Aufnahmen
  test\images\eigen\    test\masks\eigen\     deine Test-Negative (schon vorhanden) + neue Ausschnitte der 4 Test-Aufnahmen
  test\images\extern\   test\masks\extern\    die externen Testbilder (schon vorhanden)
```

1. Öffne `label_tool\output\labels.csv` in Excel. Die Spalte **`source`** zeigt die Aufnahme jedes Ausschnitts.
2. Nach Spalte `source` filtern:
   - Ausschnitte der **4 Test-Aufnahmen** kommen nach `test\images\eigen\`, die Masken mit gleichem Namen nach `test\masks\eigen\`.
   - Alle anderen kommen nach `train\images\` bzw. `train\masks\`.
3. Teile das vorhandene Testset auf: Deine nummerierten Bilder (`1.png`, `54.png` …) kommen nach `…\eigen\`, die externen (lange Namen) nach `…\extern\`.
   Masken genauso, mit gleichen Namen.
4. **Nichts umbenennen.** Am Namen `<Aufnahme>_y…_x…` erkennt das Training die Ausschnitte derselben Aufnahme.
5. Kontrolle nach dem Sortieren:
   - Jedes Bild hat eine Maske mit gleichem Namen im entsprechenden Ordner.
   - Keine Aufnahme taucht in train **und** test auf. Das Training warnt beim Start, falls doch.

---

## 3. Augmentierung einstellen (~30–60 min)

1. Lege **8 Bilder** nach `augment_tool\input\` und die passenden Masken nach `augment_tool\input\masks\`:
   3 eigene Ausschnitte mit Poren, 3 externe Bilder, 1 eigenes Negativ, 1 besonders helles oder dunkles Bild.
2. Starten:
   ```powershell
   .venv\Scripts\python augment_tool\preview.py
   ```
3. Schau dir `augment_tool\output\<Bild>_einzeln.png` an. Für jede Zeile gilt: **Sehen min und max noch aus wie echte Aufnahmen
   von deinem oder dem externen Mikroskop?**
   - Ist das zu stark, verkleinere den Bereich (`range`) in `augment_tool\augment_config.yaml`.
   - Ist es kaum sichtbar, obwohl deine Bilder stärker schwanken, vergrößere den Bereich.
   - Kommt eine Variation bei dir nie vor, setze `enabled: false` (z. B. `grayscale`, `jpeg`, wenn du nur TIF/PNG hast).
4. Danach `…_kombiniert.png`: Dürfen alle Patches als realistische Aufnahmen durchgehen? Wenn mehrere unrealistisch wirken, die Wahrscheinlichkeiten (`probability`) senken.
5. Ändern, neu starten, vergleichen. Meist reichen 3–5 Runden.
6. Kopiere am Ende den ganzen Abschnitt `augment:` aus `augment_tool\augment_config.yaml` nach `configs\study.yaml` und ersetze dort den vorhandenen Abschnitt `augment:`.

---

## 4. Config eintragen (~5 min)

In `configs\study.yaml` nur die Pfade unter `data:` anpassen:
```yaml
data:
  train:
    images: D:/daten/train/images
    masks: D:/daten/train/masks
  test:
    images: D:/daten/test/images
    masks: D:/daten/test/masks
```
Alles andere bleibt für den ersten Lauf, wie es ist (24 h, λ 0,005 und 0, Referenz `label_tool/model`).

---

## 5. Testlauf (~15 min) – unbedingt vor dem großen Lauf

```powershell
.venv\Scripts\python run.py --smoke
```
Beim ersten Start wird zusätzlich der Bilder-Cache angelegt, das dauert einige Minuten.

**Prüfen:**
- [ ] Er läuft ohne Fehler bis `=== fertig … ===`.
- [ ] In der Ausgabe gibt es **keine** `LECK`-Warnung.
- [ ] Die Zeile `train: … Bilder (… mit Poren, … negativ)` zeigt plausible Zahlen: ~300 Bilder, davon ~70 negativ plus deine neuen.
- [ ] Die Zeile `Split: … val (…)` zeigt in val Bilder **jeder Art** (Regel 1 Poren, ohne Regel Poren, ohne Regel negativ …).
- [ ] Die Hardware-Übersicht zeigt deine GPU, und die Batch-Größe ist 8 oder 16.
- [ ] `runs\smoke_…\augment_examples.png` sieht realistisch aus, und der magenta Maskenrand liegt auf den Poren.
- [ ] Im Trial-Balken stehen **GPU** über ~80 % und **Daten-Warten** unter ~20 %.
- [ ] Es gibt `runs\smoke_…\final\evaluation\evaluation.xlsx`. Die Zahlen sind nach 2 Mini-Epochen schlecht, das ist normal.

Bricht etwas ab, schick mir die letzten ~30 Zeilen der Konsole bzw. `runs\smoke_…\log.txt`.
Danach kannst du den Ordner `runs\smoke_…` löschen.

---

## 6. Training starten (24 h)

**Laptop vorbereiten:**
- Netzteil anschließen, Energiemodus auf **Höchstleistung**.
- Windows-Einstellung „Beim Zuklappen“ auf **Nichts unternehmen**. Den Standby verhindert das Skript selbst.
- **Windows Update für 2 Tage pausieren**, sonst droht ein Neustart in der Nacht.
- Andere Programme mit GPU-Last schließen (Spiele, Browser mit Video).
- **Windows Terminal** oder PowerShell benutzen, nicht die alte `cmd`.
- Mindestens ~10 GB frei auf dem Laufwerk.

**Starten:**
```powershell
.venv\Scripts\python run.py
```
- Nach einigen Minuten steht im Log die Messung: `ein Trial ~X min … ~N volle Trials`. Sinnvoll sind **≥ 20 Trials**.
  Sind es deutlich weniger (z. B. < 12), sag mir Bescheid. Dann kürzen wir `steps_per_epoch` oder `max_epochs`.
- **Zwischendurch:** Der obere Balken zeigt Zeit, Phase und den besten Wert, der untere den aktuellen Trial.
  `runs\<name>\trials.xlsx` und `progress.png` werden laufend aktualisiert.
- **Schon früher ausprobieren:** `runs\<name>\best_so_far\lambda0.005\` ist jederzeit ein fertiges Modell (in Pore_Detection oder im Label-Tool nutzbar).
- **Abbruch:** Strg+C. Mit `.venv\Scripts\python run.py` geht es an derselben Stelle weiter.

---

## 7. Ergebnis prüfen

1. Öffne `runs\<name>\final\evaluation\evaluation.xlsx`, Blatt **Summary**.
2. Die wichtigste Zeile ist **Gruppe `eigen`, Typ `Poren`**: Pixel-F1 und Objekt-F1 von `neu lambda0.005`, `neu lambda0` und `Referenz` vergleichen.
3. Dann **`eigen` / `negativ`: Phantome / Mpx**. Hier soll das neue Modell deutlich unter der Referenz liegen.
4. Blatt **Size classes**: Werden kleine Poren (0–5 px, 5–10 px) gefunden?
5. In `plots\` und `overlays\` sieht man, wo die Fehler liegen (orange = falsch erkannt, blau = übersehen).
6. **Modell auswählen:**
   - Wichtig sind dir wenige Phantome → `final\lambda0.005\`.
   - Phantome sind auch mit λ = 0 schon niedrig, und es findet mehr Poren → `final\lambda0\`.
7. **In Pore_Detection einsetzen:** den gewählten Ordner nach `Pore_Detection\models\<neuer_name>\` kopieren und in
   `config_unet.yaml` den Eintrag `model_dir: models/<neuer_name>` setzen.
8. Schick mir das Blatt **Summary** und `progress.png`. Dann stellen wir λ nach und entscheiden, ob mehr gelabelte Bilder lohnen.
