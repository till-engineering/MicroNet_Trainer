"""Label-Tool: Vorschläge des alten Modells auf großen Aufnahmen prüfen und korrigieren.

    python label_tool/label.py                          Einstellungen: label_tool/label_config.yaml
    python label_tool/label.py --config andere.yaml

Ablauf: Aufnahme aus input/ öffnen -> das Modell aus model/ rechnet sie einmal ganz durch (Ergebnis in
cache/) -> Liste von 1000 x 1000-Fenstern mit erkannten Poren (unsichere zuerst, die Pore an zufälliger
Stelle im Fenster) plus einige Fenster ohne erkannte Pore -> Maske korrigieren -> Enter speichert:

    output/images/<Aufnahme>_y<Zeile>_x<Spalte>.png   Ausschnitt, unverändert (8 / 16 bit)
    output/masks/<gleicher Name>.png                  Weiß = Pore, Schwarz = keine Pore
    output/labels.csv                                 Quelle, Position, Porenanzahl, negativ, Korrekturen …

Gespeicherte Fenster werden beim nächsten Öffnen nicht mehr vorgeschlagen.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import core  # noqa: E402  (setzt OPENCV_IO_MAX_IMAGE_PIXELS vor dem ersten cv2-Import)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "label_config.yaml", help="Einstellungen")
    args = parser.parse_args()
    settings = core.load_settings(args.config.resolve())

    from PySide6.QtWidgets import QApplication

    from gui import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow(settings)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
