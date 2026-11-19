"""Fenster des Label-Tools (PySide6). Logik: core.py."""

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QGraphicsEllipseItem, QGraphicsPathItem,
                               QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QSlider, QSpinBox, QSplitter, QToolButton, QVBoxLayout, QWidget)

import core

HELP = ("Pinsel B: links malen, rechts radieren · Radierer E · Pore löschen D: Klick entfernt eine ganze Pore · "
        "Polygon P: Punkte setzen, Doppelklick/Enter füllt (mit Shift: radiert), Rechtsklick nimmt den letzten "
        "Punkt, Esc bricht ab · Mausrad: Zoom · Strg+Mausrad oder [ ]: Pinselgröße · Leertaste/mittlere Maustaste "
        "+ ziehen: verschieben · Strg+Z/Strg+Y · M Maske · H Wahrscheinlichkeit · C Maske leeren · "
        "R Vorschlag neu · Enter speichern + weiter · N überspringen")


def qimage(array: np.ndarray) -> QImage:
    array = np.ascontiguousarray(array)
    height, width = array.shape[:2]
    fmt = QImage.Format_RGBA8888 if array.shape[2] == 4 else QImage.Format_RGB888
    return QImage(array.data, width, height, array.strides[0], fmt).copy()


# ---------------------------------------------------------------- Hintergrund

class OpenWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, settings, model, source, threshold):
        super().__init__()
        self.args = settings, model, source, threshold

    def run(self):
        try:
            settings, model, source, threshold = self.args
            opened = core.open_source(settings, model, source, threshold,
                                      lambda i, n, text: self.progress.emit(i, n, text))
            self.done.emit(opened)
        except Exception as error:      # noqa: BLE001 – jede Meldung soll im Fenster ankommen
            self.failed.emit(f"{type(error).__name__}: {error}")


# ---------------------------------------------------------------- Übersicht

class Overview(QWidget):
    """Verkleinerte Aufnahme mit Vorschlägen (gelb), gelabelten (grün) und aktuellem Fenster (rot)."""

    clicked = Signal(int, int)      # y, x in voller Auflösung

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(220)
        self.pixmap = None
        self.factor = 1.0
        self.proposals, self.labeled, self.current = [], [], None

    def set_image(self, rgb: np.ndarray, factor: float):
        self.pixmap, self.factor = QPixmap.fromImage(qimage(rgb)), factor
        self.update()

    def set_windows(self, proposals, labeled, current):
        self.proposals, self.labeled, self.current = proposals, labeled, current
        self.update()

    def _frame(self) -> tuple[float, float, float]:
        scale = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
        return (scale, (self.width() - self.pixmap.width() * scale) / 2,
                (self.height() - self.pixmap.height() * scale) / 2)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        if self.pixmap is None:
            return
        scale, ox, oy = self._frame()
        painter.drawPixmap(QRectF(ox, oy, self.pixmap.width() * scale, self.pixmap.height() * scale),
                           self.pixmap, QRectF(self.pixmap.rect()))
        f = scale / self.factor

        def rect(w):
            return QRectF(ox + w.x * f, oy + w.y * f, max(w.width * f, 2), max(w.height * f, 2))

        painter.setPen(QPen(QColor(255, 210, 0), 1))
        for w in self.proposals:
            painter.drawRect(rect(w))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 200, 80, 120))
        for w in self.labeled:
            painter.drawRect(rect(w))
        if self.current is not None:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 40, 40), 2))
            painter.drawRect(rect(self.current))

    def mousePressEvent(self, event):
        if self.pixmap is None or event.button() != Qt.LeftButton:
            return
        scale, ox, oy = self._frame()
        x = (event.position().x() - ox) / scale * self.factor
        y = (event.position().y() - oy) / scale * self.factor
        if 0 <= x < self.pixmap.width() * self.factor and 0 <= y < self.pixmap.height() * self.factor:
            self.clicked.emit(int(y), int(x))


# ---------------------------------------------------------------- Zeichenfläche

class Canvas(QGraphicsView):
    """Ein Fenster: Bild, Maske (halbtransparent, Rand deckend), optional Wahrscheinlichkeit."""

    changed = Signal()
    hover = Signal(str)

    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings
        self.setScene(QGraphicsScene(self))
        self.image_item = QGraphicsPixmapItem()
        self.heat_item = QGraphicsPixmapItem()
        self.mask_item = QGraphicsPixmapItem()
        self.path_item = QGraphicsPathItem()
        self.cursor_item = QGraphicsEllipseItem()
        for z, item in enumerate((self.image_item, self.heat_item, self.mask_item, self.path_item, self.cursor_item)):
            item.setZValue(z)
            self.scene().addItem(item)
        pen = QPen(QColor(255, 255, 255), 1)
        pen.setCosmetic(True)
        self.cursor_item.setPen(pen)
        pen = QPen(QColor(255, 80, 200), 1.5)
        pen.setCosmetic(True)
        self.path_item.setPen(pen)
        self.heat_item.setVisible(False)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setFocusPolicy(Qt.StrongFocus)

        self.mask = None            # (H, W) uint8 0 / 1
        self.prob = None            # (H, W) uint8 0..255 oder None
        self.tool = "brush"
        self.radius = settings["brush"]["radius_px"]
        self.polygon: list[tuple[int, int]] = []
        self.undo_stack, self.redo_stack = [], []
        self._stroke = None         # (Wert, letzter Punkt)
        self._pan = None
        self._space = False

    # --- Inhalt
    def set_window(self, rgb8: np.ndarray, mask: np.ndarray, prob: np.ndarray | None):
        self.mask, self.prob = mask.astype(np.uint8), prob
        self.image_item.setPixmap(QPixmap.fromImage(qimage(rgb8)))
        if prob is not None:
            heat = cv2.cvtColor(cv2.applyColorMap(np.ascontiguousarray(prob), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
            alpha = np.where(prob >= 25, 150, 0).astype(np.uint8)
            self.heat_item.setPixmap(QPixmap.fromImage(qimage(np.dstack([heat, alpha]))))
        else:
            self.heat_item.setPixmap(QPixmap())
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.polygon.clear()
        self._update_path()
        self.render_mask()
        self.scene().setSceneRect(QRectF(0, 0, rgb8.shape[1], rgb8.shape[0]))
        self.resetTransform()
        self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def set_mask(self, mask: np.ndarray, undoable: bool = True):
        if undoable:
            self.push_undo()
        self.mask = mask.astype(np.uint8)
        self.render_mask()
        self.changed.emit()

    def render_mask(self):
        if self.mask is None:
            return
        color = self.settings["display"]["mask_color"]
        alpha = int(255 * self.settings["display"]["mask_opacity"])
        filled = self.mask > 0
        edge = filled & (cv2.erode(self.mask, np.ones((3, 3), np.uint8)) == 0)
        rgba = np.zeros((*self.mask.shape, 4), np.uint8)
        rgba[filled] = (*color, alpha)
        rgba[edge] = (*color, 255)
        self.mask_item.setPixmap(QPixmap.fromImage(qimage(rgba)))

    # --- Rückgängig
    def push_undo(self):
        self.undo_stack.append(self.mask.copy())
        del self.undo_stack[:-self.settings["brush"]["undo_steps"]]
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.mask)
            self.mask = self.undo_stack.pop()
            self.render_mask()
            self.changed.emit()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.mask)
            self.mask = self.redo_stack.pop()
            self.render_mask()
            self.changed.emit()

    # --- Werkzeuge
    def set_tool(self, tool: str):
        self.tool = tool
        if tool != "polygon":
            self.cancel_polygon()
        self._update_cursor(None)

    def set_radius(self, radius: int):
        self.radius = int(np.clip(radius, 1, self.settings["brush"]["max_radius_px"]))
        self._update_cursor(None)

    def _pixel(self, event) -> tuple[int, int]:
        point = self.mapToScene(event.position().toPoint())
        return int(np.floor(point.y())), int(np.floor(point.x()))

    def _inside(self, y: int, x: int) -> bool:
        return self.mask is not None and 0 <= y < self.mask.shape[0] and 0 <= x < self.mask.shape[1]

    def _paint(self, y: int, x: int, value: int):
        last = self._stroke[1] if self._stroke else (y, x)
        cv2.line(self.mask, (last[1], last[0]), (x, y), value, thickness=2 * self.radius + 1)
        cv2.circle(self.mask, (x, y), self.radius, value, thickness=-1)
        self._stroke = (value, (y, x))
        self.render_mask()

    def _delete_pore(self, y: int, x: int):
        if not self._inside(y, x) or not self.mask[y, x]:
            return
        _, labels = cv2.connectedComponents(self.mask, connectivity=8)
        mask = self.mask.copy()
        mask[labels == labels[y, x]] = 0
        self.set_mask(mask)

    def close_polygon(self, erase: bool = False):
        if len(self.polygon) >= 3:
            mask = self.mask.copy()
            points = np.array([(x, y) for y, x in self.polygon], np.int32)
            cv2.fillPoly(mask, [points], 0 if erase else 1)
            self.set_mask(mask)
        self.cancel_polygon()

    def cancel_polygon(self):
        self.polygon.clear()
        self._update_path()

    def _update_path(self, extra=None):
        path = QPainterPath()
        points = self.polygon + ([extra] if extra and self.polygon else [])
        for i, (y, x) in enumerate(points):
            (path.moveTo if i == 0 else path.lineTo)(QPointF(x + 0.5, y + 0.5))
        self.path_item.setPath(path)

    def _update_cursor(self, pixel):
        if pixel is not None:
            self._cursor_pixel = pixel
        pixel = getattr(self, "_cursor_pixel", None)
        show = pixel is not None and self.tool in ("brush", "eraser")
        self.cursor_item.setVisible(show)
        if show:
            y, x = pixel
            r = self.radius + 0.5
            self.cursor_item.setRect(QRectF(x + 0.5 - r, y + 0.5 - r, 2 * r, 2 * r))

    # --- Maus und Tasten
    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120
        if event.modifiers() & Qt.ControlModifier:
            self.set_radius(self.radius + int(np.sign(steps)) * max(1, self.radius // 5))
            self.changed.emit()
        else:
            self.scale(1.25 ** steps, 1.25 ** steps)

    def mousePressEvent(self, event):
        if self.mask is None:
            return
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self._space):
            self._pan = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        y, x = self._pixel(event)
        left, right = event.button() == Qt.LeftButton, event.button() == Qt.RightButton
        if self.tool in ("brush", "eraser") and (left or right) and self._inside(y, x):
            value = 1 if (self.tool == "brush" and left) else 0
            self.push_undo()
            self._stroke = None
            self._paint(y, x, value)
        elif self.tool == "delete" and left:
            self._delete_pore(y, x)
        elif self.tool == "polygon":
            if left and self._inside(y, x):
                self.polygon.append((y, x))
            elif right and self.polygon:
                self.polygon.pop()
            self._update_path((y, x))

    def mouseMoveEvent(self, event):
        if self._pan is not None:
            delta = event.position() - self._pan
            self._pan = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            return
        if self.mask is None:
            return
        y, x = self._pixel(event)
        self._update_cursor((y, x))
        if self._stroke is not None and event.buttons() & (Qt.LeftButton | Qt.RightButton):
            self._paint(int(np.clip(y, 0, self.mask.shape[0] - 1)), int(np.clip(x, 0, self.mask.shape[1] - 1)),
                        self._stroke[0])
        if self.tool == "polygon":
            self._update_path((y, x))
        if self._inside(y, x):
            prob = f"   Wahrscheinlichkeit {self.prob[y, x] / 255:.2f}" if self.prob is not None else ""
            self.hover.emit(f"Pixel y {y}, x {x}{prob}")

    def mouseReleaseEvent(self, event):
        if self._pan is not None:
            self._pan = None
            self.setCursor(Qt.ArrowCursor)
            return
        if self._stroke is not None:
            self._stroke = None
            self.changed.emit()

    def mouseDoubleClickEvent(self, event):
        if self.tool == "polygon" and event.button() == Qt.LeftButton:
            self.close_polygon(bool(event.modifiers() & Qt.ShiftModifier))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space = True
            self.setCursor(Qt.OpenHandCursor)
        elif event.key() == Qt.Key_Escape:
            self.cancel_polygon()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().keyReleaseEvent(event)


# ---------------------------------------------------------------- Hauptfenster

class MainWindow(QMainWindow):
    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings
        self.model = core.Model(settings["paths"]["model"], settings)
        self.threshold = self.model.threshold
        self.opened: core.Opened | None = None
        self.current: core.Window | None = None
        self.initial = None
        self.dirty = False
        self.saved: set[int] = set()       # id() der in dieser Sitzung gespeicherten Vorschläge
        self.worker = None
        self.rng = np.random.default_rng()
        self.setWindowTitle("MicroNet Label-Tool")
        self.resize(1500, 950)
        self._build()
        self._shortcuts()
        self._fill_sources()

    # --- Aufbau
    def _build(self):
        left = QWidget()
        lay = QVBoxLayout(left)
        self.model_label = QLabel(f"Modell: {self.model.name}")
        self.model_label.setWordWrap(True)
        if not self.model.available:
            self.model_label.setText(f"Modell: keins in {self.settings['paths']['model']} – Vorschläge zufällig, "
                                     f"Masken starten leer")
        self.sources = QComboBox()
        self.open_button = QPushButton("Aufnahme öffnen")
        self.open_button.clicked.connect(self.open_source)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        self.overview = Overview()
        self.overview.clicked.connect(self.manual_window)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._row_changed)
        lay.addWidget(self.model_label)
        lay.addWidget(QLabel("Aufnahme (Ordner input):"))
        lay.addWidget(self.sources)
        lay.addWidget(self.open_button)
        lay.addWidget(self.progress)
        lay.addWidget(QLabel("Übersicht – Klick wählt ein eigenes Fenster:"))
        lay.addWidget(self.overview, 2)
        lay.addWidget(QLabel("Vorschläge (unsichere zuerst):"))
        lay.addWidget(self.list, 3)

        right = QWidget()
        rlay = QVBoxLayout(right)
        bar = QHBoxLayout()
        self.tools = QButtonGroup(self)
        for key, text, tip in (("brush", "Pinsel (B)", "links malen, rechts radieren"),
                               ("eraser", "Radierer (E)", "radieren"),
                               ("delete", "Pore löschen (D)", "Klick entfernt eine ganze Pore"),
                               ("polygon", "Polygon (P)", "Punkte setzen, Doppelklick/Enter füllt, Shift = radieren")):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setProperty("tool", key)
            self.tools.addButton(button)
            bar.addWidget(button)
        self.tools.buttons()[0].setChecked(True)
        self.tools.buttonClicked.connect(lambda b: self.canvas.set_tool(b.property("tool")))
        bar.addWidget(QLabel("Pinsel r:"))
        self.radius = QSpinBox()
        self.radius.setRange(1, self.settings["brush"]["max_radius_px"])
        self.radius.setValue(self.settings["brush"]["radius_px"])
        self.radius.valueChanged.connect(lambda v: self.canvas.set_radius(v))
        bar.addWidget(self.radius)
        bar.addSpacing(12)
        self.threshold_label = QLabel()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 99)
        self.slider.setValue(round(self.threshold * 100))
        self.slider.setFixedWidth(140)
        self.slider.setEnabled(self.model.available)
        self.slider.valueChanged.connect(self._threshold_changed)
        bar.addWidget(self.threshold_label)
        bar.addWidget(self.slider)
        self.show_mask = QCheckBox("Maske (M)")
        self.show_mask.setChecked(True)
        self.show_mask.toggled.connect(lambda on: self.canvas.mask_item.setVisible(on))
        self.show_heat = QCheckBox("Wahrscheinlichkeit (H)")
        self.show_heat.setEnabled(self.model.available)
        self.show_heat.toggled.connect(lambda on: self.canvas.heat_item.setVisible(on))
        bar.addWidget(self.show_mask)
        bar.addWidget(self.show_heat)
        bar.addStretch(1)
        for text, slot in (("Rückgängig", lambda: self.canvas.undo()), ("Wiederholen", lambda: self.canvas.redo()),
                           ("Leeren (C)", lambda: self.clear_mask()), ("Vorschlag neu (R)", lambda: self.reset_mask()),
                           ("Überspringen (N)", lambda: self.skip()),
                           ("Speichern + weiter (Enter)", lambda: self.save())):
            button = QPushButton(text)
            button.clicked.connect(slot)       # lambda: clicked(bool) soll keine Argumente setzen
            bar.addWidget(button)
        self.canvas = Canvas(self.settings)
        self.canvas.changed.connect(self._canvas_changed)
        self.canvas.hover.connect(lambda text: self.statusBar().showMessage(text))
        self.info = QLabel()
        help_label = QLabel(HELP)
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray")
        rlay.addLayout(bar)
        rlay.addWidget(self.info)
        rlay.addWidget(self.canvas, 1)
        rlay.addWidget(help_label)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([380, 1120])
        self.setCentralWidget(splitter)
        self._threshold_changed(self.slider.value())

    def _shortcuts(self):
        def key(sequence, slot):
            QShortcut(QKeySequence(sequence), self, activated=slot)

        for sequence, tool in (("B", "brush"), ("E", "eraser"), ("D", "delete"), ("P", "polygon")):
            key(sequence, lambda t=tool: self._select_tool(t))
        key("Ctrl+Z", self.canvas.undo)
        key("Ctrl+Y", self.canvas.redo)
        key("M", self.show_mask.toggle)
        key("H", lambda: self.show_heat.isEnabled() and self.show_heat.toggle())
        key("C", self.clear_mask)
        key("R", lambda: self.reset_mask())
        key("N", self.skip)
        key("[", lambda: self.radius.setValue(self.radius.value() - max(1, self.radius.value() // 5)))
        key("]", lambda: self.radius.setValue(self.radius.value() + max(1, self.radius.value() // 5)))
        for sequence in ("Return", "Enter"):
            key(sequence, self._enter)
        key("Shift+Return", lambda: self.canvas.close_polygon(erase=True))

    def _select_tool(self, tool: str):
        for button in self.tools.buttons():
            button.setChecked(button.property("tool") == tool)
        self.canvas.set_tool(tool)

    def _fill_sources(self):
        files = core.find_inputs(self.settings["paths"]["input"])
        self.source_files = files
        self.sources.addItems([p.name for p in files])
        self.open_button.setEnabled(bool(files))
        if not files:
            self.progress.setFormat(f"keine Aufnahmen in {self.settings['paths']['input']}")

    # --- Aufnahme öffnen
    def open_source(self):
        if not self._may_leave():
            return
        source = self.source_files[self.sources.currentIndex()]
        self.open_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.worker = OpenWorker(self.settings, self.model, source, self.model.threshold)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._opened)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _progress(self, index: int, total: int, text: str):
        if total > 1:
            self.progress.setRange(0, total)
            self.progress.setValue(index)
        self.progress.setFormat(text)

    def _failed(self, message: str):
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Fehler")
        self.open_button.setEnabled(True)
        QMessageBox.critical(self, "Fehler", message)

    def _opened(self, opened: core.Opened):
        self.opened = opened
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        height, width = opened.image.shape[:2]
        device = f", {self.model.device}" if self.model.device else ""
        self.progress.setFormat(f"{opened.source.name}: {width} x {height} px, {len(opened.proposals)} Vorschläge{device}")
        self.open_button.setEnabled(True)
        if self.model.device == "cpu":
            self.statusBar().showMessage("Hinweis: keine CUDA-GPU – das Modell rechnet auf der CPU")
        self.overview.set_image(opened.overview, opened.factor)
        self.list.blockSignals(True)
        self.list.clear()
        for i, w in enumerate(opened.proposals, 1):
            self.list.addItem(QListWidgetItem(self._item_text(i, w)))
        self.list.blockSignals(False)
        self.current = None
        self.dirty = False
        self.saved.clear()
        self._refresh_overview()
        if opened.proposals:
            self.list.setCurrentRow(0)

    @staticmethod
    def _item_text(i: int, w: core.Window) -> str:
        def k(v):
            return f"{v / 1000:.1f}k" if v >= 1000 else str(v)
        details = f"  unsicher {k(w.uncertain_px)}  Poren {k(w.pore_px)}" if w.kind == "Poren" else ""
        return f"{i:3d}  {w.kind}{details}   (y {w.y}, x {w.x})"

    def _refresh_overview(self):
        if self.opened:
            self.overview.set_windows(self.opened.proposals, self.opened.labeled, self.current)

    # --- Fenster wechseln
    def _may_leave(self) -> bool:
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, "Nicht gespeichert", "Die Änderungen an diesem Fenster verwerfen?")
        return answer == QMessageBox.Yes

    def _row_changed(self, row: int):
        if self.opened is None or row < 0:
            return
        window = self.opened.proposals[row]
        if window is self.current:
            return
        if not self._may_leave():
            self.list.blockSignals(True)
            if self.current in self.opened.proposals:
                self.list.setCurrentRow(self.opened.proposals.index(self.current))
            self.list.blockSignals(False)
            return
        self.show_window(window)

    def manual_window(self, y: int, x: int):
        if self.opened is None or not self._may_leave():
            return
        window = core.manual_window(y, x, self.opened.image.shape[:2], self.settings, self.rng)
        self.opened.proposals.insert(0, window)
        self.list.blockSignals(True)
        self.list.insertItem(0, QListWidgetItem(self._item_text(0, window)))
        self.list.setCurrentRow(0)
        self.list.blockSignals(False)
        self.show_window(window)

    def _crop(self, window: core.Window):
        o = self.opened
        sl = (slice(window.y, window.y + window.height), slice(window.x, window.x + window.width))
        prob = np.array(o.prob[sl]) if o.prob is not None else None
        return o.image[sl], prob

    def _proposal_mask(self, prob):
        if prob is None:
            return np.zeros((self.current.height, self.current.width), np.uint8)
        return (prob >= round(self.threshold * 255)).astype(np.uint8)

    def show_window(self, window: core.Window):
        self.current = window
        crop, prob = self._crop(window)
        self.initial = self._proposal_mask(prob)
        self.canvas.set_window(core.to_display(crop, self.settings["display"]["auto_contrast"]),
                               self.initial.copy(), prob)
        self.canvas.mask_item.setVisible(self.show_mask.isChecked())
        self.canvas.heat_item.setVisible(self.show_heat.isChecked())
        self.dirty = False
        self._update_info()
        self._refresh_overview()
        self.canvas.setFocus()

    def _update_info(self):
        w = self.current
        if w is None:
            return
        pores = cv2.connectedComponents(self.canvas.mask, connectivity=8)[0] - 1
        exists = core.exists(self.settings, self.opened.source, w)
        state = "geändert" if self.dirty else "unverändert"
        note = "   ⚠ schon gespeichert – Speichern überschreibt" if exists else ""
        self.info.setText(f"{core.label_name(self.opened.source, w)}   {w.kind}   {w.width} x {w.height} px   "
                          f"{pores} Poren in der Maske   ({state}){note}")

    def _canvas_changed(self):
        self.radius.blockSignals(True)
        self.radius.setValue(self.canvas.radius)
        self.radius.blockSignals(False)
        if self.canvas.mask is not None and self.initial is not None:
            self.dirty = bool(np.any(self.canvas.mask != self.initial))
        self._update_info()

    # --- Schwelle und Maske
    def _threshold_changed(self, value: int):
        self.threshold = value / 100
        self.threshold_label.setText(f"Schwelle {self.threshold:.2f}")
        if self.current is not None and not self.dirty:
            self.reset_mask(undoable=False)

    def reset_mask(self, undoable: bool = True):
        if self.current is None:
            return
        _, prob = self._crop(self.current)
        self.initial = self._proposal_mask(prob)
        self.canvas.set_mask(self.initial.copy(), undoable=undoable)

    def clear_mask(self):
        if self.canvas.mask is not None:
            self.canvas.set_mask(np.zeros_like(self.canvas.mask))

    # --- Speichern / weiter
    def _enter(self):
        if self.canvas.polygon:
            self.canvas.close_polygon()
        else:
            self.save()

    def _next(self):
        if self.opened is None or self.current not in self.opened.proposals:
            return
        row = self.opened.proposals.index(self.current)
        for i in range(row + 1, len(self.opened.proposals)):
            if id(self.opened.proposals[i]) not in self.saved:
                self.list.setCurrentRow(i)
                return
        self.statusBar().showMessage("Letzter Vorschlag erreicht")

    def skip(self):
        if self._may_leave():
            self.dirty = False
            self._next()

    def save(self):
        if self.current is None:
            return
        w = self.current
        if core.exists(self.settings, self.opened.source, w):
            answer = QMessageBox.question(self, "Überschreiben", "Dieses Fenster ist schon gespeichert. Überschreiben?")
            if answer != QMessageBox.Yes:
                return
        crop, _ = self._crop(w)
        try:
            name = core.save(self.settings, self.opened.source, w, crop, self.canvas.mask, self.initial,
                             self.threshold, self.model.name)
        except Exception as error:      # noqa: BLE001
            QMessageBox.critical(self, "Fehler beim Speichern", f"{type(error).__name__}: {error}")
            return
        self.opened.labeled.append(core.Window(w.y, w.x, w.height, w.width, "gelabelt"))
        self.saved.add(id(w))
        if w in self.opened.proposals:
            row = self.opened.proposals.index(w)
            item = self.list.item(row)
            item.setText("✓ " + item.text())
            item.setForeground(QColor(120, 120, 120))
        self.dirty = False
        self.statusBar().showMessage(f"gespeichert: {name}.png")
        self._refresh_overview()
        self._next()
