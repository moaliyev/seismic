"""The application windows.

Every window shares one DataStore, so the volumes and the currently selected
slice are the same wherever you look; navigating between windows only changes
what is drawn, never what is selected.
"""

import os

import numpy as np
import pyqtgraph as pg
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

import processing
import slicing
import spectrum
from core import (
    COLORMAPS,
    SLICE_AXES,
    SLOT_A,
    SLOT_B,
    TIME,
    SeismicImageView,
    SliceSelector,
    TaskRunner,
    colormap,
    display_levels,
    slice_view,
)


class BaseWindow(QtWidgets.QWidget):
    """Shared chrome: dataset label, upload button, navigation to the others."""

    #: set by subclasses that redraw when the selected slice moves
    follows_selection = False

    def __init__(self, store, title):
        super().__init__()
        self.store = store
        self.setWindowTitle(title)
        self._needs_render = False

        self.label = QtWidgets.QLabel(store.describe())
        self.label.setWordWrap(True)
        self.upload_button = QtWidgets.QPushButton("Upload Dataset")
        self.upload_button.clicked.connect(lambda: self.upload(SLOT_A))
        self.nav_layout = QtWidgets.QHBoxLayout()

        self.store.dataChanged.connect(self.on_data_changed)
        if self.follows_selection:
            self.store.selectionChanged.connect(self.schedule_render)

    # -- chrome ----------------------------------------------------------

    def add_nav(self, target, text):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(lambda: self.go_to(target))
        self.nav_layout.addWidget(button)

    def go_to(self, target):
        target.show()
        target.raise_()
        target.activateWindow()
        self.hide()

    def footer(self):
        """The upload button and navigation row every window ends with."""
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.upload_button)
        row.addStretch()
        row.addLayout(self.nav_layout)
        return row

    @QtCore.Slot()
    def upload(self, slot=SLOT_A):
        caption = "Open Volume" if slot == SLOT_A else "Open Comparison Volume"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, caption, "", "NumPy Files (*.npy)"
        )
        if not file_path:
            return
        try:
            self.store.load(file_path, slot)
        except Exception as exc:  # noqa: BLE001 - a bad file is the user's problem to see
            QtWidgets.QMessageBox.warning(self, "Could not load volume", str(exc))

    # -- redraw scheduling ------------------------------------------------

    @QtCore.Slot()
    def on_data_changed(self):
        self.label.setText(self.store.describe())
        self.schedule_render()

    @QtCore.Slot()
    def schedule_render(self):
        # A hidden window redraws when it is shown, not while it is out of sight
        if self.isVisible():
            self.render_data()
        else:
            self._needs_render = True

    def showEvent(self, event):
        super().showEvent(event)
        if self._needs_render:
            self._needs_render = False
            self.render_data()

    def render_data(self):
        raise NotImplementedError

    def shutdown(self):
        """Release anything that would otherwise outlive the application."""


def colormap_box(views, initial="gray"):
    """A colormap picker wired to one or more image views."""
    box = QtWidgets.QComboBox()
    box.addItems(COLORMAPS)

    def apply(name):
        for view in views:
            view.setColorMap(colormap(name))

    box.currentTextChanged.connect(apply)
    box.setCurrentText(initial)
    apply(initial)
    return box


def titled_panel(title, widget):
    """A widget under a caption, for the side-by-side splitters."""
    box = QtWidgets.QWidget()
    inner = QtWidgets.QVBoxLayout(box)
    inner.setContentsMargins(0, 0, 0, 0)
    inner.addWidget(QtWidgets.QLabel(title))
    inner.addWidget(widget)
    return box


class FilterControls(QtWidgets.QWidget):
    """Filter picker plus its parameters, for Gaussian smoothing and sharpening."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.filter_box = QtWidgets.QComboBox()
        self.filter_box.addItems(processing.FILTERS)

        self.sigma_spin = QtWidgets.QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 25.0)
        self.sigma_spin.setSingleStep(0.5)
        self.sigma_spin.setValue(1.5)
        self.sigma_spin.setSuffix(" samples")

        self.amount_spin = QtWidgets.QDoubleSpinBox()
        self.amount_spin.setRange(0.0, 5.0)
        self.amount_spin.setSingleStep(0.25)
        self.amount_spin.setValue(1.0)

        self.sigma_label = QtWidgets.QLabel("Sigma:")
        self.amount_label = QtWidgets.QLabel("Amount:")

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Filter:"))
        layout.addWidget(self.filter_box)
        layout.addWidget(self.sigma_label)
        layout.addWidget(self.sigma_spin)
        layout.addWidget(self.amount_label)
        layout.addWidget(self.amount_spin)

        self.filter_box.currentTextChanged.connect(self._on_filter)
        self.sigma_spin.valueChanged.connect(lambda _: self.changed.emit())
        self.amount_spin.valueChanged.connect(lambda _: self.changed.emit())
        self._on_filter(self.filter_box.currentText())

    @QtCore.Slot(str)
    def _on_filter(self, name):
        # Sigma drives both filters; amount only means something for sharpening
        active = name != processing.NONE
        for widget in (self.sigma_label, self.sigma_spin):
            widget.setEnabled(active)
        for widget in (self.amount_label, self.amount_spin):
            widget.setEnabled(name == processing.SHARPEN)
        self.changed.emit()

    @property
    def name(self):
        return self.filter_box.currentText()

    @property
    def active(self):
        return self.name != processing.NONE

    def apply(self, image):
        return processing.apply_filter(
            image, self.name, self.sigma_spin.value(), self.amount_spin.value()
        )

    def describe(self):
        return processing.describe(self.name, self.sigma_spin.value(), self.amount_spin.value())


class SliceReader(QtCore.QObject):
    """Reads the selected slice off the worker thread and caches the last one.

    Filter and colormap changes redraw from the cache; only moving the selection
    goes back to the file.
    """

    ready = QtCore.Signal(object)   # the raw slice
    failed = QtCore.Signal(str)
    busyChanged = QtCore.Signal(bool)

    def __init__(self, store, slot=SLOT_A, parent=None):
        super().__init__(parent)
        self.store = store
        self.slot = slot
        self.key = None
        self.data = None

        self.runner = TaskRunner(self)
        self.runner.finished.connect(self._on_finished)
        self.runner.failed.connect(self.failed)
        self.runner.busyChanged.connect(self.busyChanged)

    def request(self, force=False):
        volume = self.store.volume(self.slot)
        if volume is None:
            self.key = None
            self.data = None
            self.ready.emit(None)
            return

        key = (self.slot, self.store.orientation, self.store.index)
        if key == self.key and self.data is not None and not force:
            self.ready.emit(self.data)
            return

        orientation, index = self.store.orientation, self.store.index
        self.runner.submit(lambda: (key, volume.slice(orientation, index)))

    def matches_selection(self):
        """True when the cached slice is the one currently selected."""
        return self.key == (self.slot, self.store.orientation, self.store.index)

    @QtCore.Slot(object)
    def _on_finished(self, payload):
        self.key, self.data = payload
        self.ready.emit(self.data)

    def stop(self):
        self.runner.stop()


class VolumeWindow(BaseWindow):
    """Scroll through the whole volume, one frame at a time."""

    def __init__(self, store):
        super().__init__(store, "Volume View")

        self.imv = SeismicImageView()
        self.cmap_box = colormap_box([self.imv])

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Colormap:"))
        controls.addWidget(self.cmap_box)
        controls.addWidget(
            QtWidgets.QLabel("Frames run along the iline axis. Use the ROI button for a mean-amplitude trace.")
        )
        controls.addStretch()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addLayout(controls)
        layout.addWidget(self.imv)
        layout.addLayout(self.footer())

    def render_data(self):
        volume = self.store.primary
        if volume is None:
            return
        self.imv.setImage(img=volume.data)


class SliceWindow(BaseWindow):
    """A single 2D slice, optionally smoothed or sharpened."""

    follows_selection = True

    def __init__(self, store):
        super().__init__(store, "Slice View")

        self.imv = slice_view()
        self.reader = SliceReader(store, SLOT_A, self)
        self.reader.ready.connect(self.show_slice)
        self.reader.failed.connect(self.on_failed)
        self.reader.busyChanged.connect(self.on_busy)

        self.selector = SliceSelector(store)
        self.filters = FilterControls()
        self.filters.changed.connect(self.redraw)
        self.cmap_box = colormap_box([self.imv])
        self.status = QtWidgets.QLabel("")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.filters)
        controls.addWidget(QtWidgets.QLabel("Colormap:"))
        controls.addWidget(self.cmap_box)
        controls.addStretch()
        controls.addWidget(self.status)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.selector)
        layout.addLayout(controls)
        layout.addWidget(self.imv)
        layout.addLayout(self.footer())

    def render_data(self):
        self.reader.request()

    @QtCore.Slot()
    def redraw(self):
        if self.reader.data is not None:
            self.show_slice(self.reader.data)

    @QtCore.Slot(object)
    def show_slice(self, raw):
        if raw is None:
            return
        horizontal, vertical = SLICE_AXES[self.store.orientation]
        self.imv.label_axes(horizontal, vertical)
        # Levels come from the unfiltered slice so sharpening does not visibly
        # rescale the section along with everything else it changes
        levels = display_levels(raw)
        self.imv.setImage(self.filters.apply(raw), autoRange=False, autoLevels=False, levels=levels)
        self.status.setText(
            f"{self.store.selection_label()}  {raw.shape[0]}x{raw.shape[1]}  ({self.filters.describe()})"
        )

    @QtCore.Slot(bool)
    def on_busy(self, busy):
        if busy:
            self.status.setText("Reading slice...")

    @QtCore.Slot(str)
    def on_failed(self, message):
        self.status.setText(f"Slice failed: {message}")

    def shutdown(self):
        self.reader.stop()


class CompositeSliceWindow(BaseWindow):
    """A section along a user-drawn path through the volume.

    The left panel is a map view (a time slice); the polyline drawn on it is the
    trace of the section shown on the right. Drag a handle to move the path,
    click a segment to add a bend.
    """

    def __init__(self, store):
        super().__init__(store, "Composite Slice")

        self.map_view = slice_view()
        self.map_view.label_axes("iline", "xline")
        self.section_view = slice_view()
        self.section_view.label_axes("distance along path (bins)", "time")

        self.path_roi = pg.PolyLineROI([], closed=False, pen=pg.mkPen("y", width=2))
        self.path_roi.sigRegionChangeFinished.connect(self.extract)
        self.map_view.getView().addItem(self.path_roi)

        self.runner = TaskRunner(self)
        self.runner.finished.connect(self.show_section)
        self.runner.failed.connect(self.on_failed)

        self.time_spin = QtWidgets.QSpinBox()
        self.time_spin.valueChanged.connect(self.draw_map)

        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setRange(0.25, 10.0)
        self.step_spin.setSingleStep(0.25)
        self.step_spin.setValue(1.0)
        self.step_spin.setToolTip("Spacing between sampled traces along the path, in bins")
        self.step_spin.valueChanged.connect(lambda _: self.extract())

        self.reset_button = QtWidgets.QPushButton("Reset path")
        self.reset_button.clicked.connect(self.reset_path)

        self.cmap_box = colormap_box([self.map_view, self.section_view])
        self.status = QtWidgets.QLabel("")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Map at time:"))
        controls.addWidget(self.time_spin)
        controls.addWidget(QtWidgets.QLabel("Path step:"))
        controls.addWidget(self.step_spin)
        controls.addWidget(self.reset_button)
        controls.addWidget(QtWidgets.QLabel("Colormap:"))
        controls.addWidget(self.cmap_box)
        controls.addStretch()
        controls.addWidget(self.status)

        panels = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        panels.addWidget(titled_panel("Map view - drag the path", self.map_view))
        panels.addWidget(titled_panel("Section along the path", self.section_view))
        panels.setSizes([450, 550])

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addLayout(controls)
        layout.addWidget(panels, 1)
        layout.addLayout(self.footer())

    def render_data(self):
        volume = self.store.primary
        if volume is None:
            return
        n_time = volume.shape[2]
        self.time_spin.blockSignals(True)
        self.time_spin.setRange(0, n_time - 1)
        self.time_spin.setValue(n_time // 2)
        self.time_spin.blockSignals(False)
        self.draw_map()
        self.reset_path()

    @QtCore.Slot()
    def draw_map(self):
        volume = self.store.primary
        if volume is None:
            return
        image = volume.slice(TIME, self.time_spin.value())
        self.map_view.setImage(
            image, autoRange=False, autoLevels=False, levels=display_levels(image)
        )

    @QtCore.Slot()
    def reset_path(self):
        volume = self.store.primary
        if volume is None:
            return
        n_iline, n_xline, _ = volume.shape
        # A diagonal is a reasonable starting cut and shows both handle types
        self.path_roi.setPoints(
            [(0.2 * n_iline, 0.2 * n_xline), (0.8 * n_iline, 0.8 * n_xline)]
        )
        self.map_view.getView().autoRange()
        self.extract()

    def path_vertices(self):
        """Handle positions in volume coordinates, as (iline, xline) pairs."""
        image_item = self.map_view.getImageItem()
        vertices = []
        for _, scene_pos in self.path_roi.getSceneHandlePositions():
            point = image_item.mapFromScene(scene_pos)
            vertices.append((point.x(), point.y()))
        return vertices

    @QtCore.Slot()
    def extract(self):
        volume = self.store.primary
        if volume is None:
            return
        vertices = self.path_vertices()
        if len(vertices) < 2:
            return

        step = self.step_spin.value()
        self.status.setText("Extracting...")
        self.runner.submit(lambda: slicing.composite_slice(volume.data, vertices, step))

    @QtCore.Slot(object)
    def show_section(self, payload):
        section, distance = payload
        self.section_view.setImage(
            section, autoRange=True, autoLevels=False, levels=display_levels(section)
        )
        self.status.setText(
            f"{section.shape[0]} traces over {distance[-1]:.1f} bins x {section.shape[1]} samples"
        )

    @QtCore.Slot(str)
    def on_failed(self, message):
        self.status.setText(f"Extraction failed: {message}")

    def shutdown(self):
        self.runner.stop()


def pick_strides(shape, budget):
    """Per-axis step sizes that keep the decimated volume under `budget` points."""
    strides = [1, 1, 1]

    def decimated(axis):
        return -(-shape[axis] // strides[axis])  # ceil division

    while decimated(0) * decimated(1) * decimated(2) > budget:
        # Thin the axis that still has the most samples, so the volume stays cubic-ish
        axis = max(range(3), key=decimated)
        if decimated(axis) <= 2:
            break
        strides[axis] += 1
    return tuple(strides)


def decimate(data, budget):
    """Read a strided copy of the volume small enough to hand to VTK."""
    strides = pick_strides(data.shape, budget)
    sx, sy, sz = strides
    # Strided read straight off the memmap: only the kept samples are pulled in
    small = np.ascontiguousarray(data[::sx, ::sy, ::sz], dtype=np.float32)
    return small, strides


class Volume3DWindow(BaseWindow):
    """The volume in 3D with draggable orthogonal slices."""

    # Point budgets: a full 1 GB float32 volume is ~268M samples, far too many
    # to push into VTK, so the volume is decimated before it is rendered.
    QUALITY_BUDGETS = {
        "Low (~1M points)": 1_000_000,
        "Medium (~4M points)": 4_000_000,
        "High (~16M points)": 16_000_000,
    }

    def __init__(self, store):
        super().__init__(store, "3D Volume View")

        self.plotter = QtInteractor(self)
        self.runner = TaskRunner(self)
        self.runner.finished.connect(self.on_loaded)
        self.runner.failed.connect(self.on_failed)
        self.runner.busyChanged.connect(self.on_busy)

        self.cmap_box = QtWidgets.QComboBox()
        self.cmap_box.addItems(COLORMAPS)
        self.cmap_box.currentTextChanged.connect(lambda _: self.render_data())

        self.quality_box = QtWidgets.QComboBox()
        self.quality_box.addItems(list(self.QUALITY_BUDGETS))
        self.quality_box.setCurrentIndex(1)
        self.quality_box.currentTextChanged.connect(lambda _: self.render_data())

        self.status = QtWidgets.QLabel("")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Colormap:"))
        controls.addWidget(self.cmap_box)
        controls.addWidget(QtWidgets.QLabel("Detail:"))
        controls.addWidget(self.quality_box)
        controls.addWidget(QtWidgets.QLabel("Drag a slice plane to move it through the volume."))
        controls.addStretch()
        controls.addWidget(self.status)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addLayout(controls)
        layout.addWidget(self.plotter.interactor, 1)
        layout.addLayout(self.footer())

    def render_data(self):
        volume = self.store.primary
        if volume is None:
            return
        budget = self.QUALITY_BUDGETS[self.quality_box.currentText()]
        self.runner.submit(lambda: decimate(volume.data, budget))

    @QtCore.Slot(bool)
    def on_busy(self, busy):
        if busy:
            self.status.setText("Loading volume...")
        for widget in (self.cmap_box, self.quality_box, self.upload_button):
            widget.setEnabled(not busy)

    @QtCore.Slot(object)
    def on_loaded(self, payload):
        data, strides = payload

        # PyVista understands a 3D matrix as an ImageData grid
        grid = pv.ImageData()
        grid.dimensions = data.shape
        # Spacing compensates for the decimation, so the volume keeps its real shape
        grid.spacing = strides
        # Point data must be flattened in Fortran order to match PyVista's axes
        grid.point_data["Amplitude"] = data.flatten(order="F")

        self.plotter.clear()
        # Draggable orthogonal slices the user can move through the volume
        self.plotter.add_mesh_slice_orthogonal(
            grid,
            scalars="Amplitude",
            cmap=self.cmap_box.currentText(),
            show_scalar_bar=True,
        )
        self.plotter.show_axes()
        self.plotter.reset_camera()

        self.status.setText(f"{self.store.primary.shape} -> {data.shape} (step {strides})")

    @QtCore.Slot(str)
    def on_failed(self, message):
        self.status.setText(f"Load failed: {message}")
        print("3D load failed:", message)

    def shutdown(self):
        self.runner.stop()
        self.plotter.close()

    def closeEvent(self, event):
        # Release the VTK render window or the app hangs on exit
        self.shutdown()
        super().closeEvent(event)


class ComparePanel(QtWidgets.QWidget):
    """One side of the comparison: a slice with a crosshair that can be pinned."""

    positionPicked = QtCore.Signal(float, float)

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.image = None

        self.title = QtWidgets.QLabel(title)
        self.view = slice_view()

        pen = pg.mkPen("#ffd400", width=1)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        for line in (self.v_line, self.h_line):
            line.setVisible(False)
            line.setZValue(10)
            self.view.getView().addItem(line, ignoreBounds=True)

        self.view.getView().scene().sigMouseClicked.connect(self._on_click)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title)
        layout.addWidget(self.view)

    def set_image(self, image, levels, axes):
        self.image = image
        if image is None:
            self.view.clear()
            return
        self.view.label_axes(*axes)
        self.view.setImage(image, autoRange=False, autoLevels=False, levels=levels)

    @QtCore.Slot(object)
    def _on_click(self, event):
        if self.image is None or event.button() != QtCore.Qt.LeftButton:
            return
        point = self.view.getImageItem().mapFromScene(event.scenePos())
        x, y = point.x(), point.y()
        if 0 <= x < self.image.shape[0] and 0 <= y < self.image.shape[1]:
            self.positionPicked.emit(x, y)

    def set_marker(self, x, y):
        self.v_line.setPos(x)
        self.h_line.setPos(y)
        for line in (self.v_line, self.h_line):
            line.setVisible(True)

    def clear_marker(self):
        for line in (self.v_line, self.h_line):
            line.setVisible(False)

    def amplitude_at(self, x, y):
        if self.image is None:
            return None
        col, row = int(x), int(y)
        if 0 <= col < self.image.shape[0] and 0 <= row < self.image.shape[1]:
            return float(self.image[col, row])
        return None


class CompareWindow(BaseWindow):
    """Two slices side by side, sharing a slice selection and a crosshair.

    The right panel is either a filtered copy of the same slice or the matching
    slice from a second volume, so "before and after" covers both processing
    done here and processing done elsewhere.
    """

    follows_selection = True

    FILTERED = "Filtered copy of A"
    SECOND_VOLUME = "Volume B (second file)"

    def __init__(self, store):
        super().__init__(store, "Compare Slices")

        self.left = ComparePanel("A - original")
        self.right = ComparePanel("B")
        for panel in (self.left, self.right):
            panel.positionPicked.connect(self.pick_position)
        # One pan/zoom for both panels, so the sections never drift apart
        self.right.view.getView().setXLink(self.left.view.getView())
        self.right.view.getView().setYLink(self.left.view.getView())

        self.reader_a = SliceReader(store, SLOT_A, self)
        self.reader_a.ready.connect(self.on_left_ready)
        self.reader_b = SliceReader(store, SLOT_B, self)
        self.reader_b.ready.connect(self.on_right_ready)

        self.source_box = QtWidgets.QComboBox()
        self.source_box.addItems([self.FILTERED, self.SECOND_VOLUME])
        self.source_box.currentTextChanged.connect(lambda _: self.render_data())

        self.load_b_button = QtWidgets.QPushButton("Load volume B...")
        self.load_b_button.clicked.connect(lambda: self.upload(SLOT_B))

        self.selector = SliceSelector(store)
        self.filters = FilterControls()
        self.filters.changed.connect(self.redraw)
        self.cmap_box = colormap_box([self.left.view, self.right.view])

        self.readout = QtWidgets.QLabel("Click either panel to mark the same location in both.")
        self.status = QtWidgets.QLabel("")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Right panel:"))
        top.addWidget(self.source_box)
        top.addWidget(self.load_b_button)
        top.addWidget(QtWidgets.QLabel("Colormap:"))
        top.addWidget(self.cmap_box)
        top.addStretch()
        top.addWidget(self.status)

        panels = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        panels.addWidget(self.left)
        panels.addWidget(self.right)
        panels.setSizes([500, 500])

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.selector)
        layout.addWidget(self.filters)
        layout.addLayout(top)
        layout.addWidget(panels, 1)
        layout.addWidget(self.readout)
        layout.addLayout(self.footer())

    @property
    def comparing_files(self):
        return self.source_box.currentText() == self.SECOND_VOLUME

    def render_data(self):
        self.filters.setEnabled(not self.comparing_files)
        self.reader_a.request()
        if self.comparing_files:
            self.reader_b.request()
        else:
            self.redraw()

    @QtCore.Slot()
    def redraw(self):
        if self.reader_a.data is not None:
            self.on_left_ready(self.reader_a.data)

    @QtCore.Slot(object)
    def on_left_ready(self, raw):
        if raw is None:
            return
        axes = SLICE_AXES[self.store.orientation]
        levels = display_levels(raw)
        self.left.set_image(raw, levels, axes)
        self.left.title.setText(f"A - original ({self.store.selection_label()})")

        if self.comparing_files:
            # Only paint B if its cached slice is the one now selected; a slower
            # read would otherwise leave the two panels a slice apart, which is
            # exactly the mistake this window exists to prevent
            if self.reader_b.matches_selection() or self.store.secondary is None:
                self.on_right_ready(self.reader_b.data)
            return

        # Same levels on both sides: a sharpened section should look sharper,
        # not merely rescaled
        self.right.set_image(self.filters.apply(raw), levels, axes)
        self.right.title.setText(f"B - {self.filters.describe()}")
        self.status.setText(f"{raw.shape[0]}x{raw.shape[1]} samples")

    @QtCore.Slot(object)
    def on_right_ready(self, raw):
        if not self.comparing_files:
            return
        axes = SLICE_AXES[self.store.orientation]
        if raw is None:
            self.right.set_image(None, None, axes)
            self.right.title.setText("B - no second volume loaded")
            self.status.setText("Load a volume of the same shape to compare files.")
            return

        levels = display_levels(self.reader_a.data) if self.reader_a.data is not None else display_levels(raw)
        self.right.set_image(raw, levels, axes)
        self.right.title.setText(f"B - {os.path.basename(self.store.secondary.path)}")
        self.status.setText(f"{raw.shape[0]}x{raw.shape[1]} samples")

    @QtCore.Slot(float, float)
    def pick_position(self, x, y):
        self.left.set_marker(x, y)
        self.right.set_marker(x, y)

        horizontal, vertical = SLICE_AXES[self.store.orientation]
        left = self.left.amplitude_at(x, y)
        right = self.right.amplitude_at(x, y)
        parts = [f"{horizontal} {int(x)}", f"{vertical} {int(y)}"]
        parts.append("A = -" if left is None else f"A = {left:.4g}")
        parts.append("B = -" if right is None else f"B = {right:.4g}")
        if left is not None and right is not None:
            parts.append(f"difference = {right - left:.4g}")
        self.readout.setText("     ".join(parts))

    def shutdown(self):
        self.reader_a.stop()
        self.reader_b.stop()


class SpectrumWindow(BaseWindow):
    """Average frequency spectrum of a region, computed by the C# module."""

    follows_selection = True

    def __init__(self, store):
        super().__init__(store, "Spectrum Analysis")

        self.service = spectrum.SpectrumService()
        self.imv = slice_view()
        self.reader = SliceReader(store, SLOT_A, self)
        self.reader.ready.connect(self.show_slice)
        self.reader.failed.connect(self.on_failed)

        self.roi = pg.RectROI([0, 0], [10, 10], pen=pg.mkPen("#ffd400", width=2))
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.roi.setVisible(False)
        self.imv.getView().addItem(self.roi)
        self.roi.sigRegionChangeFinished.connect(self.analyse)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Average amplitude")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen=pg.mkPen("#3aa0ff", width=2))

        self.runner = TaskRunner(self)
        self.runner.finished.connect(self.show_spectrum)
        self.runner.failed.connect(self.on_failed)

        self.selector = SliceSelector(store)

        self.interval_spin = QtWidgets.QDoubleSpinBox()
        self.interval_spin.setRange(0.01, 100.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setValue(4.0)
        self.interval_spin.setSuffix(" ms")
        self.interval_spin.setToolTip(
            "Time between samples. A .npy file carries no header, so it has to be told."
        )
        self.interval_spin.valueChanged.connect(lambda _: self.analyse())

        self.window_box = QtWidgets.QComboBox()
        self.window_box.addItems(spectrum.WINDOWS)
        self.window_box.currentTextChanged.connect(lambda _: self.analyse())

        self.detrend_check = QtWidgets.QCheckBox("Remove mean")
        self.detrend_check.setChecked(True)
        self.detrend_check.toggled.connect(lambda _: self.analyse())

        self.full_button = QtWidgets.QPushButton("Whole slice")
        self.full_button.clicked.connect(self.reset_roi)
        self.analyse_button = QtWidgets.QPushButton("Recompute")
        self.analyse_button.clicked.connect(self.analyse)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Sample interval:"))
        controls.addWidget(self.interval_spin)
        controls.addWidget(QtWidgets.QLabel("Window:"))
        controls.addWidget(self.window_box)
        controls.addWidget(self.detrend_check)
        controls.addWidget(self.full_button)
        controls.addWidget(self.analyse_button)
        controls.addStretch()

        panels = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        panels.addWidget(titled_panel("Slice - drag the ROI box", self.imv))
        panels.addWidget(titled_panel("Average amplitude spectrum", self.plot))
        panels.setSizes([550, 450])

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.selector)
        layout.addLayout(controls)
        layout.addWidget(panels, 1)
        layout.addWidget(self.status)
        layout.addLayout(self.footer())

    def render_data(self):
        self.reader.request()

    @QtCore.Slot(object)
    def show_slice(self, raw):
        if raw is None:
            return
        horizontal, vertical = SLICE_AXES[self.store.orientation]
        self.imv.label_axes(horizontal, vertical)
        self.imv.setImage(raw, autoRange=False, autoLevels=False, levels=display_levels(raw))

        if not self.roi.isVisible() or self._roi_outside(raw.shape):
            self.reset_roi()
        else:
            self.analyse()

    def _roi_outside(self, shape):
        pos, size = self.roi.pos(), self.roi.size()
        return pos[0] + size[0] > shape[0] or pos[1] + size[1] > shape[1]

    @QtCore.Slot()
    def reset_roi(self):
        raw = self.reader.data
        if raw is None:
            return
        self.roi.setVisible(True)
        self.roi.blockSignals(True)
        self.roi.setPos([0, 0])
        self.roi.setSize([raw.shape[0], raw.shape[1]])
        self.roi.blockSignals(False)
        self.imv.getView().autoRange()
        self.analyse()

    @QtCore.Slot()
    def analyse(self):
        raw = self.reader.data
        if raw is None:
            return
        if self.store.orientation == TIME:
            self.curve.setData([], [])
            self.status.setText(
                "A time slice has no time axis. Pick an iline or xline slice to get a spectrum."
            )
            return

        region = self.roi.getArrayRegion(raw, self.imv.getImageItem())
        if region is None or region.size == 0:
            self.status.setText("The ROI does not overlap the slice.")
            return

        region = np.ascontiguousarray(region, dtype=np.float32)
        interval = self.interval_spin.value() / 1000.0
        window = self.window_box.currentText()
        detrend = self.detrend_check.isChecked()

        self.status.setText("Asking the C# module...")
        self.runner.submit(
            lambda: self.service.analyse(region, interval, window=window, detrend=detrend)
        )

    @QtCore.Slot(object)
    def show_spectrum(self, result):
        self.curve.setData(result.frequencies, result.amplitudes)
        self.plot.setXRange(0.0, result.nyquist, padding=0.02)
        executable = self.service.find_executable()
        self.status.setText(f"{result.summary()}   |   module: {executable}")

    @QtCore.Slot(str)
    def on_failed(self, message):
        self.curve.setData([], [])
        self.status.setText(f"Spectrum failed: {message}")

    def shutdown(self):
        self.runner.stop()
        self.reader.stop()
        self.service.close()
