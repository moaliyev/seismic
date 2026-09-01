"""Shared building blocks: the data store, the tuned image view, background work.

Axis convention follows the dataset description: a volume is indexed
``[iline, xline, time]``.
"""

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

ILINE = "iline"
XLINE = "xline"
TIME = "time"
ORIENTATIONS = (ILINE, XLINE, TIME)

ORIENTATION_LABELS = {ILINE: "Iline", XLINE: "Xline", TIME: "Time slice"}

# Volume axis each orientation steps along
ORIENTATION_AXIS = {ILINE: 0, XLINE: 1, TIME: 2}

# What the two axes of an extracted slice mean, as (horizontal, vertical)
SLICE_AXES = {
    ILINE: ("xline", "time"),
    XLINE: ("iline", "time"),
    TIME: ("iline", "xline"),
}

COLORMAPS = ("gray", "seismic", "viridis", "RdBu", "magma")

SLOT_A = "A"
SLOT_B = "B"


def colormap(name):
    """Look a colormap up by its matplotlib name."""
    return pg.colormap.get(name, source="matplotlib")


def display_levels(image, percentile=99.0):
    """A robust display range for a slice.

    Post-stack amplitudes swing either side of zero, so a range centred on zero
    keeps a diverging colormap honest. Clipping at a percentile rather than the
    extremes stops one noisy sample from washing the section out.
    """
    values = np.asarray(image).ravel()
    if values.size > 200_000:  # a percentile of the whole slice is wasted work
        values = values[:: values.size // 200_000]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0

    low = float(values.min())
    high = float(values.max())
    if low < 0.0 < high:
        limit = float(np.percentile(np.abs(values), percentile))
        limit = limit or max(abs(low), abs(high)) or 1.0
        return -limit, limit

    low, high = (float(v) for v in np.percentile(values, [100.0 - percentile, percentile]))
    if high <= low:
        high = low + 1.0
    return low, high


class Volume:
    """One loaded .npy volume, kept memmapped so a large file costs no RAM."""

    def __init__(self, data, path):
        self.data = data
        self.path = path

    @property
    def shape(self):
        return tuple(self.data.shape)

    def limit(self, orientation):
        """Highest valid index along the given orientation."""
        return self.shape[ORIENTATION_AXIS[orientation]] - 1

    def clamp(self, orientation, index):
        return int(min(max(int(index), 0), self.limit(orientation)))

    def slice(self, orientation, index):
        """One 2D slice, pulled into RAM as a contiguous float32 array.

        Taking the copy here matters: an xline or time slice is a strided view
        that reaches across the whole file, and handing that straight to a
        widget makes every repaint a fresh scattered read.
        """
        axis = ORIENTATION_AXIS[orientation]
        index = self.clamp(orientation, index)
        return np.ascontiguousarray(np.take(self.data, index, axis=axis), dtype=np.float32)


class DataStore(QtCore.QObject):
    """Holds the loaded volumes and the slice every window is looking at."""

    dataChanged = QtCore.Signal()       # a volume was loaded or dropped
    selectionChanged = QtCore.Signal()  # orientation or index moved

    def __init__(self):
        super().__init__()
        self.volumes = {SLOT_A: None, SLOT_B: None}
        self.orientation = ILINE
        self.index = 0

    @property
    def primary(self):
        return self.volumes[SLOT_A]

    @property
    def secondary(self):
        return self.volumes[SLOT_B]

    def volume(self, slot):
        return self.volumes.get(slot)

    def load(self, file_path, slot=SLOT_A):
        data = np.load(file_path, mmap_mode="r")
        if data.ndim != 3:
            raise ValueError(f"expected a 3D volume, got shape {tuple(data.shape)}")

        other_slot = SLOT_B if slot == SLOT_A else SLOT_A
        other = self.volumes[other_slot]
        if other is not None and other.shape != tuple(data.shape):
            if slot == SLOT_B:
                raise ValueError(
                    f"comparison needs matching grids: volume A is {other.shape}, "
                    f"this one is {tuple(data.shape)}"
                )
            # A new primary of a different size makes the old comparison volume
            # meaningless, so drop it rather than pair up mismatched grids
            self.volumes[other_slot] = None

        self.volumes[slot] = Volume(data, file_path)
        print(f"Loaded {slot}: {file_path} {tuple(data.shape)}")

        if slot == SLOT_A:
            self.orientation = ILINE
            self.index = data.shape[0] // 2

        self.dataChanged.emit()
        self.selectionChanged.emit()

    def clear(self, slot):
        if self.volumes.get(slot) is None:
            return
        self.volumes[slot] = None
        self.dataChanged.emit()

    def describe(self):
        parts = []
        for slot in (SLOT_A, SLOT_B):
            volume = self.volumes[slot]
            if volume is not None:
                parts.append(f"{slot}: {volume.path}  {volume.shape}")
        return "   |   ".join(parts) if parts else "No dataset selected"

    def limit(self, orientation=None):
        volume = self.primary
        if volume is None:
            return 0
        return volume.limit(orientation or self.orientation)

    def set_selection(self, orientation=None, index=None):
        volume = self.primary
        if volume is None:
            return

        if orientation is not None and orientation != self.orientation:
            self.orientation = orientation
            if index is None:
                # Land in the middle of the new axis instead of keeping an index
                # that meant something else
                index = volume.shape[ORIENTATION_AXIS[orientation]] // 2

        if index is not None:
            self.index = index

        self.index = volume.clamp(self.orientation, self.index)
        self.selectionChanged.emit()

    def current_slice(self, slot=SLOT_A):
        volume = self.volumes.get(slot)
        if volume is None:
            return None
        return volume.slice(self.orientation, self.index)

    def selection_label(self):
        return f"{ORIENTATION_LABELS[self.orientation]} {self.index}"


class SeismicImageView(pg.ImageView):
    """ImageView whose ROI readout stays usable on a memmapped seismic volume.

    pyqtgraph builds the ROI curve by averaging the ROI region over *every*
    frame, and it redoes that on each mouse-move while the ROI is dragged. On a
    memmapped multi-gigabyte volume one update is a scattered read across the
    whole file, so a drag queues up hundreds of them and the window freezes.
    Two guards fix that: coalesce the move burst into a single update, and read
    only every n-th frame so one update has a bounded cost.
    """

    ROI_UPDATE_DELAY_MS = 80        # wait for the drag to settle
    ROI_SAMPLE_BUDGET = 8_000_000   # samples pulled per ROI update
    ROI_CURVE_POINTS = 1_000        # the plot cannot resolve more than this

    _roi_timer = None  # base __init__ may fire roiChanged before ours runs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._roi_timer = QtCore.QTimer(self)
        self._roi_timer.setSingleShot(True)
        self._roi_timer.setInterval(self.ROI_UPDATE_DELAY_MS)
        self._roi_timer.timeout.connect(self.update_roi_curve)

    @QtCore.Slot()
    def roiChanged(self):
        # pyqtgraph calls this per mouse event; only the last one matters
        if self._roi_timer is None:
            super().roiChanged()
        else:
            self._roi_timer.start()

    def frame_step(self):
        """Read every n-th frame so one ROI update stays inside the budget."""
        t_axis = self.axes["t"]
        if t_axis is None:  # 2D image: the ROI only spans the visible slice
            return 1
        frames = self.getProcessedImage().shape[t_axis]
        size = self.roi.size()
        per_frame = max(1.0, abs(size[0]) * abs(size[1]))
        # Two ceilings: how much of the file one update may pull in, and how
        # many points the curve can actually show. A deep volume hits the
        # second one, a wide ROI the first.
        by_samples = -(-frames * per_frame // self.ROI_SAMPLE_BUDGET)
        by_points = -(-frames // self.ROI_CURVE_POINTS)
        step = int(max(by_samples, by_points))
        return max(1, min(step, frames))

    @QtCore.Slot()
    def update_roi_curve(self):
        if self.image is None:
            return

        step = self.frame_step()
        if step == 1:
            super().roiChanged()
            return

        # Hand the base implementation a strided view instead: same x/y extent,
        # so the ROI still maps onto it, but only a fraction of the frames are
        # touched. The full arrays go back afterwards for display and export.
        t_axis = self.axes["t"]
        full_image = self.getProcessedImage()
        full_tvals = self.tVals

        frames = [slice(None)] * full_image.ndim
        frames[t_axis] = slice(None, None, step)

        self.imageDisp = full_image[tuple(frames)]
        self.tVals = full_tvals[::step]
        try:
            super().roiChanged()
        finally:
            self.imageDisp = full_image
            self.tVals = full_tvals

    def label_axes(self, horizontal, vertical):
        plot = self.getView()
        if isinstance(plot, pg.PlotItem):
            plot.setLabel("bottom", horizontal)
            plot.setLabel("left", vertical)


def slice_view(show_buttons=False):
    """An image view set up for a single 2D slice rather than a stack.

    A PlotItem view gives the slice real axes; the built-in ROI and menu buttons
    are hidden because the windows that need an ROI place their own.
    """
    view = SeismicImageView(view=pg.PlotItem())
    view.ui.roiBtn.setVisible(show_buttons)
    view.ui.menuBtn.setVisible(show_buttons)
    view.setColorMap(colormap("gray"))
    return view


class _Worker(QtCore.QObject):
    """Runs one callable and reports back; lives on the worker thread."""

    done = QtCore.Signal(bool, object)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @QtCore.Slot()
    def run(self):
        try:
            self.done.emit(True, self.fn())
        except Exception as exc:  # noqa: BLE001 - anything here belongs in the UI
            self.done.emit(False, exc)


class TaskRunner(QtCore.QObject):
    """Runs one background callable at a time; a newer request replaces the queued one.

    Reading or resampling a slice out of a multi-gigabyte memmap is slow enough
    to stall the event loop, and the controls that trigger it (sliders, combo
    boxes) fire far faster than the work completes. Superseding the queued
    request keeps the view current without piling up stale reads.
    """

    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    busyChanged = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._worker = None
        self._pending = None

    @property
    def busy(self):
        return self._thread is not None

    def submit(self, fn):
        if self.busy:
            self._pending = fn
            return

        self._pending = None
        self._thread = QtCore.QThread(self)
        self._worker = _Worker(fn)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._thread.start()
        self.busyChanged.emit(True)

    @QtCore.Slot(bool, object)
    def _on_done(self, ok, payload):
        self._teardown()

        pending, self._pending = self._pending, None
        if pending is not None:  # settings moved on while we were working
            self.submit(pending)
            return

        self.busyChanged.emit(False)
        if ok:
            self.finished.emit(payload)
        else:
            self.failed.emit(str(payload))

    def stop(self):
        self._pending = None
        self._teardown()

    def _teardown(self):
        if self._thread is None:
            return
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None


class SliceSelector(QtWidgets.QWidget):
    """Orientation and index controls bound to the shared selection.

    Every window that shows a slice uses one of these, so picking a slice in one
    window is the same act as picking it in another.
    """

    PUSH_DELAY_MS = 60  # let a slider drag settle before reading the volume

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self._syncing = False

        self.orientation_box = QtWidgets.QComboBox()
        for orientation in ORIENTATIONS:
            self.orientation_box.addItem(ORIENTATION_LABELS[orientation], orientation)

        self.spin = QtWidgets.QSpinBox()
        self.spin.setMaximumWidth(90)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.limit_label = QtWidgets.QLabel("of -")

        self._push_timer = QtCore.QTimer(self)
        self._push_timer.setSingleShot(True)
        self._push_timer.setInterval(self.PUSH_DELAY_MS)
        self._push_timer.timeout.connect(self._push)

        self.orientation_box.currentIndexChanged.connect(self._on_orientation)
        self.spin.valueChanged.connect(self._on_index)
        self.slider.valueChanged.connect(self._on_index)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Slice:"))
        layout.addWidget(self.orientation_box)
        layout.addWidget(self.spin)
        layout.addWidget(self.limit_label)
        layout.addWidget(self.slider, 1)

        store.dataChanged.connect(self.sync)
        store.selectionChanged.connect(self.sync)
        self.sync()

    @QtCore.Slot()
    def sync(self):
        """Pull the current selection back into the controls."""
        volume = self.store.primary
        self._syncing = True
        try:
            self.setEnabled(volume is not None)
            limit = volume.limit(self.store.orientation) if volume else 0
            self.limit_label.setText(f"of {limit}" if volume else "of -")
            self.orientation_box.setCurrentIndex(
                self.orientation_box.findData(self.store.orientation)
            )
            for widget in (self.spin, self.slider):
                widget.setRange(0, limit)
                widget.setValue(self.store.index)
        finally:
            self._syncing = False

    @QtCore.Slot(int)
    def _on_orientation(self, _index):
        if self._syncing:
            return
        self.store.set_selection(orientation=self.orientation_box.currentData())

    @QtCore.Slot(int)
    def _on_index(self, value):
        if self._syncing:
            return
        self._syncing = True
        try:  # keep the spin box and the slider showing the same number
            self.spin.setValue(value)
            self.slider.setValue(value)
        finally:
            self._syncing = False
        self._push_timer.start()

    @QtCore.Slot()
    def _push(self):
        self.store.set_selection(index=self.spin.value())
