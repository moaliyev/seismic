import sys
from PySide6 import QtCore, QtWidgets
import numpy as np
import pyqtgraph as pg
import pyvista as pv
from pyvistaqt import QtInteractor


class DataStore(QtCore.QObject):
    """Holds the loaded volume so every window stays in sync."""

    dataChanged = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.data = None
        self.path = ""

    def load(self, file_path):
        self.data = np.load(file_path, mmap_mode="r")
        self.path = file_path
        print("Selected:", file_path)
        print("Data shape:", self.data.shape)
        self.dataChanged.emit()


class BaseWindow(QtWidgets.QWidget):
    """Common upload button + status label + navigation to the other windows."""

    def __init__(self, store, title):
        super().__init__()
        self.store = store
        self.setWindowTitle(title)
        self._needs_render = False

        self.label = QtWidgets.QLabel("No dataset selected")
        self.upload_button = QtWidgets.QPushButton("Upload Dataset")
        self.nav_layout = QtWidgets.QHBoxLayout()

        self.upload_button.clicked.connect(self.open_file_dialog)
        self.store.dataChanged.connect(self.on_data_changed)

    def add_nav(self, target, text):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(lambda: self.go_to(target))
        self.nav_layout.addWidget(button)

    def go_to(self, target):
        target.show()
        target.raise_()
        target.activateWindow()
        self.hide()

    @QtCore.Slot()
    def open_file_dialog(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Dataset",
            "",
            "Numpy Files (*.npy)"
        )
        if file_path:
            self.store.load(file_path)

    @QtCore.Slot()
    def on_data_changed(self):
        self.label.setText(self.store.path)
        # Only redraw hidden windows once they are actually shown
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


class VolumeWindow(BaseWindow):
    """First window: load a dataset and scroll through the whole volume."""

    def __init__(self, store):
        super().__init__(store, "Volume View")

        self.imv = pg.ImageView()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.imv)
        layout.addWidget(self.upload_button)
        layout.addLayout(self.nav_layout)

    def render_data(self):
        if self.store.data is None:
            return
        self.imv.setImage(img=self.store.data)


class SliceWindow(BaseWindow):
    """Second window: pick an iline or xline index and view that 2D slice."""

    def __init__(self, store):
        super().__init__(store, "Slice View")

        self.imv = pg.ImageView()

        self.iline_input = QtWidgets.QLineEdit()
        self.iline_input.setPlaceholderText("Enter iline index and press Enter...")

        self.xline_input = QtWidgets.QLineEdit()
        self.xline_input.setPlaceholderText("Enter xline index and press Enter...")

        self.iline_input.returnPressed.connect(self.update_iline_slice)
        self.xline_input.returnPressed.connect(self.update_xline_slice)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.iline_input)
        layout.addWidget(self.xline_input)
        layout.addWidget(self.imv)
        layout.addWidget(self.upload_button)
        layout.addLayout(self.nav_layout)

    def render_data(self):
        # Start on the middle iline of the real dataset, no hardcoded index
        if self.store.data is None:
            return
        idx = self.store.data.shape[0] // 2
        self.iline_input.setText(str(idx))
        self.imv.setImage(self.store.data[idx, :, :])

    def _read_index(self, line_edit, axis):
        if self.store.data is None:
            print("Please load a dataset first.")
            return None
        try:
            idx = int(line_edit.text())
        except ValueError:
            print("Invalid input! Please enter an integer.")
            return None

        # Prevent crashing if the index is too large or negative
        max_idx = self.store.data.shape[axis] - 1
        return max(0, min(idx, max_idx))

    @QtCore.Slot()
    def update_iline_slice(self):
        idx = self._read_index(self.iline_input, 0)
        if idx is None:
            return
        self.imv.setImage(self.store.data[idx, :, :])
        print(f"Showing iline at index {idx}")

    @QtCore.Slot()
    def update_xline_slice(self):
        idx = self._read_index(self.xline_input, 1)
        if idx is None:
            return
        self.imv.setImage(self.store.data[:, idx, :])
        print(f"Showing xline at index {idx}")


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


class VolumeLoader(QtCore.QObject):
    """Reads and decimates the volume off the GUI thread."""

    finished = QtCore.Signal(object, tuple)  # decimated array, strides used
    failed = QtCore.Signal(str)

    def __init__(self, data, budget):
        super().__init__()
        self.data = data
        self.budget = budget

    @QtCore.Slot()
    def run(self):
        try:
            strides = pick_strides(self.data.shape, self.budget)
            sx, sy, sz = strides
            # Strided read straight off the memmap: only the kept samples are pulled in
            small = np.ascontiguousarray(
                self.data[::sx, ::sy, ::sz], dtype=np.float32
            )
            self.finished.emit(small, strides)
        except Exception as exc:  # noqa: BLE001 - report anything back to the UI
            self.failed.emit(str(exc))


class Volume3DWindow(BaseWindow):
    """Third window: the volume in 3D with draggable orthogonal slices."""

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
        self.worker_thread = None
        self.loader = None
        self.pending = False

        self.cmap_box = QtWidgets.QComboBox()
        self.cmap_box.addItems(["gray", "seismic", "viridis"])
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
        controls.addWidget(self.status)
        controls.addStretch()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addLayout(controls)
        layout.addWidget(self.plotter.interactor)
        layout.addWidget(self.upload_button)
        layout.addLayout(self.nav_layout)

    def render_data(self):
        if self.store.data is None:
            return

        # A load is already running: queue one more pass instead of racing it
        if self.worker_thread is not None:
            self.pending = True
            return

        self.pending = False
        self.status.setText("Loading volume...")
        self.set_controls_enabled(False)

        budget = self.QUALITY_BUDGETS[self.quality_box.currentText()]
        self.worker_thread = QtCore.QThread(self)
        self.loader = VolumeLoader(self.store.data, budget)
        self.loader.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.loader.run)
        self.loader.finished.connect(self.on_loaded)
        self.loader.failed.connect(self.on_failed)
        self.worker_thread.start()

    def set_controls_enabled(self, enabled):
        self.cmap_box.setEnabled(enabled)
        self.quality_box.setEnabled(enabled)
        self.upload_button.setEnabled(enabled)

    def stop_thread(self):
        if self.worker_thread is None:
            return
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.worker_thread = None
        self.loader = None
        self.set_controls_enabled(True)

    @QtCore.Slot(object, tuple)
    def on_loaded(self, data, strides):
        self.stop_thread()

        if self.pending:  # settings changed while loading, redo with the new ones
            self.render_data()
            return

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

        full = self.store.data.shape
        self.status.setText(
            f"{full} -> {data.shape} (step {strides})"
        )

    @QtCore.Slot(str)
    def on_failed(self, message):
        self.stop_thread()
        self.pending = False
        self.status.setText(f"Load failed: {message}")
        print("3D load failed:", message)

    def closeEvent(self, event):
        # Release the worker and the VTK render window or the app hangs on exit
        self.stop_thread()
        self.plotter.close()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    store = DataStore()

    volume_window = VolumeWindow(store)
    slice_window = SliceWindow(store)
    volume_3d_window = Volume3DWindow(store)

    volume_window.add_nav(slice_window, "Slice View →")
    volume_window.add_nav(volume_3d_window, "3D View →")

    slice_window.add_nav(volume_window, "← Volume View")
    slice_window.add_nav(volume_3d_window, "3D View →")

    volume_3d_window.add_nav(volume_window, "← Volume View")
    volume_3d_window.add_nav(slice_window, "← Slice View")

    for window in (volume_window, slice_window, volume_3d_window):
        window.resize(900, 700)

    volume_window.show()

    sys.exit(app.exec())
