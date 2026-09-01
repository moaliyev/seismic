"""Seismic volume visualisation and analysis tool.

Run with `python main.py`. Every window shares one data store, so a volume
loaded anywhere is a volume loaded everywhere, and the slice selected in one
window is the slice the others show.
"""

import sys

from PySide6 import QtWidgets

from core import DataStore
from views import (
    CompareWindow,
    CompositeSliceWindow,
    SliceWindow,
    SpectrumWindow,
    Volume3DWindow,
    VolumeWindow,
)

WINDOW_SPECS = (
    ("Volume", VolumeWindow),
    ("Slice", SliceWindow),
    ("Composite", CompositeSliceWindow),
    ("3D", Volume3DWindow),
    ("Compare", CompareWindow),
    ("Spectrum", SpectrumWindow),
)


def build_windows(store):
    """Create every window and give each one a button to reach the others."""
    windows = [(name, window_class(store)) for name, window_class in WINDOW_SPECS]
    for _, window in windows:
        for name, target in windows:
            if target is not window:
                window.add_nav(target, name)
    return [window for _, window in windows]


def main():
    app = QtWidgets.QApplication(sys.argv)

    store = DataStore()
    windows = build_windows(store)
    for window in windows:
        window.resize(1150, 800)

    # Only one window is visible at a time, so closing it ends the session:
    # give the hidden ones a chance to release threads and child processes
    app.aboutToQuit.connect(lambda: [window.shutdown() for window in windows])

    windows[0].show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
