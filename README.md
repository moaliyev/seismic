# Seismic Volume Visualisation and Analysis

A desktop tool for inspecting post-stack seismic volumes stored as `.npy`, with
a separate C# module for frequency analysis.

Volumes are indexed `[iline, xline, time]`, matching the dataset description.

## Running

```bash
cd seismic
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python main.py
```

The spectrum window needs the .NET SDK (8.0 or newer) the first time it runs, to
build the C# module. It builds it automatically; to do it up front:

```bash
dotnet build SpectrumService/SpectrumService.csproj -c Release
```

Large files are memory-mapped, so a multi-gigabyte volume opens immediately and
only the samples actually drawn are read.

## Windows

Every window shares one data store: a volume loaded anywhere is loaded
everywhere, and the slice selected in one window is the slice the others show.
The buttons along the bottom of each window switch between them.

| Window | What it does |
| --- | --- |
| **Volume** | Scrolls through the whole volume frame by frame. The `ROI` button adds a region whose mean amplitude is plotted against the frame axis. |
| **Slice** | One 2D slice — iline, xline or time — with Gaussian smoothing and sharpening. |
| **Composite** | A section along a path drawn across the map view. |
| **3D** | The volume in 3D with three draggable orthogonal slice planes. |
| **Compare** | Two slices side by side with a shared crosshair, pan and zoom. |
| **Spectrum** | Average frequency spectrum of a selected region, computed by the C# module. |

### Slice selection

The `Slice:` row (orientation, index box, slider) appears in every window that
shows a slice and drives all of them at once. Switching orientation re-centres
the index on the new axis.

### 2D and 3D interaction

* Mouse wheel zooms, left-drag pans, right-drag scales one axis.
* The histogram to the right of each 2D view sets the colour range — drag the
  ends of the region, or drag the gradient bar itself.
* The `Colormap` box switches between `gray`, `seismic`, `viridis`, `RdBu` and
  `magma`.
* In the 3D view, drag a slice plane's handle to move it through the volume.
  `Detail` trades resolution for interactivity: a full 1 GB volume is ~268M
  samples, far more than VTK will render smoothly, so the volume is decimated to
  a point budget and the grid spacing is scaled to keep its true shape.

### Gaussian smoothing and sharpening

In the **Slice** and **Compare** windows:

* **Gaussian smoothing** — `cv2.GaussianBlur` with the kernel derived from
  `Sigma` (in samples), reflecting at the edges so no dark rim appears.
* **Sharpening** — an unsharp mask: `image + amount * (image - blurred)`.
  `Sigma` sets the scale of detail restored, `Amount` how strongly.

Display levels are computed from the *unfiltered* slice and reused, so a
sharpened section reads as genuinely sharper rather than merely rescaled.
Changing a filter redraws from the cached slice; only moving the selection goes
back to the file.

### Composite slice

The left panel is a map view (a time slice). The yellow polyline on it is the
trace of the section shown on the right.

* Drag a handle to move an end of the path.
* Click a segment to add a bend there; right-click a handle to remove it.
* `Map at time` changes which time slice the map shows.
* `Path step` is the spacing between sampled traces, in bins.
* `Reset path` restores the starting diagonal.

Each output trace is bilinearly interpolated from the four surrounding traces,
so the section is smooth rather than staircased.

### Slice synchronisation

The **Compare** window puts two versions of the same slice side by side. The
right panel is either:

* **Filtered copy of A** — the same slice with the smoothing or sharpening
  above applied, i.e. before and after processing; or
* **Volume B (second file)** — the matching slice from a second `.npy`, loaded
  with `Load volume B...`. It must have the same shape as volume A.

Both panels share one pan and zoom, and both use volume A's colour range.
Clicking either panel pins a crosshair at that location in *both*, and the
readout underneath reports the coordinates, the amplitude in each volume and the
difference between them.

### Spectrum analysis

The slice is on the left with a yellow ROI box; drag it or its corner handle to
choose the region. `Whole slice` resets it to everything. The spectrum on the
right updates when the drag ends.

* `Sample interval` — a `.npy` file has no header, so the time step has to be
  supplied. It sets the frequency axis: Nyquist is `1 / (2 * interval)`.
* `Window` — `hann` (default) or `none`.
* `Remove mean` — detrends each trace so a DC offset does not swamp the low end.

Every trace in the region is transformed separately and the magnitudes are
averaged. A time slice has no time axis, so no spectrum is offered for one.

## Layout

```
seismic/
  main.py            entry point: builds the windows and wires navigation
  core.py            Volume, DataStore, SeismicImageView, TaskRunner, SliceSelector
  processing.py      Gaussian smoothing and unsharp-mask sharpening
  slicing.py         composite slice extraction along a polyline
  spectrum.py        client for the C# module (the Python end of the pipe)
  views.py           the six windows
  SpectrumService/   the C# module — independent, no reference to the app
    Program.cs         request loop and self-test
    Protocol.cs        frame reading and writing
    SpectrumAnalyzer.cs  windowing, averaging, normalisation
    Fft.cs             radix-2 Cooley-Tukey FFT
```

## The C# spectrum module

`SpectrumService` is a standalone .NET console executable. It has no UI, no
dependency on the Python application and no NuGet packages — the FFT is
implemented directly so the module builds with nothing but the SDK. It can be
driven by anything that can write to a pipe.

Check it independently:

```bash
dotnet run --project SpectrumService -c Release -- --selftest
```

This transforms a 25 Hz sine sampled at 4 ms and confirms the peak lands in the
25 Hz bin at the sine's own amplitude.

### Communication method

The application starts the executable once as a **child process** and keeps it
alive, exchanging frames over its **stdin and stdout pipes**:

* request frames are written to the module's stdin;
* response frames are read from its stdout;
* diagnostics go to stderr, which keeps stdout purely binary;
* closing stdin is the shutdown signal, and the module exits with status 0.

The process is reused across requests, so only the first analysis pays the
startup cost — subsequent ones take about a millisecond. A malformed *request*
is answered with an error frame and the pipe stays usable; a malformed *frame*
cannot be resynchronised, so the module exits and the next request restarts it.

### Data exchange format

Both directions use the same shape: a 4-byte magic, a little-endian `uint32`
header length, a UTF-8 JSON header, then a raw block of little-endian `float32`.

```
request    "SPQ1" | uint32 headerLength | header JSON | float32[rows * cols]
response   "SPR1" | uint32 headerLength | header JSON | float32[bins]
```

Request header:

```json
{"rows": 240, "cols": 800, "sampleInterval": 0.004, "window": "hann", "detrend": true}
```

| Field | Meaning |
| --- | --- |
| `rows` | number of traces in the region |
| `cols` | samples per trace |
| `sampleInterval` | time between samples, in seconds |
| `window` | `hann` or `none` |
| `detrend` | remove each trace's mean before transforming |

The payload is the region itself, row-major: one trace per row, time running
along the row. That is exactly how a slice comes out of the store, so neither
side has to transpose or re-pack anything.

Response header, on success:

```json
{"status": "ok", "bins": 513, "binWidth": 0.244140625, "nfft": 1024, "traces": 240}
```

and on failure:

```json
{"status": "error", "message": "unknown window 'triangle', expected 'hann' or 'none'"}
```

The payload is `bins` amplitudes; bin `k` is at `k * binWidth` Hz, so the axis
runs from 0 to Nyquist. Traces are zero-padded to the next power of two
(`nfft`), which is why `binWidth` is `1 / (nfft * sampleInterval)` rather than
`1 / (cols * sampleInterval)`.

Amplitudes are divided by the trace count and the window's coherent gain, and
the negative frequencies are folded onto the positive ones, so a pure sine reads
back at its own amplitude rather than at an arbitrary scale.

## Performance notes

A 1 GB volume is far larger than the widgets expect, so a few things are
deliberate:

* **ROI curves are bounded.** pyqtgraph averages the ROI region over *every*
  frame and redoes it on each mouse-move, which on a memmap is a scattered read
  of the whole file per event. `SeismicImageView` coalesces the burst into one
  update and reads only every n-th frame, keeping one update inside a fixed
  sample budget. On a 1.15 GB volume this took a 150x150 ROI drag from 25
  seconds to 0.4.
* **Slices are copied into RAM.** An xline or time slice is a strided view
  reaching across the whole file; taking a contiguous copy once beats re-reading
  it on every repaint.
* **Slow reads run off the GUI thread.** Slice reads, composite extraction, 3D
  decimation and spectrum requests all go through `TaskRunner`, which runs one
  at a time and lets a newer request supersede a queued one, so dragging a
  slider does not pile up stale work.
