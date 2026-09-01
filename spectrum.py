"""Client for the standalone C# spectrum module.

The analysis itself lives in SpectrumService/, a separate .NET console
executable with no knowledge of this application. This module owns the other
end of the pipe: it locates (and if necessary builds) the executable, keeps one
instance running as a child process, and exchanges frames with it.

Wire format, little-endian throughout:

    request   b"SPQ1" | uint32 header_len | header JSON (utf-8) | float32[rows * cols]
    response  b"SPR1" | uint32 header_len | header JSON (utf-8) | float32[bins]

    request header   {"rows": 240, "cols": 800, "sampleInterval": 0.004,
                      "window": "hann", "detrend": true}
    response header  {"status": "ok", "bins": 513, "binWidth": 0.244,
                      "nfft": 1024, "traces": 240}
                     {"status": "error", "message": "..."}

The sample block is row-major with one trace per row and time running along the
row, which is exactly how a slice comes out of the store, so neither side has to
transpose or re-pack anything.
"""

import collections
import json
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

REQUEST_MAGIC = b"SPQ1"
RESPONSE_MAGIC = b"SPR1"

PROJECT_DIR = Path(__file__).resolve().parent / "SpectrumService"
EXECUTABLE_NAME = "SpectrumService.exe" if sys.platform == "win32" else "SpectrumService"

HANN = "hann"
NO_WINDOW = "none"
WINDOWS = (HANN, NO_WINDOW)


class SpectrumError(RuntimeError):
    """Anything that stops the module from returning a spectrum."""


class SpectrumResult:
    """A spectrum plus the frequency axis the module derived it on."""

    def __init__(self, frequencies, amplitudes, traces, nfft, bin_width):
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.traces = traces
        self.nfft = nfft
        self.bin_width = bin_width

    @property
    def nyquist(self):
        return float(self.frequencies[-1]) if len(self.frequencies) else 0.0

    def summary(self):
        peak = int(np.argmax(self.amplitudes)) if len(self.amplitudes) else 0
        return (
            f"{self.traces} traces, nfft {self.nfft}, "
            f"df {self.bin_width:.3f} Hz, peak {self.frequencies[peak]:.1f} Hz"
        )


class SpectrumService:
    """Owns the child process and the conversation with it."""

    BUILD_TIMEOUT_S = 300

    def __init__(self, project_dir=None):
        self.project_dir = Path(project_dir) if project_dir else PROJECT_DIR
        self._process = None
        self._executable = None
        self._stderr_tail = collections.deque(maxlen=40)

    # -- locating and building -------------------------------------------

    def find_executable(self):
        """Newest build output for the module, or None if it was never built."""
        candidates = sorted(
            self.project_dir.glob(f"bin/*/net*/{EXECUTABLE_NAME}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def build(self):
        """Compile the module with the .NET SDK. Slow, so callers do it once."""
        project = self.project_dir / "SpectrumService.csproj"
        if not project.exists():
            raise SpectrumError(f"C# module not found at {project}")

        try:
            completed = subprocess.run(
                ["dotnet", "build", str(project), "-c", "Release", "--nologo", "-v", "quiet"],
                capture_output=True,
                text=True,
                timeout=self.BUILD_TIMEOUT_S,
                creationflags=_no_window(),
            )
        except FileNotFoundError as exc:
            raise SpectrumError(
                "the .NET SDK is not on PATH, so the spectrum module cannot be built; "
                "install it from https://dotnet.microsoft.com/download"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SpectrumError("timed out building the spectrum module") from exc

        if completed.returncode != 0:
            detail = (completed.stdout or completed.stderr or "").strip().splitlines()
            raise SpectrumError("building the spectrum module failed: " + " / ".join(detail[-3:]))

        executable = self.find_executable()
        if executable is None:
            raise SpectrumError("the build reported success but produced no executable")
        return executable

    def ensure_executable(self):
        if self._executable is not None and self._executable.exists():
            return self._executable
        self._executable = self.find_executable() or self.build()
        return self._executable

    # -- process lifetime ------------------------------------------------

    @property
    def running(self):
        return self._process is not None and self._process.poll() is None

    def start(self):
        if self.running:
            return

        executable = self.ensure_executable()
        self._stderr_tail.clear()
        self._process = subprocess.Popen(
            [str(executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.project_dir),
            creationflags=_no_window(),
        )
        # Drain stderr in the background: it is only diagnostics, but a full pipe
        # would block the module mid-response
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def close(self):
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()  # closing the pipe is the shutdown signal
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()

    def _drain_stderr(self):
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in iter(process.stderr.readline, b""):
            self._stderr_tail.append(line.decode("utf-8", "replace").rstrip())

    def _diagnostics(self):
        return " / ".join(self._stderr_tail) or "no output on stderr"

    # -- the request itself ----------------------------------------------

    def analyse(self, traces, sample_interval, window=HANN, detrend=True):
        """Send a block of traces to the module and get its average spectrum.

        `traces` is 2D with one trace per row and time along the row.
        `sample_interval` is in seconds.
        """
        traces = np.ascontiguousarray(traces, dtype=np.float32)
        if traces.ndim != 2:
            raise SpectrumError(f"expected a 2D block of traces, got shape {traces.shape}")
        rows, cols = traces.shape
        if rows < 1 or cols < 2:
            raise SpectrumError(f"region of {rows}x{cols} is too small for a spectrum")
        if sample_interval <= 0:
            raise SpectrumError("sample interval must be positive")

        self.start()

        header = {
            "rows": int(rows),
            "cols": int(cols),
            "sampleInterval": float(sample_interval),
            "window": window,
            "detrend": bool(detrend),
        }
        try:
            self._write_frame(header, traces)
            response_header, payload = self._read_frame()
        except (OSError, ValueError, EOFError) as exc:
            self.close()
            raise SpectrumError(f"lost contact with the spectrum module: {exc}") from exc

        if response_header.get("status") != "ok":
            raise SpectrumError(response_header.get("message", "the module reported an error"))

        bin_width = float(response_header["binWidth"])
        frequencies = np.arange(len(payload), dtype=np.float64) * bin_width
        return SpectrumResult(
            frequencies=frequencies,
            amplitudes=payload,
            traces=int(response_header.get("traces", rows)),
            nfft=int(response_header.get("nfft", 0)),
            bin_width=bin_width,
        )

    def _write_frame(self, header, payload):
        blob = json.dumps(header).encode("utf-8")
        stdin = self._process.stdin
        stdin.write(REQUEST_MAGIC)
        stdin.write(struct.pack("<I", len(blob)))
        stdin.write(blob)
        stdin.write(payload.tobytes(order="C"))
        stdin.flush()

    def _read_frame(self):
        magic = self._read_exact(4)
        if magic != RESPONSE_MAGIC:
            raise ValueError(f"expected {RESPONSE_MAGIC!r}, got {magic!r} ({self._diagnostics()})")

        (header_length,) = struct.unpack("<I", self._read_exact(4))
        header = json.loads(self._read_exact(header_length).decode("utf-8"))

        bins = int(header.get("bins", 0)) if header.get("status") == "ok" else 0
        payload = np.frombuffer(self._read_exact(bins * 4), dtype="<f4").copy()
        return header, payload

    def _read_exact(self, count):
        if count == 0:
            return b""
        chunks = []
        remaining = count
        stdout = self._process.stdout
        while remaining:
            chunk = stdout.read(remaining)
            if not chunk:
                raise EOFError(
                    f"module closed the pipe after {count - remaining} of {count} bytes "
                    f"({self._diagnostics()})"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def _no_window():
    """Keep a console window from flashing up on Windows."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)
