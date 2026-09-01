"""Arbitrary (composite) slice extraction along a polyline through the volume.

An iline or xline slice is a plane; a composite slice is whatever vertical
section a user-drawn path in the (iline, xline) map cuts out. The path is
resampled to roughly one bin per step, then every trace along it is bilinearly
interpolated from the four surrounding traces.
"""

import numpy as np


def path_points(vertices, step=1.0):
    """Densify a polyline in (iline, xline) to about one point per `step` bins.

    Returns the sampled points and their distance along the path, which is the
    horizontal axis the extracted section is plotted against.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("path vertices must be an (n, 2) array of iline/xline pairs")

    # Drop repeated vertices, which would otherwise contribute zero-length legs
    keep = np.ones(len(vertices), dtype=bool)
    keep[1:] = np.any(np.diff(vertices, axis=0) != 0, axis=1)
    vertices = vertices[keep]
    if len(vertices) < 2:
        raise ValueError("a composite slice needs at least two distinct points")

    legs = np.hypot(*np.diff(vertices, axis=0).T)
    travelled = np.concatenate([[0.0], np.cumsum(legs)])
    total = travelled[-1]

    count = max(2, int(round(total / float(step))) + 1)
    distance = np.linspace(0.0, total, count)
    iline = np.interp(distance, travelled, vertices[:, 0])
    xline = np.interp(distance, travelled, vertices[:, 1])
    return np.column_stack([iline, xline]), distance


def composite_slice(data, vertices, step=1.0):
    """Extract the vertical section under a polyline.

    `data` is the [iline, xline, time] volume (a memmap is fine). Returns the
    section as (points along the path, time) plus the distance axis.
    """
    points, distance = path_points(vertices, step)
    n_iline, n_xline, _ = data.shape

    iline = np.clip(points[:, 0], 0, n_iline - 1)
    xline = np.clip(points[:, 1], 0, n_xline - 1)

    i0 = np.floor(iline).astype(np.intp)
    j0 = np.floor(xline).astype(np.intp)
    i1 = np.minimum(i0 + 1, n_iline - 1)
    j1 = np.minimum(j0 + 1, n_xline - 1)

    fi = (iline - i0).astype(np.float32)[:, None]
    fj = (xline - j0).astype(np.float32)[:, None]

    # Fancy indexing gathers all the traces for one corner in a single pass,
    # which keeps this to four reads of the memmap instead of four per point
    corner_00 = np.asarray(data[i0, j0], dtype=np.float32)
    corner_10 = np.asarray(data[i1, j0], dtype=np.float32)
    corner_01 = np.asarray(data[i0, j1], dtype=np.float32)
    corner_11 = np.asarray(data[i1, j1], dtype=np.float32)

    section = (
        (1.0 - fi) * (1.0 - fj) * corner_00
        + fi * (1.0 - fj) * corner_10
        + (1.0 - fi) * fj * corner_01
        + fi * fj * corner_11
    )
    return section.astype(np.float32, copy=False), distance
