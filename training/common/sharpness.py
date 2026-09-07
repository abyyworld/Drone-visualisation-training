"""Focus measurement, shared by tools/image_quality.py and tools/rebuild_turbine.py.

One implementation on purpose: the audit reports a number and the rebuild filters on that
same number, so if the two ever computed it differently the tool would drop a different set
of images than the one it just told you about.

Unlike the rest of `training/common`, this needs numpy and Pillow - image decoding cannot be
done from the standard library. Callers import it lazily so the dependency-free tools stay
dependency-free.
"""

from __future__ import annotations

from pathlib import Path

# Below this an image is soft. It is a flag to look at, NOT a verdict - the metric cannot
# tell "out of focus" from "genuinely smooth content", and a blade against clear sky is
# legitimately low-texture.
SOFT = 100.0

# Below this the image is a smear. A bounding box drawn on one cannot teach localisation.
UNUSABLE = 20.0


def laplacian_variance(path: Path) -> float:
    """Variance of the Laplacian - the standard focus measure. Higher is sharper.

    Computed on the greyscale image at native resolution. Note this is scale-dependent:
    comparing values across datasets with different resolutions is meaningless. Every image
    in this export is 640x640, so comparisons within it are valid.
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    from PIL import Image

    with Image.open(path) as handle:
        array = np.asarray(handle.convert("L"), dtype=np.float32)

    if min(array.shape) < 3:
        return 0.0

    # 4-neighbour Laplacian kernel.
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    windows = sliding_window_view(array, (3, 3))
    return float((windows * kernel).sum(axis=(-1, -2)).var())
