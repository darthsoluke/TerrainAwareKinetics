"""Lossless numeric encoding for a CPU-only recording worker."""

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


def encode_npz(arrays):
    with BytesIO() as buffer:
        with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
            for name, value in arrays.items():
                with archive.open(name+".npy", "w", force_zip64=True) as entry:
                    np.lib.format.write_array(entry, np.asarray(value), allow_pickle=False)
        return buffer.getvalue()
