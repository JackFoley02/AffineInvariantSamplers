"""Compressed HDF5 storage for large sampler-series arrays.

The file layout is::

    results_series.h5
    |-- attributes
    |   |-- format_version = 1
    |   |-- thinning = <positive integer>
    |   `-- axis_order = "walker,draw,coordinate"
    `-- samplers
        |-- Dense-mass_NUTS
        |   `-- series
        |-- Hamiltonian_Walk_Move
        |   `-- series
        |-- Stretch_Move
        |   `-- series
        `-- HMC
            `-- series

Groups behave like directories, datasets behave like NumPy arrays, and
attributes hold small metadata values.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


HDF5_FILENAME = "results_series.h5"
HDF5_FORMAT_VERSION = 1
HDF5_DTYPE = np.dtype("float32")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _sampler_key(name: str) -> str:
    """Return a stable HDF5 group name for a sampler."""
    return name.replace("/", "_").replace(" ", "_")


def save_series_hdf5(
    results: Mapping[str, Mapping[str, Any]],
    outdir: str | os.PathLike[str],
    thin: int = 1,
    block_draws: int = 256,
) -> Path:
    """Save post-burn-in sampler series as compressed HDF5 datasets.

    Parameters
    ----------
    results
        Mapping from sampler name to a result mapping containing a three-axis
        ``series`` array in ``(walker, draw, coordinate)`` order.
    outdir
        Per-run output directory. It is created if necessary.
    thin
        Storage-only thinning factor. ``thin=1`` retains every draw.
    block_draws
        Number of retained draws written per block.

    Returns
    -------
    pathlib.Path
        Path to ``results_series.h5``.

    Notes
    -----
    The file is first written beside the destination and then atomically
    replaced. An interrupted write therefore does not corrupt an existing
    completed file.
    """
    thin = _positive_integer(thin, "thin")
    block_draws = _positive_integer(block_draws, "block_draws")
    if not isinstance(results, Mapping):
        raise TypeError("results must be a mapping from sampler names to results")

    output_directory = Path(outdir)
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / HDF5_FILENAME

    temporary = tempfile.NamedTemporaryFile(
        dir=output_directory,
        prefix=f".{path.stem}_",
        suffix=".h5.tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()

    try:
        with h5py.File(temporary_path, "w") as h5:
            h5.attrs["format_version"] = HDF5_FORMAT_VERSION
            h5.attrs["thinning"] = thin
            h5.attrs["axis_order"] = "walker,draw,coordinate"
            samplers_group = h5.create_group("samplers")

            for name, result in results.items():
                if not isinstance(name, str):
                    raise TypeError("sampler names must be strings")
                if "series" not in result:
                    raise KeyError(f"result for {name!r} has no 'series' array")

                series = np.asarray(result["series"])
                if series.ndim != 3:
                    raise ValueError(
                        f"series for {name!r} must have shape "
                        "(walker, draw, coordinate); "
                        f"got {series.shape}"
                    )
                n_walkers, n_draws, n_dim = series.shape
                if n_walkers < 1 or n_dim < 1:
                    raise ValueError(
                        f"series for {name!r} must contain at least one walker "
                        f"and coordinate; got {series.shape}"
                    )
                if np.iscomplexobj(series):
                    raise TypeError(f"series for {name!r} must be real-valued")

                n_saved = (n_draws + thin - 1) // thin
                key = _sampler_key(name)
                if key in samplers_group:
                    raise ValueError(
                        f"sampler names produce duplicate HDF5 key {key!r}"
                    )

                group = samplers_group.create_group(key)
                group.attrs["sampler_name"] = name
                group.attrs["original_draws"] = n_draws
                group.attrs["saved_draws"] = n_saved
                group.attrs["source_dtype"] = str(series.dtype)

                if n_saved == 0:
                    group.create_dataset(
                        "series", shape=(n_walkers, 0, n_dim), dtype=HDF5_DTYPE
                    )
                    continue

                dataset = group.create_dataset(
                    "series",
                    shape=(n_walkers, n_saved, n_dim),
                    dtype=HDF5_DTYPE,
                    chunks=(
                        min(n_walkers, 4),
                        min(n_saved, block_draws),
                        n_dim,
                    ),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    fletcher32=True,
                )

                # Iterate over retained-draw indices. Using n_saved here is
                # essential when thin > 1.
                for start in range(0, n_saved, block_draws):
                    stop = min(start + block_draws, n_saved)
                    dataset[:, start:stop, :] = series[
                        :, start * thin : stop * thin : thin, :
                    ]

        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return path
