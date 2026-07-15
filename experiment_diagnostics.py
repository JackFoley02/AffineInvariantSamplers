"""Shared diagnostics and storage helpers for condition-number experiments."""

import json
from pathlib import Path

import numpy as np

from autocorrelation_func import integrated_autocorr_time


def to_rest_frame(samples, transform):
    samples = np.asarray(samples)
    if not transform.get("affine", False):
        return samples
    A = np.asarray(transform["A"])
    B = np.asarray(transform["B"])
    return (samples - B) @ np.linalg.inv(A).T


def worst_coordinate_ess(samples, transform):
    """Conservative ESS: sum chain ESS, then take the worst rest-frame coordinate.

    A chain with an invalid IAT (most commonly a constant/frozen chain, or a
    chain too short for the conservative window) contributes zero ESS. This
    penalizes failures without turning an otherwise usable run into NaN. The
    returned validity fractions remain available as a quality warning.
    """
    rest = to_rest_frame(samples, transform)
    ess_by_coordinate = []
    tau_by_coordinate = []
    valid_fraction_by_coordinate = []
    for k in range(rest.shape[-1]):
        chain_ess = []
        chain_tau = []
        for chain in rest:
            tau, _, ess = integrated_autocorr_time(chain[:, k])
            if np.isfinite(tau) and tau > 0.0 and np.isfinite(ess) and ess > 0.0:
                # Finite-chain negative correlations can produce tau < 1 and
                # ESS > number of draws. Cap these noisy estimates so the
                # efficiency plot cannot claim more independent draws than
                # were actually retained.
                tau = max(float(tau), 1.0)
                chain_tau.append(tau)
                # Recompute ESS from the validated/capped tau. Never reuse a
                # potentially negative or inconsistent value returned by the
                # lower-level window estimator.
                chain_ess.append(float(chain.shape[0]) / tau)
        valid_fraction = len(chain_tau) / rest.shape[0]
        valid_fraction_by_coordinate.append(valid_fraction)
        total_ess = float(np.sum(chain_ess))
        if total_ess <= 0.0:
            ess_by_coordinate.append(0.0)
            tau_by_coordinate.append(np.nan)
        else:
            ess_by_coordinate.append(total_ess)
            # Equivalent ensemble-wide IAT. Invalid chains have contributed
            # zero ESS, so they increase this tau rather than disappearing
            # from a median over only the successful chains.
            nominal_draws = rest.shape[0] * rest.shape[1]
            tau_by_coordinate.append(nominal_draws / total_ess)

    finite = np.isfinite(ess_by_coordinate)
    if not np.any(finite):
        return (np.nan, np.nan, np.asarray(ess_by_coordinate),
                np.asarray(tau_by_coordinate), np.asarray(valid_fraction_by_coordinate))
    worst = int(np.nanargmin(ess_by_coordinate))
    return (float(ess_by_coordinate[worst]), float(tau_by_coordinate[worst]),
            np.asarray(ess_by_coordinate), np.asarray(tau_by_coordinate),
            np.asarray(valid_fraction_by_coordinate))


def evaluation_count(name, n_chains, n_saved, n_thin, n_leapfrog):
    """Post-warmup evaluation count; warmup is deliberately excluded from both numerator and denominator."""
    transitions = int(n_saved) * int(n_thin)
    if name == "Stretch Move":
        return n_chains * transitions, "function"
    if n_leapfrog is None or not np.isfinite(n_leapfrog):
        return np.nan, "gradient"
    # NUTS reports mean actual leapfrog steps. Fixed-length methods require
    # an initial gradient plus one gradient after each position update.
    per_transition = float(n_leapfrog) if name == "Dense-mass NUTS" else float(n_leapfrog) + 1.0
    return n_chains * transitions * per_transition, "gradient"


def update_seed_manifest(base_dir, seed, run_dir, metadata):
    """Maintain a lightweight index without mixing arrays from different seeds."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "seeds_manifest.json"
    if path.exists():
        with path.open() as f:
            manifest = json.load(f)
    else:
        manifest = {"runs": []}
    entry = {"seed": int(seed), "directory": str(Path(run_dir).relative_to(base)), **metadata}
    manifest["runs"] = [r for r in manifest["runs"] if r["seed"] != int(seed)]
    manifest["runs"].append(entry)
    manifest["runs"].sort(key=lambda r: r["seed"])
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
