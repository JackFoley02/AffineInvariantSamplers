"""Shared diagnostics and storage helpers for condition-number experiments."""

import json
from pathlib import Path

import numpy as np

from autocorrelation_func import integrated_autocorr_time


SWEEP_PARAMETERS = (
    "dimension",
    "condition",
    "n-chains",
    "warmup",
    "acceptance-rate",
)
FIXED_STANDARD_CHAINS = 64
MIN_ENSEMBLE_WALKERS = 64
ENSEMBLE_WALKERS_PER_DIMENSION = 4


def distribution_results_root(distribution, output_dir=None):
    """Return a distribution's master results directory or one compartment.

    ``output_dir`` is deliberately relative so every experiment result remains
    below ``<distribution>Results``.
    """
    root = Path(f"{distribution}Results")
    if output_dir in (None, "", "."):
        return root

    compartment = Path(output_dir)
    if compartment.is_absolute() or ".." in compartment.parts:
        raise ValueError(
            "output_dir must be a relative subdirectory of "
            f"{root}; got {output_dir!r}"
        )
    return root / compartment


def scaled_ensemble_walkers(dim, requested=None):
    """Return an ensemble size of at least ``max(64, 4 * dim)``.

    ``requested`` may be used to run a larger ensemble, but it cannot lower
    the dimension-scaled floor shared by HWM and Stretch Move experiments.
    """
    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be a positive integer")

    required = max(MIN_ENSEMBLE_WALKERS, ENSEMBLE_WALKERS_PER_DIMENSION * dim)
    if requested is None:
        return required
    return max(required, int(requested))


def sweep_output_base(output_dir, sweep_parameter, dim, condition_number,
                      affine=False, n_chains=None, target_accept=None):
    """Return the result directory for one point in a benchmark sweep.

    Dimension sweeps use ``<output_dir>/<dim>d`` (or ``<dim>d_af`` for an
    affine run), condition sweeps use ``<output_dir>/<condition>c``, and
    chain-count sweeps use ``<output_dir>/<n_chains>chains``, and target
    acceptance-rate sweeps use ``<output_dir>/<target_accept>accept``. Each
    directory is the stable parent for that sweep point's seed outputs.
    """
    output_root = Path(output_dir).expanduser()
    if sweep_parameter == "dimension":
        suffix = "_af" if affine else ""
        sweep_dir = f"{int(dim)}d{suffix}"
    elif sweep_parameter == "condition":
        sweep_dir = f"{float(condition_number):g}c"
    elif sweep_parameter == "n-chains":
        if n_chains is None or int(n_chains) < 1:
            raise ValueError("n_chains must be a positive integer")
        sweep_dir = f"{int(n_chains)}chains"
    elif sweep_parameter == "acceptance-rate":
        if (
            target_accept is None
            or not np.isfinite(target_accept)
            or not 0.0 < float(target_accept) < 1.0
        ):
            raise ValueError(
                "target_accept must be a finite value between 0 and 1"
            )
        sweep_dir = f"{float(target_accept):g}accept"
    else:
        choices = ", ".join(SWEEP_PARAMETERS)
        raise ValueError(
            f"sweep_parameter must be one of: {choices}; got {sweep_parameter!r}"
        )
    return output_root / sweep_dir


def to_rest_frame(samples, transform):
    samples = np.asarray(samples)
    if not transform.get("affine", False):
        return samples
    A = np.asarray(transform["A"])
    B = np.asarray(transform["B"])
    return (samples - B) @ np.linalg.inv(A).T


def worst_coordinate_ess(samples, transform):
    """Mean ESS across walkers, then across rest-frame coordinates.

    ESS is computed independently for every walker-coordinate trajectory.
    For each coordinate, walker ESS values are averaged; the scalar ESS
    returned to callers is then the arithmetic mean across coordinates.  A
    walker with an invalid IAT (most commonly because its trajectory is
    constant or too short for the conservative window) contributes zero ESS,
    preserving a penalty for failed diagnostics without making an otherwise
    usable run NaN.

    The function name is retained for compatibility with existing experiment
    scripts; its aggregation is no longer a worst-coordinate calculation.
    """
    rest = to_rest_frame(samples, transform)
    ess_by_coordinate = []
    tau_by_coordinate = []
    valid_fraction_by_coordinate = []
    for k in range(rest.shape[-1]):
        walker_ess = []
        n_valid = 0
        for walker in rest:
            tau, _, ess = integrated_autocorr_time(walker[:, k])
            if np.isfinite(tau) and tau > 0.0 and np.isfinite(ess) and ess > 0.0:
                # Finite-chain negative correlations can produce tau < 1 and
                # ESS above the number of retained draws. Cap these estimates.
                tau = max(float(tau), 1.0)
                walker_ess.append(float(walker.shape[0]) / tau)
                n_valid += 1
            else:
                walker_ess.append(0.0)

        coordinate_ess = float(np.mean(walker_ess))
        ess_by_coordinate.append(coordinate_ess)
        valid_fraction_by_coordinate.append(n_valid / rest.shape[0])
        tau_by_coordinate.append(
            float(rest.shape[1] / coordinate_ess)
            if coordinate_ess > 0.0 else np.nan
        )

    ess_by_coordinate = np.asarray(ess_by_coordinate)
    tau_by_coordinate = np.asarray(tau_by_coordinate)
    valid_fraction_by_coordinate = np.asarray(valid_fraction_by_coordinate)
    if not np.any(valid_fraction_by_coordinate):
        return (np.nan, np.nan, np.asarray(ess_by_coordinate),
                np.asarray(tau_by_coordinate), np.asarray(valid_fraction_by_coordinate))
    mean_ess = float(np.mean(ess_by_coordinate))
    # Report the IAT corresponding to the scalar mean ESS so that the two
    # aggregate values retain the identity ESS = retained_draws / IAT.
    mean_tau = float(rest.shape[1] / mean_ess) if mean_ess > 0.0 else np.nan
    return (mean_ess, mean_tau, ess_by_coordinate, tau_by_coordinate,
            valid_fraction_by_coordinate)


def evaluation_count(name, n_chains, n_saved, n_thin, n_leapfrog):
    """Total post-warmup evaluations across the complete ensemble.

    This is an ensemble-total cost.  Use :func:`mean_walker_efficiency` when
    pairing it with the mean-per-walker ESS returned by
    :func:`worst_coordinate_ess`.
    """
    transitions = int(n_saved) * int(n_thin)
    if name == "Stretch Move":
        return n_chains * transitions, "standard"
    if n_leapfrog is None or not np.isfinite(n_leapfrog):
        return np.nan, "gradient"
    # NUTS reports mean actual leapfrog steps. Fixed-length methods require
    # an initial gradient plus one gradient after each position update.
    per_transition = float(n_leapfrog) if name == "Dense-mass NUTS" else float(n_leapfrog) + 1.0
    return n_chains * transitions * per_transition, "gradient"


def mean_walker_efficiency(mean_walker_ess, total_evaluations, n_walkers):
    """Return ESS/evaluation on a consistent per-walker basis.

    ``mean_walker_ess`` is averaged over walkers, whereas
    ``total_evaluations`` is summed over walkers.  Dividing the latter by the
    actual walker count puts numerator and denominator on the same scale.

    Returns
    -------
    ess_per_eval : float
        Mean-walker ESS divided by evaluations per walker.
    evaluations_per_walker : float
        The total ensemble evaluation count divided by ``n_walkers``.
    """
    if (not np.isfinite(mean_walker_ess)
            or not np.isfinite(total_evaluations)
            or not np.isfinite(n_walkers)
            or int(n_walkers) <= 0):
        return np.nan, np.nan

    evaluations_per_walker = float(total_evaluations) / int(n_walkers)
    if evaluations_per_walker <= 0.0:
        return np.nan, evaluations_per_walker
    return float(mean_walker_ess) / evaluations_per_walker, evaluations_per_walker


def _diagnostic_subsample(samples, max_draws_per_chain=200):
    """Bound the memory cost of multivariate diagnostics on large runs."""
    samples = np.asarray(samples)
    stride = max(1, int(np.ceil(samples.shape[1] / max_draws_per_chain)))
    return samples[:, ::stride, :]


def sample_health_diagnostics(samples, transform):
    """Cheap correctness/health checks that complement coordinate-wise ESS."""
    rest = to_rest_frame(_diagnostic_subsample(samples), transform)
    finite = np.isfinite(rest)
    finite_draws = np.all(finite, axis=-1)
    flat = rest.reshape(-1, rest.shape[-1])
    flat = flat[np.all(np.isfinite(flat), axis=1)]
    if flat.shape[0] < 2:
        rank = 0
    else:
        rank = int(np.linalg.matrix_rank(np.cov(flat, rowvar=False)))
    return {
        "actual_n_chains": int(rest.shape[0]),
        "diagnostic_draws_per_chain": int(rest.shape[1]),
        "sample_finite_fraction": float(np.mean(finite)),
        "finite_draw_fraction": float(np.mean(finite_draws)),
        "sample_covariance_rank": rank,
        "sample_covariance_full_rank": bool(rank == rest.shape[-1]),
    }


def rosenbrock_moment_errors(samples, transform, a=1.0, b=100.0):
    """Rest-frame mean/covariance errors for the paired Rosenbrock target."""
    rest = to_rest_frame(_diagnostic_subsample(samples), transform)
    flat = rest.reshape(-1, rest.shape[-1])
    flat = flat[np.all(np.isfinite(flat), axis=1)]
    if flat.shape[0] < 2:
        return {"mean_mse": np.nan, "cov_mse": np.nan}

    target_mean = np.empty(rest.shape[-1])
    target_mean[0::2] = a
    target_mean[1::2] = a**2 + 0.5

    pair_cov = np.array([
        [0.5, a],
        [a, 2.0 * a**2 + 0.5 + 0.5 / b],
    ])
    target_cov = np.kron(np.eye(rest.shape[-1] // 2), pair_cov)
    sample_mean = np.mean(flat, axis=0)
    sample_cov = np.cov(flat, rowvar=False)
    return {
        "mean_mse": float(np.mean((sample_mean - target_mean) ** 2)),
        "cov_mse": float(
            np.sum((sample_cov - target_cov) ** 2) / np.sum(target_cov**2)
        ),
    }


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
