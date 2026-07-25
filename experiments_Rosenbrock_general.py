import numpy as np
import matplotlib.pyplot as plt
import time
from plotTools.benchmark_corner import benchmark_corner
from plotTools.benchmark_trends import benchmark_trends
from plotTools.benchmark_autocorrelation import benchmark_autocorrelation
from matplotlib import rc
import os
os.environ.setdefault("JAX_ENABLE_X64", "True")
import json
import traceback
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

rc('text', usetex=False)

import argparse
from samplers.sampler_nuts import hmc_nuts
from samplers.sampler_peachees import hamiltonian_walk_chees
from samplers.sampler_chees import hmc_chees
from samplers.samplers import stretch_move
from autocorrelation_func import autocorrelation_fft, integrated_autocorr_time
from experiment_diagnostics import (worst_coordinate_ess, evaluation_count,
                                    sample_health_diagnostics,
                                    rosenbrock_moment_errors, update_seed_manifest)

N_CHAINS = 150
ROSENBROCK_SIGMA = 1.5
INITIALIZATION_CHOICES = ("oracle", "agnostic")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark tests of different MCMC sampling algorithms, applied to a multidimensional Rosenbrock distribution. # of dimensions MUST be even.'
    )
    parser.add_argument(
        '--dim',
        type = int,
        default = 2,
        help="Number of dimensions for the Gaussian benchmark (default: 2)"
    )
    parser.add_argument(
        '--af',
        action = 'store_true',
        help='If True, apply an arbitrary affine transformation to the target distribution. Helpful to test affine-invariance of ensemble samplers.'
    )
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--gpu", type = int, default=None)
    parser.add_argument('--cond', type =float, default = 50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--burn-in', type=int, default=1000)
    parser.add_argument('--n-warmup', type=int, default=1000)
    parser.add_argument('--n-chains', type=int, default=N_CHAINS)
    parser.add_argument('--hwm-walkers', type=int, default=300,
                        help='Walker count used only by Hamiltonian Walk Move.')
    parser.add_argument('--n-thin', type=int, default=1,
                        help='Transitions per saved draw; use 1 for reliable ESS estimates.')
    parser.add_argument(
        '--initialization',
        choices=INITIALIZATION_CHOICES,
        default=os.environ.get("ROSENBROCK_INITIALIZATION", "agnostic").lower(),
        help=("Initial ensemble in the intrinsic Rosenbrock frame: agnostic "
              "uses exact target draws; 'agnostic' uses independent N(0, 2^2) "
              "coordinates (default: %(default)s)."),
    )

    return parser.parse_args()


args = parse_args()
d = args.dim
af = args.af
gpu = args.gpu
cond = args.cond
seed = args.seed
INITIALIZATION_MODE = args.initialization

if gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

#import jax after setting default CUDA device 
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

if (d % 2) != 0:
    raise ValueError('# of dimensions must be an even number.')

def affine_transform(dim, max_dim=128, seed=4321, trans_scale=2.0, cond=cond):
    rng = np.random.default_rng(seed)

    B_full = rng.uniform(-trans_scale, trans_scale, size=max_dim)

    M = rng.standard_normal((max_dim, max_dim))
    Q, R = np.linalg.qr(M)

    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    U_full = Q @ np.diag(signs)

    if np.linalg.det(U_full) < 0:
        U_full[:, 0] *= -1

    # Take the leading dim-dimensional block
    U = U_full[:dim, :dim]

    # Re-orthogonalize after slicing
    U, R = np.linalg.qr(U)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    U = U @ np.diag(signs)

    if np.linalg.det(U) < 0:
        U[:, 0] *= -1

    s_vals = np.geomspace(1.0, cond, dim)
    S = np.diag(s_vals)

    A = U @ S
    B = B_full[:dim]
    Ainv = np.linalg.inv(A)

    return A, B, Ainv

def benchmark_samplers_Rosenbrock_general(dim=2, n_samples=10000, burn_in=1000, sigma=1.5, a = 1.0, b = 100.0, affine = False, seed=0, n_warmup=1000, n_chains=N_CHAINS, n_thin=1):
    """
    Benchmark different MCMC samplers on a Rosenbrock distribution of any EVEN dimension.
    
    Parameters:
    -----------
    dim : int
        Dimension of the distribution
    n_samples : int
        Number of samples to generate after burn-in
    burn_in : int
        Number of initial samples to discard as burn-in
    sigma : float
        Width parameter of the ring (smaller values make a sharper ring)
    a : float
        Parameter for the Rosenbrock Distribution
    b : float
        Parameter for the Rosenbrock Distribution
    rot : bool
        If set to true, perform an affine transformation to rotate the distribution.
        Tests performance of affine-invariant and non-affine-invariant samplers. 
    """

    if affine: #Apply an affine transformation 
        #i.e. y = Ax + b
        #Affine-invariant samplers should behave equally well under any rotation, which is what this tests
        A, B, Ai = affine_transform(dim = dim, trans_scale=2.0, cond=cond) 
        AiT = Ai.T
    else:
        A = np.eye(dim) #identity matrix that is dim x dim
        B = np.zeros(dim) #just a null vector
        Ai = np.eye(dim)
        AiT = Ai.T

    A_jax = jnp.asarray(A)
    B_jax = jnp.asarray(B)
    Ai_jax = jnp.asarray(Ai)
    AiT_jax = jnp.asarray(AiT)


    def Rosenbrock_paired(params: np.ndarray) -> np.ndarray:
        """Rosenbrock Distribution for any (even) number of dimensions. Generated as a sum of separable Rosenbrocks.
        Arguments:
        ----------
        params:
            An array of length 'dim' consisting of all dim parameter values.

        Returns:
        ----------
        The Rosenbrock distribution at this point in dim-space. 
        
        """
        x = np.asarray(params, dtype = float)

        if (x.shape[-1] % 2) != 0:
            raise ValueError('Invalid dimensionality. Must have an even number dimension.') #This form of the Rosenbrock requires an even number of dimensions
        params_odd = x[..., 0::2] #all odd parameters
        params_even = x[..., 1::2] #all even parameters
        return np.sum(b * (params_odd **2 - params_even)**2 + (params_odd - a)**2, axis = -1)

    def grad_Rosenbrock_paired(params: np.ndarray) -> np.ndarray:
        """
        Gradient function for the paired Rosenbrock distribution, for use in gradient-aware/Hamiltonian MC samplers

        Arguments:
        ----------
        params:
            array of the parameter values at the particular sampler location. Should be 1D and contain an even number of values.
        
        Returns:
        ----------
        grad:
            gradient array of the paired Rosenbrock using the particular parameter values
        """
        x = np.asarray(params, dtype = float)
 
        if x.shape[-1] % 2 != 0: #make sure the array contains an even number of values
            raise ValueError('Must use an even number of parameters/dimensions')
        
        x_o = x[..., 0::2] # odd parameters
        x_e = x[..., 1::2] # even parameters

        grad = np.zeros_like(x) #defining placehold gradient array

        grad[..., 0::2] = -400.0 * x_o * (x_e - x_o**2) + 2.0 * (x_o - 1) #defining even pairing values
        grad[..., 1::2] = 200.0 * (x_e - x_o**2)

        return grad

    # Define the distribution log-density
    def log_density(z):
        """Log density [log(p(z))] of the Rosenbrock distribution (p(z))"""
        z = np.asarray(z)
        #u: original, non-rotated frame. z: rotated frame.
        u = (z - B) @ AiT
        R = Rosenbrock_paired(u) #solve for the Rosenbrock Distribution in its rest frame at point/vector u = (x, y)
        logden = -0.5 * R / (sigma**2)
        return logden

    # Define the gradient of the negative log density
    def gradient(z):
        """Gradient of the negative log density (potential gradient/potential energy)"""
        z = np.asarray(z)
        u = (z - B) @ AiT
        gu = 0.5 * grad_Rosenbrock_paired(u) / (sigma**2)
        gz = gu @ Ai
        return gz
    

    def log_density_jax(x):
        # Close over the benchmark parameters so all samplers and diagnostics
        # use the same requested Rosenbrock width.
        x = jnp.asarray(x)

        u = (x - B_jax) @ AiT_jax

        params_odd = u[0::2] #all odd parameters
        params_even = u[1::2] #all even parameters

        R = jnp.sum(b * (params_even - params_odd**2)**2 + (params_odd -a)**2)

        return -0.5 * R / sigma**2

    def sample_initial_ensemble(n_chains, seed):
        rng = np.random.default_rng(seed)

        if INITIALIZATION_MODE == "agnostic":
            # Do not encode the Rosenbrock valley y ~= x**2.  Every intrinsic
            # coordinate starts independently in an isotropic, overdispersed
            # cloud, so warmup/burn-in must discover the target geometry.
            return 2.0 * rng.standard_normal((n_chains, dim))

        # Oracle initialization: exact draws from the paired Rosenbrock target.
        u = np.empty((n_chains, dim), dtype=float)
        x_odd = a + sigma * rng.standard_normal((n_chains, dim // 2))
        x_even = x_odd**2 + (sigma / np.sqrt(b)) * rng.standard_normal((n_chains, dim // 2))
        u[:, 0::2] = x_odd
        u[:, 1::2] = x_even

        return u

    # Draw the same intrinsic ensemble for every condition number, then map
    # the complete ensemble into sampling coordinates.
    hwm_walkers = int(args.hwm_walkers)
    if hwm_walkers < 4 or hwm_walkers % 2:
        raise ValueError("--hwm-walkers must be an even integer >= 4")
    u0_all = sample_initial_ensemble(
        n_chains=max(n_chains, hwm_walkers), seed=seed
    )
    initial = u0_all[:n_chains] @ A.T + B
    initial_hwm = u0_all[:hwm_walkers] @ A.T + B

    
    # Dictionary to store results
    results = {}
    
    # Define samplers to benchmark with burn-in
    total_samples = n_samples + burn_in
    
    # Define samplers to benchmark - parameters tuned for the ring distribution
    samplers = {
        "Dense-mass NUTS": lambda: hmc_nuts(log_density_jax, initial, total_samples, epsilon=0.1, n_chains=n_chains, n_warmup=n_warmup, n_thin=n_thin, max_tree_depth=8, seed=seed, progress_bar=True),
        "Hamiltonian Walk Move": lambda: hamiltonian_walk_chees(
            log_density_jax, initial_hwm, total_samples,
            n_walkers=hwm_walkers, epsilon=0.1, L=10, n_warmup=n_warmup, max_L=1000, n_thin=n_thin, seed=seed
        ),        
        "Stretch Move": lambda: stretch_move(log_density, initial, total_samples, n_walkers=n_chains, a=1.0+2.151/np.sqrt(dim), n_thin=n_thin),
        "HMC": lambda: hmc_chees(log_density_jax, initial, total_samples, epsilon=0.1, L=10, n_chains=n_chains, n_warmup=n_warmup, max_L=1000, n_thin=n_thin, seed=seed),
 
    }

    
    # Benchmark each sampler with careful error handling
    for name, sampler_func in samplers.items():
        # stretch_move uses NumPy's legacy global RNG; reseeding here also
        # makes sampler ordering irrelevant.
        np.random.seed(seed)
        print(f"Running {name}...")
        start_time = time.time()
        
        try:

            out = sampler_func()
            samples, acceptance_rates = out[:2]

            eps_final = out[2] if len(out) > 2 else 0.0
            eps_hist = out[3] if len(out) > 3 else [0.0, 0.0, 0.0]
            if len(out) > 4:
                n_leapfrog, sampler_n_warmup, target_accept, gamma, t0, kappa = out[4]
            else:
                n_leapfrog = sampler_n_warmup = target_accept = gamma = t0 = kappa = None
            sampler_info = out[5] if len(out) > 5 and isinstance(out[5], dict) else {}
            # Apply burn-in: discard the first burn_in samples
            post_burn_in_samples = samples[:, burn_in:, :]
            series = post_burn_in_samples
            burn_in_samps = samples[:, :burn_in, :]
            
            d = np.linalg.norm(np.diff(post_burn_in_samples[0, :, :], axis=0), axis=1)
            print("  median step size:", np.median(d), " max step:", np.max(d))
            # Flatten samples from all chains
            flat_samples = post_burn_in_samples.reshape(-1, dim)
            
            # Calculate mean distance from ring
            mean = np.mean(flat_samples, axis = 0)
            cov = np.cov(flat_samples.T)
            
            # Compute autocorrelation for first dimension
            acf = autocorrelation_fft(np.mean(post_burn_in_samples[:, :, 0],axis=0))
            
            transform_now = {"affine": affine, "A": A, "B": B}
            ess, tau, ess_by_coordinate, tau_by_coordinate, ess_valid_fraction = worst_coordinate_ess(
                post_burn_in_samples, transform_now
            )
            tau_std = float(np.nanstd(tau_by_coordinate))
            
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
            traceback.print_exc()
            # Create dummy data in case of error
            flat_samples = np.zeros((10, dim))
            acceptance_rates = np.zeros(dim)
            series = np.zeros((1, 0, dim))
            burn_in_samps = np.zeros((1, 0, dim))
            mean = np.full(dim, np.nan)
            cov = np.full((dim, dim), np.nan)
            acf = np.zeros(100)
            tau, tau_std, ess = np.nan, np.nan, np.nan
            ess_by_coordinate = np.full(dim, np.nan)
            ess_valid_fraction = np.zeros(dim)
            eps_final, eps_hist = 0.0, np.asarray([])
            n_leapfrog = sampler_n_warmup = target_accept = gamma = t0 = kappa = None
            sampler_info = {}
        
        elapsed = time.time() - start_time
        n_evaluations, evaluation_type = evaluation_count(
            name, len(acceptance_rates), n_samples, n_thin, n_leapfrog
        )
        ess_per_eval = ess / n_evaluations if np.isfinite(ess) and np.isfinite(n_evaluations) else np.nan
        transform_now = {"affine": affine, "A": A, "B": B}
        health = sample_health_diagnostics(series, transform_now)
        moment_errors = rosenbrock_moment_errors(
            series, transform_now, a=a, b=b, sigma=sigma
        )
        
        # Store results
        results[name] = {
            "distribution":'Paired Rosenbrock',
            "samples": flat_samples,
            "acceptance_rates": acceptance_rates,
            "series": series,
            "burn_in":burn_in_samps,
            "mean": mean,
            "cov": cov,
            "autocorrelation": acf,
            "tau": tau,
            "tau_std":tau_std,
            "ess": ess,
            "ess_by_coordinate": ess_by_coordinate,
            "ess_valid_fraction": ess_valid_fraction,
            "n_evaluations": n_evaluations,
            "evaluation_type": evaluation_type,
            "ess_per_eval": ess_per_eval,
            "time": elapsed,
            "labels": None,
            "sigma":sigma,
            "epsilon":eps_final,
            "epsilon_history":eps_hist,
            "n_leapfrog":n_leapfrog,
            "n_warmup":sampler_n_warmup,
            "target_accept":target_accept,
            "gamma":gamma,
            "t0":t0,
            "kappa":kappa,
            "cond":cond
            ,"acceptance_rate_mean": float(np.nanmean(acceptance_rates))
            ,"acceptance_rate_min": float(np.nanmin(acceptance_rates))
            ,"acceptance_rate_max": float(np.nanmax(acceptance_rates))
            ,"n_divergent": sampler_info.get("n_divergent")
            ,**health
            ,**moment_errors
        }
        
        print(f"  Acceptance rate: {np.mean(acceptance_rates):.2f}")
        print(f"  Means: {np.round(mean, 3)}")
        #print(f"  Covariance : {cov}")
        print(f"  Integrated autocorrelation time: {tau:.2f}" if np.isfinite(tau) else "  Integrated autocorrelation time: NaN")
        # print(f"  Effective sample size: {ess:.2f}" if np.isfinite(ess) else "  Effective sample size: NaN")
        # print(f"  ESS/sec: {ess/elapsed:.2f}" if np.isfinite(ess) else "  ESS/sec: NaN")
        print(f"  Time: {elapsed:.2f} seconds")
    
    #storing transformation info into separate dict
    transformation = {"affine":affine, "A":A, 'B':B}

    return results, sigma, log_density, transformation

def plot_Rosenbrock_results(results, log_density, dim=2, sigma=1.5, transform = {'affine':False}, output_dir=None):
    """Plot comparison of sampler results for ring distribution"""
    samplers = list(results.keys())

    if transform['affine']:
        rotstring = "_af" #suffix to append to file name if affine transformation has been applied
    else:
        rotstring = '' #no transformation? set suffix to empty string

    if dim < 2:
        print("Skipping Rosenbrock overlay plot: need at least 2 dimensions.")
        return

    def rosenbrock_2d_log_density(x, y, sigma=1.5, a=1.0, b=100.0):
        R = (a - x)**2 + b * (y - x**2)**2
        return -0.5 * R / (sigma**2)


    def to_rest_frame(points, transform):
        """
        Undo the affine transformation so the samples are plotted in the
        intrinsic Rosenbrock frame.
        """
        if transform is None or not transform.get("affine", False):
            return points

        A = np.asarray(transform["A"])
        B = np.asarray(transform["B"])
        AiT = np.linalg.inv(A).T
        return (points - B) @ AiT

    # grid for background density
    x = np.linspace(-3, 3, 400)
    y = np.linspace(-2, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock_2d_log_density(X, Y, sigma=sigma, a=1, b=100)

    n_panels = len(samplers)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)

    for ax, name in zip(axes.flat, samplers):
        series = np.asarray(results[name]["series"])

        if series.ndim != 3 or series.shape[-1] < 2:
            ax.set_title(f"{name}\nInvalid series shape: {series.shape}")
            ax.axis("off")
            continue

        rng = np.random.default_rng(123)
        n_show = min(4, series.shape[0])
        show_idx = rng.choice(series.shape[0], size=n_show, replace=False)

        pts_full = series[show_idx, :, :]
        starts_full = series[show_idx, 0, :]

        # map back to unrotated Rosenbrock frame if needed
        pts_rest_full = to_rest_frame(pts_full.reshape(-1, series.shape[-1]), transform).reshape(pts_full.shape)
        starts_rest_full = to_rest_frame(starts_full, transform)

        pts_rest = pts_rest_full[..., :2]
        starts_rest = starts_rest_full[..., :2]

        # background density
        ax.contour(X, Y, Z, levels=12, linewidths=0.7)

        # all thinned samples
        colors = plt.cm.tab10(np.linspace(0, 1, pts_rest.shape[0]))

        for j in range(pts_rest.shape[0]):
            ax.scatter(
                pts_rest[j, :, 0],
                pts_rest[j, :, 1],
                s=4,
                alpha=0.5,
                color=colors[j],
                rasterized=True, zorder = 1
            )

            ax.scatter(
                starts_rest[j, 0],
                starts_rest[j, 1],
                marker='x',
                s=40,
                linewidths=1.5,
                color=colors[j], zorder = 10
            )
            ax.set_title(name)
            ax.set_xlabel("x0")
            ax.set_ylabel("x1")
            ax.set_xlim(-3, 3)
            ax.set_ylim(-2, 10)


    # hide unused axes
    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    fig.suptitle("Sampler projections onto first two Rosenbrock dimensions", y=0.995)
    fig.tight_layout()
    output_dir = output_dir or f"RosenbrockResultsC/{cond}c"
    fig.savefig(os.path.join(output_dir, "Rosenbrock_overlay_first2.png"), dpi=220, bbox_inches="tight")
    plt.close()
    colors = ['black', 'black', 'purple', 'blue', 'green', 'y', 'orange', 'red']

    plt.figure(figsize=(12, 6))

    for i, name in enumerate(samplers):
        if results[name]['epsilon'] == 0.0: #skip samplers without recorded epsilon histories
            continue
        else:
            plt.plot(np.arange(0, len(results[name]['epsilon_history'])), results[name]['epsilon_history'],
                     color = colors[i], linestyle = ':', label = f"{name}: epsilon = {np.round(results[name]['epsilon'], 3)}")
            
    plt.title('StepSizeTuner Results')
    plt.xlabel('Warm-up Step Number')
    plt.ylabel('Epsilon value')
    plt.ylim(0, 1)
    plt.xlim(0, results[samplers[0]]['n_warmup'])
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'StepSize.pdf'))
    plt.close()


    
    
    return

# Run the benchmark for the Rosenbrock distribution
# Note: You need to have the sampler functions (side_move, stretch_move, etc.) defined elsewhere
results, sigma, log_density, transform = benchmark_samplers_Rosenbrock_general(
    dim=d, n_samples=args.n_samples, burn_in=args.burn_in,
    sigma=ROSENBROCK_SIGMA,
    affine=af, seed=seed, n_warmup=args.n_warmup, n_chains=args.n_chains,
    n_thin=args.n_thin
)

if transform['affine']:
    afstring = '_af'
else:
    afstring = ''


# Keep agnostic runs separate so they cannot overwrite oracle condition runs.
results_root = (
    "RosenbrockResultsC_agnostic"
    if INITIALIZATION_MODE == "agnostic"
    else "RosenbrockResultsC"
)
cond_string = f"{cond:g}c"
base_outdir = f"{results_root}/{cond_string}"
outdir = os.path.join(base_outdir, "seeds", f"seed_{seed:05d}")
corner_dir = os.path.join(outdir, "corner")
trends_dir = os.path.join(outdir, "trends")
os.makedirs(outdir, exist_ok=True)
os.makedirs(corner_dir, exist_ok=True)
os.makedirs(trends_dir, exist_ok=True)

overlay_rosenbrock = {'a':1.0, 'b':100.0, 'sigma':sigma}



def save_light_results(results, transform, overlay, outdir, dim, af):
    os.makedirs(outdir, exist_ok=True)

    summary = {}

    arrays = {
        "A": np.asarray(transform["A"]),
        "B": np.asarray(transform["B"]),
    }

    for name, r in results.items():
        key = name.replace(" ", "_")

        # Keep only thinned samples for plotting/corners/trends
        series = np.asarray(r["series"])
        arrays[f"{key}_series_thin10"] = series[:, ::10, :].astype(np.float32)

        # Keep short diagnostics
        arrays[f"{key}_acceptance_rates"] = np.asarray(r["acceptance_rates"], dtype=np.float32)
        arrays[f"{key}_mean"] = np.asarray(r["mean"], dtype=np.float32)
        arrays[f"{key}_cov"] = np.asarray(r["cov"], dtype=np.float32)
        arrays[f"{key}_autocorrelation"] = np.asarray(r["autocorrelation"], dtype=np.float32)
        arrays[f"{key}_epsilon_history"] = np.asarray(r["epsilon_history"], dtype=np.float32)

        summary[name] = {
            "distribution": r["distribution"],
            "tau": float(r["tau"]),
            "tau_std": float(r['tau_std']),
            "ess": float(r["ess"]),
            "ess_by_coordinate": np.asarray(r["ess_by_coordinate"]).tolist(),
            "ess_valid_fraction": np.asarray(r["ess_valid_fraction"]).tolist(),
            "n_evaluations": float(r["n_evaluations"]),
            "evaluation_type": r["evaluation_type"],
            "ess_per_eval": float(r["ess_per_eval"]),
            "time": float(r["time"]),
            "sigma": float(r["sigma"]),
            "epsilon": float(r["epsilon"]),
            "n_leapfrog": r["n_leapfrog"],
            "n_warmup": r["n_warmup"],
            "target_accept": r["target_accept"],
            "cond":r['cond']
            ,"mean_mse": float(r["mean_mse"])
            ,"cov_mse": float(r["cov_mse"])
            ,"acceptance_rate_mean": r["acceptance_rate_mean"]
            ,"acceptance_rate_min": r["acceptance_rate_min"]
            ,"acceptance_rate_max": r["acceptance_rate_max"]
            ,"n_divergent": r["n_divergent"]
            ,"actual_n_chains": r["actual_n_chains"]
            ,"diagnostic_draws_per_chain": r["diagnostic_draws_per_chain"]
            ,"sample_finite_fraction": r["sample_finite_fraction"]
            ,"finite_draw_fraction": r["finite_draw_fraction"]
            ,"sample_covariance_rank": r["sample_covariance_rank"]
            ,"sample_covariance_full_rank": r["sample_covariance_full_rank"]
            
        }

    metadata = {
        "dim": dim,
        "seed": seed,
        "n_samples": args.n_samples,
        "burn_in": args.burn_in,
        "n_thin": args.n_thin,
        "n_chains": args.n_chains,
        "hwm_walkers_requested": args.hwm_walkers,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "af": af,
        "initialization_mode": INITIALIZATION_MODE,
        "initialization_description": (
            "independent N(0, 2^2) coordinates in the intrinsic Rosenbrock frame"
            if INITIALIZATION_MODE == "agnostic"
            else "exact draws from the paired Rosenbrock target"
        ),
        "transform_affine": bool(transform["affine"]),
        "overlay_rosenbrock": overlay,
        "summary": summary,
    }

    np.savez_compressed(os.path.join(outdir, "results_arrays_light.npz"), **arrays)

    with open(os.path.join(outdir, "results_summary.json"), "w") as f:
        json.dump(metadata, f, indent=2)

save_light_results(
    results=results,
    transform=transform,
    overlay=overlay_rosenbrock,
    outdir=outdir,
    dim=d,
    af=af,
)
update_seed_manifest(
    base_outdir, seed, outdir,
    {"dim": d, "condition_number": cond, "affine": af,
     "n_samples": args.n_samples, "burn_in": args.burn_in,
     "n_thin": args.n_thin, "n_chains": args.n_chains,
     "hwm_walkers": args.hwm_walkers, "jax_enable_x64": True,
     "initialization_mode": INITIALIZATION_MODE},
)

# Plot the results
if not args.no_plots:

    plot_Rosenbrock_results(
        results, log_density, dim=d, sigma=ROSENBROCK_SIGMA,
        transform=transform, output_dir=outdir
    )
    benchmark_corner(results, corner_dir, thin = 10,  overlay_rosenbrock=overlay_rosenbrock, transform=transform)
    benchmark_trends(results, trends_dir, 'RosenbrockC')
    benchmark_autocorrelation(results, outdir, 'RosenbrockC')

if not args.no_report:
    from generate_report import SamplerReport

    report = SamplerReport(results=results, label = 'RosenbrockC', transform=transform, overlay=overlay_rosenbrock)
    report.compile_pdf(texname = os.path.join(outdir, f'RosenbrockC_SamplerReport_{d}d{afstring}.tex'), template_dir='templates', latex_compiler='pdflatex')

print("Rosenbrock distribution benchmark complete. Check the output directory for plots.")
