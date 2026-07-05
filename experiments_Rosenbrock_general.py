import numpy as np
import matplotlib.pyplot as plt
import time
from plotTools.benchmark_corner import benchmark_corner
from plotTools.benchmark_trends import benchmark_trends
from plotTools.benchmark_autocorrelation import benchmark_autocorrelation
from matplotlib import rc
import os
import json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import jax
import jax.numpy as jnp
rc('text', usetex=False)

import argparse
from samplers.samplers import side_move, stretch_move
from samplers.sampler_chees import hmc_chees
from samplers.sampler_peachees import hamiltonian_walk_chees
from autocorrelation_func import autocorrelation_fft, integrated_autocorr_time


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
    parser.add_argument("--cond")


    return parser.parse_args()


args = parse_args()
d = args.dim
af = args.af


if (d % 2) != 0:
    raise ValueError('# of dimensions must be an even number.')

def affine_transform(dim, max_dim=128, seed=4321, trans_scale=2.0, cond=50.0):
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

def benchmark_samplers_Rosenbrock_general(dim=2, n_samples=10000, burn_in=1000, sigma=0.7, a = 1.0, b = 100.0, affine = False):
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
        A, B, Ai = affine_transform(dim = dim, trans_scale=2.0, cond=50) 
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
    def potential(z):
        """potential energy, V(z) = -log(p(z)) of the Rosenbrock distribution (p(z))"""
        z = np.asarray(z)
        u = (z - B) @ AiT
        R = Rosenbrock_paired(u)
        pot = 0.5 * R / (sigma**2)
        return pot

    # Define the gradient of the negative log density
    def gradient(z):
        """Gradient of the negative log density (potential gradient/potential energy)"""
        z = np.asarray(z)
        u = (z - B) @ AiT
        gu = 0.5 * grad_Rosenbrock_paired(u) / (sigma**2)
        gz = gu @ Ai
        return gz
    

    def log_density_jax(x, a = 1.0, b = 100.0, sigma = 0.7):
        x = jnp.asarray(x)

        u = (x - B_jax) @ AiT_jax

        params_odd = u[0::2] #all odd parameters
        params_even = u[1::2] #all even parameters

        R = jnp.sum(b * (params_even - params_odd**2)**2 + (params_odd -a)**2)

        return -0.5 * R / sigma**2


    u0 = np.ones(dim) + (sigma / np.sqrt(dim)) * np.random.randn(dim)

    if affine:
        initial = u0 @ A.T + B
    else:
        initial = u0
    
    # Dictionary to store results
    results = {}
    
    # Define samplers to benchmark with burn-in
    total_samples = n_samples + burn_in
    
    # Define samplers to benchmark - parameters tuned for the ring distribution
    samplers = {
        "HMC": lambda: hmc_chees(log_density_jax, initial, total_samples, epsilon=0.02, L=20, n_chains=260, n_warmup = 10000, max_L = 1000, n_thin = 10),
        "Hamiltonian Walk Move": lambda: hamiltonian_walk_chees(
            log_density_jax, initial, total_samples,
            n_walkers = 260, epsilon=0.02, L=20, n_warmup = 10000, max_L = 1000, n_thin = 10
        ),
    }

    
    # Benchmark each sampler with careful error handling
    for name, sampler_func in samplers.items():
        print(f"Running {name}...")
        start_time = time.time()
        
        try:

            out = sampler_func()
            samples, acceptance_rates = out[:2]

            eps_final = out[2] if len(out) > 2 else 0.0
            eps_hist = out[3] if len(out) > 3 else [0.0, 0.0, 0.0]
            if len(out) > 4:
                n_leapfrog, n_warmup, target_accept, gamma, t0, kappa = out[4]
            else:
                n_leapfrog = n_warmup = target_accept = gamma = t0 = kappa = None            
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
            
            # Compute integrated autocorrelation time for first dimension
            try:
                taus = []
                esses = []

                for k in range(dim):
                    coord_series = np.mean(post_burn_in_samples[:, :, k], axis=0)
                    tau_k, _, ess_k = integrated_autocorr_time(coord_series)

                    if np.isfinite(tau_k):
                        taus.append(tau_k)
                        esses.append(ess_k)

                tau = np.median(taus)   # robust average
                ess = np.median(esses)
                tau_std = np.std(taus)
            except:
                tau, ess = np.nan, np.nan
                print("  Warning: Could not compute integrated autocorrelation time")
            
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
            # Create dummy data in case of error
            flat_samples = np.zeros((10, dim))
            acceptance_rates = np.zeros(dim)
            mean = np.full(dim, np.nan)
            cov = np.full((dim, dim), np.nan)
            acf = np.zeros(100)
            tau, ess = np.nan, np.nan
        
        elapsed = time.time() - start_time
        
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
            "time": elapsed,
            "labels": None,
            "sigma":sigma,
            "epsilon":eps_final,
            "epsilon_history":eps_hist,
            "n_leapfrog":n_leapfrog,
            "n_warmup":n_warmup,
            "target_accept":target_accept,
            "gamma":gamma,
            "t0":t0,
            "kappa":kappa
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

def plot_Rosenbrock_results(results, log_density, dim=2, sigma=0.7, transform = {'affine':False}):
    """Plot comparison of sampler results for ring distribution"""
    samplers = list(results.keys())

    if transform['affine']:
        rotstring = "_af" #suffix to append to file name if affine transformation has been applied
    else:
        rotstring = '' #no transformation? set suffix to empty string

    if dim < 2:
        print("Skipping Rosenbrock overlay plot: need at least 2 dimensions.")
        return

    def rosenbrock_2d_log_density(x, y, sigma=0.7, a=1.0, b=100.0):
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
    fig.savefig(f"RosenbrockResults/{dim}d{rotstring}/Rosenbrock_overlay_first2.png", dpi=220, bbox_inches="tight")
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
    plt.savefig(f'RosenbrockResults/{d}d{rotstring}/StepSize.pdf')
    plt.close()


    
    
    return

# Run the benchmark for the Rosenbrock distribution
# Note: You need to have the sampler functions (side_move, stretch_move, etc.) defined elsewhere
results, sigma, log_density, transform = benchmark_samplers_Rosenbrock_general(dim=d, n_samples= 100000, burn_in=10000, sigma=0.7, affine = af)

if transform['affine']:
    afstring = '_af'
else:
    afstring = ''


#make directories
outdir = f"RosenbrockResults/{d}d{afstring}"
corner_dir = f"RosenbrockResults/{d}d{afstring}/corner"
trends_dir = f"RosenbrockResults/{d}d{afstring}/trends"
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
            "time": float(r["time"]),
            "sigma": float(r["sigma"]),
            "epsilon": float(r["epsilon"]),
            "n_leapfrog": r["n_leapfrog"],
            "n_warmup": r["n_warmup"],
            "target_accept": r["target_accept"],
            "gamma": r["gamma"],
            "t0": r["t0"],
            "kappa": r["kappa"],
        }

    metadata = {
        "dim": dim,
        "af": af,
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

# Plot the results
if not args.no_plots:

    plot_Rosenbrock_results(results, log_density, dim=d, sigma=0.7, transform = transform)
    benchmark_corner(results, corner_dir, thin = 10,  overlay_rosenbrock=overlay_rosenbrock, transform=transform)
    benchmark_trends(results, trends_dir, 'RosenbrockTuned')
    benchmark_autocorrelation(results, outdir, 'RosenbrockTuned')

if not args.no_report:
    from generate_report import SamplerReport

    report = SamplerReport(results=results, label = 'RosenbrockTuned', transform=transform, overlay=overlay_rosenbrock)
    report.compile_pdf(texname = os.path.join(outdir, f'RosenbrockTuned_SamplerReport_{d}d{afstring}.tex'), template_dir='templates', latex_compiler='pdflatex')

print("Rosenbrock distribution benchmark complete. Check the output directory for plots.")