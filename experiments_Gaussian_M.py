import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import argparse
import json
import traceback
from samplers.sampler_nuts import hmc_nuts
from samplers.samplers import stretch_move
from samplers.sampler_chees import hmc_chees
from samplers.sampler_peachees import hamiltonian_walk_chees
from plotTools.benchmark_autocorrelation import benchmark_autocorrelation
from plotTools.benchmark_corner import benchmark_corner
from plotTools.benchmark_trends import benchmark_trends
from autocorrelation_func import autocorrelation_fft, integrated_autocorr_time
from experiment_diagnostics import worst_coordinate_ess, evaluation_count, update_seed_manifest
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

N_CHAINS = 100

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
    parser.add_argument('--cond', type = int, default = 50.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--burn-in', type=int, default=1000)
    parser.add_argument('--n-warmup', type=int, default=1000)
    parser.add_argument('--n-chains', type=int, default=N_CHAINS)
    parser.add_argument('--n-thin', type=int, default=1)

    return parser.parse_args()


args = parse_args()
d = args.dim
af = args.af
gpu = args.gpu
cond = args.cond
seed = args.seed

if gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

#importing JAX after setting default device
import jax
import jax.numpy as jnp

def create_high_dim_precision(dim, condition_number=1.0):
    """Create a high-dimensional precision matrix with given condition number."""
    # Create random eigenvectors (orthogonal matrix)
    np.random.seed(42)  # For reproducibility
    H = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(H)
    
    # Create eigenvalues with desired condition number
    eigenvalues = 0.1* np.linspace(1, condition_number, dim)
    
    # Construct precision matrix: Q @ diag(eigenvalues) @ Q.T
    precision = Q @ np.diag(eigenvalues) @ Q.T
    
    # Ensure it's symmetric (fix numerical issues)
    precision = 0.5 * (precision + precision.T)
    
    return precision

def affine_transform(dim, max_dim=128, seed=1234, trans_scale=2.0, cond=cond):
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


def benchmark_samplers(dim=40, n_samples=10000, burn_in=1000, condition_number=cond,
                       affine=False, seed=0, n_warmup=1000, n_chains=N_CHAINS, n_thin=1):
    """
    Benchmark different MCMC samplers on a high-dimensional Gaussian.
    """
    # Create precision matrix (inverse covariance)
    precision_matrix = create_high_dim_precision(dim, 1)
    
    # Compute covariance matrix for reference (needed for evaluation)
    cov_matrix = np.linalg.inv(precision_matrix)
    
    true_mean = np.ones(dim)

    precision_jax = jnp.asarray(precision_matrix)
    true_mean_jax = jnp.asarray(true_mean)


    if affine:
        A, B, Ai = affine_transform(dim, trans_scale=2, cond = condition_number)
    else:
        A = np.eye(dim)
        B = np.zeros(dim)
        Ai = np.eye(dim)

    B_jax = jnp.asarray(B)
    Ai_jax = jnp.asarray(Ai)

    @jax.jit
    def log_density_jax(x):
        x = jnp.asarray(x)

        u = (x - B_jax) @ Ai_jax.T
        centered = u - true_mean_jax

        return -0.5 * centered @ precision_jax @ centered

    def log_density(x):
        """Vectorized log density of the multivariate Gaussian"""
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        u = (x - B) @ Ai.T

        # Vectorized operation for all samples
        centered = u - true_mean
        # Using einsum for efficient batch matrix multiplication
        result = -0.5 * np.einsum('ij,jk,ik->i', centered, precision_matrix, centered)
            
        return result
    
    def gradient(x):
        """Vectorized gradient of the negative log density (potential gradient)"""
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        u = (x - B) @ Ai.T

        # Vectorized operation for all samples
        centered = u - true_mean
        # Matrix multiplication for each sample in the batch
        result = np.einsum('jk,ij->ik', precision_matrix, centered)
            
        return result @ Ai #Chain rule pulls out an additional Ainv
    
    def potential(x):
        """Vectorized negative log density (potential energy)"""
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        u = (x - B) @ Ai.T
        # Vectorized operation for all samples
        centered = u - true_mean
        result = 0.5 * np.einsum('ij,jk,ik->i', centered, precision_matrix, centered)
            
        return result
    
    # Initial state
    rng = np.random.default_rng(seed)
    u0 = rng.multivariate_normal(true_mean, cov_matrix, size=n_chains)
    initial = u0 @ A.T + B
    
    # Dictionary to store results
    results = {}
    
    # Define samplers to benchmark with burn-in
    total_samples = n_samples + burn_in
    
    # Define samplers to benchmark - adjust parameters for high-dimensional case
    samplers = {
        "Dense-mass NUTS": lambda: hmc_nuts(log_density_jax, initial, total_samples, epsilon=0.1, n_chains=n_chains, n_warmup=n_warmup, n_thin=n_thin, max_tree_depth=13, seed=seed, progress_bar=False),
        "Hamiltonian Walk Move": lambda: hamiltonian_walk_chees(
            log_density_jax, initial, total_samples,
            n_walkers=n_chains, epsilon=0.1, L=10, n_warmup=n_warmup, max_L=1000, n_thin=n_thin, seed=seed
        ),        
        "Stretch Move": lambda: stretch_move(log_density, initial, total_samples, n_walkers=n_chains, a=1.0+2.151/np.sqrt(dim), n_thin=n_thin),
        "HMC": lambda: hmc_chees(log_density_jax, initial, total_samples, epsilon=0.1, L=10, n_chains=n_chains, n_warmup=n_warmup, max_L=1000, n_thin=n_thin, seed=seed),
 
    }
    for name, sampler_func in samplers.items():
        np.random.seed(seed)
        print(f"Running {name}...")
        start_time = time.time()

        out = sampler_func()
        samples, acceptance_rates = out[:2]
        eps_final = out[2] if len(out) > 2 else 0.0
        eps_hist = out[3] if len(out) > 3 else [0.0, 0.0, 0.0]

        if len(out) > 4:
            n_leapfrog, sampler_n_warmup, target_accept, gamma, t0, kappa = out[4]
        else:
            n_leapfrog = sampler_n_warmup = target_accept = gamma = t0 = kappa = None

        
        # Apply burn-in: discard the first burn_in samples
        post_burn_in_samples = samples[:, burn_in:, :]
        burn_in_samps = samples[:, :burn_in, :]
        
        # Flatten samples from all chains
        flat_samples = post_burn_in_samples.reshape(-1, dim)
        
        # Compute sample mean and covariance
        rest_samples = (post_burn_in_samples - B) @ Ai.T if affine else post_burn_in_samples
        flat_rest_samples = rest_samples.reshape(-1, dim)
        sample_mean = np.mean(flat_rest_samples, axis=0)
        sample_cov = np.cov(flat_rest_samples, rowvar=False)
        # Calculate mean squared error for mean and covariance
        mean_mse = np.mean((sample_mean - true_mean)**2) / np.mean(true_mean**2)
        cov_mse = np.sum((sample_cov - cov_matrix)**2) / np.sum(cov_matrix**2)
        
        # Compute autocorrelation for first dimension
        # Average over chains to compute autocorrelation
        acf = autocorrelation_fft(np.mean(samples[:, :, 0],axis=0))
        
        # Compute integrated autocorrelation time for first dimension
        transform_now = {"affine": affine, "A": A, "B": B}
        ess, tau, ess_by_coordinate, tau_by_coordinate, ess_valid_fraction = worst_coordinate_ess(post_burn_in_samples, transform_now)
        tau_std = float(np.nanstd(tau_by_coordinate))

        elapsed = time.time() - start_time
        n_evaluations, evaluation_type = evaluation_count(name, len(acceptance_rates), n_samples, n_thin, n_leapfrog)
        ess_per_eval = ess / n_evaluations if np.isfinite(ess) and np.isfinite(n_evaluations) else np.nan
        
        # Store results
        results[name] = {
            "distribution":'Gaussian',
            "samples": flat_samples,
            "series":post_burn_in_samples,
            "burn_in":burn_in_samps,
            "acceptance_rates": acceptance_rates,
            "mean_mse": mean_mse,
            "mean":sample_mean,
            "cov_mse": cov_mse,
            "cov": sample_cov,
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
            "sigma":0.0,
            "epsilon":eps_final,
            "epsilon_history":eps_hist,
            "n_leapfrog":n_leapfrog,
            "n_warmup":sampler_n_warmup,
            "target_accept":target_accept,
            "gamma":gamma,
            "t0":t0,
            "kappa":kappa
        }
        
        print(f"  Acceptance rate: {np.mean(acceptance_rates):.2f}")
        print(f"  Mean MSE: {mean_mse:.6f}")
        print(f"  Covariance MSE: {cov_mse:.6f}")
        print(f"  Integrated autocorrelation time: {tau:.2f}")
        print(f"  Time: {elapsed:.2f} seconds")

    transform = {"affine":affine, 'A':A, 'B':B} 
    
    return results, true_mean, cov_matrix, transform

results, true_mean, cov, transform = benchmark_samplers(
    dim=d, n_samples=args.n_samples, burn_in=args.burn_in,
    condition_number=cond, affine=af, seed=seed,
    n_warmup=args.n_warmup, n_chains=args.n_chains, n_thin=args.n_thin,
)

samplers = list(results.keys())

if transform['affine']:
    print('NOTE: an affine transformation has been applied.')
    afstring = '_af'
else:
    afstring = ''
# Plot the results
colors = ['black', 'black', 'purple', 'blue', 'green', 'y', 'orange', 'red']

for i, name in enumerate(samplers):
        if results[name]['epsilon'] == 0.0: #skip samplers without recorded epsilon histories
            continue
        else:
            plt.plot(np.arange(0, len(results[name]['epsilon_history'])), results[name]['epsilon_history'],
                     color = colors[i], linestyle = ':', label = f"{name}: epsilon = {np.round(results[name]['epsilon'], 3)}")


base_outdir = f"GaussianResultsM/{d}d{afstring}"
outdir = os.path.join(base_outdir, "seeds", f"seed_{seed:05d}")
corner_dir = os.path.join(outdir, "corner")
trends_dir = os.path.join(outdir, "trends")
os.makedirs(outdir, exist_ok=True)
os.makedirs(corner_dir, exist_ok=True)
os.makedirs(trends_dir, exist_ok=True)

plt.title('StepSizeTuner Results')
plt.xlabel('Warm-up Step Number')
plt.ylabel('Epsilon value')
plt.ylim(0, 1.8)
plt.xlim(left = 0.0)
plt.legend()
plt.savefig(os.path.join(outdir, 'StepSize.pdf'))
plt.close()
#pickling output to compile locally
def save_light_results(results, transform, outdir, dim, af):
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
            "gamma": r["gamma"],
            "t0": r["t0"],
            "kappa": r["kappa"],
        }

    metadata = {
        "dim": dim,
        "seed": seed,
        "condition_number": cond,
        "n_samples": args.n_samples,
        "burn_in": args.burn_in,
        "n_thin": args.n_thin,
        "n_chains": args.n_chains,
        "af": af,
        "transform_affine": bool(transform["affine"]),
        "summary": summary,
    }

    np.savez_compressed(os.path.join(outdir, "results_arrays_light.npz"), **arrays)

    with open(os.path.join(outdir, "results_summary.json"), "w") as f:
        json.dump(metadata, f, indent=2)

save_light_results(
    results=results,
    transform=transform,
    outdir=outdir,
    dim=d,
    af=af,
)
update_seed_manifest(
    base_outdir, seed, outdir,
    {"dim": d, "condition_number": cond, "affine": af,
     "n_samples": args.n_samples, "burn_in": args.burn_in,
     "n_thin": args.n_thin, "n_chains": args.n_chains},
)

# Plot the results
if not args.no_plots:

    benchmark_corner(results, corner_dir, thin = 10, transform=transform)
    benchmark_trends(results, trends_dir, 'GaussianM')
    benchmark_autocorrelation(results, outdir, 'GaussianM')

if not args.no_report:
    from generate_report import SamplerReport

    report = SamplerReport(results=results, label = 'GaussianM', transform=transform)
    report.compile_pdf(texname = os.path.join(outdir, f'GaussianM_SamplerReport_{d}d{afstring}.tex'), template_dir='templates', latex_compiler='pdflatex')

print("Gaussian distribution benchmark complete. Check the output directory for plots.")
