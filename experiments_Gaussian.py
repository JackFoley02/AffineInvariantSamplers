import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import argparse
import json
from samplers.sampler_nuts import hmc_nuts
from samplers.samplers import stretch_move
from samplers.sampler_chees import hmc_chees
from samplers.sampler_peachees import hamiltonian_walk_chees
from plotTools.benchmark_autocorrelation import benchmark_autocorrelation
from plotTools.benchmark_corner import benchmark_corner
from plotTools.benchmark_trends import benchmark_trends
from autocorrelation_func import autocorrelation_fft, integrated_autocorr_time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

N_CHAINS = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark tests of different MCMC sampling algorithms applied to a multidimensional Gaussian distribution."
    )
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--af", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cond", type=float, default=50.0)
    return parser.parse_args()


args = parse_args()
d = args.dim
af = args.af
gpu = args.gpu
cond = args.cond

if gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

import jax
import jax.numpy as jnp


def create_high_dim_precision(dim, condition_number=1.0):
    np.random.seed(42)
    H = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(H)

    eigenvalues = 0.1 * np.linspace(1, condition_number, dim)

    precision = Q @ np.diag(eigenvalues) @ Q.T
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

    U = U_full[:dim, :dim]

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


def benchmark_samplers(dim=40, n_samples=10000, burn_in=1000, condition_number=cond, affine=False):
    precision_matrix = create_high_dim_precision(dim, 1)

    cov_matrix = np.linalg.inv(precision_matrix)
    true_mean = np.ones(dim)

    precision_jax = jnp.asarray(precision_matrix)
    true_mean_jax = jnp.asarray(true_mean)

    if affine:
        A, B, Ai = affine_transform(dim, trans_scale=2, cond=condition_number)
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
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        u = (x - B) @ Ai.T
        centered = u - true_mean
        result = -0.5 * np.einsum("ij,jk,ik->i", centered, precision_matrix, centered)

        return result

    def gradient(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        u = (x - B) @ Ai.T
        centered = u - true_mean
        result = np.einsum("jk,ij->ik", precision_matrix, centered)

        return result @ Ai

    def potential(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        u = (x - B) @ Ai.T
        centered = u - true_mean
        result = 0.5 * np.einsum("ij,jk,ik->i", centered, precision_matrix, centered)

        return result

    u0 = np.random.multivariate_normal(true_mean, cov_matrix)
    initial = u0 @ A.T + B if affine else u0

    results = {}
    total_samples = n_samples + burn_in

    samplers = {
        "Dense-mass NUTS": lambda: hmc_nuts(
            log_density_jax,
            initial,
            total_samples,
            epsilon=0.1,
            n_chains=N_CHAINS,
            n_warmup=1000,
            n_thin=10,
            max_tree_depth=13,
        ),
        "Hamiltonian Walk Move": lambda: hamiltonian_walk_chees(
            log_density_jax,
            initial,
            total_samples,
            n_walkers=N_CHAINS,
            epsilon=0.1,
            L=10,
            n_warmup=1000,
            max_L=1000,
            n_thin=10,
        ),
        "Stretch Move": lambda: stretch_move(
            log_density,
            initial,
            total_samples,
            n_walkers=N_CHAINS,
            a=1.0 + 2.151 / np.sqrt(dim),
            n_thin=10,
        ),
        "HMC": lambda: hmc_chees(
            log_density_jax,
            initial,
            total_samples,
            epsilon=0.1,
            L=10,
            n_chains=N_CHAINS,
            n_warmup=1000,
            max_L=1000,
            n_thin=10,
        ),
    }

    for name, sampler_func in samplers.items():
        print(f"Running {name}...")
        start_time = time.time()

        out = sampler_func()
        samples, acceptance_rates = out[:2]
        eps_final = out[2] if len(out) > 2 else 0.0
        eps_hist = out[3] if len(out) > 3 else [0.0, 0.0, 0.0]

        if len(out) > 4:
            n_leapfrog, n_warmup, target_accept, gamma, t0, kappa = out[4]
        else:
            n_leapfrog = n_warmup = target_accept = gamma = t0 = kappa = None

        post_burn_in_samples = samples[:, burn_in:, :]
        burn_in_samps = samples[:, :burn_in, :]

        flat_samples = post_burn_in_samples.reshape(-1, dim)

        sample_mean = np.mean(flat_samples, axis=0)
        sample_cov = np.cov(flat_samples, rowvar=False)

        mean_mse = np.mean((sample_mean - true_mean) ** 2) / np.mean(true_mean**2)
        cov_mse = np.sum((sample_cov - cov_matrix) ** 2) / np.sum(cov_matrix**2)

        acf = autocorrelation_fft(np.mean(samples[:, :, 0], axis=0))

        try:
            taus = []
            esses = []

            for k in range(dim):
                coord_series = np.mean(post_burn_in_samples[:, :, k], axis=0)
                tau_k, _, ess_k = integrated_autocorr_time(coord_series)

                if np.isfinite(tau_k):
                    taus.append(tau_k)
                    esses.append(ess_k)

            tau = np.median(taus)
            ess = np.median(esses)
            tau_std = np.std(taus)
        except:
            tau, ess = np.nan, np.nan
            tau_std = np.nan

        elapsed = time.time() - start_time

        results[name] = {
            "distribution": "Gaussian",
            "samples": flat_samples,
            "series": post_burn_in_samples,
            "burn_in": burn_in_samps,
            "acceptance_rates": acceptance_rates,
            "mean_mse": mean_mse,
            "mean": sample_mean,
            "cov_mse": cov_mse,
            "cov": sample_cov,
            "autocorrelation": acf,
            "tau": tau,
            "tau_std": tau_std,
            "ess": ess,
            "time": elapsed,
            "labels": None,
            "sigma": 0.0,
            "epsilon": eps_final,
            "epsilon_history": eps_hist,
            "n_leapfrog": n_leapfrog,
            "n_warmup": n_warmup,
            "target_accept": target_accept,
            "gamma": gamma,
            "t0": t0,
            "kappa": kappa,
        }

        print(f"  Acceptance rate: {np.mean(acceptance_rates):.2f}")
        print(f"  Mean MSE: {mean_mse:.6f}")
        print(f"  Covariance MSE: {cov_mse:.6f}")
        print(f"  Integrated autocorrelation time: {tau:.2f}")
        print(f"  Time: {elapsed:.2f} seconds")

    transform = {"affine": affine, "A": A, "B": B}

    return results, true_mean, cov_matrix, transform


results, true_mean, cov, transform = benchmark_samplers(
    dim=d,
    n_samples=10000,
    burn_in=1000,
    condition_number=cond,
    affine=af,
)

samplers = list(results.keys())

if transform["affine"]:
    print("NOTE: an affine transformation has been applied.")
    afstring = "_af"
else:
    afstring = ""

cond_string = f"{cond:g}c"

outdir = f"GaussianResultsC/{cond_string}/{d}d{afstring}"
corner_dir = f"{outdir}/corner"
trends_dir = f"{outdir}/trends"

os.makedirs(outdir, exist_ok=True)
os.makedirs(corner_dir, exist_ok=True)
os.makedirs(trends_dir, exist_ok=True)

colors = ["black", "black", "purple", "blue", "green", "y", "orange", "red"]

for i, name in enumerate(samplers):
    if results[name]["epsilon"] == 0.0:
        continue
    else:
        plt.plot(
            np.arange(0, len(results[name]["epsilon_history"])),
            results[name]["epsilon_history"],
            color=colors[i],
            linestyle=":",
            label=f"{name}: epsilon = {np.round(results[name]['epsilon'], 3)}",
        )

plt.title("StepSizeTuner Results")
plt.xlabel("Warm-up Step Number")
plt.ylabel("Epsilon value")
plt.ylim(0, 1.8)
plt.xlim(left=0.0)
plt.legend()
plt.savefig(os.path.join(outdir, "StepSize.pdf"))
plt.close()


def save_light_results(results, transform, outdir, dim, af, cond):
    os.makedirs(outdir, exist_ok=True)

    summary = {}

    arrays = {
        "A": np.asarray(transform["A"]),
        "B": np.asarray(transform["B"]),
    }

    for name, r in results.items():
        key = name.replace(" ", "_")

        series = np.asarray(r["series"])
        arrays[f"{key}_series_thin10"] = series[:, ::10, :].astype(np.float32)

        arrays[f"{key}_acceptance_rates"] = np.asarray(r["acceptance_rates"], dtype=np.float32)
        arrays[f"{key}_mean"] = np.asarray(r["mean"], dtype=np.float32)
        arrays[f"{key}_cov"] = np.asarray(r["cov"], dtype=np.float32)
        arrays[f"{key}_autocorrelation"] = np.asarray(r["autocorrelation"], dtype=np.float32)
        arrays[f"{key}_epsilon_history"] = np.asarray(r["epsilon_history"], dtype=np.float32)

        summary[name] = {
            "distribution": r["distribution"],
            "tau": float(r["tau"]),
            "tau_std": float(r["tau_std"]),
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
        "cond": cond,
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
    cond=cond,
)

if not args.no_plots:
    benchmark_corner(results, corner_dir, thin=10, transform=transform)
    benchmark_trends(results, trends_dir, "GaussianC")
    benchmark_autocorrelation(results, outdir, "GaussianC")

if not args.no_report:
    from generate_report import SamplerReport

    report = SamplerReport(results=results, label="GaussianC", transform=transform)
    report.compile_pdf(
        texname=os.path.join(outdir, f"GaussianC_SamplerReport_{d}d{afstring}.tex"),
        template_dir="templates",
        latex_compiler="pdflatex",
    )

print("Gaussian distribution benchmark complete. Check the output directory for plots.")