import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

from samplers import side_move, stretch_move
from samplers_dualAvg import hmc_sst, hamiltonian_walk_move_sst
from plotTools.benchmark_autocorrelation import benchmark_autocorrelation
from plotTools.benchmark_corner import benchmark_corner
from plotTools.benchmark_trends import benchmark_trends
from autocorrelation_func import autocorrelation_fft, integrated_autocorr_time
from generate_report import SamplerReport

#parser to identify the number of dimensions from command line
def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark tests of different MCMC sampling algorithms, applied to a multidimensional Gaussina distribution.'
    )
    parser.add_argument(
        '--dim',
        type = int,
        default = 2,
        help="Number of dimensions for the Gaussian benchmark (default: 2)"
    )
    parser.add_argument(
        '--af',
        action='store_true',
        help = 'Apply an arbitrary affine transformation to the target distribution. Useful for testing affine-invariance of ensemble sampling methods.'
    )
    return parser.parse_args()


args = parse_args()
d = args.dim
af = args.af

def create_high_dim_precision(dim, condition_number=100):
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

def affine_transform(dim: int, trans_scale: float = 2.0, cond: float = 10.0):
        """
        Generates an arbitrary transformation matrix A, and translation vector B, to perform
        an affine-invariant transformation upon the distribution of choice, i.e. 

        y = Ax + B
        
        Arguments:
        ----------
        dim:
            integer number of dimensions for the particular distribution. 
        trans_scale:
            The degree of translation applied to the distribution. Generally, this shouldn't impact the 
            ability of samplers on the distribution, regardless of their algorithmic affine-invariance
        cond:
            The length of the longest axis divided by the length of the smallest axis (which is set to 1). 
            Encodes the degree to which different axes are stretched. A larger value will generate a more anisotropic distribution

        Returns:
        ----------
        A:
            The transformation matrix (dim x dim).
        B: 
            The translation vector (dim x 1).
        Ainv:
            The inverse of A (dim x dim).
        """

        rng = np.random.default_rng(42)
        
        B = rng.uniform(-trans_scale, trans_scale, size = dim) #define B vector

        #A bunch of matrix calculations to follow. First, let's get some rotation matrices.

        Mu = rng.standard_normal((dim, dim)) #Define entirely random square matrix
        Qu, Ru = np.linalg.qr(Mu) #Take QR decomposition of Mu. Qu gives the orthonormal basis, Ru gives coefficients of projection

        #QR Decomp. is not unique! We need to ensure there is no ambiguity in the sign of diag(Q)
        #This is important for saving time when checking if Q is a true rotation matrix (i.e. det(Q) = +1)

        signsu = np.sign(np.diag(Ru)) #Take signs from each diagonal R entry
        signsu[signsu == 0] = 1.0 #convert zeroes (from dependent axes) to ones
        U = Qu @ np.diag(signsu) #left w/ transformation matrix U, with unambiguous signage
        if np.linalg.det(U) < 0: #Check to make sure det(S) = +1, fix it otherwise
            U[:, 0] *= -1

        #We can represent A, an arbitrary affine transformation, in terms of its singular value decomposition:
        # A = U @ S for singular matrix S. This encodes the stretching along each coordinate axis. 
        s_vals = np.geomspace(1.0, cond, dim)
        S = np.diag(s_vals)

        A = U @ S
        Ainv = np.linalg.inv(A)
        
        return A, B, Ainv


def benchmark_samplers(dim=40, n_samples=10000, burn_in=1000, condition_number=100, affine = False):
    """
    Benchmark different MCMC samplers on a high-dimensional Gaussian.
    """
    # Create precision matrix (inverse covariance)
    precision_matrix = create_high_dim_precision(dim, condition_number)
    
    # Compute covariance matrix for reference (needed for evaluation)
    cov_matrix = np.linalg.inv(precision_matrix)
    
    true_mean = np.ones(dim)

 
    if affine:
        A, B, Ai = affine_transform(dim, 2, 50)
    else:
        A = np.eye(dim)
        B = np.zeros(dim)
        Ai = np.eye(dim)

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
    initial = np.zeros(dim)
    
    
    # Dictionary to store results
    results = {}
    
    # Define samplers to benchmark with burn-in
    total_samples = n_samples + burn_in
    
    # Define samplers to benchmark - adjust parameters for high-dimensional case
    samplers = {
        "Side Move": lambda: side_move(log_density, initial, total_samples, n_walkers=min(dim*20, 200), gamma=1.687),
        "Stretch Move": lambda: stretch_move(log_density, initial, total_samples, n_walkers=min(dim*20, 200), a=1.0+2.151/np.sqrt(dim)),
        "HMC n=10": lambda: hmc_sst(log_density, initial, total_samples, gradient, epsilon=0.1, L=10, n_chains=1, emax = 1.8),
        "HMC n=2": lambda: hmc_sst(log_density, initial, total_samples, gradient, epsilon=0.5, L=2, n_chains=1, emax = 1.8),
        "Hamiltonian Walk Move n=10": lambda: hamiltonian_walk_move_sst(gradient, potential, initial, total_samples, 
                                                        n_chains_per_group=min(dim*10, 100), epsilon=0.1, n_leapfrog=10, beta=1.0, emax = 1.8),
        "Hamiltonian Walk Move n=2": lambda: hamiltonian_walk_move_sst(gradient, potential, initial, total_samples, 
                                                        n_chains_per_group=min(dim*10, 100), epsilon=0.5, n_leapfrog=2, beta=1.0, emax = 1.8),
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

        
        # Apply burn-in: discard the first burn_in samples
        post_burn_in_samples = samples[:, burn_in:, :]
        burn_in_samps = samples[:, :burn_in, :]
        
        # Flatten samples from all chains
        flat_samples = post_burn_in_samples.reshape(-1, dim)
        
        # Compute sample mean and covariance
        sample_mean = np.mean(flat_samples, axis=0)
        sample_cov = np.cov(flat_samples, rowvar=False)
        # Calculate mean squared error for mean and covariance
        mean_mse = np.mean((sample_mean - true_mean)**2) / np.mean(true_mean**2)
        cov_mse = np.sum((sample_cov - cov_matrix)**2) / np.sum(cov_matrix**2)
        
        # Compute autocorrelation for first dimension
        # Average over chains to compute autocorrelation
        acf = autocorrelation_fft(np.mean(samples[:, :, 0],axis=0))
        
        # Compute integrated autocorrelation time for first dimension
        try:
            tau, _, ess = integrated_autocorr_time(np.mean(samples[:, :, 0],axis=0))
        except:
            tau, ess = np.nan, np.nan

        elapsed = time.time() - start_time
        
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
            "ess": ess,
            "time": elapsed,
            "labels": None,
            "sigma":0.0,
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
        print(f"  Mean MSE: {mean_mse:.6f}")
        print(f"  Covariance MSE: {cov_mse:.6f}")
        print(f"  Integrated autocorrelation time: {tau:.2f}")
        print(f"  Time: {elapsed:.2f} seconds")

    transform = {"affine":affine, 'A':A, 'B':B} 
    
    return results, true_mean, cov_matrix, transform

results, true_mean, cov, transform = benchmark_samplers(dim = d, n_samples = 10000, burn_in = 1000, condition_number = 100, affine = af)

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


outdir = f"GaussianResults/{d}d{afstring}"
corner_dir = f"GaussianResults/{d}d{afstring}/corner"
trends_dir = f"GaussianResults/{d}d{afstring}/trends"
os.makedirs(outdir, exist_ok=True)
os.makedirs(corner_dir, exist_ok=True)
os.makedirs(trends_dir, exist_ok=True)

plt.title('StepSizeTuner Results')
plt.xlabel('Warm-up Step Number')
plt.ylabel('Epsilon value')
plt.ylim(0, 1.8)
plt.xlim(left = 0.0)
plt.legend()
plt.savefig(f'GaussianResults/{d}d{afstring}/StepSize.pdf')
plt.close()

overlay_gaussian = {'mu':true_mean, 'cov':cov} #encodes gaussian parameters for expected marginal overlays

benchmark_corner(results, corner_dir, thin = 10, overlay_gaussian=overlay_gaussian, transform=transform)
benchmark_trends(results,  trends_dir, 'Gaussian')
benchmark_autocorrelation(results, outdir, 'Gaussian')

report = SamplerReport(results=results, label = 'Gaussian', transform=transform, overlay=overlay_gaussian)
report.compile_pdf(texname = os.path.join(outdir, f'Gaussian_SamplerReport_{d}d{afstring}.tex'), template_dir='templates', latex_compiler='pdflatex')

print("Benchmark complete. Check the output directory for plots.")