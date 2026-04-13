# AffineInvariantSamplers

Paper: https://arxiv.org/abs/2505.02987

This project endeavors to test the efficacy and performance of gradient-aware, affine-invariant ensemble samplers on various distributions.

Standard Markov Chain Monte Carlo (MCMC) samplers are an extremely valuable tool for effectively sampling complex distributions where direct evaluation is computational expensive or downright impossible. The current industry standards are affine-invariant ensemble samplers (such as the Stretch Move used in the emcee package), which use the information of many walkers in an ensemble to sample distributions quickly and independently. These sampling algorithms are affine-invariant, meaning they can handle highly anisotropic distributions very well, as long as said distribution could be affine-transformed to a more isotropic form. However, these samplers suffer from long autocorrelation times when applied to distributions with high curvature (which can't be affine-transformed into something without said curvature) and high dimensionality (where the typical set the sampler draws from becomes extremely narrow, shrinking step sizes and reducing acceptance probabilities). 

This may be remedied by using gradient-aware sampling algorithms, called Hamiltonian Monte Carlo (HMC). As the name suggests, these algorithms treat the typical set as an energy level set in Hamiltonian dynamics. By enforcing conservation of energy (the typical set), inducing a momentum, and integrating across the level set using Hamilton's equations, these algorithms use the local gradient of the distribution to direct walker trajectories along the level set, in analogy to real physical systems with conserved energy. These samplers generally perform better on curved or high-dimensional distributions, but require extensive preconditioning (the 'mass' of the walker) to do so. With poor or no preconditioning, HMC samplers cannot perform as well on highly anisotropic distributions.

As such, an algorithm that is both affine-invariant and gradient-aware would be extremely valuable, synergizing both benefits from affine-invariance (sampling highly anisotropic distributions) and gradient-awareness (sampling highly curved or high dimensional datasets). This experiment and accompanying code tests two such samplers, the Hamiltonian Walk Move (HWM) and Hamiltonian Side Move (HSM) algorithms.

# Usage

Standard MCMC samplers are all in `samplers.py.` Hamiltonian/gradient-aware samplers are in `samplers_dualAvg.py`, which includes the ability to iteratively select the optimal step size during the warm-up phase. 

So far, report generation is set up for the Rosenbrock and Gaussian distributions. In the AffineInvariantSamplers directory, experiments may be run in the CLI using the following:

```
python experiments_Rosenbrock_general.py
python experiments_Gaussian.py
```
Additional parser arguments can be passed to each of these experiments. The `--dim` argument specifies the number of dimensions of the chosen distribution (default: 2). Passing the `--af` flag will apply an arbitrary affine transformation to the target distribution, which is useful for testing affine-invariance.
```
python experiments_Rosenbrock_general.py --dim 10
python experiments_Gaussian.py --dim 20 --af
```

# Reports

Each experiment generates a LaTeX compiled report which contains all of the benchmark tests' information, including:
- Parameter means, standard deviations, and covariances
- Sampler initialization and run parameters
- Hamiltonian step size tuning (if applicable)

The reports also include the following figures:
- Corner plots for each sampler
- Trend plots for each sampler
- Autocorrelation comparsion across samplers
- Step Size tuning results.

Please email jackfoley1929@g.ucla.edu if there are any questions, concerns, or major bugs. 

```
@article{chen2025new,
  title={New affine invariant ensemble samplers and their dimensional scaling},
  author={Chen, Yifan},
  journal={arXiv preprint arXiv:2505.02987},
  year={2025}
}
```
