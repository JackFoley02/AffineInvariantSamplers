"""
NumPyro NUTS sampler wrapper for the benchmark scripts.

This keeps the same return convention as the HMC/ChEES samplers:
    samples, acceptance_rates, final_epsilon, epsilon_history, parmslist

Unlike sampler_chees_tuned, this runs NUTS end to end. NumPyro adapts the
step size and dense mass matrix during warmup, while NUTS dynamically chooses
the trajectory length for each post-warmup transition.
"""

import numpy as np
import jax
import jax.numpy as jnp
from inspect import signature
from numpyro.infer import MCMC, NUTS


def prepare_initial_ensemble(initial, n_chains, key, noise_scale=0.01):
    """
    Accept one initial point with shape (dim,) or a full ensemble with shape
    (n_chains, dim). This mirrors the benchmark-facing behavior of the ChEES
    samplers so the same experiment scripts can call either implementation.
    """
    initial = jnp.asarray(initial, dtype=float)

    if initial.ndim == 1:
        dim = int(initial.shape[0])
        if n_chains < 1:
            n_chains = 1
        q0 = jnp.tile(initial[None, :], (n_chains, 1))
        if n_chains > 1 and noise_scale > 0.0:
            q0 = q0 + noise_scale * jax.random.normal(key, shape=(n_chains, dim))
        return q0, int(n_chains), dim

    if initial.ndim == 2:
        n_chains_from_initial, dim = initial.shape
        if n_chains_from_initial < 1:
            raise ValueError("Initial ensemble must contain at least one chain.")
        return initial, int(n_chains_from_initial), int(dim)

    raise ValueError("initial must have shape (dim,) or (n_chains, dim).")


def _as_chain_array(x, n_chains):
    """
    NumPyro sometimes returns scalar diagnostics as shape (n_samples,) for a
    single chain and (n_chains, n_samples) for multiple chains.
    """
    x = np.asarray(x)
    if n_chains == 1 and x.ndim == 1:
        return x[None, :]
    return x


def _get_extra_fields(mcmc, n_chains):
    try:
        fields = mcmc.get_extra_fields(group_by_chain=True)
    except TypeError:
        fields = mcmc.get_extra_fields()

    return {name: _as_chain_array(value, n_chains) for name, value in fields.items()}


def _supported_kwargs(callable_obj, kwargs):
    supported = signature(callable_obj).parameters
    return {name: value for name, value in kwargs.items() if name in supported}


def hmc_nuts(
    log_prob,
    initial,
    n_samples,
    epsilon=0.1,
    L=None,
    n_chains=1,
    n_thin=1,
    n_warmup=1000,
    target_accept=0.8,
    max_L=None,
    max_tree_depth=10,
    dense_mass=True,
    seed=0,
    init_noise=0.01,
    progress_bar=True,
    chain_method="vectorized",
):
    """
    Run NUTS with dense mass-matrix adaptation.

    Parameters kept for compatibility:
        L, max_L:
            Accepted but not used. NUTS has no fixed leapfrog count; use
            max_tree_depth to cap trajectory expansion instead.

    Returns:
        samples:
            Array with shape (n_chains, n_samples, dim).
        acceptance_rates:
            Mean post-warmup NUTS acceptance probability per chain.
        final_epsilon:
            Adapted final step size.
        epsilon_history:
            Post-warmup step-size values when available, otherwise final step
            size repeated once.
        parmslist:
            Drop-in parameter tuple. The first value is mean post-warmup tree
            depth, used as a trajectory-cost proxy in place of fixed L.
    """
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    init_params, n_chains, _ = prepare_initial_ensemble(
        initial,
        n_chains=n_chains,
        key=init_key,
        noise_scale=init_noise,
    )

    def potential_fn(q):
        return -log_prob(q)

    nuts_kwargs = _supported_kwargs(
        NUTS,
        {
            "potential_fn": potential_fn,
            "step_size": epsilon,
            "adapt_step_size": True,
            "adapt_mass_matrix": True,
            "dense_mass": dense_mass,
            "target_accept_prob": target_accept,
            "max_tree_depth": max_tree_depth,
        },
    )
    kernel = NUTS(**nuts_kwargs)

    mcmc_kwargs = _supported_kwargs(
        MCMC,
        {
            "num_warmup": n_warmup,
            "num_samples": n_samples,
            "num_chains": n_chains,
            "thinning": n_thin,
            "chain_method": chain_method,
            "progress_bar": progress_bar,
        },
    )
    mcmc = MCMC(kernel, **mcmc_kwargs)

    extra_fields = ("accept_prob", "num_steps", "diverging", "adapt_state.step_size")
    try:
        mcmc.run(key, init_params=init_params, extra_fields=extra_fields)
    except (AttributeError, TypeError, ValueError):
        fallback_fields = ("accept_prob", "num_steps", "diverging")
        mcmc.run(key, init_params=init_params, extra_fields=fallback_fields)

    samples = np.asarray(mcmc.get_samples(group_by_chain=True))
    fields = _get_extra_fields(mcmc, n_chains)

    accept_prob = fields.get("accept_prob")
    if accept_prob is None:
        acceptance_rates = np.full(n_chains, np.nan)
    else:
        acceptance_rates = np.mean(accept_prob, axis=1)

    step_size_hist = fields.get("adapt_state.step_size")
    if step_size_hist is None:
        final_eps = float(jax.device_get(mcmc.last_state.adapt_state.step_size))
        eps_hist = np.asarray([final_eps])
    else:
        eps_hist = np.asarray(step_size_hist)
        final_eps = float(np.ravel(eps_hist)[-1])

    num_steps = fields.get("num_steps")
    if num_steps is None:
        mean_tree_depth = None
        mean_num_steps = None
    else:
        mean_num_steps = float(np.mean(num_steps))
        mean_tree_depth = int(np.round(np.mean(np.log2(np.maximum(num_steps, 1)))))

    divergent = fields.get("diverging")
    n_divergent = int(np.sum(divergent)) if divergent is not None else 0

    print("  final epsilon:", final_eps)
    if mean_num_steps is not None:
        print("  mean NUTS leapfrog steps:", round(mean_num_steps, 3))
        print("  mean NUTS tree depth:", mean_tree_depth)
    print("  divergences:", n_divergent)

    parmslist = [mean_tree_depth, n_warmup, target_accept, 0.05, 10, 0.75]

    return samples, acceptance_rates, final_eps, eps_hist, parmslist


# Alias for drop-in experiments that expect the old HMC function name.
nuts = hmc_nuts
