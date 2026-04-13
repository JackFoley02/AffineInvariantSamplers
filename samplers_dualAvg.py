import numpy as np

# ============================================================
# Dual averaging step-size tuner
# ============================================================

class StepSizeTuner:
    """
    Dual-averaging step size adaptation.
    """

    def __init__(
        self,
        epsilon_init,
        target_accept=0.65,
        gamma=0.05,
        t0=10.0,
        kappa=0.75,
        emin=1e-5,
        emax=0.4,
    ):
        epsilon_init = float(np.clip(epsilon_init, emin, emax))

        self.mu = np.log(10.0 * epsilon_init)
        self.target_accept = float(target_accept)
        self.gamma = float(gamma)
        self.t0 = float(t0)
        self.kappa = float(kappa)
        self.emin = float(emin)
        self.emax = float(emax)

        self.log_epsilon = np.log(epsilon_init)
        self.log_epsilon_bar = np.log(epsilon_init)
        self.H_bar = 0.0
        self.m = 0

    def update(self, accept_stat):
        """
        accept_stat should be a scalar in [0, 1].
        """
        accept_stat = float(accept_stat)
        if not np.isfinite(accept_stat):
            accept_stat = 0.0
        accept_stat = np.clip(accept_stat, 0.0, 1.0)

        self.m += 1
        eta = 1.0 / (self.m + self.t0)

        self.H_bar = (1.0 - eta) * self.H_bar + eta * (self.target_accept - accept_stat)

        self.log_epsilon = self.mu - (np.sqrt(self.m) / self.gamma) * self.H_bar
        self.log_epsilon = np.clip(self.log_epsilon, np.log(self.emin), np.log(self.emax))

        eta_bar = self.m ** (-self.kappa)
        self.log_epsilon_bar = (1.0 - eta_bar) * self.log_epsilon_bar + eta_bar * self.log_epsilon
        self.log_epsilon_bar = np.clip(self.log_epsilon_bar, np.log(self.emin), np.log(self.emax))

    @property
    def epsilon(self):
        return float(np.clip(np.exp(self.log_epsilon), self.emin, self.emax))

    @property
    def epsilon_bar(self):
        return float(np.clip(np.exp(self.log_epsilon_bar), self.emin, self.emax))


# ============================================================
# Shared utilities
# ============================================================
#These are a bunch of functions that ensure the output sampling is not infinite or discontinuous.

def _is_finite_rows(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return np.isfinite(x)
    return np.all(np.isfinite(x), axis=tuple(range(1, x.ndim)))

def _safe_grad_eval(grad_fn, x, reshape_prefix=None):
    """
    Evaluate gradient without hiding NaNs/Infs.
    Returns:
        grad_flat : (n, d) array
        finite_mask : (n,) bool array
    """
    try:
        if reshape_prefix is None:
            grad = grad_fn(x)
        else:
            grad = grad_fn(x.reshape(*reshape_prefix))
    except Exception:
        n = x.shape[0]
        d = x.shape[1]
        return np.full((n, d), np.nan), np.zeros(n, dtype=bool)

    grad = np.asarray(grad)
    grad = grad.reshape(x.shape[0], -1)
    finite_mask = _is_finite_rows(grad)
    return grad, finite_mask

def _safe_potential_eval(potential_fn, x, reshape_prefix):
    """
    Returns:
        U : (n,) array
        finite_mask : (n,) bool array
    """
    try:
        U = np.asarray(potential_fn(x.reshape(*reshape_prefix)))
    except Exception:
        U = np.full(x.shape[0], np.nan)

    finite_mask = np.isfinite(U)
    return U, finite_mask

def _safe_logprob_eval(log_prob, x):
    try:
        lp = np.asarray(log_prob(x))
    except Exception:
        lp = np.full(x.shape[0], -np.inf)

    finite_mask = np.isfinite(lp)
    return lp, finite_mask

def _kinetic_from_vector_momentum(p):
    """
    p shape: (n,)
    """
    K = 0.5 * p**2
    finite_mask = np.isfinite(K)
    K = np.where(finite_mask, K, np.inf)
    return K, finite_mask

def _kinetic_from_matrix_momentum(p):
    """
    p shape: (n, m)
    """
    sq = np.sum(p * p, axis=1)
    K = 0.5 * sq
    finite_mask = np.isfinite(K)
    K = np.where(finite_mask, K, np.inf)
    return K, finite_mask

def _accept_probs_from_energy(current_U, current_K, proposed_U, proposed_K, extra_finite_mask):
    """
    Returns:
        accept_probs : (n,)
        finite_mask  : (n,)
        dH           : (n,)
    """
    finite_mask = (
        np.isfinite(current_U)
        & np.isfinite(current_K)
        & np.isfinite(proposed_U)
        & np.isfinite(proposed_K)
        & extra_finite_mask
    )

    dH = np.full_like(current_U, np.inf, dtype=float)
    dH[finite_mask] = (proposed_U[finite_mask] + proposed_K[finite_mask]) - (
        current_U[finite_mask] + current_K[finite_mask]
    )

    accept_probs = np.zeros_like(current_U, dtype=float)

    neg = finite_mask & (dH <= 0.0)
    pos = finite_mask & (dH > 0.0)

    accept_probs[neg] = 1.0
    accept_probs[pos] = np.exp(-np.clip(dH[pos], 0.0, 100.0))

    return accept_probs, finite_mask, dH

def _draw_accepts(accept_probs):
    u = np.random.random(size=accept_probs.shape[0])
    return u < accept_probs


# ============================================================
# Standard HMC with dual averaging
# ============================================================

def hmc_sst(
    log_prob,
    initial,
    n_samples,
    grad_fn,
    epsilon=0.1,
    L=10,
    n_chains=1,
    n_thin=1,
    n_warmup=1000,
    target_accept=0.65,
    gamma=0.05,
    t0=10,
    kappa=0.75,
    divergence_threshold=1e3,
    emax = 0.4
):
    """
    Vectorized HMC with dual-averaging warmup.
    Returns:
        samples: (n_chains, n_samples, dim)
        acceptance_rates: (n_chains,)
        final_epsilon: scalar
        step_size_hist: (n_warmup,)
    """

    initial = np.asarray(initial, dtype=float)
    dim = len(initial)

    chains = np.tile(initial, (n_chains, 1)) + 0.1 * np.random.randn(n_chains, dim)
    chain_log_probs, lp_finite = _safe_logprob_eval(log_prob, chains)
    chain_log_probs = np.where(lp_finite, chain_log_probs, -np.inf)

    total_sampling_iterations = n_samples * n_thin
    total_iterations = n_warmup + total_sampling_iterations

    samples = np.zeros((n_chains, n_samples, dim), dtype=float)
    accepts_sampling = np.zeros(n_chains, dtype=float)
    sample_idx = 0

    sst = StepSizeTuner(
        epsilon_init=epsilon,
        target_accept=target_accept,
        gamma=gamma,
        t0=t0,
        kappa=kappa,
        emax=emax
    )
    step_size_hist = []
    final_epsilon = float(np.clip(epsilon, sst.emin, sst.emax))

    for i in range(total_iterations):
        is_warmup = i < n_warmup
        current_epsilon = sst.epsilon if is_warmup else final_epsilon

        p = np.random.normal(size=(n_chains, dim))
        current_p = p.copy()
        x = chains.copy()

        trajectory_ok = np.ones(n_chains, dtype=bool)

        # Initial half-step
        x_grad, grad_finite = _safe_grad_eval(grad_fn, x)
        trajectory_ok &= grad_finite

        # Only update valid rows
        valid = trajectory_ok.copy()
        p[valid] -= 0.5 * current_epsilon * x_grad[valid]

        # Leapfrog
        for j in range(L):
            valid = trajectory_ok.copy()
            x[valid] += current_epsilon * p[valid]

            row_finite = _is_finite_rows(x)
            row_bounded = np.max(np.abs(x), axis=1) <= divergence_threshold
            trajectory_ok &= row_finite & row_bounded

            if j < L - 1:
                valid = trajectory_ok.copy()
                if np.any(valid):
                    x_grad, grad_finite = _safe_grad_eval(grad_fn, x)
                    trajectory_ok &= grad_finite

                    valid2 = trajectory_ok.copy()
                    p[valid2] -= current_epsilon * x_grad[valid2]

        # Final half-step
        valid = trajectory_ok.copy()
        if np.any(valid):
            x_grad, grad_finite = _safe_grad_eval(grad_fn, x)
            trajectory_ok &= grad_finite

            valid2 = trajectory_ok.copy()
            p[valid2] -= 0.5 * current_epsilon * x_grad[valid2]

        p = -p

        proposal_log_probs, prop_lp_finite = _safe_logprob_eval(log_prob, x)

        current_K, currentK_finite = _kinetic_from_matrix_momentum(current_p)
        proposal_K, proposalK_finite = _kinetic_from_matrix_momentum(p)

        finite_mask = (
            trajectory_ok
            & prop_lp_finite
            & currentK_finite
            & proposalK_finite
            & _is_finite_rows(x)
            & _is_finite_rows(p)
            & np.isfinite(chain_log_probs)
        )

        log_accept_prob = np.full(n_chains, -np.inf, dtype=float)
        log_accept_prob[finite_mask] = np.minimum(
            0.0,
            proposal_log_probs[finite_mask]
            - chain_log_probs[finite_mask]
            - proposal_K[finite_mask]
            + current_K[finite_mask]
        )

        acceptance_prob = np.zeros(n_chains, dtype=float)
        acceptance_prob[finite_mask] = np.exp(log_accept_prob[finite_mask])

        log_u = np.log(np.random.uniform(size=n_chains))
        accept_mask = finite_mask & (log_u < log_accept_prob)

        chains[accept_mask] = x[accept_mask]
        chain_log_probs[accept_mask] = proposal_log_probs[accept_mask]

        if is_warmup:
            mean_acc = float(np.mean(acceptance_prob))
            if not np.isfinite(mean_acc):
                mean_acc = 0.0
            sst.update(mean_acc)
            step_size_hist.append(sst.epsilon)

            if i == n_warmup - 1:
                final_epsilon = sst.epsilon_bar
        else:
            accepts_sampling += accept_mask.astype(float)

        if (not is_warmup) and ((i - n_warmup) % n_thin == 0) and (sample_idx < n_samples):
            samples[:, sample_idx, :] = chains
            sample_idx += 1

    parmslist = [L, n_warmup, target_accept, gamma, t0, kappa]

    acceptance_rates = accepts_sampling / total_sampling_iterations
    return samples, acceptance_rates, final_epsilon, np.array(step_size_hist), parmslist


# ============================================================
# Hamiltonian side move with dual averaging
# ============================================================

def hamiltonian_side_move_sst(
    gradient_func,
    potential_func,
    initial,
    n_samples,
    n_chains_per_group=5,
    epsilon=0.01,
    n_leapfrog=10,
    beta=1.0,
    n_thin=1,
    n_warmup=1000,
    target_accept=0.65,
    gamma=0.05,
    t0=10,
    kappa=0.75,
    divergence_threshold=1e3,
):
    """
    Vectorized ensemble Hamiltonian side move with dual averaging.
    """

    initial = np.asarray(initial, dtype=float)
    orig_dim = initial.shape
    flat_dim = int(np.prod(orig_dim))
    total_chains = 2 * n_chains_per_group

    states = np.tile(initial.flatten(), (total_chains, 1)) + 0.1 * np.random.randn(total_chains, flat_dim)

    group1_indices = np.arange(n_chains_per_group)
    group2_indices = np.arange(n_chains_per_group, total_chains)

    total_sampling_iterations = n_samples * n_thin
    total_iterations = n_warmup + total_sampling_iterations

    samples = np.zeros((total_chains, n_samples, flat_dim), dtype=float)
    accepts_sampling = np.zeros(total_chains, dtype=float)
    sample_idx = 0

    sst = StepSizeTuner(
        epsilon_init=epsilon,
        target_accept=target_accept,
        gamma=gamma,
        t0=t0,
        kappa=kappa,
    )
    step_size_hist = []
    final_epsilon = float(np.clip(epsilon, sst.emin, sst.emax))

    reshape_prefix = (n_chains_per_group, *orig_dim)

    for i in range(total_iterations):
        is_warmup = i < n_warmup
        current_epsilon = sst.epsilon if is_warmup else final_epsilon

        beta_eps = beta * current_epsilon
        beta_eps_half = 0.5 * beta_eps

        # --------------------------------------------------
        # First group update
        # --------------------------------------------------
        r1 = np.random.choice(group2_indices, size=n_chains_per_group, replace=True)
        r2 = np.random.choice(group2_indices, size=n_chains_per_group, replace=True)
        for j in range(n_chains_per_group):
            while r2[j] == r1[j]:
                r2[j] = np.random.choice(group2_indices)

        selected1 = states[r1]
        selected2 = states[r2]

        # Paper-consistent normalization: sqrt(2d)
        diff_group2 = (selected1 - selected2) / np.sqrt(2.0 * flat_dim)

        current_q1 = states[group1_indices].copy()
        current_U1, currentU1_finite = _safe_potential_eval(potential_func, current_q1, reshape_prefix)

        p1 = np.random.randn(n_chains_per_group)
        current_K1, currentK1_finite = _kinetic_from_vector_momentum(p1)

        q1 = current_q1.copy()
        p1_current = p1.copy()
        traj1_ok = currentU1_finite & currentK1_finite & _is_finite_rows(diff_group2)

        grad1, grad1_finite = _safe_grad_eval(gradient_func, q1, reshape_prefix)
        traj1_ok &= grad1_finite

        valid = traj1_ok.copy()
        if np.any(valid):
            proj = np.sum(grad1[valid] * diff_group2[valid], axis=1)
            proj_finite = np.isfinite(proj)
            bad_rows = np.where(valid)[0][~proj_finite]
            traj1_ok[bad_rows] = False
            good_rows = np.where(valid)[0][proj_finite]
            p1_current[good_rows] -= beta_eps_half * proj[proj_finite]

        for _ in range(n_leapfrog):
            valid = traj1_ok.copy()
            if np.any(valid):
                q1[valid] += beta_eps * (p1_current[valid, None] * diff_group2[valid])

            row_finite = _is_finite_rows(q1)
            row_bounded = np.max(np.abs(q1), axis=1) <= divergence_threshold
            traj1_ok &= row_finite & row_bounded

            # middle momentum step except on last step handled by loop structure below
            grad1, grad1_finite = _safe_grad_eval(gradient_func, q1, reshape_prefix)
            traj1_ok &= grad1_finite

            valid = traj1_ok.copy()
            if np.any(valid):
                proj = np.sum(grad1[valid] * diff_group2[valid], axis=1)
                proj_finite = np.isfinite(proj)
                bad_rows = np.where(valid)[0][~proj_finite]
                traj1_ok[bad_rows] = False
                good_rows = np.where(valid)[0][proj_finite]
                p1_current[good_rows] -= beta_eps * proj[proj_finite]

        # remove one extra full step, add back half-step for exact leapfrog bookkeeping
        valid = traj1_ok.copy()
        if np.any(valid):
            grad1, grad1_finite = _safe_grad_eval(gradient_func, q1, reshape_prefix)
            traj1_ok &= grad1_finite

            valid2 = traj1_ok.copy()
            if np.any(valid2):
                proj = np.sum(grad1[valid2] * diff_group2[valid2], axis=1)
                proj_finite = np.isfinite(proj)
                bad_rows = np.where(valid2)[0][~proj_finite]
                traj1_ok[bad_rows] = False
                good_rows = np.where(valid2)[0][proj_finite]
                p1_current[good_rows] += 0.5 * beta_eps * proj[proj_finite]

        proposed_U1, proposedU1_finite = _safe_potential_eval(potential_func, q1, reshape_prefix)
        proposed_K1, proposedK1_finite = _kinetic_from_vector_momentum(p1_current)

        extra1 = traj1_ok & proposedU1_finite & proposedK1_finite & _is_finite_rows(q1) & np.isfinite(p1_current)
        accept_probs1, _, _ = _accept_probs_from_energy(current_U1, current_K1, proposed_U1, proposed_K1, extra1)
        accepts1 = _draw_accepts(accept_probs1)

        states[group1_indices[accepts1]] = q1[accepts1]

        # --------------------------------------------------
        # Second group update
        # --------------------------------------------------
        r1 = np.random.choice(group1_indices, size=n_chains_per_group, replace=True)
        r2 = np.random.choice(group1_indices, size=n_chains_per_group, replace=True)
        for j in range(n_chains_per_group):
            while r2[j] == r1[j]:
                r2[j] = np.random.choice(group1_indices)

        selected1 = states[r1]
        selected2 = states[r2]
        diff_group1 = (selected1 - selected2) / np.sqrt(2.0 * flat_dim)

        current_q2 = states[group2_indices].copy()
        current_U2, currentU2_finite = _safe_potential_eval(potential_func, current_q2, reshape_prefix)

        p2 = np.random.randn(n_chains_per_group)
        current_K2, currentK2_finite = _kinetic_from_vector_momentum(p2)

        q2 = current_q2.copy()
        p2_current = p2.copy()
        traj2_ok = currentU2_finite & currentK2_finite & _is_finite_rows(diff_group1)

        grad2, grad2_finite = _safe_grad_eval(gradient_func, q2, reshape_prefix)
        traj2_ok &= grad2_finite

        valid = traj2_ok.copy()
        if np.any(valid):
            proj = np.sum(grad2[valid] * diff_group1[valid], axis=1)
            proj_finite = np.isfinite(proj)
            bad_rows = np.where(valid)[0][~proj_finite]
            traj2_ok[bad_rows] = False
            good_rows = np.where(valid)[0][proj_finite]
            p2_current[good_rows] -= beta_eps_half * proj[proj_finite]

        for _ in range(n_leapfrog):
            valid = traj2_ok.copy()
            if np.any(valid):
                q2[valid] += beta_eps * (p2_current[valid, None] * diff_group1[valid])

            row_finite = _is_finite_rows(q2)
            row_bounded = np.max(np.abs(q2), axis=1) <= divergence_threshold
            traj2_ok &= row_finite & row_bounded

            grad2, grad2_finite = _safe_grad_eval(gradient_func, q2, reshape_prefix)
            traj2_ok &= grad2_finite

            valid = traj2_ok.copy()
            if np.any(valid):
                proj = np.sum(grad2[valid] * diff_group1[valid], axis=1)
                proj_finite = np.isfinite(proj)
                bad_rows = np.where(valid)[0][~proj_finite]
                traj2_ok[bad_rows] = False
                good_rows = np.where(valid)[0][proj_finite]
                p2_current[good_rows] -= beta_eps * proj[proj_finite]

        valid = traj2_ok.copy()
        if np.any(valid):
            grad2, grad2_finite = _safe_grad_eval(gradient_func, q2, reshape_prefix)
            traj2_ok &= grad2_finite

            valid2 = traj2_ok.copy()
            if np.any(valid2):
                proj = np.sum(grad2[valid2] * diff_group1[valid2], axis=1)
                proj_finite = np.isfinite(proj)
                bad_rows = np.where(valid2)[0][~proj_finite]
                traj2_ok[bad_rows] = False
                good_rows = np.where(valid2)[0][proj_finite]
                p2_current[good_rows] += 0.5 * beta_eps * proj[proj_finite]

        proposed_U2, proposedU2_finite = _safe_potential_eval(potential_func, q2, reshape_prefix)
        proposed_K2, proposedK2_finite = _kinetic_from_vector_momentum(p2_current)

        extra2 = traj2_ok & proposedU2_finite & proposedK2_finite & _is_finite_rows(q2) & np.isfinite(p2_current)
        accept_probs2, _, _ = _accept_probs_from_energy(current_U2, current_K2, proposed_U2, proposed_K2, extra2)
        accepts2 = _draw_accepts(accept_probs2)

        states[group2_indices[accepts2]] = q2[accepts2]

        mean_accept_prob = float((np.sum(accept_probs1) + np.sum(accept_probs2)) / total_chains)

        if is_warmup:
            if not np.isfinite(mean_accept_prob):
                mean_accept_prob = 0.0
            sst.update(mean_accept_prob)
            step_size_hist.append(sst.epsilon)

            if i == n_warmup - 1:
                final_epsilon = sst.epsilon_bar
        else:
            accepts_sampling[group1_indices] += accepts1.astype(float)
            accepts_sampling[group2_indices] += accepts2.astype(float)

            if ((i - n_warmup) % n_thin == 0) and (sample_idx < n_samples):
                samples[:, sample_idx] = states
                sample_idx += 1

    samples = samples.reshape((total_chains, n_samples) + orig_dim)
    acceptance_rates = accepts_sampling / total_sampling_iterations
    return samples, acceptance_rates, final_epsilon, np.array(step_size_hist)


# ============================================================
# Hamiltonian walk move with dual averaging
# ============================================================

def hamiltonian_walk_move_sst(
    gradient_func,
    potential_func,
    initial,
    n_samples,
    n_chains_per_group=5,
    epsilon=0.01,
    n_leapfrog=10,
    beta=0.05,
    n_thin=1,
    n_warmup=1000,
    target_accept=0.65,
    gamma=0.05,
    t0=10,
    kappa=0.75,
    divergence_threshold=1e3,
    emax = 0.4
):
    """
    Vectorized Hamiltonian walk move with dual averaging.
    """

    initial = np.asarray(initial, dtype=float)
    orig_dim = initial.shape
    flat_dim = int(np.prod(orig_dim))
    total_chains = 2 * n_chains_per_group

    states = np.tile(initial.flatten(), (total_chains, 1)) + 0.1 * np.random.randn(total_chains, flat_dim)

    group1 = np.arange(0, n_chains_per_group)
    group2 = np.arange(n_chains_per_group, total_chains)

    total_sampling_iterations = n_samples * n_thin
    total_iterations = n_warmup + total_sampling_iterations

    samples = np.zeros((total_chains, n_samples, flat_dim), dtype=float)
    accepts_sampling = np.zeros(total_chains, dtype=float)
    sample_idx = 0

    sst = StepSizeTuner(
        epsilon_init=epsilon,
        target_accept=target_accept,
        gamma=gamma,
        t0=t0,
        kappa=kappa,
        emax=emax
    )
    step_size_hist = []
    final_epsilon = float(np.clip(epsilon, sst.emin, sst.emax))

    reshape_prefix = (n_chains_per_group, *orig_dim)

    for i in range(total_iterations):
        is_warmup = i < n_warmup
        current_epsilon = sst.epsilon if is_warmup else final_epsilon

        beta_eps = beta * current_epsilon
        beta_eps_half = 0.5 * beta_eps

        # --------------------------------------------------
        # Group 1 update using centered group 2
        # --------------------------------------------------
        centered2 = (states[group2] - np.mean(states[group2], axis=0)) / np.sqrt(n_chains_per_group)

        current_q1 = states[group1].copy()
        current_U1, currentU1_finite = _safe_potential_eval(potential_func, current_q1, reshape_prefix)

        p1 = np.random.randn(n_chains_per_group, n_chains_per_group)
        current_K1, currentK1_finite = _kinetic_from_matrix_momentum(p1)

        q1 = current_q1.copy()
        p1_current = p1.copy()
        traj1_ok = currentU1_finite & currentK1_finite & _is_finite_rows(centered2)

        grad1, grad1_finite = _safe_grad_eval(gradient_func, q1, reshape_prefix)
        traj1_ok &= grad1_finite

        valid = traj1_ok.copy()
        if np.any(valid):
            proj = grad1[valid] @ centered2.T
            proj_finite = _is_finite_rows(proj)
            bad_rows = np.where(valid)[0][~proj_finite]
            traj1_ok[bad_rows] = False
            good_rows = np.where(valid)[0][proj_finite]
            p1_current[good_rows] -= beta_eps_half * proj[proj_finite]

        for _ in range(n_leapfrog):
            valid = traj1_ok.copy()
            if np.any(valid):
                q1[valid] += beta_eps * (p1_current[valid] @ centered2)

            row_finite = _is_finite_rows(q1)
            row_bounded = np.max(np.abs(q1), axis=1) <= divergence_threshold
            traj1_ok &= row_finite & row_bounded

            grad1, grad1_finite = _safe_grad_eval(gradient_func, q1, reshape_prefix)
            traj1_ok &= grad1_finite

            valid = traj1_ok.copy()
            if np.any(valid):
                proj = grad1[valid] @ centered2.T
                proj_finite = _is_finite_rows(proj)
                bad_rows = np.where(valid)[0][~proj_finite]
                traj1_ok[bad_rows] = False
                good_rows = np.where(valid)[0][proj_finite]
                p1_current[good_rows] -= beta_eps * proj[proj_finite]

        valid = traj1_ok.copy()
        if np.any(valid):
            grad1, grad1_finite = _safe_grad_eval(gradient_func, q1, reshape_prefix)
            traj1_ok &= grad1_finite

            valid2 = traj1_ok.copy()
            if np.any(valid2):
                proj = grad1[valid2] @ centered2.T
                proj_finite = _is_finite_rows(proj)
                bad_rows = np.where(valid2)[0][~proj_finite]
                traj1_ok[bad_rows] = False
                good_rows = np.where(valid2)[0][proj_finite]
                p1_current[good_rows] += 0.5 * beta_eps * proj[proj_finite]

        proposed_U1, proposedU1_finite = _safe_potential_eval(potential_func, q1, reshape_prefix)
        proposed_K1, proposedK1_finite = _kinetic_from_matrix_momentum(p1_current)

        extra1 = traj1_ok & proposedU1_finite & proposedK1_finite & _is_finite_rows(q1) & _is_finite_rows(p1_current)
        accept_probs1, _, _ = _accept_probs_from_energy(current_U1, current_K1, proposed_U1, proposed_K1, extra1)
        accepts1 = _draw_accepts(accept_probs1)

        states[group1[accepts1]] = q1[accepts1]

        # --------------------------------------------------
        # Group 2 update using centered group 1
        # --------------------------------------------------
        centered1 = (states[group1] - np.mean(states[group1], axis=0)) / np.sqrt(n_chains_per_group)

        current_q2 = states[group2].copy()
        current_U2, currentU2_finite = _safe_potential_eval(potential_func, current_q2, reshape_prefix)

        p2 = np.random.randn(n_chains_per_group, n_chains_per_group)
        current_K2, currentK2_finite = _kinetic_from_matrix_momentum(p2)

        q2 = current_q2.copy()
        p2_current = p2.copy()
        traj2_ok = currentU2_finite & currentK2_finite & _is_finite_rows(centered1)

        grad2, grad2_finite = _safe_grad_eval(gradient_func, q2, reshape_prefix)
        traj2_ok &= grad2_finite

        valid = traj2_ok.copy()
        if np.any(valid):
            proj = grad2[valid] @ centered1.T
            proj_finite = _is_finite_rows(proj)
            bad_rows = np.where(valid)[0][~proj_finite]
            traj2_ok[bad_rows] = False
            good_rows = np.where(valid)[0][proj_finite]
            p2_current[good_rows] -= beta_eps_half * proj[proj_finite]

        for _ in range(n_leapfrog):
            valid = traj2_ok.copy()
            if np.any(valid):
                q2[valid] += beta_eps * (p2_current[valid] @ centered1)

            row_finite = _is_finite_rows(q2)
            row_bounded = np.max(np.abs(q2), axis=1) <= divergence_threshold
            traj2_ok &= row_finite & row_bounded

            grad2, grad2_finite = _safe_grad_eval(gradient_func, q2, reshape_prefix)
            traj2_ok &= grad2_finite

            valid = traj2_ok.copy()
            if np.any(valid):
                proj = grad2[valid] @ centered1.T
                proj_finite = _is_finite_rows(proj)
                bad_rows = np.where(valid)[0][~proj_finite]
                traj2_ok[bad_rows] = False
                good_rows = np.where(valid)[0][proj_finite]
                p2_current[good_rows] -= beta_eps * proj[proj_finite]

        valid = traj2_ok.copy()
        if np.any(valid):
            grad2, grad2_finite = _safe_grad_eval(gradient_func, q2, reshape_prefix)
            traj2_ok &= grad2_finite

            valid2 = traj2_ok.copy()
            if np.any(valid2):
                proj = grad2[valid2] @ centered1.T
                proj_finite = _is_finite_rows(proj)
                bad_rows = np.where(valid2)[0][~proj_finite]
                traj2_ok[bad_rows] = False
                good_rows = np.where(valid2)[0][proj_finite]
                p2_current[good_rows] += 0.5 * beta_eps * proj[proj_finite]

        proposed_U2, proposedU2_finite = _safe_potential_eval(potential_func, q2, reshape_prefix)
        proposed_K2, proposedK2_finite = _kinetic_from_matrix_momentum(p2_current)

        extra2 = traj2_ok & proposedU2_finite & proposedK2_finite & _is_finite_rows(q2) & _is_finite_rows(p2_current)
        accept_probs2, _, _ = _accept_probs_from_energy(current_U2, current_K2, proposed_U2, proposed_K2, extra2)
        accepts2 = _draw_accepts(accept_probs2)

        states[group2[accepts2]] = q2[accepts2]

        mean_accept_prob = float((np.sum(accept_probs1) + np.sum(accept_probs2)) / total_chains)

        if is_warmup:
            if not np.isfinite(mean_accept_prob):
                mean_accept_prob = 0.0
            sst.update(mean_accept_prob)
            step_size_hist.append(sst.epsilon)

            if i == n_warmup - 1:
                final_epsilon = sst.epsilon_bar
        else:
            accepts_sampling[group1] += accepts1.astype(float)
            accepts_sampling[group2] += accepts2.astype(float)

            if ((i - n_warmup) % n_thin == 0) and (sample_idx < n_samples):
                samples[:, sample_idx] = states
                sample_idx += 1

    parmslist = [n_leapfrog, n_warmup, target_accept, gamma, t0, kappa]    

    samples = samples.reshape((total_chains, n_samples) + orig_dim)
    acceptance_rates = accepts_sampling / total_sampling_iterations
    return samples, acceptance_rates, final_epsilon, np.array(step_size_hist), parmslist