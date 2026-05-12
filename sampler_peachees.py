import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable, Tuple
from functools import partial


Array = jax.Array


def make_batched_fns(log_prob_sca: Callable[[Array], Array]):
    """Create batched log_prob, grad_log_prob, and grad_U=-grad log_prob."""
    log_prob = jax.vmap(log_prob_sca)
    grad_log_prob = jax.vmap(jax.grad(log_prob_sca))

    def grad_U(x: Array) -> Array:
        return -grad_log_prob(x)

    return log_prob, grad_log_prob, grad_U


class DAState(NamedTuple):
    iteration: Array
    log_eps: Array
    log_eps_bar: Array
    H_bar: Array


class CHEESState(NamedTuple):
    log_T: Array
    log_T_bar: Array
    m: Array
    v: Array
    iteration: Array
    halton: Array


@jax.jit
def halton(n: int, base: int = 2) -> Array:
    """JAX-compatible scalar Halton sequence value."""
    i = jnp.asarray(n, jnp.int32)
    b = jnp.asarray(base, jnp.int32)

    def cond(state):
        i, _, _ = state
        return i > 0

    def body(state):
        i, f, r = state
        f = f / b
        r = r + f * jnp.mod(i, b)
        i = i // b
        return i, f, r

    _, _, r = jax.lax.while_loop(cond, body, (i, 1.0, 0.0))
    return r


def init_da_state(epsilon_init: float) -> DAState:
    log_eps0 = jnp.log(epsilon_init)
    return DAState(iteration=jnp.array(0), log_eps=log_eps0, log_eps_bar=log_eps0, H_bar=0.0)


def update_da(
    state: DAState,
    accept_prob: Array,
    log_eps0: Array,
    target_accept: float = 0.651,
    gamma: float = 0.05,
    t0: float = 10.0,
    kappa: float = 0.75,
    emin: float = 1e-5,
    emax: float = 1.0,
) -> DAState:
    """Dual averaging update for the scalar step size."""
    it = state.iteration + 1
    accept_prob = jnp.clip(accept_prob, 0.0, 1.0)

    eta = 1.0 / (it + t0)
    H_bar = (1.0 - eta) * state.H_bar + eta * (target_accept - accept_prob)

    mu = log_eps0 + jnp.log(10.0)
    log_eps = mu - (jnp.sqrt(it) / gamma) * H_bar
    log_eps = jnp.clip(log_eps, jnp.log(emin), jnp.log(emax))

    eta_bar = it ** (-kappa)
    log_eps_bar = (1.0 - eta_bar) * state.log_eps_bar + eta_bar * log_eps

    return DAState(iteration=it, log_eps=log_eps, log_eps_bar=log_eps_bar, H_bar=H_bar)


def init_chees(epsilon_init: float, L_init: int) -> CHEESState:
    T_init = epsilon_init * L_init
    return CHEESState(
        log_T=jnp.log(T_init),
        log_T_bar=jnp.log(T_init),
        m=0.0,
        v=0.0,
        iteration=jnp.array(1),
        halton=halton(1),
    )


def chees_L(
    state: CHEESState,
    epsilon: Array,
    jitter: float = 0.6,
    use_bar: bool = False,
    max_L: int = 100,
    apply_jit: bool = True,
) -> Array:
    """Convert ChEES trajectory time T to an integer leapfrog count L."""
    log_T = state.log_T_bar if use_bar else state.log_T
    T = jnp.exp(log_T)
    T_eff = jnp.where(apply_jit, (1.0 - jitter) * T + jitter * state.halton * T, T)
    L = jnp.ceil(T_eff / epsilon)
    return jnp.clip(L, 1, max_L).astype(jnp.int32)


def update_chees(
    state: CHEESState,
    accept_prob: Array,
    q_cur: Array,
    q_prop: Array,
    velocity: Array,
    lr: float = 0.025,
    beta1: float = 0.0,
    beta2: float = 0.95,
    reg: float = 1e-7,
    T_min: float = 0.25,
    T_max: float = 10.0,
    T_interp: float = 0.9,
) -> CHEESState:
    """
    ChEES update for log trajectory time.

    velocity should be dx/dt at the end of the proposal. For the Hamiltonian
    walk move this is B @ p, implemented as p @ B because B is stored as
    rows with shape (half_walkers, dim).
    """
    c_cur = q_cur - jnp.mean(q_cur, axis=0)
    c_prop = q_prop - jnp.mean(q_prop, axis=0)

    diff_sq = jnp.sum(c_prop**2, axis=1) - jnp.sum(c_cur**2, axis=1)
    inner = jnp.sum(c_prop * velocity, axis=1)

    g_all = state.halton * jnp.exp(state.log_T) * diff_sq * inner
    valid = (accept_prob > 1e-4) & jnp.isfinite(g_all)
    g_all = jnp.where(valid, g_all, 0.0)

    g = jnp.sum(accept_prob * g_all) / (jnp.sum(accept_prob) + reg)

    it = state.iteration + 1
    g = jnp.where(jnp.isfinite(g), g, 0.0)

    m = beta1 * state.m + (1.0 - beta1) * g
    v = beta2 * state.v + (1.0 - beta2) * g * g

    m = jnp.where(jnp.isfinite(m), m, 0.0)
    v = jnp.where(jnp.isfinite(v), v, 0.0)

    m_hat = jnp.where(beta1 == 0.0, m, m / (1.0 - beta1**it))
    v_hat = v / (1.0 - beta2**it)

    m_hat = jnp.where(jnp.isfinite(m_hat), m_hat, 0.0)
    v_hat = jnp.where(jnp.isfinite(v_hat), v_hat, 0.0)

    delta = lr * m_hat / jnp.sqrt(v_hat + reg)
    delta = jnp.where(jnp.isfinite(delta), delta, 0.0)
    delta = jnp.clip(delta, -0.05, 0.05)

    old_log_T = jnp.where(jnp.isfinite(state.log_T), state.log_T, jnp.log(T_min))
    old_log_T_bar = jnp.where(jnp.isfinite(state.log_T_bar), state.log_T_bar, jnp.log(T_min))

    log_T = jnp.clip(old_log_T + delta, jnp.log(T_min), jnp.log(T_max))

    log_T_bar = jnp.logaddexp(
        jnp.log(T_interp) + old_log_T_bar,
        jnp.log(1.0 - T_interp) + log_T,
    )
    log_T_bar = jnp.where(jnp.isfinite(log_T_bar), log_T_bar, log_T)
    log_T_bar = jnp.clip(log_T_bar, jnp.log(T_min), jnp.log(T_max))
    return CHEESState(log_T=log_T, log_T_bar=log_T_bar, m=m, v=v, iteration=it, halton=halton(it))


def centered_B(complement: Array) -> Array:
    """
    Normalized centered ensemble B, stored as rows: shape (half_walkers, dim).
    The paper writes B as dim x half_walkers; using rows lets q update as p @ B.
    """
    half = complement.shape[0]
    return (complement - jnp.mean(complement, axis=0)) / jnp.sqrt(half)


def walk_leapfrog_group(q: Array, p: Array, B_rows: Array, grad_U, eps: Array, L: Array) -> Tuple[Array, Array, Array]:
    """
    Leapfrog for one active group under Hamiltonian walk dynamics:
        dq/dt = B p
        dp/dt = -B^T grad_U(q)
    with B stored as rows, so B p == p @ B_rows.
    """
    p = p - 0.5 * eps * (grad_U(q) @ B_rows.T)

    def body(i, state):
        q, p = state
        q = q + eps * (p @ B_rows)
        p = jax.lax.cond(
            i < L - 1,
            lambda pp: pp - eps * (grad_U(q) @ B_rows.T),
            lambda pp: pp,
            p,
        )
        return q, p

    q, p = jax.lax.fori_loop(0, L, body, (q, p))
    p = p - 0.5 * eps * (grad_U(q) @ B_rows.T)
    velocity = p @ B_rows
    return q, p, velocity


def update_first_half(key: Array, q: Array, log_prob, grad_U, eps: Array, L: Array):
    n_walkers, _ = q.shape
    half = n_walkers // 2
    q_active = q[:half]
    q_comp = q[half:]
    B = centered_B(q_comp)

    key_p, key_a = jax.random.split(key)
    p0 = jax.random.normal(key_p, shape=(half, half))

    U_cur = -log_prob(q_active)
    K_cur = 0.5 * jnp.sum(p0**2, axis=1)

    q_prop, p_prop, vel = walk_leapfrog_group(q_active, p0, B, grad_U, eps, L)
    p_flip = -p_prop

    U_prop = -log_prob(q_prop)
    K_prop = 0.5 * jnp.sum(p_flip**2, axis=1)
    finite = (
        jnp.isfinite(U_cur)
        & jnp.isfinite(K_cur)
        & jnp.isfinite(U_prop)
        & jnp.isfinite(K_prop)
        & jnp.all(jnp.isfinite(q_prop), axis=1)
        & jnp.all(jnp.isfinite(p_prop), axis=1)
    )

    raw_log_accept = U_cur + K_cur - U_prop - K_prop
    log_accept_prob = jnp.where(finite, raw_log_accept, -jnp.inf)
    log_accept_prob = jnp.minimum(0.0, log_accept_prob)

    accept_prob = jnp.exp(log_accept_prob)
    accept_prob = jnp.where(jnp.isfinite(accept_prob), accept_prob, 0.0)
    accept = jnp.log(jax.random.uniform(key_a, shape=(half,))) < log_accept_prob

    q_new_active = jnp.where(accept[:, None], q_prop, q_active)
    q_new = jnp.concatenate([q_new_active, q_comp], axis=0)

    return q_new, q_prop, vel, accept, accept_prob, log_accept_prob


def update_second_half(key: Array, q: Array, log_prob, grad_U, eps: Array, L: Array):
    n_walkers, _ = q.shape
    half = n_walkers // 2
    q_comp = q[:half]       # already-updated first half, held fixed
    q_active = q[half:]
    B = centered_B(q_comp)

    key_p, key_a = jax.random.split(key)
    p0 = jax.random.normal(key_p, shape=(half, half))

    U_cur = -log_prob(q_active)
    K_cur = 0.5 * jnp.sum(p0**2, axis=1)

    q_prop, p_prop, vel = walk_leapfrog_group(q_active, p0, B, grad_U, eps, L)
    p_flip = -p_prop

    U_prop = -log_prob(q_prop)
    K_prop = 0.5 * jnp.sum(p_flip**2, axis=1)
    finite = (
        jnp.isfinite(U_cur)
        & jnp.isfinite(K_cur)
        & jnp.isfinite(U_prop)
        & jnp.isfinite(K_prop)
        & jnp.all(jnp.isfinite(q_prop), axis=1)
        & jnp.all(jnp.isfinite(p_prop), axis=1)
    )

    raw_log_accept = U_cur + K_cur - U_prop - K_prop
    log_accept_prob = jnp.where(finite, raw_log_accept, -jnp.inf)
    log_accept_prob = jnp.minimum(0.0, log_accept_prob)

    accept_prob = jnp.exp(log_accept_prob)
    accept_prob = jnp.where(jnp.isfinite(accept_prob), accept_prob, 0.0)
    accept = jnp.log(jax.random.uniform(key_a, shape=(half,))) < log_accept_prob

    q_new_active = jnp.where(accept[:, None], q_prop, q_active)
    q_new = jnp.concatenate([q_comp, q_new_active], axis=0)

    return q_new, q_prop, vel, accept, accept_prob, log_accept_prob


@partial(jax.jit, static_argnames=("log_prob", "grad_U"))
def hamiltonian_walk_step(key: Array, q: Array, log_prob, grad_U, eps: Array, L: Array):
    """One full parallel Hamiltonian walk move: update first half, then second half."""
    key1, key2 = jax.random.split(key)
    q_mid, q_prop_1, vel_1, accept_1, accept_prob_1, log_alpha_1 = update_first_half(
        key1, q, log_prob, grad_U, eps, L
    )
    q_new, q_prop_2, vel_2, accept_2, accept_prob_2, log_alpha_2 = update_second_half(
        key2, q_mid, log_prob, grad_U, eps, L
    )

    q_prop = jnp.concatenate([q_prop_1, q_prop_2], axis=0)
    velocity = jnp.concatenate([vel_1, vel_2], axis=0)
    accept = jnp.concatenate([accept_1, accept_2], axis=0)
    accept_prob = jnp.concatenate([accept_prob_1, accept_prob_2], axis=0)
    log_alpha = jnp.concatenate([log_alpha_1, log_alpha_2], axis=0)

    return q_new, accept, accept_prob, log_alpha, q_prop, velocity


def walk_chees_warmup(
    key: Array,
    q0: Array,
    log_prob,
    grad_U,
    eps0: float,
    L0: int,
    n_warmup: int,
    max_L: int = 100,
    target_accept: float = 0.651,
    emin: float = 1e-5,
    emax: float = 1.0,
):
    """Warmup both epsilon by dual averaging and trajectory time by ChEES."""
    da = init_da_state(eps0)
    log_eps0 = jnp.log(eps0)
    chees = init_chees(eps0, L0)

    def step(carry, _):
        key, q, da, chees = carry
        eps = jnp.exp(da.log_eps)
        current_L = chees_L(chees, eps, max_L=max_L)

        key, subkey = jax.random.split(key)
        q_new, accept, accept_prob, _, q_prop, velocity = hamiltonian_walk_step(
            subkey, q, log_prob, grad_U, eps, current_L
        )

        mean_accept = jnp.mean(accept_prob)
        mean_accept = jnp.where(jnp.isfinite(mean_accept), mean_accept, 0.0)
        da_new = update_da(
            da, mean_accept, log_eps0,
            target_accept=target_accept, emin=emin, emax=emax,
        )
        chees_new = update_chees(
            chees, accept_prob=accept_prob, q_cur=q, q_prop=q_prop, velocity=velocity
        )
        return (key, q_new, da_new, chees_new), (eps, mean_accept, current_L)

    (key, q_final, da_final, chees_final), (eps_hist, accept_hist, L_hist) = jax.lax.scan(
        step, (key, q0, da, chees), xs=None, length=n_warmup
    )

    final_eps = jnp.exp(da_final.log_eps_bar)
    final_T = jnp.exp(chees_final.log_T_bar)
    raw_final_L = jnp.ceil(final_T / final_eps)
    final_L = raw_final_L.astype(jnp.int32)
    final_L = jnp.clip(final_L, 1, max_L)

    print("  final epsilon:", float(jax.device_get(final_eps)))
    print("  final log_T_bar:", float(jax.device_get(chees_final.log_T_bar)))
    print("  final T:", float(jax.device_get(jnp.exp(chees_final.log_T_bar))))
    print("  raw final L:", float(jax.device_get(raw_final_L)))
    print("  final L:", int(jax.device_get(final_L)))
    
    return key, q_final, final_eps, final_L, eps_hist, accept_hist, L_hist


def walk_sample(
    key: Array,
    q0: Array,
    log_prob,
    grad_U,
    eps: Array,
    L: Array,
    n_samples: int,
    n_thin: int,
):
    steps = n_samples * n_thin

    def step(carry, _):
        key, q, accepts_sum = carry
        key, subkey = jax.random.split(key)
        q_new, accept, _, _, _, _ = hamiltonian_walk_step(subkey, q, log_prob, grad_U, eps, L)
        accepts_sum = accepts_sum + accept.astype(jnp.float32)
        return (key, q_new, accepts_sum), q_new

    accepts0 = jnp.zeros(q0.shape[0], dtype=jnp.float32)
    (key, q_final, accepts_sum), all_samps = jax.lax.scan(
        step, (key, q0, accepts0), xs=None, length=steps
    )

    samples = all_samps[::n_thin]
    acceptance_rates = accepts_sum / steps
    return key, samples, acceptance_rates


def hamiltonian_walk_chees(
    log_prob: Callable[[Array], Array],
    initial,
    n_samples: int,
    epsilon: float = 0.1,
    L: int = 10,
    n_walkers: int = 20,
    n_thin: int = 1,
    n_warmup: int = 1000,
    target_accept: float = 0.651,
    max_L: int = 100,
    seed: int = 0,
):
    """
    Affine-invariant Hamiltonian walk move with JAX dual averaging + ChEES.

    Notes
    -----
    * n_walkers must be even. If odd, it is incremented by one.
    * For the full-rank Hamiltonian walk preconditioner, use roughly n_walkers >= 2*dim
      when possible, matching the usual complementary-ensemble requirement.
    """
    initial = jnp.asarray(initial, dtype=float)
    dim = int(initial.shape[0])

    if n_walkers < 4:
        n_walkers = 4
    if n_walkers % 2 != 0:
        n_walkers += 1

    log_prob_batched, _, grad_U = make_batched_fns(log_prob)

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    q0 = jnp.tile(initial[None, :], (n_walkers, 1))
    q0 = q0 + 0.1 * jax.random.normal(init_key, shape=(n_walkers, dim))

    key, q_warm, final_eps, final_L, eps_hist, accept_hist, L_hist = walk_chees_warmup(
        key=key,
        q0=q0,
        log_prob=log_prob_batched,
        grad_U=grad_U,
        eps0=epsilon,
        L0=L,
        n_warmup=n_warmup,
        max_L=max_L,
        target_accept=target_accept,
    )

    key, samples_jax, acceptance_rates_jax = walk_sample(
        key=key,
        q0=q_warm,
        log_prob=log_prob_batched,
        grad_U=grad_U,
        eps=final_eps,
        L=final_L,
        n_samples=n_samples,
        n_thin=n_thin,
    )

    samples = np.asarray(samples_jax).transpose(1, 0, 2)
    acceptance_rates = np.asarray(acceptance_rates_jax)
    final_eps_float = float(final_eps)
    final_L_int = int(final_L)

    parmslist = [final_L_int, n_warmup, target_accept, 0.05, 10, 0.75]

    print("  final epsilon:", final_eps_float)
    print("  n_leapfrog:", L)
    print("  eps_hist min/max:", np.nanmin(eps_hist), np.nanmax(eps_hist))
    print(accept_hist)
    return samples, acceptance_rates, final_eps_float, np.asarray(eps_hist), parmslist


