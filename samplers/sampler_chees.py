"""
Standard HMC Sampler, refit using JAX for efficiency and to incorporate
ChEES criterion to tune integration length.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple


#Helper Functions
def make_batched_fns(log_prob_sca):
    """
    Converts a scalar log probability function into a batched, vectorized one for JAX compilation.

    Arguments:
        log_prob_sca : a function that takes one point of shape (dim,) and returns one scalar probability.

    Returns:
        log_prob: 
            function that can take in many points w/ shape (n_chains, dim) and returns a vector of scalar probabilities of shape (n_chains,).

        grad_log_prob: 
            function that can take in many points w/ shape (n_chains, dim) and returns the gradient of the scalar probability function for each dimension, of shape (n_chains, dim).

        grad_U: 
            -1 * grad_log_prob.

    """
    log_prob = jax.vmap(log_prob_sca)
    grad_log_prob = jax.vmap(jax.grad(log_prob_sca))

    def grad_U(x):
        return -grad_log_prob(x)
    
    return log_prob, grad_log_prob, grad_U

def leapfrog(q, p, grad_U, eps, L):
    """
    Helper Function to perform leapfrog integration that is JAX/JIT compatible

    Arguments:
        q:
            current position vector within the target distribution
        p: 
            associated conjugate momenta of the positions in the target distribution.
        grad_U:
            gradient of potential energy function required for advancing the momentum during the leapfrog integration
        eps:
            the 'time step' length
        L:
            the number of leapfrog integrations to perform
    """
    #advance p by half step
    p = p - 0.5 * eps * grad_U(q)

    #begin integration chain
    #defined for loop body to be compatible w/ JAX
    def fl_body(i, state):
        q, p = state #define current position and momenta

        #advance the position forward one step:
        q = q + eps * p

        #Momentum update, w/ JAX condition to skip final momentum update step:
        p = jax.lax.cond(i < L - 1, lambda p: p - eps * grad_U(q), lambda p:p, p)
        #JAX if statements are contained in these cond objects, and have the following syntax:
        #jax.lax.cond(condition, true_func, false_func, operand)
        #condition: condition to evaluate
        #true_func: if condition is true, evaluate true_func using operand
        #false_func: if condition is false, evaluate false_func using operand
        #operand: the variable/value to perform operations with

        #finally, return the state of the main loop body
        return (q, p)
    
    #run loop to update q and p from their initial states:
    q, p = jax.lax.fori_loop(0, L, fl_body, (q, p))

    #perform final momentum integration step:
    p = p - 0.5 * eps * grad_U(q)

    return q, p

def hmc_step(key, q, log_prob, grad_U, eps, L):
    """
    Helper function to run each individual HMC step. Vectorized for all chains.

    Arguments:
        key:
            PRNG key
        q: 
            position vector within the target distribution. Has shape (n_chains, dim)
        log_prob:
            batched log probability distribution
        grad_U:
            batched negative gradient of log_prob
        eps:
            epsilon, 'time step'/integration length parameter
        L:
            number of leapfrog steps
        
    """
    n_chains, dim = q.shape #grab dimensions and number of chains here

    #Split JAX randomization keys
    key_p, key_accept = jax.random.split(key)

    #Draw random momentum sample
    p0 = jax.random.normal(key_p, shape = q.shape)

    #Compute energies of current step
    U_current = -log_prob(q)
    K_current = 0.5 * jnp.sum(p0**2, axis = 1)

    #generate leapfrog step proposal
    q_prop, p_prop = leapfrog(q, p0, grad_U, eps=eps, L=L)

    #return velocity for ChEES
    vel = p_prop

    #flip momentum for reversibility/detailed balance
    p_prop = -p_prop

    #Compute new energies at the proposal position
    U_prop = -log_prob(q_prop)
    K_prop = 0.5 * jnp.sum(p_prop**2, axis = 1)

    #Generate Metropolis acceptance probability with some protection against discontinuities
    finite = (
        jnp.isfinite(U_current)
        & jnp.isfinite(K_current)
        & jnp.isfinite(U_prop)
        & jnp.isfinite(K_prop)
        & jnp.all(jnp.isfinite(q_prop), axis=1)
        & jnp.all(jnp.isfinite(p_prop), axis=1)
    )

    raw_log_accept = U_current + K_current - U_prop - K_prop
    log_accept_prob = jnp.where(finite, raw_log_accept, -jnp.inf)
    log_accept_prob = jnp.minimum(0.0, log_accept_prob)

    accept_prob = jnp.exp(log_accept_prob)
    accept_prob = jnp.where(jnp.isfinite(accept_prob), accept_prob, 0.0)

    #Draw some uniform random numbers for each walker
    log_uniform = jnp.log(jax.random.uniform(key_accept, shape = (n_chains,)))

    accept = log_uniform < log_accept_prob

    #select the new states, leave states unchanged if acceptance not met for the particular chain:
    q_new = jnp.where(accept[:, None], q_prop, q)

    return q_new, accept, accept_prob, log_accept_prob, q_prop, vel

class DAState(NamedTuple):
    """
    Class that contains the current dual averaging state and update functions.
    """

    iteration:int 
    log_eps:float
    log_eps_bar:float
    H_bar:float

def init_da_state(epsilon_init): #Initializes DA state
        log_eps0 = jnp.log(epsilon_init)
        return DAState(
            iteration=0,
            log_eps=log_eps0,
            log_eps_bar=log_eps0,
            H_bar = 0.0,
        )
    
def update_da( #update the dual averaging state
      state,
      accept_prob, #Should be scalar mean acceptance probability across walkers?
      log_eps0,
      target_accept = 0.651,
      gamma = 0.05,
      t0 = 10,
      kappa = 0.75,
      emin = 1e-4,
      emax = 0.1,

    ):
        it = state.iteration + 1
        accept_prob = jnp.clip(accept_prob, 0.0, 1.0)

        eta = 1 / (it + t0)

        H_bar = (1.0 - eta) * state.H_bar + eta * (target_accept - accept_prob)
        
        mu = log_eps0+jnp.log(10)
        
        log_eps = mu - (jnp.sqrt(it) / gamma) * H_bar
        log_eps = jnp.clip(log_eps, jnp.log(emin), jnp.log(emax))

        eta_bar = it **(-kappa)

        log_eps_bar = (1-eta_bar) * state.log_eps_bar + eta_bar * log_eps

        return DAState(
            iteration=it,
            log_eps=log_eps,
            log_eps_bar=log_eps_bar,
            H_bar=H_bar
        )
    
class CHEESState(NamedTuple):
     log_T: float
     log_T_bar: float
     m: float
     v: float
     iteration: int
     halton: float

@jax.jit
def halton(n, base = 2): #deterministic jitter function to shift the integration length around. n is the iteration index within the warmup sequence
     i = jnp.asarray(n, jnp.int32)
     b = jnp.asarray(base, jnp.int32)

     def cond(state): #condition for loop
          i, f, r = state
          return i > 0
     
     def loop_body(state): #main loop body, just a halton sequence
          i, f, r = state
          f = f / b
          r = r + f * jnp.mod(i, b)
          i = i // b
          return i, f, r
     
     _, _, r = jax.lax.while_loop(cond, loop_body, (i, 1.0, 0.0))
     return r

def init_chees(epsilon_init, L_init):
    T_init = epsilon_init * L_init
    return CHEESState(log_T=jnp.log(T_init), log_T_bar=jnp.log(T_init), m = 0.0, v = 0.0, iteration = 1, halton=halton(1))

def chees_L(state, epsilon, jitter = 0.6, use_bar = False, max_L = 5000, apply_jit=True):
     log_T = state.log_T_bar if use_bar else state.log_T
     T = jnp.exp(log_T) #grab current T
     if apply_jit:
        T_jit = (1-jitter) * T + jitter * state.halton * T #jitter the current T
     else:
        T_jit = T

     L = jnp.ceil(T_jit / epsilon)
     L = jnp.clip(L, 1, max_L)

     return L.astype(jnp.int32)

def update_chees(
    state,
    accept_prob,
    q_cur,
    q_prop,
    velocity,
    lr = 0.01,
    beta1 = 0.0,
    beta2 = 0.95,
    reg = 1e-7,
    T_min = 0.01,
    T_max = 0.2,
    T_interp = 0.9
):
     #Centering position vectors for ChEES criterion
    c_cur = q_cur - jnp.mean(q_cur, axis = 0)
    c_prop = q_prop - jnp.mean(q_prop, axis = 0)

     #Calculating difference of squares
    diff_sq = jnp.sum(c_prop**2, axis = 1) - jnp.sum(c_cur**2, axis = 1)
    inner = jnp.sum(c_prop * velocity, axis = 1)

     #calculate the g statistic for ChEES
    g_all = state.halton * jnp.exp(state.log_T) * diff_sq * inner

     #Prevent discontinuous behavior:
    valid = (accept_prob > 1e-4) & jnp.isfinite(g_all)
    g_all = jnp.where(valid, g_all, 0.0) #replace discontinuous values and essentially zero probabilites to zero.

    #calculate acceptance-weighted g statistic (add small regularization to prevent division by zero)
    g = jnp.sum(accept_prob * g_all) / (jnp.sum(accept_prob) + reg)
    g = jnp.where(jnp.isfinite(g), g, 0.0)

    it = state.iteration + 1

    m = beta1 * state.m + (1.0 - beta1) * g
    v = beta2 * state.v + (1.0 - beta2) * g * g

    m = jnp.where(jnp.isfinite(m), m, 0.0)
    v = jnp.where(jnp.isfinite(v), v, 0.0)

    m_hat = m if beta1 == 0.0 else m / (1.0 - beta1**it)
    v_hat = v / (1.0 - beta2**it)

    m_hat = jnp.where(jnp.isfinite(m_hat), m_hat, 0.0)
    v_hat = jnp.where(jnp.isfinite(v_hat), v_hat, 0.0)

    delta = lr * m_hat / jnp.sqrt(v_hat + reg)
    delta = jnp.where(jnp.isfinite(delta), delta, 0.0)
    delta = jnp.clip(delta, -0.05, 0.05)

    #Finally, at long last, update log T
    old_log_T = jnp.where(jnp.isfinite(state.log_T), state.log_T, jnp.log(T_min))
    old_log_T_bar = jnp.where(jnp.isfinite(state.log_T_bar), state.log_T_bar, jnp.log(T_min))

    log_T = old_log_T + delta
    log_T = jnp.clip(log_T, jnp.log(T_min), jnp.log(T_max))

    log_T_bar = jnp.logaddexp(
        jnp.log(T_interp) + old_log_T_bar,
        jnp.log(1.0 - T_interp) + log_T,
    )
    log_T_bar = jnp.where(jnp.isfinite(log_T_bar), log_T_bar, log_T)
    log_T_bar = jnp.clip(log_T_bar, jnp.log(T_min), jnp.log(T_max))    

    return CHEESState(
          log_T = log_T,
          log_T_bar = log_T_bar,
          m = m,
          v = v,
          iteration = it,
          halton = halton(it)
     )

def stretch_warmup_step(key, q0, log_prob, a = 2):
    """A single parallelized stretch-move step for a pre-warmup conditioning of the initial walker ensemble.
    
    Arguments:
    ----------
    key : 
        JAX randomness key
    q0:
        initial walker ensemble, has shape (n_chains, dim)
    log_prob:
        JAX-batched log-probability function of shape (n_chains,)
    
    """

    n_chains, dim = q0.shape #grab chains and dim from ensemble shape
    half = n_chains // 2 #split up ensemble into complementary ensembles

    if n_chains % 2 != 0:
        raise ValueError('The number of chains must be even for warm-up parallelization.')
    
    def update_half(key, q, active_idx, comp_idx):
        """Helper function to update one of the ensembles with the other ensemble.
        
        Arguments:
        ----------
        key :
            JAX randomness key
        q : 
            Ensemble vector
        active_idx : 
            Index array (within q) of the current walker to update
        comp_idx : 
            Index array (within q) of the complimentary walker for the update.

        Returns:
        ----------
        q :
            Updated ensemble vector.
        accept :
            Acceptance probability.
        """

        key_select, key_z, key_accept = jax.random.split(key, 3) #prep keys

        active = q[active_idx] #grab walker to update
        comp = q[comp_idx]

        # Select one complementary walker for each active walker.
        selected = jax.random.randint(
            key_select,
            shape=(half,),
            minval=0,
            maxval=half,
        )
        comp_selected = comp[selected]

        # Goodman-Weare stretch factor:
        # z in [1/a, a], density proportional to 1/sqrt(z)
        u = jax.random.uniform(key_z, shape=(half,))
        z = ((a - 1.0) * u + 1.0) ** 2 / a

        proposals = comp_selected + z[:, None] * (active - comp_selected)

        current_lp = log_prob(active)
        proposal_lp = log_prob(proposals)

        finite = (
            jnp.isfinite(current_lp)
            & jnp.isfinite(proposal_lp)
            & jnp.all(jnp.isfinite(proposals), axis=1)
        )

        log_accept = (dim - 1) * jnp.log(z) + proposal_lp - current_lp
        log_accept = jnp.where(finite, log_accept, -jnp.inf)

        log_u = jnp.log(jax.random.uniform(key_accept, shape=(half,)))
        accept = log_u < log_accept

        updated_active = jnp.where(accept[:, None], proposals, active)
        q = q.at[active_idx].set(updated_active)

        return q, accept

    idx1 = jnp.arange(0, half)
    idx2 = jnp.arange(half, n_chains)

    key1, key2 = jax.random.split(key)

    q, accept1 = update_half(key1, q0, idx1, idx2)
    q, accept2 = update_half(key2, q, idx2, idx1)

    accept = jnp.concatenate([accept1, accept2])

    return q, accept

def stretch_warmup(key, q0, log_prob, n_steps = 100, a = 2.0):
    """Perform the full 50 step stretch-move warmup algorithm
    
    Arguments:
    ----------
    key : 
        JAX randomness key
    q0:
        Initial Ensemble
    log_prob : 
        JAX-batched log probablity function
    n_steps : 
        Number of stretch-move steps or orchestrate during 
    
    Returns:
    ----------
    key : 
        JAX randomness key
    q_final : 
        Newly conditioned initial ensemble for extensive HMC warmup
    """

    def step(carry, _):
        "Run single step while managing key transfer"
        key, q = carry #grabbing carry-over from other steps
        key, subkey = jax.random.split(key)

        q_new, accept = stretch_warmup_step(key=subkey, q0 = q, log_prob=log_prob, a = a)

        return (key, q_new), jnp.mean(accept)
    
    (key, q_final), accept_history = jax.lax.scan(step, (key, q0), xs=None, length=n_steps)

    return(key, q_final, accept_history)
        
def hmc_warmup(
        key,
        q0,
        log_prob,
        grad_U, 
        eps0,
        L0,
        n_warmup,
        max_L = 5000,
        target_accept = 0.651,
        emin = 1e-3,
        emax = 0.1,
        chees_lr = 0.01,
        T_min = 0.01,
        T_max = 0.2
):
    """
    Function to perform the warmup step size and integration length tuning
    """

    #initialize dual averaging state:
    da = init_da_state(eps0)
    log_eps0 = jnp.log(eps0)

    #Initialize CHEES State:
    chees = init_chees(eps0, L0)

    #Function to produce each warmup step:
    def step(carry:tuple, _):
        key, q, da, chees = carry #carry is a tuple that stores all of the information carried between warm-up steps (i.e. information fed into dual averaging scheme at the start of each step)
        eps = jnp.exp(da.log_eps)
        current_L = chees_L(chees, eps, max_L = max_L)

        key, subkey = jax.random.split(key) #splitting key for randomness reproducibility

        #perform the hmc integration step
        q_new, accept, accept_prob, log_accept_prob, q_prop, vel = hmc_step(subkey, q, log_prob, grad_U, eps, current_L)
         
        #determine mean acceptance probability from this integration step:
        mean_accept = jnp.mean(accept_prob)
        mean_accept = jnp.where(jnp.isfinite(mean_accept), mean_accept, 0.0)

        #Update dual averaging state using the new positions and acceptance probability
        da_new = update_da(da, mean_accept, log_eps0, target_accept=target_accept, emin=emin, emax=emax)

        #update ChEES state
        q_prop_safe = jnp.where(jnp.isfinite(q_prop), q_prop, q)
        vel_safe = jnp.where(jnp.isfinite(vel), vel, 0.0)

        chees_new = update_chees(
            chees,
            accept_prob=accept_prob,
            q_cur=q,
            q_prop=q_prop_safe,
            velocity=vel_safe,
            lr=chees_lr,
            T_min=T_min,
            T_max=T_max,
        )

        #populate the new carry tuple and pass on the epsilon and acceptance rate (for history)
        new_carry = (key, q_new, da_new, chees_new)
        new_epshist = (eps, mean_accept, current_L)

        return new_carry, new_epshist
    
    #generate complete warmup integration smoothly with JAX:
    (key, q_final, da_final, chees_final), (eps_hist, accept_hist, L_hist) = jax.lax.scan(
         step, #perform this function
         (key, q0, da, chees), #starting with these initial conditions
         xs = None, #no need to reference an existing array
         length = n_warmup #do this stepping for this many steps
    )

    final_eps = jnp.exp(da_final.log_eps_bar) #store final epsilon value
    final_eps = jnp.clip(final_eps, emin, emax)
    final_T = jnp.exp(chees_final.log_T_bar)
    raw_final_L = jnp.ceil(final_T / final_eps)

    final_L = jnp.asarray(raw_final_L, dtype=jnp.int32)
    final_L = jnp.maximum(final_L, jnp.array(1, dtype=jnp.int32))
    final_L = jnp.minimum(final_L, jnp.array(max_L, dtype=jnp.int32))

    print("  final epsilon:", float(jax.device_get(final_eps)))
    print("  final log_T_bar:", float(jax.device_get(chees_final.log_T_bar)))
    print("  final T:", float(jax.device_get(jnp.exp(chees_final.log_T_bar))))
    print("  raw final L:", float(jax.device_get(raw_final_L)))
    print("  final L:", int(jax.device_get(final_L)))
    print("  warmup L min/max:", int(jax.device_get(jnp.min(L_hist))), int(jax.device_get(jnp.max(L_hist))))
    if bool(jax.device_get(jnp.any(L_hist >= max_L))):
        print("  Warning: ChEES hit max_L during warmup; raise emin, lower T_max, or add mass-matrix preconditioning.")
    
    return key, q_final, final_eps, final_L, eps_hist, accept_hist

def hmc_sample(
    key,
    q0,
    log_prob,
    grad_U,
    eps,
    L,
    n_samples,
    n_thin,
):
    def one_step(carry, _):
        key, q = carry
        key, subkey = jax.random.split(key)

        q_new, accept, *_ = hmc_step(
            subkey, q, log_prob, grad_U, eps, L
        )

        return (key, q_new), accept

    def one_saved_sample(carry, _):
        # Run n_thin HMC transitions, but only keep the final q
        (key, q), accepts_block = jax.lax.scan(
            one_step,
            carry,
            xs=None,
            length=n_thin,
        )

        # Store only one sample per block
        accept_block = jnp.mean(accepts_block, axis=0)

        return (key, q), (q, accept_block)

    (key, q_final), (samples, accepts) = jax.lax.scan(
        one_saved_sample,
        (key, q0),
        xs=None,
        length=n_samples,
    )

    return key, samples, accepts

def hmc_chees(
        log_prob,
        initial,
        n_samples,
        epsilon = 0.1,
        L = 10,
        n_chains = 2,
        n_thin = 1,
        n_warmup = 1000,
        target_accept = 0.651,
        max_L = 5000,
        seed = 0,
        emin = 1e-3,
        emax = 0.1,
        chees_lr = 0.01,
        T_min = 0.01,
        T_max = 0.2
):
    
    """
    Vectorized HMC w/ dual-averaging + trajectory length tuning warmup. Written in JAX
    """
    #generating initial conditions
    initial = jnp.asarray(initial, dtype=float)
    dim = int(initial.shape[-1])

    #batching probability density functions for JAX
    log_prob, grad_log_prob, grad_U = make_batched_fns(log_prob_sca=log_prob)

    #generate and split JAX key
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)

    if initial.ndim == 2:
        q0 = initial
        n_chains = int(initial.shape[0])
    else:
        q0 = jnp.tile(initial[None, :], (n_chains, 1))
        q0 = q0 + 0.1 * jax.random.normal(init_key, shape=(n_chains, dim))

    #run 50-step stretch move warmup before ChEES
    key, q0, stretch_accept_hist = stretch_warmup(
        key=key,
        q0=q0,
        log_prob=log_prob,
        n_steps=50,
        a=2.0,
    )

    print(f"Stretch-Move warmup complete, with acceptance {float(jax.device_get(jnp.round(jnp.mean(stretch_accept_hist), 3)))}")
    
    #run warmup loop
    key, q_warm, final_eps, final_L, eps_hist, accept_hist = hmc_warmup(
         key=key,
         q0=q0,
         log_prob=log_prob,
         grad_U=grad_U,
         eps0=epsilon,
         L0=L,
         n_warmup=n_warmup,
         max_L=max_L,
         target_accept=target_accept,
         emin=emin,
         emax=emax,
         chees_lr=chees_lr,
         T_min=T_min,
         T_max=T_max,
    )

    #use warmup loop parameters to run main sampling loop
    key, samples_jax, accepts = hmc_sample(
         key,
         q0=q_warm,
         log_prob = log_prob,
         grad_U = grad_U,
         eps = final_eps,
         L = final_L,
         n_samples=n_samples,
         n_thin=n_thin
    )

    #transpose JAX scan (n_samples, n_chains, dim) to standard ouput shape (n_chains, n_samples, dim)
    #also converting back to numpy for congruency w/ plotting and report scripts
    samples = np.asarray(samples_jax).transpose(1, 0, 2) #swap places of 0th and 1st axis.

    #convert outputs to numpy-compatible values
    accepts = np.asarray(jnp.mean(accepts, axis=0))
    final_eps = float(final_eps)
    eps_hist = np.asarray(eps_hist)

    parmslist = [int(final_L), n_warmup, target_accept, 0.05, 10, 0.75]


    return samples, accepts, final_eps, eps_hist, parmslist


#adding this so I can commit it again
