
import numpy as np

def autocorrelation_fft(x, max_lag=None):

    """
    Efficiently compute autocorrelation function using FFT.
    
    Parameters:
    -----------
    x : array
        1D array of samples
    max_lag : int, optional
        Maximum lag to compute (default: len(x)//3)
        
    Returns:
    --------
    acf : array
        Autocorrelation function values
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 3, 20000)  # Cap at 20000 to prevent slow computation

    if n == 0 or max_lag <= 0:
        return np.array([], dtype=float)
    
    # Remove mean and normalize
    x_norm = x - np.mean(x)
    var = np.var(x_norm)

    if (not np.isfinite(var)) or var <= 0.0:
        acf = np.full(min(max_lag, n), np.nan, dtype=float)
        if len(acf) > 0:
            acf[0] = 1.0
        return acf
    
    # Compute autocorrelation using FFT
    # Pad the signal with zeros to avoid circular correlation
    fft = np.fft.fft(x_norm, n=2*n)
    acf = np.fft.ifft(fft * np.conjugate(fft))[:n]
    acf = acf.real

    # Normalize by acf[0] rather than by n*var to ensure acf[0] == 1.
    # The biased normalization is intentional here; it is stable for IAT
    # windowing and avoids large noisy tail corrections.
    if acf[0] <= 0.0 or not np.isfinite(acf[0]):
        return np.full(min(max_lag, n), np.nan, dtype=float)
    acf = acf / acf[0]
    
    return acf[:max_lag]

def _auto_window(taus, c):
    """Return the first index satisfying Sokal's self-consistent window."""
    m = np.arange(len(taus)) < c * taus
    if np.any(~m):
        return int(np.argmin(m))
    return None

def integrated_autocorr_time(x, M=5, c=10):
    """
    Estimate the integrated autocorrelation time using a self-consistent window.
    Based on the algorithm described by Goodman and Weare.
    
    Parameters:
    -----------
    x : array
        1D array of samples
    M : int, default=5
        Deprecated compatibility argument. The estimate now uses the
        self-consistent window controlled by c.
    c : int, default=10
        Window criterion. A larger value is more conservative and requires
        more samples before reporting a finite estimate.
        
    Returns:
    --------
    tau : float
        Integrated autocorrelation time
    acf : array
        Autocorrelation function values
    ess : float
        Effective sample size
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < 2:
        acf = np.full(n, np.nan, dtype=float)
        return np.nan, acf, 0.0

    acf = autocorrelation_fft(x)

    if len(acf) < 2 or not np.all(np.isfinite(acf[:2])):
        return np.nan, acf, 0.0

    # tau[t] is the IAT estimate using lags 0..t:
    # tau = 1 + 2 * sum_{lag=1}^t rho_lag.
    taus = 2.0 * np.cumsum(acf) - 1.0

    # Negative or non-finite partial sums indicate that the noisy tail has
    # taken over; stop before trusting it.
    valid = np.isfinite(taus) & (taus > 0.0)
    if not np.any(valid):
        return np.nan, acf, 0.0

    last_valid = int(np.where(valid)[0][-1])
    window = _auto_window(taus[: last_valid + 1], c)

    if window is None or window <= 0:
        return np.nan, acf, 0.0

    tau = float(taus[window])

    # The estimate is not trustworthy unless the chain is many autocorrelation
    # times long. Returning NaN is better than manufacturing a capped value.
    if n < c * tau:
        return np.nan, acf, 0.0

    ess = n / tau
    
    return tau, acf, ess
