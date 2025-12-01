"""
Spectral preprocessing techniques

Implements individual preprocessing operations:
- S: Savitzky-Golay smoothing
- A: airPLS baseline correction
- 0: Remove negative values
- M: Min-Max normalization
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve


def savgol_smoothing(data, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay smoothing filter.
    
    Parameters:
    -----------
    data : np.ndarray
        Input spectral data (samples x wavelengths)
    window_length : int
        Length of the smoothing window (must be odd)
    polyorder : int
        Order of polynomial to fit
    
    Returns:
    --------
    np.ndarray
        Smoothed spectral data
    """
    if window_length % 2 == 0:
        window_length += 1
    
    smoothed = np.zeros_like(data)
    for i in range(data.shape[0]):
        smoothed[i] = savgol_filter(data[i], window_length, polyorder)
    
    return smoothed


def airpls_baseline(data, lam=100, porder=1, itermax=15):
    """
    Asymmetric least squares smoothing (airPLS) for baseline correction.
    
    Parameters:
    -----------
    data : np.ndarray
        Input spectral data (samples x wavelengths)
    lam : float
        Smoothing parameter (lambda)
    porder : int
        Order of polynomial
    itermax : int
        Maximum iterations
    
    Returns:
    --------
    np.ndarray
        Baseline-corrected spectral data
    """
    corrected = np.zeros_like(data, dtype=float)
    
    for i in range(data.shape[0]):
        y = data[i].astype(float)
        m = y.shape[0]
        
        # Create difference matrix
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(m - 2, m))
        w = np.ones(m)
        
        for iteration in range(itermax):
            W = sparse.diags(w)
            Z = W + lam * D.T @ D
            baseline = spsolve(Z, w * y)
            
            # Update weights
            residual = y - baseline
            w = np.where(residual > 0, np.exp(-residual / (2 * np.std(residual[residual < 0]))), 1)
        
        corrected[i] = y - baseline
    
    return corrected


def remove_negatives(data, method='zero'):
    """
    Remove or handle negative values in spectral data.
    
    Parameters:
    -----------
    data : np.ndarray
        Input spectral data
    method : str
        Method to handle negatives: 'zero', 'shift', 'abs'
    
    Returns:
    --------
    np.ndarray
        Data with negatives removed/handled
    """
    result = data.copy()
    
    if method == 'zero':
        result[result < 0] = 0
    elif method == 'shift':
        min_val = result.min(axis=1, keepdims=True)
        result = np.where(min_val < 0, result - min_val, result)
    elif method == 'abs':
        result = np.abs(result)
    
    return result


def minmax_normalize(data, feature_range=(0, 1)):
    """
    Min-Max normalization to scale data to [0, 1].
    
    Parameters:
    -----------
    data : np.ndarray
        Input spectral data (samples x wavelengths)
    feature_range : tuple
        Target range (min, max)
    
    Returns:
    --------
    np.ndarray
        Normalized spectral data
    """
    min_val = data.min(axis=1, keepdims=True)
    max_val = data.max(axis=1, keepdims=True)
    
    # Avoid division by zero
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    
    # Normalize to [0, 1]
    normalized = (data - min_val) / range_val
    
    # Scale to target range
    min_range, max_range = feature_range
    normalized = normalized * (max_range - min_range) + min_range
    
    return normalized
