"""
Hospital capacity functions for admission gating.

This module handles:
- Hill function gating for capacity-constrained admissions
- Both scalar and vectorized implementations for numerical stability
"""

import numpy as np


def hill_gate(occupancy: float, capacity: float, hill_coef: float) -> float:
    """
    Calculate Hill function gating factor for capacity-constrained admissions.
    
    Parameters
    ----------
    occupancy : float
        Current occupancy level. Must be non-negative.
    capacity : float
        Maximum capacity. Must be positive.
    hill_coef : float
        Hill coefficient controlling steepness of gating. Must be non-negative.
    
    Returns
    -------
    float
        Gating factor between 0 and 1.
    
    Raises
    ------
    ValueError
        If occupancy is negative, capacity is non-positive, or hill_coef is negative.
    
    Notes
    -----
    g(H) = 1 / (1 + (H/K)^n)
    - When H << K: g ≈ 1 (admissions unrestricted)
    - When H = K: g = 0.5 (admissions halved)
    - When H >> K: g → 0 (admissions severely limited)
    
    Uses log-domain computation for large hill_coef values to prevent overflow.

    Mathematical Mapping
    --------------------
    - occupancy -> H
    - capacity -> K
    - hill_coef -> n
    - Formula: g(H) = 1 / (1 + (H/K)^n)
    """
    # Input validation
    if occupancy < 0:
        raise ValueError(f"occupancy must be non-negative, got {occupancy}")
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    if hill_coef < 0:
        raise ValueError(f"hill_coef must be non-negative, got {hill_coef}")
    
    # Edge cases
    if occupancy == 0:
        return 1.0
    if hill_coef == 0:
        return 0.5
    
    ratio = occupancy / capacity
    
    # Use log-domain for numerical stability with large hill_coef
    if ratio >= 1.0:
        # For ratio >= 1, result is <= 0.5
        log_ratio = np.log(ratio)
        exponent = hill_coef * log_ratio
        if exponent > 700:  # exp(700) ≈ 1e304, near float max
            return 0.0  # Overwhelmed capacity
        power = np.exp(exponent)
    else:
        # For ratio < 1, (ratio)^n approaches 0, safe to compute directly
        power = ratio ** hill_coef
    
    return 1.0 / (1.0 + power)


def hill_gate_vectorized(occupancy: float, capacity: float, hill_coef: float) -> float:
    """
    Vectorized Hill function gating with numerical stability.
    
    This is an internal function called from within the ODE solver.
    Input validation is performed in the outer simulate function.
    Uses log-domain computation for large hill_coef to prevent overflow.
    
    Parameters
    ----------
    occupancy : float
        Current total occupancy. Assumed non-negative.
    capacity : float
        Maximum capacity.
    hill_coef : float
        Hill coefficient. Assumed non-negative.
    
    Returns
    -------
    float
        Gating factor between 0 and 1.
    """
    # Edge cases
    if capacity <= 0:
        return 0.0
    if occupancy <= 0:
        return 1.0
    if hill_coef == 0:
        return 0.5
    
    ratio = occupancy / capacity
    
    # Use log-domain for numerical stability with large hill_coef
    if ratio >= 1.0:
        # For ratio >= 1, result is <= 0.5
        log_ratio = np.log(ratio)
        exponent = hill_coef * log_ratio
        if exponent > 700:  # exp(700) ≈ 1e304, near float max
            return 0.0  # Overwhelmed capacity
        power = np.exp(exponent)
    else:
        # For ratio < 1, (ratio)^n approaches 0, safe to compute directly
        power = ratio ** hill_coef
    
    return 1.0 / (1.0 + power)
