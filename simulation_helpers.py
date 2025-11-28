def _validate_age_structured_inputs(age_params, contact_matrix, age_pops, coverage):
    """Basic shape validation to catch common configuration mistakes early."""
    n_ages = len(age_pops)
    if len(age_params) != n_ages:
        raise ValueError(f"age_params length ({len(age_params)}) must match age_pops ({n_ages}).")
    if contact_matrix.shape != (n_ages, n_ages):
        raise ValueError(f"contact_matrix must be shape {(n_ages, n_ages)}, got {contact_matrix.shape}.")
    if isinstance(coverage, list) and len(coverage) != n_ages:
        raise ValueError(f"coverage length ({len(coverage)}) must match age_pops ({n_ages}).")


def _coerce_initial_vector(initial_conditions, key, n_ages, fallback):
    """Return an initial-condition vector of length n_ages."""
    values = initial_conditions.get(key, fallback)
    # ensure list copy to avoid side effects
    values = list(values)
    if len(values) < n_ages:
        values.extend([0] * (n_ages - len(values)))
    return values[:n_ages]


def hill_gate(occupancy, capacity, hill_coef):
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
    
    import numpy as np
    
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