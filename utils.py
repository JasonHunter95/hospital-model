"""
Utility functions for the hospital model simulations.

This module handles:
- State vector packing/unpacking for ODE solvers
- Input validation for age-structured simulations
- Initial condition coercion
"""

import numpy as np
from typing import Dict, List, Any, Union


# ============================================================================
# STATE VECTOR PACKING/UNPACKING
# ============================================================================
# Order of compartments in the flattened state vector.
# For n_ages age groups, the state vector has 20 * n_ages elements.
# Layout: [S_0..S_{n-1}, E_0..E_{n-1}, ..., D_vax_0..D_vax_{n-1}]
#
# X compartment is split into X_queued (waiting for admission, untreated mortality)
# and X_admitted (secured ward spot, treated mortality, can recover or flow to H_ward)

STATE_ORDER = [
    'S', 'E', 'I', 'X_queued', 'X_admitted', 'H_ward', 'H_icu', 'R', 'D',
    'S_vax', 'E_vax', 'I_vax', 'X_queued_vax', 'X_admitted_vax', 'H_ward_vax', 'H_icu_vax', 'R_vax', 'D_vax'
]
NUM_COMPARTMENTS = len(STATE_ORDER)  # 18

# Additional tracked variables (differential mortality, demographics) - these are integrated
# separately as part of the state vector for accumulation
TRACKED_ORDER = ['D_treated', 'D_untreated', 'D_vax_treated', 'D_vax_untreated', 
                 'cum_breakthrough', 'cum_births', 'cum_background_deaths']
NUM_TRACKED = len(TRACKED_ORDER)  # 7


def pack_state(compartments: Dict[str, np.ndarray], n_ages: int) -> np.ndarray:
    """
    Pack compartment dictionaries into a flat 1D numpy array for ODE solvers.
    
    Parameters
    ----------
    compartments : dict
        Dictionary mapping compartment names to lists of values per age group.
        Keys should match STATE_ORDER and TRACKED_ORDER.
    n_ages : int
        Number of age groups.
    
    Returns
    -------
    np.ndarray
        Flattened state vector of shape ((NUM_COMPARTMENTS + NUM_TRACKED) * n_ages,)
    
    Notes
    -----
    State vector layout:
    [S_0, S_1, ..., S_{n-1}, E_0, E_1, ..., D_vax_{n-1}, 
     D_treated_0, ..., cum_breakthrough_{n-1}]
    """
    y = np.zeros((NUM_COMPARTMENTS + NUM_TRACKED) * n_ages)
    
    # Pack main compartments
    for i, name in enumerate(STATE_ORDER):
        start = i * n_ages
        y[start:start + n_ages] = compartments[name]
    
    # Pack tracked variables
    base = NUM_COMPARTMENTS * n_ages
    for i, name in enumerate(TRACKED_ORDER):
        start = base + i * n_ages
        y[start:start + n_ages] = compartments[name]
    
    return y


def unpack_state(y: np.ndarray, n_ages: int) -> Dict[str, np.ndarray]:
    """
    Unpack flat state vector into compartment dictionary.
    
    Parameters
    ----------
    y : np.ndarray
        Flattened state vector of shape ((NUM_COMPARTMENTS + NUM_TRACKED) * n_ages,)
    n_ages : int
        Number of age groups.
    
    Returns
    -------
    dict
        Dictionary mapping compartment names to numpy arrays of values per age group.
    """
    compartments = {}
    
    # Unpack main compartments
    for i, name in enumerate(STATE_ORDER):
        start = i * n_ages
        compartments[name] = y[start:start + n_ages]
    
    # Unpack tracked variables
    base = NUM_COMPARTMENTS * n_ages
    for i, name in enumerate(TRACKED_ORDER):
        start = base + i * n_ages
        compartments[name] = y[start:start + n_ages]
    
    return compartments


def validate_age_structured_inputs(
    age_params: List[Dict],
    contact_matrix: np.ndarray,
    age_pops: List[float],
    coverage: Union[float, List[float]]
) -> None:
    """
    Basic shape validation to catch common configuration mistakes early.
    
    Parameters
    ----------
    age_params : list of dict
        Age-specific disease parameters.
    contact_matrix : np.ndarray
        Contact matrix between age groups.
    age_pops : list
        Population sizes by age group.
    coverage : float or list
        Vaccination coverage (scalar or per-age-group).
    
    Raises
    ------
    ValueError
        If shapes are inconsistent.
    """
    n_ages = len(age_pops)
    if len(age_params) != n_ages:
        raise ValueError(f"age_params length ({len(age_params)}) must match age_pops ({n_ages}).")
    if contact_matrix.shape != (n_ages, n_ages):
        raise ValueError(f"contact_matrix must be shape {(n_ages, n_ages)}, got {contact_matrix.shape}.")
    if isinstance(coverage, list) and len(coverage) != n_ages:
        raise ValueError(f"coverage length ({len(coverage)}) must match age_pops ({n_ages}).")


def coerce_initial_vector(
    initial_conditions: Dict[str, Any],
    key: str,
    n_ages: int,
    fallback: List[float]
) -> List[float]:
    """
    Return an initial-condition vector of length n_ages.
    
    Parameters
    ----------
    initial_conditions : dict
        Dictionary of initial condition arrays.
    key : str
        Key to look up in initial_conditions.
    n_ages : int
        Required length of the output vector.
    fallback : list
        Default values if key is not found.
    
    Returns
    -------
    list
        Initial condition vector of length n_ages.
    """
    values = initial_conditions.get(key, fallback)
    # ensure list copy to avoid side effects
    values = list(values)
    if len(values) < n_ages:
        values.extend([0] * (n_ages - len(values)))
    return values[:n_ages]
