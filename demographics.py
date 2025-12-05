"""
Demographic helper functions for hospital model simulations.

This module handles:
- Birth rate computations
- Background (non-disease) mortality
- Validation of demographic parameters
"""

import numpy as np
from typing import Dict, Optional, List, Union


def compute_birth_rate(
    total_live_pop: float,
    birth_rate: float,
    age_pops: Union[List[float], np.ndarray],
    birth_age_distribution: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute age-distributed birth inflow to susceptible compartments.
    
    Births are proportional to the total live population and distributed
    according to birth_age_distribution (defaults to newborns entering youngest age group).
    
    Parameters
    ----------
    total_live_pop : float
        Total current live population (sum across all compartments except D).
    birth_rate : float
        Per-capita birth rate (births per person per day).
        Typical value: ~0.00003 (≈12 births per 1000 per year).
    age_pops : array-like
        Initial population sizes by age group (used for scaling).
    birth_age_distribution : array-like, optional
        Fraction of births entering each age group. Default: [1, 0, 0, ...]
        (all births enter youngest age group).
    
    Returns
    -------
    np.ndarray
        Birth inflow rate per age group (persons per day).
    
    Notes
    -----
    Total births = birth_rate x total_live_pop
    Births per age group = total_births x birth_age_distribution
    
    The model assumes births enter the S (unvaccinated susceptible) compartment.
    For neonatal vaccination, use the neonatal_vaccination_rate parameter in
    simulate_master_hospital_model() to route a fraction of births to S_vax.
    """
    n_ages = len(age_pops)
    
    if birth_rate <= 0:
        return np.zeros(n_ages)
    
    total_births = birth_rate * total_live_pop
    
    if birth_age_distribution is None:
        # Default: all births enter youngest age group
        birth_age_distribution = np.zeros(n_ages)
        birth_age_distribution[0] = 1.0
    else:
        birth_age_distribution = np.array(birth_age_distribution)
        # Normalize to sum to 1
        if birth_age_distribution.sum() > 0:
            birth_age_distribution = birth_age_distribution / birth_age_distribution.sum()
    
    return total_births * birth_age_distribution


def compute_background_death_rate(
    compartment_pop: np.ndarray,
    mu_background: np.ndarray
) -> np.ndarray:
    """
    Compute background (non-disease) mortality outflow from a compartment.
    
    Parameters
    ----------
    compartment_pop : np.ndarray
        Current population in compartment by age group.
    mu_background : np.ndarray
        Age-specific background mortality rate (deaths per person per day).
        Examples of some typical values: [0.00001, 0.00005, 0.0003] for young, middle, elderly.
    
    Returns
    -------
    np.ndarray
        Background death outflow rate per age group (persons per day).
    
    Notes
    -----
    Background mortality applies to all living compartments:
    S, E, I, X_queued, X_admitted, H_ward, H_icu, R, and their vaccinated counterparts.
    
    This represents deaths from causes other than the modeled disease
    (accidents, other diseases, aging, etc.).
    
    Age-specific rates allow realistic modeling where elderly have higher
    background mortality than young individuals.
    """
    return mu_background * compartment_pop


def validate_demographic_params(
    demographic_params: Optional[Dict],
    n_ages: int
) -> Dict:
    """
    Validate demographic parameters for simulation.
    
    Parameters
    ----------
    demographic_params : dict or None
        Dictionary containing:
        - 'birth_rate': float, per-capita birth rate
        - 'mu_background': float or list, background mortality rate(s)
        - 'birth_age_distribution': list, optional
        - 'neonatal_vaccination_rate': float, optional
    n_ages : int
        Number of age groups.
    
    Returns
    -------
    dict
        Validated and normalized demographic parameters.
    
    Raises
    ------
    ValueError
        If parameters are invalid or inconsistent.
    """
    if demographic_params is None:
        # Return defaults (no demographics)
        return {
            'birth_rate': 0.0,
            'mu_background': np.zeros(n_ages),
            'birth_age_distribution': None,
            'neonatal_vaccination_rate': 0.0,
        }
    
    validated = {}
    
    # Birth rate
    birth_rate = demographic_params.get('birth_rate', 0.0)
    if birth_rate < 0:
        raise ValueError(f"birth_rate must be non-negative, got {birth_rate}")
    validated['birth_rate'] = birth_rate
    
    # Background mortality (age-specific)
    mu_background = demographic_params.get('mu_background', 0.0)
    if isinstance(mu_background, (int, float)):
        # Uniform rate across all ages
        validated['mu_background'] = np.full(n_ages, float(mu_background))
    else:
        mu_background = np.array(mu_background)
        if len(mu_background) != n_ages:
            raise ValueError(
                f"mu_background length ({len(mu_background)}) must match n_ages ({n_ages})"
            )
        if np.any(mu_background < 0):
            raise ValueError("mu_background values must be non-negative")
        validated['mu_background'] = mu_background
    
    # Birth age distribution
    birth_age_dist = demographic_params.get('birth_age_distribution', None)
    if birth_age_dist is not None:
        birth_age_dist = np.array(birth_age_dist)
        if len(birth_age_dist) != n_ages:
            raise ValueError(
                f"birth_age_distribution length ({len(birth_age_dist)}) must match n_ages ({n_ages})"
            )
        if np.any(birth_age_dist < 0):
            raise ValueError("birth_age_distribution values must be non-negative")
        # Normalize
        if birth_age_dist.sum() > 0:
            birth_age_dist = birth_age_dist / birth_age_dist.sum()
        validated['birth_age_distribution'] = birth_age_dist
    else:
        validated['birth_age_distribution'] = None
    
    # Neonatal vaccination rate
    neonatal_vax_rate = demographic_params.get('neonatal_vaccination_rate', 0.0)
    if not (0.0 <= neonatal_vax_rate <= 1.0):
        raise ValueError(
            f"neonatal_vaccination_rate must be between 0 and 1, got {neonatal_vax_rate}"
        )
    validated['neonatal_vaccination_rate'] = neonatal_vax_rate
    
    return validated
