"""
Time-varying extensions for our model.
This module provides helper functions for adding various time-varying transmission rates including:
- Seasonality
- Policy interventions (lockdowns/relaxations)
- Waning immunity
"""

import numpy as np

# ========================================
# Helper Functions for Time-Varying Parameters
# ========================================

def seasonal_forcing(t, beta_base, amplitude=0.3, period=365, peak_day=0):
    """
    Calculates seasonally-varying transmission rate.
    
    Parameters
    ----------
    t : float
        Current time in days.
    beta_base : float
        Baseline transmission rate.
    amplitude : float, optional
        Seasonal amplitude (0-1), default 0.3.
    period : float, optional
        Period in days, default 365.
    peak_day : float, optional
        Day when transmission peaks, default 0.
    
    Returns
    -------
    float
        Time-varying beta value.
    
    Notes
    -----
    beta(t) = beta_base * (1 + amplitude * cos(2*pi*(t - peak_day)/period))
    """
    if amplitude > 1.0:
        raise ValueError(f"amplitude={amplitude} > 1.0 can produce negative transmission rates")
    return beta_base * (1 + amplitude * np.cos(2 * np.pi * (t - peak_day) / period))


def policy_multiplier(t, interventions):
    """
    Calculate transmission multiplier based on active policy interventions.
    
    Parameters
    ----------
    t : float
        Current time in days.
    interventions : list of dict
        List of interventions, each with 'start_day', 'end_day', 'transmission_reduction'.
    
    Returns
    -------
    float
        Transmission multiplier (1.0 = no intervention, <1.0 = reduced transmission).
    
    Notes
    -----
    If multiple interventions overlap, the strongest reduction is applied.
    """
    if not interventions:
        return 1.0
    
    # find strongest active intervention
    max_reduction = 0.0
    for intervention in interventions:
        reduction = intervention['transmission_reduction']
        if not (0.0 <= reduction <= 1.0):
            raise ValueError(f"transmission_reduction={reduction} must be in [0, 1]")
        if intervention['start_day'] <= t <= intervention['end_day']:
            max_reduction = max(max_reduction, reduction)
    
    return 1.0 - max_reduction

## add waning immunity function here
