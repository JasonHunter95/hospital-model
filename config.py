"""
Configuration module for hospital SIXHRD model simulations.

This module centralizes all simulation parameters, age-specific disease parameters,
contact matrices, and vaccination strategies for improved maintainability and reusability.
"""

import numpy as np


# ========================================
# Standard Simulation Parameters
# ========================================

DEFAULT_SIM_PARAMS = {
    'Tmax': 200,          # simulation duration in days
    'time_step': 0.1,     # euler integration time step
    'hill_coef': 4,       # hill coefficient for admission gating
    'theta_X': 0.5,       # relative infectiousness of X compartment
    'theta_H': 0.3,       # relative infectiousness of H (ward + ICU) compartment
    'VE': 0.7             # vaccine efficacy (leaky model)
}


# ========================================
# Hospital Capacity Parameters
# ========================================

DEFAULT_CAPACITY_PARAMS = {
    'ward_capacity': 80,      # general ward bed capacity
    'icu_capacity': 20,       # ICU bed capacity
    'total_capacity': 100,    # for backward compatibility (ward + ICU)
    'hill_coef_ward': 4,      # hill coefficient for ward admission gating
    'hill_coef_icu': 4        # hill coefficient for ICU admission gating
}


# ========================================
# Differential Mortality Parameters
# ========================================
# Mortality multipliers for patients who cannot access care due to capacity constraints.
# These multipliers are applied to base mortality rates when admission is denied.
# A multiplier of 2.0 means untreated patients die at twice the rate of treated patients.

DIFFERENTIAL_MORTALITY_PARAMS = {
    # Multiplier for X compartment mortality when hospital admission is denied
    'mu_X_untreated_multiplier': 2.0,
    
    # Multiplier for ward patients who need ICU but cannot be admitted
    'mu_ward_denied_icu_multiplier': 1.5,
    
    # Age-specific untreated mortality multipliers (higher for elderly)
    'mu_X_untreated_multiplier_young': 1.5,     # young can cope better without care
    'mu_X_untreated_multiplier_middle': 2.0,    # baseline
    'mu_X_untreated_multiplier_elderly': 3.0,   # elderly most vulnerable without care
    
    # Age-specific ICU denial multipliers
    'mu_ward_denied_icu_multiplier_young': 1.3,
    'mu_ward_denied_icu_multiplier_middle': 1.5,
    'mu_ward_denied_icu_multiplier_elderly': 2.0
}


# ========================================
# Time-Varying Parameters
# ========================================

# seasonal transmission parameters
SEASONAL_PARAMS = {
    'amplitude': 0.3,     # seasonal amplitude (0 = no seasonality, 1 = full amplitude)
    'period': 365,        # period in days (365 = annual cycle)
    'peak_day': 0         # day of year when transmission is highest (0 = start of simulation)
}

# waning immunity parameters
WANING_PARAMS = {
    'default_omega': 0.0,    # default no waning
    'omega': 0.005,          # rate of immunity loss (1/days), 0 = no waning
    'omega_young': 0.001,    # age-specific waning for young
    'omega_middle': 0.0025,  # age-specific waning for middle
    'omega_elderly': 0.1     # age-specific waning for elderly
}

# policy intervention parameters
# each intervention is a dict with 'start_day', 'end_day', 'transmission_reduction'
POLICY_INTERVENTIONS = [
    # example: lockdown from day 50 to day 100, reducing transmission by 60%
    # {'start_day': 50, 'end_day': 100, 'transmission_reduction': 0.6}
]

# example intervention scenarios
LOCKDOWN_SCENARIO = [
    {'start_day': 50, 'end_day': 100, 'transmission_reduction': 0.6}
]

MULTIPLE_WAVES_SCENARIO = [
    {'start_day': 50, 'end_day': 80, 'transmission_reduction': 0.5},   # first lockdown
    {'start_day': 150, 'end_day': 180, 'transmission_reduction': 0.4}  # second lockdown
]


# ========================================
# Age Group Definitions
# ========================================

AGE_LABELS = ['Young (0-19)', 'Middle (20-64)', 'Elderly (65+)']
AGE_LABELS_SHORT = ['Young', 'Middle', 'Elderly']

# default population sizes for each age group
AGE_POPS_DEFAULT = [3000, 5000, 2000]  # total population: 10,000
AGE_POPS_LARGE_DEFAULT = [600000, 1000000, 400000]  # total population: 200,000


# ========================================
# Age-Specific Disease Parameters
# ========================================

# young (0-19): low severity, low mortality
YOUNG_PARAMS = {
    'sigma': 0.1,        # progression rate to severe cases
    'eta': 0.2,          # hospitalization need rate (ward)
    'eta_icu': 0.05,     # ICU need rate (fraction of ward patients needing ICU)
    'gamma_I': 0.12,     # recovery rate from I
    'mu_I': 0.001,       # mortality rate in I (very low)
    'gamma_X': 0.18,     # recovery rate from X
    'mu_X': 0.01,        # mortality rate in X (low) - treated baseline
    'mu_X_untreated': 0.015,  # mortality rate in X when care denied (1.5x)
    'gamma_ward': 0.25,  # recovery rate from ward
    'mu_ward': 0.003,    # mortality rate in ward (low)
    'mu_ward_denied_icu': 0.004,  # ward mortality when ICU denied (1.3x)
    'gamma_icu': 0.15,   # recovery rate from ICU (slower)
    'mu_icu': 0.02,      # mortality rate in ICU (higher than ward)
    # backward compatibility aliases
    'gamma_H': 0.25,     # recovery rate from H (legacy)
    'mu_H': 0.005        # mortality rate in H (legacy)
}

# middle (20-64): moderate severity and mortality
MIDDLE_PARAMS = {
    'sigma': 0.2,
    'eta': 0.3,          # hospitalization need rate (ward)
    'eta_icu': 0.15,     # ICU need rate (fraction of ward patients)
    'gamma_I': 0.1,
    'mu_I': 0.01,
    'gamma_X': 0.15,
    'mu_X': 0.05,        # mortality rate in X - treated baseline
    'mu_X_untreated': 0.10,  # mortality rate in X when care denied (2x)
    'gamma_ward': 0.2,   # recovery rate from ward
    'mu_ward': 0.01,     # mortality rate in ward
    'mu_ward_denied_icu': 0.015,  # ward mortality when ICU denied (1.5x)
    'gamma_icu': 0.12,   # recovery rate from ICU (slower)
    'mu_icu': 0.06,      # mortality rate in ICU (higher)
    # backward compatibility aliases
    'gamma_H': 0.2,
    'mu_H': 0.02
}

# elderly (65+): high severity, high mortality
ELDERLY_PARAMS = {
    'sigma': 0.3,        # higher progression to severe
    'eta': 0.5,          # higher hospitalization rate (ward)
    'eta_icu': 0.3,      # higher ICU need rate
    'gamma_I': 0.08,     # slower recovery
    'mu_I': 0.03,        # higher mortality in I
    'gamma_X': 0.12,     # slower recovery
    'mu_X': 0.15,        # much higher mortality in X - treated baseline
    'mu_X_untreated': 0.45,  # mortality rate in X when care denied (3x)
    'gamma_ward': 0.15,  # slower recovery from ward
    'mu_ward': 0.04,     # higher mortality in ward
    'mu_ward_denied_icu': 0.08,  # ward mortality when ICU denied (2x)
    'gamma_icu': 0.08,   # much slower ICU recovery
    'mu_icu': 0.20,      # much higher ICU mortality
    # backward compatibility aliases
    'gamma_H': 0.15,
    'mu_H': 0.08
}

AGE_PARAMS_DEFAULT = [YOUNG_PARAMS, MIDDLE_PARAMS, ELDERLY_PARAMS]


# ========================================
# Contact Matrices
# ========================================

# default contact matrix - assortative mixing by age
# rows = infector age, cols = infectee age
# young people have higher contact rates overall
CONTACT_MATRIX_DEFAULT = np.array([
    [10.0, 3.0, 1.0],   # young contacts: mostly with young
    [3.0, 8.0, 2.0],    # middle contacts: mostly with middle
    [1.0, 2.0, 4.0]     # elderly contacts: lower overall rates
])

# homogeneous mixing - all age groups mix equally
CONTACT_MATRIX_HOMOGENEOUS = np.array([
    [8.0, 8.0, 8.0],
    [8.0, 8.0, 8.0],
    [8.0, 8.0, 8.0]
])

# strong assortative mixing - age groups mostly interact within themselves
CONTACT_MATRIX_ASSORTATIVE = np.array([
    [15.0, 1.0, 0.5],
    [1.0, 12.0, 1.0],
    [0.5, 1.0, 6.0]
])


# ========================================
# Vaccination Strategies
# ========================================

# predefined vaccination strategies for comparison
# each strategy specifies coverage [young, middle, elderly]
VACCINATION_STRATEGIES = {
    'No vaccination': [0.0, 0.0, 0.0],
    'Uniform 30%': [0.3, 0.3, 0.3],
    'Elderly priority': [0.1, 0.2, 0.7],
    'Young priority': [0.7, 0.2, 0.1],
    'Middle priority': [0.1, 0.7, 0.2]
}


# ========================================
# Helper Functions for Time-Varying Parameters
# ========================================

def seasonal_forcing(t, beta_base, amplitude=0.3, period=365, peak_day=0):
    """
    Calculate seasonally-varying transmission rate.
    
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
        if intervention['start_day'] <= t <= intervention['end_day']:
            max_reduction = max(max_reduction, intervention['transmission_reduction'])
    
    return 1.0 - max_reduction
