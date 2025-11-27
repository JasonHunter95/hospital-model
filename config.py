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

REGIONAL_CAPACITY_PARAMS = {
    'ward_capacity': 1600,
    'icu_capacity': 400,
    'total_capacity': 2000,
    'hill_coef_ward': 4,
    'hill_coef_icu': 4
}

METROPOLITAN_CAPACITY_PARAMS = {
    'ward_capacity': 3200,
    'icu_capacity': 800,
    'total_capacity': 4000,
    'hill_coef_ward': 4,
    'hill_coef_icu': 4
}

LARGE_CAPACITY_PARAMS = {
    'ward_capacity': 16000,
    'icu_capacity': 4000,
    'total_capacity': 20000,
    'hill_coef_ward': 4,
    'hill_coef_icu': 4
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
# Initial Conditions
# ========================================

DEFAULT_INITIAL_CONDITIONS = {
    'I_single': 10,
    'E_single': 0,
    'X_single': 0,
    'H_single': 0,
    'R_single': 0,
    'D_single': 0,
    # Age-structured defaults (length should match number of age groups)
    'E_by_age': [0, 0, 0],
    'I_by_age': [10, 0, 0],
    'X_by_age': [0, 0, 0],
    'H_ward_by_age': [0, 0, 0],
    'H_icu_by_age': [0, 0, 0],
    'R_by_age': [0, 0, 0],
    'D_by_age': [0, 0, 0]
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
AGE_POPS_LARGE_DEFAULT = [600000, 1000000, 400000]  # total population: 2,000,000
AGE_POPS_METROPOLITAN_DEFAULT = [120000, 200000, 80000]  # total population: 400,000
AGE_POPS_REGIONAL_DEFAULT = [60000, 100000, 40000]  # total population: 200,000



# ========================================
# Age-Specific Disease Parameters
# ========================================

# young (0-19): very low severity
YOUNG_PARAMS_EMPIRICAL = {
    'alpha': 0.2,        # E → I rate (1/latent period, ~5 days)
    'sigma': 0.02,       # Only 2% progress to severe (vs 10% before)
    'eta': 0.05,         # 5% of severe cases need ward (vs 20%)
    'eta_icu': 0.02,     # 2% of ward patients need ICU (rare in young)
    'gamma_I': 0.14,     # ~7 day infectious period (1/0.14)
    'mu_I': 0.0001,      # Near-zero community mortality
    'gamma_X': 0.2,      # ~5 day severe illness
    'mu_X': 0.002,       # 0.2% daily mortality if severe but treated
    'mu_X_untreated': 0.006,  # 3x when denied care
    'gamma_ward': 0.2,   # ~5 day ward stay
    'mu_ward': 0.001,    # Very low ward mortality
    'mu_ward_denied_icu': 0.02,  # 2% daily if ICU denied (rare scenario)
    'gamma_icu': 0.1,    # ~10 day ICU stay
    'mu_icu': 0.005,     # 0.5% daily ICU mortality
    # Legacy aliases
    'gamma_H': 0.2,
    'mu_H': 0.001
}

# middle (20-64): moderate severity - subdivide if needed (20-44, 45-64)
MIDDLE_PARAMS_EMPIRICAL = {
    'alpha': 0.2,        # E → I rate (1/latent period, ~5 days)
    'sigma': 0.08,       # 8% progress to severe (vs 20%)
    'eta': 0.15,         # 15% of severe need ward admission
    'eta_icu': 0.10,     # 10% of ward patients need ICU
    'gamma_I': 0.12,     # ~8 day infectious period
    'mu_I': 0.001,       # 0.1% community mortality (rare)
    'gamma_X': 0.15,     # ~7 day severe phase
    'mu_X': 0.008,       # 0.8% daily mortality if severe, treated
    'mu_X_untreated': 0.024,  # 2.4% when denied care (3x)
    'gamma_ward': 0.14,  # ~7 day ward stay
    'mu_ward': 0.005,    # 0.5% daily ward mortality
    'mu_ward_denied_icu': 0.06,  # 6% daily if ICU denied
    'gamma_icu': 0.08,   # ~12 day ICU stay (longer than young)
    'mu_icu': 0.02,      # 2% daily ICU mortality
    # Legacy aliases
    'gamma_H': 0.14,
    'mu_H': 0.005
}

# elderly (65+): high severity - this is where most deaths occur
ELDERLY_PARAMS_EMPIRICAL = {
    'alpha': 0.18,       # E → I rate (1/latent period, ~5.5 days - slightly longer for elderly)
    'sigma': 0.15,       # 15% progress to severe (vs 30%)
    'eta': 0.35,         # 35% of severe need ward admission
    'eta_icu': 0.25,     # 25% of ward patients need ICU
    'gamma_I': 0.10,     # ~10 day infectious period (slower clearance)
    'mu_I': 0.005,       # 0.5% community mortality
    'gamma_X': 0.10,     # ~10 day severe phase
    'mu_X': 0.025,       # 2.5% daily mortality if severe, treated
    'mu_X_untreated': 0.10,   # 10% when denied care (4x)
    'gamma_ward': 0.10,  # ~10 day ward stay
    'mu_ward': 0.015,    # 1.5% daily ward mortality
    'mu_ward_denied_icu': 0.12,  # 12% daily if ICU denied
    'gamma_icu': 0.05,   # ~20 day ICU stay (prolonged)
    'mu_icu': 0.04,      # 4% daily ICU mortality
    # Legacy aliases
    'gamma_H': 0.10,
    'mu_H': 0.015
}

AGE_PARAMS_EMPIRICAL = [YOUNG_PARAMS_EMPIRICAL, MIDDLE_PARAMS_EMPIRICAL, ELDERLY_PARAMS_EMPIRICAL]

# young (0-19): low severity, low mortality
YOUNG_PARAMS = {
    'alpha': 0.2,        # E → I rate (1/latent period, ~5 days)
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
    'mu_ward_denied_icu': 0.025,  # ward mortality when ICU denied (8x baseline - these people need the ICU)
    'gamma_icu': 0.15,   # recovery rate from ICU (slower)
    'mu_icu': 0.008,      # mortality rate in ICU (higher than ward)
    # backward compatibility aliases
    'gamma_H': 0.25,     # recovery rate from H (legacy)
    'mu_H': 0.005        # mortality rate in H (legacy)
}

# middle (20-64): moderate severity and mortality
MIDDLE_PARAMS = {
    'alpha': 0.2,        # E → I rate (1/latent period, ~5 days)
    'sigma': 0.2,
    'eta': 0.3,          # hospitalization need rate (ward)
    'eta_icu': 0.15,     # ICU need rate (fraction of ward patients)
    'gamma_I': 0.1,
    'mu_I': 0.01,
    'gamma_X': 0.15,
    'mu_X': 0.02,        # mortality rate in X - treated baseline
    'mu_X_untreated': 0.06,  # mortality rate in X when care denied (3x)
    'gamma_ward': 0.2,   # recovery rate from ward
    'mu_ward': 0.01,     # mortality rate in ward
    'mu_ward_denied_icu': 0.12,  # ward mortality when ICU denied (12x - critical patients)
    'gamma_icu': 0.12,   # recovery rate from ICU (slower)
    'mu_icu': 0.04,      # mortality rate in ICU (lower than denied)
    # backward compatibility aliases
    'gamma_H': 0.2,
    'mu_H': 0.02
}

# elderly (65+): high severity, high mortality
ELDERLY_PARAMS = {
    'alpha': 0.18,       # E → I rate (1/latent period, ~5.5 days - slightly longer for elderly)
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
    'mu_ward_denied_icu': 0.35,  # ward mortality when ICU denied (very high)
    'gamma_icu': 0.08,   # much slower ICU recovery
    'mu_icu': 0.12,      # ICU mortality (high but saves lives vs. denial)
    # backward compatibility aliases
    'gamma_H': 0.15,
    'mu_H': 0.08
}

# ========================================
# Parameter Set Selection
# ========================================
# Set to True for realistic COVID-like epidemiology
# Set to False for teaching mode (high visibility, exaggerated effects)
USE_EMPIRICAL_PARAMS = True

# Legacy parameter sets (high mortality - for teaching)
AGE_PARAMS_TEACHING = [YOUNG_PARAMS, MIDDLE_PARAMS, ELDERLY_PARAMS]

# Empirical parameter sets (COVID-calibrated)
AGE_PARAMS_EMPIRICAL = [YOUNG_PARAMS_EMPIRICAL, MIDDLE_PARAMS_EMPIRICAL, ELDERLY_PARAMS_EMPIRICAL]

# Default selection based on mode
AGE_PARAMS_DEFAULT = AGE_PARAMS_EMPIRICAL if USE_EMPIRICAL_PARAMS else AGE_PARAMS_TEACHING

# Also export individual params for backward compatibility
if USE_EMPIRICAL_PARAMS:
    YOUNG_PARAMS_ACTIVE = YOUNG_PARAMS_EMPIRICAL
    MIDDLE_PARAMS_ACTIVE = MIDDLE_PARAMS_EMPIRICAL
    ELDERLY_PARAMS_ACTIVE = ELDERLY_PARAMS_EMPIRICAL
else:
    YOUNG_PARAMS_ACTIVE = YOUNG_PARAMS
    MIDDLE_PARAMS_ACTIVE = MIDDLE_PARAMS
    ELDERLY_PARAMS_ACTIVE = ELDERLY_PARAMS


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


