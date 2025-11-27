"""
Configuration module for hospital SEIXHRD model simulations.

This module centralizes all simulation parameters, age-specific disease parameters,
contact matrices, vaccination strategies, and complete scenario configurations for
the master hospital model. Configurations are organized hierarchically:

1. CORE PARAMETERS - Fundamental simulation and model parameters
2. AGE-SPECIFIC DISEASE PARAMETERS - Epidemiological rates by age group  
3. CONTACT MATRICES - Mixing patterns between age groups
4. HEALTHCARE SYSTEM CONFIGURATIONS - Bundled capacity + population presets
5. TIME-VARYING PARAMETERS - Seasonality, interventions, waning immunity
6. VACCINATION STRATEGIES - Coverage allocation patterns
7. COMPLETE SCENARIO BUNDLES - Ready-to-use simulation configurations
8. HELPER FUNCTIONS - Utilities for scenario management

Usage:
    from config import SCENARIO_COVID_DELTA, get_scenario_params
    params = get_scenario_params('covid_delta')
    results = simulate_master_hospital_model(**params)
"""

import numpy as np
from typing import Dict, List, Any, Optional
from copy import deepcopy


# ============================================================================
# SECTION 1: CORE SIMULATION PARAMETERS
# ============================================================================

DEFAULT_SIM_PARAMS = {
    'Tmax': 200,          # simulation duration in days
    'time_step': 0.1,     # Euler integration time step
    'hill_coef': 4,       # Hill coefficient for admission gating (legacy)
    'theta_X': 0.5,       # relative infectiousness of X compartment
    'theta_H': 0.3,       # relative infectiousness of H (ward + ICU) compartment
    'VE': 0.7             # vaccine efficacy (leaky model)
}

# Extended simulation durations for different study types
SIM_DURATION_PRESETS = {
    'short': 100,         # short outbreak analysis
    'standard': 200,      # standard epidemic wave
    'extended': 365,      # full year for seasonal effects
    'endemic': 730,       # two years for endemic equilibrium
    'long_term': 1095,    # three years for long-term dynamics
}


# ============================================================================
# SECTION 2: TRANSMISSION RATE (BETA) PRESETS
# ============================================================================
# Beta values calibrated to approximate R0 values assuming typical contact patterns.
# R0 ≈ beta * avg_contacts * infectious_period / N_scaling
# These are starting points; actual R0 depends on contact matrix and age structure.

TRANSMISSION_PRESETS = {
    'very_mild': {
        'beta_base': 0.12,
        'approx_R0': 1.2,
        'description': 'Subcritical or barely spreading (seasonal cold)',
    },
    'mild': {
        'beta_base': 0.18,
        'approx_R0': 1.5,
        'description': 'Low transmission (seasonal influenza)',
    },
    'moderate': {
        'beta_base': 0.28,
        'approx_R0': 2.5,
        'description': 'Moderate transmission (early COVID-19 Wuhan strain)',
    },
    'high': {
        'beta_base': 0.38,
        'approx_R0': 3.5,
        'description': 'High transmission (COVID-19 Delta variant)',
    },
    'severe': {
        'beta_base': 0.45,
        'approx_R0': 4.5,
        'description': 'Very high transmission (COVID-19 Omicron, measles-like)',
    },
    'extreme': {
        'beta_base': 0.55,
        'approx_R0': 6.0,
        'description': 'Extreme transmission (measles, pertussis)',
    },
}


# ============================================================================
# SECTION 3: AGE GROUP DEFINITIONS
# ============================================================================

AGE_LABELS = ['Young (0-19)', 'Middle (20-64)', 'Elderly (65+)']
AGE_LABELS_SHORT = ['Young', 'Middle', 'Elderly']
NUM_AGE_GROUPS = 3


# ============================================================================
# SECTION 4: AGE-SPECIFIC DISEASE PARAMETERS
# ============================================================================
# Two parameter sets available:
# - EMPIRICAL: COVID-calibrated, realistic epidemiology
# - TEACHING: Exaggerated effects for educational demonstrations

# --------------------------------------
# 4.1 Empirical Parameters (COVID-calibrated)
# --------------------------------------

YOUNG_PARAMS_EMPIRICAL = {
    # Latent period (E → I)
    'alpha': 0.2,             # ~5 day latent period
    
    # Disease progression
    'sigma': 0.02,            # 2% progress to severe
    'eta': 0.05,              # 5% of severe need ward
    'eta_icu': 0.02,          # 2% of ward need ICU
    
    # Recovery rates
    'gamma_I': 0.14,          # ~7 day infectious period
    'gamma_X': 0.2,           # ~5 day severe phase
    'gamma_ward': 0.2,        # ~5 day ward stay
    'gamma_icu': 0.1,         # ~10 day ICU stay
    
    # Mortality rates (daily probability)
    'mu_I': 0.0001,           # near-zero community mortality
    'mu_X': 0.002,            # 0.2% if severe but treated
    'mu_X_untreated': 0.006,  # 0.6% when denied care (3x)
    'mu_ward': 0.001,         # 0.1% ward mortality
    'mu_ward_denied_icu': 0.02,  # 2% if ICU denied
    'mu_icu': 0.005,          # 0.5% ICU mortality
    
    # Legacy aliases for backward compatibility
    'gamma_H': 0.2,
    'mu_H': 0.001
}

MIDDLE_PARAMS_EMPIRICAL = {
    'alpha': 0.2,
    'sigma': 0.08,            # 8% progress to severe
    'eta': 0.15,              # 15% of severe need ward
    'eta_icu': 0.10,          # 10% of ward need ICU
    'gamma_I': 0.12,          # ~8 day infectious period
    'gamma_X': 0.15,          # ~7 day severe phase
    'gamma_ward': 0.14,       # ~7 day ward stay
    'gamma_icu': 0.08,        # ~12 day ICU stay
    'mu_I': 0.001,
    'mu_X': 0.008,
    'mu_X_untreated': 0.024,  # 3x when denied care
    'mu_ward': 0.005,
    'mu_ward_denied_icu': 0.06,
    'mu_icu': 0.02,
    'gamma_H': 0.14,
    'mu_H': 0.005
}

ELDERLY_PARAMS_EMPIRICAL = {
    'alpha': 0.18,            # slightly longer latent period
    'sigma': 0.15,            # 15% progress to severe
    'eta': 0.35,              # 35% of severe need ward
    'eta_icu': 0.25,          # 25% of ward need ICU
    'gamma_I': 0.10,          # ~10 day infectious period
    'gamma_X': 0.10,          # ~10 day severe phase
    'gamma_ward': 0.10,       # ~10 day ward stay
    'gamma_icu': 0.05,        # ~20 day ICU stay
    'mu_I': 0.005,
    'mu_X': 0.025,
    'mu_X_untreated': 0.10,   # 4x when denied care
    'mu_ward': 0.015,
    'mu_ward_denied_icu': 0.12,
    'mu_icu': 0.04,
    'gamma_H': 0.10,
    'mu_H': 0.015
}

AGE_PARAMS_EMPIRICAL = [YOUNG_PARAMS_EMPIRICAL, MIDDLE_PARAMS_EMPIRICAL, ELDERLY_PARAMS_EMPIRICAL]


# --------------------------------------
# 4.2 Teaching Parameters (exaggerated effects)
# --------------------------------------

YOUNG_PARAMS_TEACHING = {
    'alpha': 0.2,
    'sigma': 0.1,
    'eta': 0.2,
    'eta_icu': 0.05,
    'gamma_I': 0.12,
    'gamma_X': 0.18,
    'gamma_ward': 0.25,
    'gamma_icu': 0.15,
    'mu_I': 0.001,
    'mu_X': 0.01,
    'mu_X_untreated': 0.015,
    'mu_ward': 0.003,
    'mu_ward_denied_icu': 0.025,
    'mu_icu': 0.008,
    'gamma_H': 0.25,
    'mu_H': 0.005
}

MIDDLE_PARAMS_TEACHING = {
    'alpha': 0.2,
    'sigma': 0.2,
    'eta': 0.3,
    'eta_icu': 0.15,
    'gamma_I': 0.1,
    'gamma_X': 0.15,
    'gamma_ward': 0.2,
    'gamma_icu': 0.12,
    'mu_I': 0.01,
    'mu_X': 0.02,
    'mu_X_untreated': 0.06,
    'mu_ward': 0.01,
    'mu_ward_denied_icu': 0.12,
    'mu_icu': 0.04,
    'gamma_H': 0.2,
    'mu_H': 0.02
}

ELDERLY_PARAMS_TEACHING = {
    'alpha': 0.18,
    'sigma': 0.3,
    'eta': 0.5,
    'eta_icu': 0.3,
    'gamma_I': 0.08,
    'gamma_X': 0.12,
    'gamma_ward': 0.15,
    'gamma_icu': 0.08,
    'mu_I': 0.03,
    'mu_X': 0.15,
    'mu_X_untreated': 0.45,
    'mu_ward': 0.04,
    'mu_ward_denied_icu': 0.35,
    'mu_icu': 0.12,
    'gamma_H': 0.15,
    'mu_H': 0.08
}

AGE_PARAMS_TEACHING = [YOUNG_PARAMS_TEACHING, MIDDLE_PARAMS_TEACHING, ELDERLY_PARAMS_TEACHING]


# --------------------------------------
# 4.3 Parameter Set Selection
# --------------------------------------

USE_EMPIRICAL_PARAMS = True  # Toggle between empirical and teaching modes

AGE_PARAMS_DEFAULT = AGE_PARAMS_EMPIRICAL if USE_EMPIRICAL_PARAMS else AGE_PARAMS_TEACHING

# Active individual params for backward compatibility
YOUNG_PARAMS_ACTIVE = YOUNG_PARAMS_EMPIRICAL if USE_EMPIRICAL_PARAMS else YOUNG_PARAMS_TEACHING
MIDDLE_PARAMS_ACTIVE = MIDDLE_PARAMS_EMPIRICAL if USE_EMPIRICAL_PARAMS else MIDDLE_PARAMS_TEACHING
ELDERLY_PARAMS_ACTIVE = ELDERLY_PARAMS_EMPIRICAL if USE_EMPIRICAL_PARAMS else ELDERLY_PARAMS_TEACHING

# Legacy aliases
YOUNG_PARAMS = YOUNG_PARAMS_TEACHING
MIDDLE_PARAMS = MIDDLE_PARAMS_TEACHING
ELDERLY_PARAMS = ELDERLY_PARAMS_TEACHING


# ============================================================================
# SECTION 5: DIFFERENTIAL MORTALITY PARAMETERS
# ============================================================================
# Mortality multipliers for patients denied care due to capacity constraints.
# Applied when Hill gating restricts admissions.

DIFFERENTIAL_MORTALITY_PARAMS = {
    # Base multipliers (applied uniformly if age-specific not specified)
    'mu_X_untreated_multiplier': 2.0,
    'mu_ward_denied_icu_multiplier': 1.5,
    
    # Age-specific multipliers for X compartment (denied hospital admission)
    'mu_X_untreated_multiplier_young': 1.5,      # young compensate better
    'mu_X_untreated_multiplier_middle': 2.0,     # baseline
    'mu_X_untreated_multiplier_elderly': 3.0,    # elderly most vulnerable
    
    # Age-specific multipliers for ward patients denied ICU
    'mu_ward_denied_icu_multiplier_young': 1.3,
    'mu_ward_denied_icu_multiplier_middle': 1.5,
    'mu_ward_denied_icu_multiplier_elderly': 2.0
}


# ============================================================================
# SECTION 6: CONTACT MATRICES
# ============================================================================
# Contact rates C[a,b] = contacts per day from age group a to b
# Rows = infector age group, Cols = infectee age group

CONTACT_MATRIX_DEFAULT = np.array([
    [10.0, 3.0, 1.0],    # young: high within-group, low with elderly
    [3.0, 8.0, 2.0],     # middle: moderate all-around
    [1.0, 2.0, 4.0]      # elderly: lower overall contact rates
])

CONTACT_MATRIX_HOMOGENEOUS = np.array([
    [8.0, 8.0, 8.0],
    [8.0, 8.0, 8.0],
    [8.0, 8.0, 8.0]
])

CONTACT_MATRIX_ASSORTATIVE = np.array([
    [15.0, 1.0, 0.5],    # strong within-group mixing
    [1.0, 12.0, 1.0],
    [0.5, 1.0, 6.0]
])

# School closure scenario: reduced young-to-all contacts
CONTACT_MATRIX_SCHOOL_CLOSURE = np.array([
    [4.0, 1.5, 0.5],     # young contacts reduced ~60%
    [1.5, 8.0, 2.0],     # middle unchanged
    [0.5, 2.0, 4.0]      # elderly unchanged
])

# Work-from-home scenario: reduced middle adult contacts
CONTACT_MATRIX_WORK_FROM_HOME = np.array([
    [10.0, 1.5, 1.0],
    [1.5, 4.0, 1.0],     # middle-middle reduced ~50%
    [1.0, 1.0, 4.0]
])

# Elderly shielding: reduced elderly contacts
CONTACT_MATRIX_ELDERLY_SHIELDING = np.array([
    [10.0, 3.0, 0.3],
    [3.0, 8.0, 0.6],
    [0.3, 0.6, 2.0]      # elderly contacts reduced ~50%
])


# ============================================================================
# SECTION 7: HEALTHCARE SYSTEM CONFIGURATIONS
# ============================================================================
# Bundled capacity + population presets for different healthcare settings.
# Each configuration represents a coherent healthcare system type.

HEALTHCARE_SYSTEM_SMALL = {
    'name': 'Small Community Hospital',
    'description': 'Single small hospital serving rural community',
    'ward_capacity': 40,
    'icu_capacity': 8,
    'hill_coef_ward': 4,
    'hill_coef_icu': 4,
    'age_pops': [12000, 20000, 8000],  # 40,000 total
    'beds_per_1000': 1.2,
}

HEALTHCARE_SYSTEM_RURAL = {
    'name': 'Rural Hospital Network',
    'description': 'Regional rural health system with limited resources',
    'ward_capacity': 80,
    'icu_capacity': 20,
    'hill_coef_ward': 4,
    'hill_coef_icu': 4,
    'age_pops': [30000, 50000, 20000],  # 100,000 total
    'beds_per_1000': 1.0,
}

HEALTHCARE_SYSTEM_SUBURBAN = {
    'name': 'Suburban Hospital',
    'description': 'Mid-sized suburban hospital system',
    'ward_capacity': 160,
    'icu_capacity': 40,
    'hill_coef_ward': 4,
    'hill_coef_icu': 4,
    'age_pops': [60000, 100000, 40000],  # 200,000 total
    'beds_per_1000': 1.0,
}

HEALTHCARE_SYSTEM_URBAN = {
    'name': 'Urban Medical Center',
    'description': 'Major urban hospital with regional referral capacity',
    'ward_capacity': 400,
    'icu_capacity': 100,
    'hill_coef_ward': 4,
    'hill_coef_icu': 4,
    'age_pops': [120000, 200000, 80000],  # 400,000 total
    'beds_per_1000': 1.25,
}

HEALTHCARE_SYSTEM_METROPOLITAN = {
    'name': 'Metropolitan Health System',
    'description': 'Large metropolitan area with multiple hospitals',
    'ward_capacity': 1600,
    'icu_capacity': 400,
    'hill_coef_ward': 4,
    'hill_coef_icu': 4,
    'age_pops': [600000, 1000000, 400000],  # 2,000,000 total
    'beds_per_1000': 1.0,
}

HEALTHCARE_SYSTEM_WELL_RESOURCED = {
    'name': 'Well-Resourced System',
    'description': 'High-income region with abundant healthcare resources',
    'ward_capacity': 2400,
    'icu_capacity': 600,
    'hill_coef_ward': 4,
    'hill_coef_icu': 4,
    'age_pops': [600000, 1000000, 400000],  # 2,000,000 total
    'beds_per_1000': 1.5,
}

HEALTHCARE_SYSTEM_RESOURCE_LIMITED = {
    'name': 'Resource-Limited System',
    'description': 'Low-resource setting with constrained healthcare',
    'ward_capacity': 200,
    'icu_capacity': 25,
    'hill_coef_ward': 6,   # sharper constraint
    'hill_coef_icu': 6,
    'age_pops': [600000, 1000000, 400000],  # 2,000,000 total
    'beds_per_1000': 0.11,
}

# Surge capacity scenarios (temporary expansions)
HEALTHCARE_SYSTEM_SURGE_MILD = {
    'name': 'Mild Surge Capacity',
    'description': 'Urban system with 25% surge capacity activated',
    'ward_capacity': 500,   # +25%
    'icu_capacity': 125,    # +25%
    'hill_coef_ward': 4,
    'hill_coef_icu': 4,
    'age_pops': [120000, 200000, 80000],
    'beds_per_1000': 1.56,
}

HEALTHCARE_SYSTEM_SURGE_MAJOR = {
    'name': 'Major Surge Capacity',
    'description': 'Urban system with 50% surge capacity (field hospitals)',
    'ward_capacity': 600,   # +50%
    'icu_capacity': 150,    # +50%
    'hill_coef_ward': 5,    # slightly less efficient
    'hill_coef_icu': 5,
    'age_pops': [120000, 200000, 80000],
    'beds_per_1000': 1.88,
}

# For backward compatibility
DEFAULT_CAPACITY_PARAMS = {
    'ward_capacity': 80,
    'icu_capacity': 20,
    'total_capacity': 100,
    'hill_coef_ward': 4,
    'hill_coef_icu': 4
}

# Legacy aliases
REGIONAL_CAPACITY_PARAMS = {
    'ward_capacity': HEALTHCARE_SYSTEM_METROPOLITAN['ward_capacity'],
    'icu_capacity': HEALTHCARE_SYSTEM_METROPOLITAN['icu_capacity'],
    'total_capacity': 2000,
    'hill_coef_ward': 4,
    'hill_coef_icu': 4
}


# ============================================================================
# SECTION 8: INITIAL CONDITIONS
# ============================================================================

DEFAULT_INITIAL_CONDITIONS = {
    # Single-population model
    'I_single': 10,
    'E_single': 0,
    'X_single': 0,
    'H_single': 0,
    'R_single': 0,
    'D_single': 0,
    
    # Age-structured model
    'E_by_age': [0, 0, 0],
    'I_by_age': [10, 0, 0],      # seed in young (school outbreak)
    'X_by_age': [0, 0, 0],
    'H_ward_by_age': [0, 0, 0],
    'H_icu_by_age': [0, 0, 0],
    'R_by_age': [0, 0, 0],
    'D_by_age': [0, 0, 0]
}

# Alternative initial condition scenarios
INITIAL_CONDITIONS_PRESETS = {
    'young_seed': {
        'description': 'Outbreak starting in young population (school)',
        'E_by_age': [5, 0, 0],
        'I_by_age': [10, 0, 0],
    },
    'middle_seed': {
        'description': 'Outbreak starting in working-age adults (workplace)',
        'E_by_age': [0, 5, 0],
        'I_by_age': [0, 10, 0],
    },
    'elderly_seed': {
        'description': 'Outbreak starting in elderly (care home)',
        'E_by_age': [0, 0, 5],
        'I_by_age': [0, 0, 10],
    },
    'multi_source': {
        'description': 'Multiple simultaneous introductions',
        'E_by_age': [3, 3, 2],
        'I_by_age': [5, 5, 3],
    },
    'established_outbreak': {
        'description': 'Simulation starting mid-outbreak',
        'E_by_age': [50, 80, 30],
        'I_by_age': [100, 150, 60],
        'X_by_age': [5, 20, 15],
        'H_ward_by_age': [1, 8, 6],
        'H_icu_by_age': [0, 2, 3],
        'R_by_age': [200, 300, 100],
    },
    'post_wave': {
        'description': 'Starting after first wave with immunity',
        'E_by_age': [2, 3, 1],
        'I_by_age': [5, 8, 3],
        'R_by_age': [3000, 5000, 2000],  # significant immunity
    },
}


# ============================================================================
# SECTION 9: TIME-VARYING PARAMETERS
# ============================================================================

# --------------------------------------
# 9.1 Seasonal Transmission Patterns
# --------------------------------------

SEASONAL_PARAMS_NONE = {
    'amplitude': 0.0,
    'period': 365,
    'peak_day': 0,
    'description': 'No seasonal variation',
}

SEASONAL_PARAMS_MILD = {
    'amplitude': 0.15,
    'period': 365,
    'peak_day': 0,
    'description': 'Mild winter peak (subtropical climate)',
}

SEASONAL_PARAMS_MODERATE = {
    'amplitude': 0.25,
    'period': 365,
    'peak_day': 0,
    'description': 'Moderate seasonality (temperate climate)',
}

SEASONAL_PARAMS_STRONG = {
    'amplitude': 0.40,
    'period': 365,
    'peak_day': 0,
    'description': 'Strong winter peak (continental/cold climate)',
}

# Legacy alias
SEASONAL_PARAMS = SEASONAL_PARAMS_MODERATE


# --------------------------------------
# 9.2 Waning Immunity Presets
# --------------------------------------

WANING_NONE = {
    'omega': 0.0,
    'description': 'No waning immunity (permanent protection)',
    'mean_duration_days': float('inf'),
}

WANING_SLOW = {
    'omega': 0.001,
    'description': 'Slow waning (~3 years)',
    'mean_duration_days': 1000,
}

WANING_MODERATE = {
    'omega': 0.003,
    'description': 'Moderate waning (~1 year)',
    'mean_duration_days': 333,
}

WANING_FAST = {
    'omega': 0.005,
    'description': 'Fast waning (~6 months)',
    'mean_duration_days': 200,
}

WANING_VERY_FAST = {
    'omega': 0.01,
    'description': 'Very fast waning (~3 months)',
    'mean_duration_days': 100,
}

# Age-differential waning (elderly lose immunity faster)
WANING_AGE_DIFFERENTIAL = {
    'omega_young': 0.002,    # ~500 days
    'omega_middle': 0.003,   # ~333 days
    'omega_elderly': 0.006,  # ~167 days
    'description': 'Age-dependent waning (elderly faster)',
}

# Legacy alias
WANING_PARAMS = {
    'default_omega': 0.0,
    'omega': WANING_FAST['omega'],
    'omega_young': WANING_AGE_DIFFERENTIAL['omega_young'],
    'omega_middle': WANING_AGE_DIFFERENTIAL['omega_middle'],
    'omega_elderly': WANING_AGE_DIFFERENTIAL['omega_elderly'],
}


# --------------------------------------
# 9.3 Policy Intervention Templates
# --------------------------------------
# Each intervention: {'start_day', 'end_day', 'transmission_reduction'}

INTERVENTION_NONE = []

INTERVENTION_EARLY_STRONG = [
    {'start_day': 14, 'end_day': 60, 'transmission_reduction': 0.6},
]

INTERVENTION_EARLY_MODERATE = [
    {'start_day': 21, 'end_day': 75, 'transmission_reduction': 0.4},
]

INTERVENTION_DELAYED_STRONG = [
    {'start_day': 45, 'end_day': 105, 'transmission_reduction': 0.6},
]

INTERVENTION_DELAYED_MODERATE = [
    {'start_day': 60, 'end_day': 120, 'transmission_reduction': 0.4},
]

# Tiered restrictions (escalating)
INTERVENTION_TIERED_ESCALATING = [
    {'start_day': 21, 'end_day': 35, 'transmission_reduction': 0.2},
    {'start_day': 35, 'end_day': 56, 'transmission_reduction': 0.4},
    {'start_day': 56, 'end_day': 84, 'transmission_reduction': 0.6},
]

# Tiered restrictions (de-escalating after peak)
INTERVENTION_TIERED_DEESCALATING = [
    {'start_day': 30, 'end_day': 60, 'transmission_reduction': 0.6},
    {'start_day': 60, 'end_day': 90, 'transmission_reduction': 0.4},
    {'start_day': 90, 'end_day': 120, 'transmission_reduction': 0.2},
]

# On-off cycling (intermittent lockdowns)
INTERVENTION_CYCLICAL = [
    {'start_day': 30, 'end_day': 51, 'transmission_reduction': 0.5},
    {'start_day': 72, 'end_day': 93, 'transmission_reduction': 0.5},
    {'start_day': 114, 'end_day': 135, 'transmission_reduction': 0.5},
    {'start_day': 156, 'end_day': 177, 'transmission_reduction': 0.5},
]

# Prolonged moderate restrictions
INTERVENTION_SUSTAINED_MODERATE = [
    {'start_day': 30, 'end_day': 150, 'transmission_reduction': 0.3},
]

# Short sharp lockdown
INTERVENTION_SHORT_SHARP = [
    {'start_day': 25, 'end_day': 39, 'transmission_reduction': 0.7},
]

# Multiple waves response
INTERVENTION_MULTI_WAVE = [
    {'start_day': 30, 'end_day': 60, 'transmission_reduction': 0.5},
    {'start_day': 120, 'end_day': 150, 'transmission_reduction': 0.4},
    {'start_day': 210, 'end_day': 240, 'transmission_reduction': 0.35},
]

# Legacy aliases
LOCKDOWN_SCENARIO = [
    {'start_day': 50, 'end_day': 100, 'transmission_reduction': 0.6}
]

MULTIPLE_WAVES_SCENARIO = [
    {'start_day': 50, 'end_day': 80, 'transmission_reduction': 0.5},
    {'start_day': 150, 'end_day': 180, 'transmission_reduction': 0.4}
]


# ============================================================================
# SECTION 10: VACCINATION STRATEGIES
# ============================================================================
# Coverage arrays: [young, middle, elderly]

VACCINATION_STRATEGIES = {
    'none': {
        'coverage': [0.0, 0.0, 0.0],
        'description': 'No vaccination',
    },
    'uniform_low': {
        'coverage': [0.2, 0.2, 0.2],
        'description': 'Low uniform coverage (20%)',
    },
    'uniform_moderate': {
        'coverage': [0.4, 0.4, 0.4],
        'description': 'Moderate uniform coverage (40%)',
    },
    'uniform_high': {
        'coverage': [0.7, 0.7, 0.7],
        'description': 'High uniform coverage (70%)',
    },
    'elderly_priority': {
        'coverage': [0.1, 0.3, 0.8],
        'description': 'Prioritize elderly (high-risk)',
    },
    'elderly_only': {
        'coverage': [0.0, 0.1, 0.9],
        'description': 'Focus on elderly only',
    },
    'working_age_priority': {
        'coverage': [0.2, 0.7, 0.5],
        'description': 'Prioritize working-age adults (transmission reduction)',
    },
    'young_priority': {
        'coverage': [0.7, 0.3, 0.4],
        'description': 'Prioritize young (school/transmission)',
    },
    'balanced_risk': {
        'coverage': [0.3, 0.5, 0.8],
        'description': 'Balanced by mortality risk',
    },
    'herd_immunity_target': {
        'coverage': [0.6, 0.75, 0.85],
        'description': 'High coverage targeting herd immunity',
    },
    # Legacy format for backward compatibility
    'No vaccination': [0.0, 0.0, 0.0],
    'Uniform 30%': [0.3, 0.3, 0.3],
    'Elderly priority': [0.1, 0.2, 0.7],
    'Young priority': [0.7, 0.2, 0.1],
    'Middle priority': [0.1, 0.7, 0.2],
}

# Vaccine efficacy presets
VACCINE_EFFICACY_PRESETS = {
    'low': 0.50,          # early/suboptimal vaccines
    'moderate': 0.65,     # typical flu vaccine
    'high': 0.80,         # mRNA vaccines against infection
    'very_high': 0.90,    # optimal conditions
    'sterilizing': 0.95,  # near-complete protection
}


# ============================================================================
# SECTION 11: COMPLETE SCENARIO BUNDLES
# ============================================================================
# Ready-to-use configurations combining all parameters.
# Use get_scenario_params() to extract for simulate_master_hospital_model().

SCENARIO_BASELINE = {
    'name': 'Baseline',
    'description': 'Default moderate outbreak in urban setting, no interventions',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_NONE,
    'vaccination': VACCINATION_STRATEGIES['none'],
    'VE': VACCINE_EFFICACY_PRESETS['high'],
    'Tmax': 200,
}

SCENARIO_COVID_EARLY_2020 = {
    'name': 'COVID-19 Early 2020',
    'description': 'Early pandemic wave, no vaccines, delayed lockdown',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_DELAYED_STRONG,
    'vaccination': VACCINATION_STRATEGIES['none'],
    'VE': 0.0,
    'Tmax': 200,
}

SCENARIO_COVID_DELTA = {
    'name': 'COVID-19 Delta Wave',
    'description': 'High transmission Delta variant with partial vaccination',
    'beta_base': TRANSMISSION_PRESETS['high']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_MILD,
    'waning_params': WANING_FAST,
    'interventions': INTERVENTION_TIERED_ESCALATING,
    'vaccination': VACCINATION_STRATEGIES['balanced_risk'],
    'VE': VACCINE_EFFICACY_PRESETS['moderate'],  # reduced against Delta
    'Tmax': 300,
}

SCENARIO_COVID_OMICRON = {
    'name': 'COVID-19 Omicron Wave',
    'description': 'Very high transmission but lower severity, high immunity',
    'beta_base': TRANSMISSION_PRESETS['severe']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,  # would need reduced severity params
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_MODERATE,
    'waning_params': WANING_VERY_FAST,
    'interventions': INTERVENTION_EARLY_MODERATE,
    'vaccination': VACCINATION_STRATEGIES['herd_immunity_target'],
    'VE': VACCINE_EFFICACY_PRESETS['low'],  # immune escape
    'Tmax': 200,
}

SCENARIO_SEASONAL_FLU = {
    'name': 'Seasonal Influenza',
    'description': 'Typical flu season with moderate vaccination',
    'beta_base': TRANSMISSION_PRESETS['mild']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_STRONG,
    'waning_params': WANING_MODERATE,
    'interventions': INTERVENTION_NONE,
    'vaccination': VACCINATION_STRATEGIES['elderly_priority'],
    'VE': VACCINE_EFFICACY_PRESETS['moderate'],
    'Tmax': 365,
}

SCENARIO_ENDEMIC_EQUILIBRIUM = {
    'name': 'Endemic Equilibrium',
    'description': 'Long-term endemic dynamics with waning immunity',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_MODERATE,
    'waning_params': WANING_MODERATE,
    'interventions': INTERVENTION_NONE,
    'vaccination': VACCINATION_STRATEGIES['balanced_risk'],
    'VE': VACCINE_EFFICACY_PRESETS['moderate'],
    'Tmax': 730,  # 2 years
}

SCENARIO_HOSPITAL_STRESS_TEST = {
    'name': 'Hospital Stress Test',
    'description': 'Severe outbreak overwhelming limited healthcare capacity',
    'beta_base': TRANSMISSION_PRESETS['high']['beta_base'],
    'age_params': AGE_PARAMS_TEACHING,  # exaggerated severity
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_RURAL,  # limited capacity
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_DELAYED_MODERATE,
    'vaccination': VACCINATION_STRATEGIES['none'],
    'VE': 0.0,
    'Tmax': 200,
}

SCENARIO_OPTIMAL_RESPONSE = {
    'name': 'Optimal Response',
    'description': 'Early intervention + high vaccination, well-resourced system',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_WELL_RESOURCED,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_SLOW,
    'interventions': INTERVENTION_EARLY_STRONG,
    'vaccination': VACCINATION_STRATEGIES['herd_immunity_target'],
    'VE': VACCINE_EFFICACY_PRESETS['high'],
    'Tmax': 300,
}

SCENARIO_RESOURCE_LIMITED = {
    'name': 'Resource-Limited Setting',
    'description': 'Low-resource healthcare system with moderate outbreak',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_RESOURCE_LIMITED,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_DELAYED_MODERATE,
    'vaccination': VACCINATION_STRATEGIES['uniform_low'],
    'VE': VACCINE_EFFICACY_PRESETS['moderate'],
    'Tmax': 200,
}

SCENARIO_SCHOOL_OUTBREAK = {
    'name': 'School Outbreak',
    'description': 'Outbreak seeded in schools with school closure response',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_SCHOOL_CLOSURE,  # after closure
    'healthcare_system': HEALTHCARE_SYSTEM_SUBURBAN,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_EARLY_MODERATE,
    'vaccination': VACCINATION_STRATEGIES['young_priority'],
    'VE': VACCINE_EFFICACY_PRESETS['high'],
    'initial_conditions': INITIAL_CONDITIONS_PRESETS['young_seed'],
    'Tmax': 150,
}

SCENARIO_CARE_HOME_OUTBREAK = {
    'name': 'Care Home Outbreak',
    'description': 'Outbreak starting in elderly care facility',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_ELDERLY_SHIELDING,
    'healthcare_system': HEALTHCARE_SYSTEM_SUBURBAN,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_EARLY_STRONG,
    'vaccination': VACCINATION_STRATEGIES['elderly_only'],
    'VE': VACCINE_EFFICACY_PRESETS['moderate'],
    'initial_conditions': INITIAL_CONDITIONS_PRESETS['elderly_seed'],
    'Tmax': 150,
}

SCENARIO_SURGE_CAPACITY = {
    'name': 'Surge Capacity Activation',
    'description': 'Testing impact of surge capacity during severe wave',
    'beta_base': TRANSMISSION_PRESETS['high']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_SURGE_MAJOR,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_TIERED_ESCALATING,
    'vaccination': VACCINATION_STRATEGIES['balanced_risk'],
    'VE': VACCINE_EFFICACY_PRESETS['moderate'],
    'Tmax': 200,
}

SCENARIO_CYCLICAL_POLICY = {
    'name': 'Cyclical Intervention Policy',
    'description': 'Testing on-off lockdown strategies',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_CYCLICAL,
    'vaccination': VACCINATION_STRATEGIES['uniform_moderate'],
    'VE': VACCINE_EFFICACY_PRESETS['high'],
    'Tmax': 250,
}


# Registry of all scenarios for easy access
SCENARIO_REGISTRY = {
    'baseline': SCENARIO_BASELINE,
    'covid_early_2020': SCENARIO_COVID_EARLY_2020,
    'covid_delta': SCENARIO_COVID_DELTA,
    'covid_omicron': SCENARIO_COVID_OMICRON,
    'seasonal_flu': SCENARIO_SEASONAL_FLU,
    'endemic': SCENARIO_ENDEMIC_EQUILIBRIUM,
    'stress_test': SCENARIO_HOSPITAL_STRESS_TEST,
    'optimal_response': SCENARIO_OPTIMAL_RESPONSE,
    'resource_limited': SCENARIO_RESOURCE_LIMITED,
    'school_outbreak': SCENARIO_SCHOOL_OUTBREAK,
    'care_home_outbreak': SCENARIO_CARE_HOME_OUTBREAK,
    'surge_capacity': SCENARIO_SURGE_CAPACITY,
    'cyclical_policy': SCENARIO_CYCLICAL_POLICY,
}


# ============================================================================
# SECTION 12: SENSITIVITY ANALYSIS RANGES
# ============================================================================
# Parameter ranges for systematic exploration

PARAMETER_RANGES = {
    'beta_base': {
        'min': 0.1,
        'max': 0.6,
        'default': 0.28,
        'description': 'Baseline transmission rate',
    },
    'ward_capacity': {
        'min': 20,
        'max': 500,
        'default': 100,
        'description': 'General ward bed capacity',
    },
    'icu_capacity': {
        'min': 5,
        'max': 125,
        'default': 25,
        'description': 'ICU bed capacity',
    },
    'VE': {
        'min': 0.3,
        'max': 0.95,
        'default': 0.7,
        'description': 'Vaccine efficacy',
    },
    'coverage_elderly': {
        'min': 0.0,
        'max': 0.95,
        'default': 0.5,
        'description': 'Elderly vaccination coverage',
    },
    'seasonal_amplitude': {
        'min': 0.0,
        'max': 0.5,
        'default': 0.25,
        'description': 'Seasonal transmission amplitude',
    },
    'omega': {
        'min': 0.0,
        'max': 0.02,
        'default': 0.003,
        'description': 'Immunity waning rate',
    },
    'hill_coef': {
        'min': 2,
        'max': 8,
        'default': 4,
        'description': 'Hill coefficient (admission gating steepness)',
    },
}


# ============================================================================
# SECTION 13: HELPER FUNCTIONS
# ============================================================================

def get_scenario_params(scenario_name: str) -> Dict[str, Any]:
    """
    Extract parameters from a scenario bundle for simulate_master_hospital_model().
    
    Args:
        scenario_name: Key from SCENARIO_REGISTRY (e.g., 'covid_delta')
        
    Returns:
        Dictionary of parameters ready to unpack into simulation function.
        
    Example:
        params = get_scenario_params('covid_delta')
        results = simulate_master_hospital_model(**params)
    """
    if scenario_name not in SCENARIO_REGISTRY:
        available = list(SCENARIO_REGISTRY.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")
    
    scenario = deepcopy(SCENARIO_REGISTRY[scenario_name])
    healthcare = scenario.pop('healthcare_system')
    vaccination = scenario.pop('vaccination')
    seasonal = scenario.pop('seasonal_params', {})
    waning = scenario.pop('waning_params', {})
    
    # Handle vaccination - can be dict with 'coverage' key or direct list
    if isinstance(vaccination, dict):
        coverage = vaccination.get('coverage', [0.0, 0.0, 0.0])
    else:
        coverage = vaccination
    
    # Build parameter dict for simulate_master_hospital_model
    params = {
        'beta_base': scenario['beta_base'],
        'age_params': scenario['age_params'],
        'contact_matrix': scenario.get('contact_matrix', CONTACT_MATRIX_DEFAULT),
        'age_pops': healthcare['age_pops'],
        'ward_capacity': healthcare['ward_capacity'],
        'icu_capacity': healthcare['icu_capacity'],
        'hill_coef_ward': healthcare.get('hill_coef_ward', 4),
        'hill_coef_icu': healthcare.get('hill_coef_icu', 4),
        'coverage': coverage,
        'VE': scenario.get('VE', 0.7),
        'Tmax': scenario.get('Tmax', 200),
        'interventions': scenario.get('interventions', []),
    }
    
    # Add seasonal parameters
    if seasonal and seasonal.get('amplitude', 0) > 0:
        params['seasonal_params'] = {
            'amplitude': seasonal['amplitude'],
            'period': seasonal.get('period', 365),
            'peak_day': seasonal.get('peak_day', 0),
        }
    
    # Add waning immunity (pass as waning_params dict for master model)
    if waning:
        omega = waning.get('omega', 0.0)
        if omega > 0:
            params['waning_params'] = {'omega': omega}
        # Check for age-specific waning
        elif 'omega_young' in waning:
            params['waning_params'] = {
                'omega_young': waning['omega_young'],
                'omega_middle': waning['omega_middle'],
                'omega_elderly': waning['omega_elderly'],
            }
    
    # Add initial conditions if specified
    if 'initial_conditions' in scenario:
        ic = scenario['initial_conditions']
        if 'E_by_age' in ic:
            params['E0_by_age'] = ic['E_by_age']
        if 'I_by_age' in ic:
            params['I0_by_age'] = ic['I_by_age']
        if 'R_by_age' in ic:
            params['R0_by_age'] = ic['R_by_age']
    
    return params


def list_scenarios() -> List[str]:
    """Return list of available scenario names."""
    return list(SCENARIO_REGISTRY.keys())


def describe_scenario(scenario_name: str) -> str:
    """Return detailed description of a scenario."""
    if scenario_name not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    s = SCENARIO_REGISTRY[scenario_name]
    healthcare = s['healthcare_system']
    vaccination = s['vaccination']
    
    # Handle vaccination format
    if isinstance(vaccination, dict):
        vax_desc = vaccination.get('description', 'Custom')
        vax_cov = vaccination.get('coverage', [0, 0, 0])
    else:
        vax_desc = 'Custom'
        vax_cov = vaccination
    
    lines = [
        f"=== {s['name']} ===",
        f"Description: {s['description']}",
        f"",
        f"Transmission:",
        f"  β_base = {s['beta_base']:.2f}",
        f"  Seasonality: {s.get('seasonal_params', {}).get('description', 'None')}",
        f"",
        f"Healthcare System: {healthcare['name']}",
        f"  Ward capacity: {healthcare['ward_capacity']}",
        f"  ICU capacity: {healthcare['icu_capacity']}",
        f"  Population: {sum(healthcare['age_pops']):,}",
        f"",
        f"Vaccination: {vax_desc}",
        f"  Coverage: {vax_cov}",
        f"  Efficacy: {s.get('VE', 0.7):.0%}",
        f"",
        f"Interventions: {len(s.get('interventions', []))} phase(s)",
        f"Duration: {s.get('Tmax', 200)} days",
    ]
    return '\n'.join(lines)


def get_healthcare_systems() -> Dict[str, Dict]:
    """Return dictionary of available healthcare system configurations."""
    return {
        'small': HEALTHCARE_SYSTEM_SMALL,
        'rural': HEALTHCARE_SYSTEM_RURAL,
        'suburban': HEALTHCARE_SYSTEM_SUBURBAN,
        'urban': HEALTHCARE_SYSTEM_URBAN,
        'metropolitan': HEALTHCARE_SYSTEM_METROPOLITAN,
        'well_resourced': HEALTHCARE_SYSTEM_WELL_RESOURCED,
        'resource_limited': HEALTHCARE_SYSTEM_RESOURCE_LIMITED,
        'surge_mild': HEALTHCARE_SYSTEM_SURGE_MILD,
        'surge_major': HEALTHCARE_SYSTEM_SURGE_MAJOR,
    }


def get_vaccination_strategies() -> Dict[str, List[float]]:
    """
    Return vaccination strategies as {name: coverage_list} format.
    
    This normalizes VACCINATION_STRATEGIES to always return coverage arrays,
    compatible with compare_vaccination_strategies() and other helper functions.
    
    Returns:
        Dictionary mapping strategy names to [young, middle, elderly] coverage lists.
        
    Example:
        strategies = get_vaccination_strategies()
        # {'none': [0.0, 0.0, 0.0], 'elderly_priority': [0.1, 0.3, 0.8], ...}
    """
    result = {}
    for name, value in VACCINATION_STRATEGIES.items():
        if isinstance(value, dict):
            result[name] = value.get('coverage', [0.0, 0.0, 0.0])
        else:
            result[name] = value
    return result


def validate_age_params(age_params: List[Dict]) -> bool:
    """
    Validate that age parameter dictionaries contain all required keys.
    
    Returns True if valid, raises ValueError with details if not.
    """
    required_keys = {
        'alpha', 'sigma', 'eta', 'eta_icu',
        'gamma_I', 'gamma_X', 'gamma_ward', 'gamma_icu',
        'mu_I', 'mu_X', 'mu_ward', 'mu_icu',
    }
    
    for i, params in enumerate(age_params):
        missing = required_keys - set(params.keys())
        if missing:
            age_label = AGE_LABELS_SHORT[i] if i < len(AGE_LABELS_SHORT) else f"Age group {i}"
            raise ValueError(f"{age_label} params missing keys: {missing}")
    
    return True


def create_custom_scenario(
    name: str,
    beta_base: float,
    healthcare_system: Dict,
    vaccination_coverage: List[float],
    **kwargs
) -> Dict[str, Any]:
    """
    Create a custom scenario configuration.
    
    Args:
        name: Scenario name
        beta_base: Baseline transmission rate
        healthcare_system: Healthcare system config dict
        vaccination_coverage: [young, middle, elderly] coverage rates
        **kwargs: Additional parameters (age_params, interventions, etc.)
        
    Returns:
        Scenario configuration dictionary
    """
    scenario = {
        'name': name,
        'description': kwargs.get('description', f'Custom scenario: {name}'),
        'beta_base': beta_base,
        'age_params': kwargs.get('age_params', AGE_PARAMS_DEFAULT),
        'contact_matrix': kwargs.get('contact_matrix', CONTACT_MATRIX_DEFAULT),
        'healthcare_system': healthcare_system,
        'seasonal_params': kwargs.get('seasonal_params', SEASONAL_PARAMS_NONE),
        'waning_params': kwargs.get('waning_params', WANING_NONE),
        'interventions': kwargs.get('interventions', INTERVENTION_NONE),
        'vaccination': {
            'coverage': vaccination_coverage,
            'description': 'Custom coverage',
        },
        'VE': kwargs.get('VE', 0.7),
        'Tmax': kwargs.get('Tmax', 200),
    }
    
    if 'initial_conditions' in kwargs:
        scenario['initial_conditions'] = kwargs['initial_conditions']
    
    return scenario


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================
# Preserve commonly-used legacy exports

AGE_POPS_DEFAULT = [3000, 5000, 2000]  # Original default (10,000 total)
AGE_POPS_REGIONAL_DEFAULT = HEALTHCARE_SYSTEM_SUBURBAN['age_pops']


