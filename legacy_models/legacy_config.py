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
    'VE': 0.7,            # vaccine efficacy (leaky model, legacy - use VACCINE_EFFICACY_PARAMS for three-factor)
    'theta_vax': 0.5,     # relative infectiousness of vaccinated infected individuals (breakthrough)
}


# ============================================================================
# SECTION 1B: THREE-FACTOR VACCINE MODEL PARAMETERS
# ============================================================================
# Three-factor vaccine model separates efficacy into:
# - VE_infection: Efficacy against infection (reduces susceptibility/force of infection)
# - VE_severe: Efficacy against severe disease (reduces progression I → X)
# - VE_death: Efficacy against death (reduces mortality rates)
#
# This provides more realistic modeling of real-world vaccines where
# protection against death often exceeds protection against infection.

VACCINE_EFFICACY_PARAMS = {
    # Three-factor efficacy parameters
    'VE_infection': 0.6,    # 60% reduction in susceptibility to infection
    'VE_severe': 0.8,       # 80% reduction in progression to severe disease
    'VE_death': 0.9,        # 90% reduction in mortality
    
    # Vaccinated breakthrough infectiousness
    'theta_vax': 0.5,       # breakthrough infections are 50% as infectious as unvaccinated
}

# Vaccination rate parameters for dynamic vaccination
VACCINATION_RATE_PARAMS = {
    'vaccination_rate': 0.0,          # daily rate of vaccination (fraction of S → S_vax)
    'vaccination_rate_by_age': None,  # age-specific rates [young, middle, elderly] or None for uniform
}

# Vaccine immunity waning parameters
VACCINE_WANING_PARAMS = {
    'omega_vax': 0.0,                 # vaccine immunity waning rate (1/days), 0 = no waning
    'omega_vax_by_age': None,         # age-specific waning [young, middle, elderly] or None for uniform
    'waning_destination': 'S',        # 'S' = return to fully susceptible, 'S_vax' = partial protection
}

# Preset vaccine profiles for different vaccine types
VACCINE_PROFILES = {
    'mrna_original': {
        'description': 'mRNA vaccine (Pfizer/Moderna) vs original strain',
        'VE_infection': 0.80,
        'VE_severe': 0.90,
        'VE_death': 0.95,
        'theta_vax': 0.3,
        'omega_vax': 0.002,   # ~500 day waning
    },
    'mrna_omicron': {
        'description': 'mRNA vaccine vs Omicron variant (immune escape)',
        'VE_infection': 0.30,
        'VE_severe': 0.70,
        'VE_death': 0.85,
        'theta_vax': 0.6,
        'omega_vax': 0.004,   # ~250 day waning
    },
    'adenovirus': {
        'description': 'Adenovirus vaccine (AZ/J&J)',
        'VE_infection': 0.65,
        'VE_severe': 0.85,
        'VE_death': 0.90,
        'theta_vax': 0.4,
        'omega_vax': 0.003,   # ~333 day waning
    },
    'inactivated': {
        'description': 'Inactivated virus vaccine (Sinovac/Sinopharm)',
        'VE_infection': 0.50,
        'VE_severe': 0.70,
        'VE_death': 0.80,
        'theta_vax': 0.5,
        'omega_vax': 0.004,   # ~250 day waning
    },
    'influenza_typical': {
        'description': 'Typical seasonal influenza vaccine',
        'VE_infection': 0.40,
        'VE_severe': 0.60,
        'VE_death': 0.75,
        'theta_vax': 0.6,
        'omega_vax': 0.003,   # ~333 day waning
    },
    'ideal': {
        'description': 'Ideal/sterilizing vaccine',
        'VE_infection': 0.95,
        'VE_severe': 0.99,
        'VE_death': 0.99,
        'theta_vax': 0.1,
        'omega_vax': 0.0,     # no waning
    },
    'minimal': {
        'description': 'Minimal/suboptimal vaccine',
        'VE_infection': 0.30,
        'VE_severe': 0.50,
        'VE_death': 0.60,
        'theta_vax': 0.7,
        'omega_vax': 0.005,   # ~200 day waning
    },
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
    'gamma_X': 0.2,           # ~5 day severe phase (recovery from X)
    'gamma_X_admit': 0.5,     # admission rate from X_queued to X_admitted (~2 day wait if capacity available)
    'gamma_ward': 0.2,        # ~5 day ward stay
    'gamma_icu': 0.1,         # ~10 day ICU stay
    
    # Mortality rates (daily probability)
    'mu_I': 0.0001,           # near-zero community mortality
    'mu_X': 0.002,            # 0.2% if severe and admitted (treated)
    'mu_X_untreated': 0.006,  # 0.6% if queued/untreated (3x treated rate)
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
    'gamma_X_admit': 0.5,     # admission rate from X_queued to X_admitted
    'gamma_ward': 0.14,       # ~7 day ward stay
    'gamma_icu': 0.08,        # ~12 day ICU stay
    'mu_I': 0.001,
    'mu_X': 0.008,            # treated mortality
    'mu_X_untreated': 0.024,  # 3x when untreated
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
    'gamma_X_admit': 0.5,     # admission rate from X_queued to X_admitted
    'gamma_ward': 0.10,       # ~10 day ward stay
    'gamma_icu': 0.05,        # ~20 day ICU stay
    'mu_I': 0.005,
    'mu_X': 0.025,            # treated mortality
    'mu_X_untreated': 0.10,   # 4x when untreated
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
    'gamma_X_admit': 0.5,     # admission rate from X_queued to X_admitted
    'gamma_ward': 0.25,
    'gamma_icu': 0.15,
    'mu_I': 0.001,
    'mu_X': 0.01,             # treated mortality
    'mu_X_untreated': 0.015,  # untreated mortality
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
    'gamma_X_admit': 0.5,     # admission rate from X_queued to X_admitted
    'gamma_ward': 0.2,
    'gamma_icu': 0.12,
    'mu_I': 0.01,
    'mu_X': 0.02,             # treated mortality
    'mu_X_untreated': 0.06,   # untreated mortality
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
    'gamma_X_admit': 0.5,     # admission rate from X_queued to X_admitted
    'gamma_ward': 0.15,
    'gamma_icu': 0.08,
    'mu_I': 0.03,
    'mu_X': 0.15,             # treated mortality
    'mu_X_untreated': 0.45,   # untreated mortality
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
    
    # Age-structured model (unvaccinated compartments)
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

# ============================================================================
# 9.1 Seasonal Transmission Patterns
# ============================================================================

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


# ============================================================================
# 9.2 IMMUNITY WANING SCENARIOS
# ============================================================================

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


# ============================================================================
# 9.3 INTERVENTION SCENARIOS
# ============================================================================
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
    # Legacy format for backward compatibility
    'No vaccination': [0.0, 0.0, 0.0],
    'Uniform 30%': [0.3, 0.3, 0.3],
    'Elderly priority': [0.1, 0.2, 0.7],
    'Young priority': [0.7, 0.2, 0.1],
    'Middle priority': [0.1, 0.7, 0.2],
}


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================
# Preserve commonly-used legacy exports

AGE_POPS_DEFAULT = [3000, 5000, 2000]  # Original default (10,000 total)
AGE_POPS_REGIONAL_DEFAULT = HEALTHCARE_SYSTEM_SUBURBAN['age_pops']

# ============================================================================
# Legacy Config Helpers
# ============================================================================
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


