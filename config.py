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

DEFAULT_CAPACITY_PARAMS = {
    'ward_capacity': 80,
    'icu_capacity': 20,
    'total_capacity': 100,
    'hill_coef_ward': 4,
    'hill_coef_icu': 4
}

AGE_POPS_DEFAULT = [3000, 5000, 2000]  # Original default (10,000 total)


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
    'waning_destination': 'S_vax',        # 'S' = return to vaccination susceptibility, 'S_vax' = partial protection
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

# =======================================================================
# 4.1 Empirical Parameters (COVID-calibrated)
# =======================================================================

YOUNG_PARAMS_EMPIRICAL = {
    # Latent period (E → I)
    'alpha': 0.2,             # ~5 day latent period
    
    # Disease progression
    'sigma': 0.02,            # 2% progress to severe per day
    'eta': 0.05,              # ward admission attempt rate (per day) for severe cases
    'eta_icu': 0.02,          # ICU escalation rate (per day) for ward patients
    
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
}

MIDDLE_PARAMS_EMPIRICAL = {
    'alpha': 0.2,
    'sigma': 0.08,            # 8% progress to severe per day
    'eta': 0.15,              # ward admission attempt rate (per day)
    'eta_icu': 0.10,          # ICU escalation rate (per day)
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
    'mu_icu': 0.02
}

ELDERLY_PARAMS_EMPIRICAL = {
    'alpha': 0.18,            # slightly longer latent period
    'sigma': 0.15,            # 15% progress to severe per day
    'eta': 0.35,              # ward admission attempt rate (per day)
    'eta_icu': 0.25,          # ICU escalation rate (per day)
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
    'mu_icu': 0.04
}

AGE_PARAMS_EMPIRICAL = [YOUNG_PARAMS_EMPIRICAL, MIDDLE_PARAMS_EMPIRICAL, ELDERLY_PARAMS_EMPIRICAL]


# =======================================================================
# 4.2 Teaching Parameters (exaggerated effects)
# =======================================================================

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
    'mu_icu': 0.008
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
    'mu_icu': 0.04
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
    'mu_icu': 0.12
}

AGE_PARAMS_TEACHING = [YOUNG_PARAMS_TEACHING, MIDDLE_PARAMS_TEACHING, ELDERLY_PARAMS_TEACHING]


# =======================================================================
# 4.3 Parameter Set Selection
# =======================================================================

USE_EMPIRICAL_PARAMS = True  # Toggle between empirical and teaching modes

AGE_PARAMS_DEFAULT = AGE_PARAMS_EMPIRICAL if USE_EMPIRICAL_PARAMS else AGE_PARAMS_TEACHING

# Active individual params for backward compatibility
YOUNG_PARAMS_ACTIVE = YOUNG_PARAMS_EMPIRICAL if USE_EMPIRICAL_PARAMS else YOUNG_PARAMS_TEACHING
MIDDLE_PARAMS_ACTIVE = MIDDLE_PARAMS_EMPIRICAL if USE_EMPIRICAL_PARAMS else MIDDLE_PARAMS_TEACHING
ELDERLY_PARAMS_ACTIVE = ELDERLY_PARAMS_EMPIRICAL if USE_EMPIRICAL_PARAMS else ELDERLY_PARAMS_TEACHING


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
    'ward_capacity': 800,
    'icu_capacity': 200,
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
    'D_by_age': [0, 0, 0],
    
    # Vaccinated compartments (Three-Factor Model)
    'S_vax_by_age': [0, 0, 0],   # initially susceptible vaccinated
    'E_vax_by_age': [0, 0, 0],   # exposed vaccinated (breakthrough)
    'I_vax_by_age': [0, 0, 0],   # infectious vaccinated (breakthrough)
    'X_vax_by_age': [0, 0, 0],   # severe vaccinated
    'H_ward_vax_by_age': [0, 0, 0],  # ward vaccinated
    'H_icu_vax_by_age': [0, 0, 0],   # ICU vaccinated
    'R_vax_by_age': [0, 0, 0],   # recovered vaccinated
    'D_vax_by_age': [0, 0, 0],   # dead vaccinated
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

# =======================================================================
# 9.1 Seasonal Transmission Patterns
# =======================================================================

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


# =======================================================================
# 9.2 Waning Immunity Presets
# =======================================================================

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


# =======================================================================
# 9.4 Demographic Parameters (Births and Background Deaths)
# =======================================================================
# For long-term simulations, open population dynamics with births and
# non-disease (background) mortality can be enabled.

DEMOGRAPHIC_PARAMS_NONE = None  # Closed population (default)

DEMOGRAPHIC_PARAMS_DEFAULT = {
    # Birth rate: daily births per 1000 population (crude birth rate)
    # Global average ~18/1000/year = 0.049/day per 1000 = 0.000049 per capita/day
    'birth_rate': 0.000049,
    
    # Age distribution of births (where newborns enter S compartment)
    # Default: all births enter youngest age group
    'birth_age_distribution': [1.0, 0.0, 0.0],
    
    # Background mortality rate: age-specific daily death rates (non-disease)
    # Based on typical life tables (per capita per day)
    # Young (0-19): ~0.5/1000/year = 0.0000014/day
    # Middle (20-64): ~3/1000/year = 0.0000082/day  
    # Elderly (65+): ~40/1000/year = 0.00011/day
    'mu_background': [0.0000014, 0.0000082, 0.00011],
    
    # Optional: neonatal vaccination rate (fraction of newborns vaccinated)
    # If > 0, births are split between S and S_vax
    'neonatal_vaccination_rate': 0.0,
}

# High-income country demographics (lower birth/death rates)
DEMOGRAPHIC_PARAMS_HIGH_INCOME = {
    'birth_rate': 0.000030,  # ~11/1000/year
    'birth_age_distribution': [1.0, 0.0, 0.0],
    'mu_background': [0.0000008, 0.0000055, 0.00012],  # Lower young/middle, similar elderly
    'neonatal_vaccination_rate': 0.0,
    'description': 'High-income country demographics (low birth rate, low mortality)',
}

# Low-income country demographics (higher birth/death rates)
DEMOGRAPHIC_PARAMS_LOW_INCOME = {
    'birth_rate': 0.000082,  # ~30/1000/year
    'birth_age_distribution': [1.0, 0.0, 0.0],
    'mu_background': [0.0000055, 0.00011, 0.00022],  # Higher mortality all ages
    'neonatal_vaccination_rate': 0.0,
    'description': 'Low-income country demographics (high birth rate, high mortality)',
}

# Endemic equilibrium demographics (balanced births and deaths)
DEMOGRAPHIC_PARAMS_EQUILIBRIUM = {
    'birth_rate': 0.000049,  # Matched to approximate death rate
    'birth_age_distribution': [1.0, 0.0, 0.0],
    'mu_background': [0.0000014, 0.0000082, 0.00011],
    'neonatal_vaccination_rate': 0.0,
    'description': 'Demographics balanced for stable population (birth rate ≈ death rate)',
}

# Neonatal vaccination scenario
DEMOGRAPHIC_PARAMS_NEONATAL_VAX = {
    'birth_rate': 0.000049,
    'birth_age_distribution': [1.0, 0.0, 0.0],
    'mu_background': [0.0000014, 0.0000082, 0.00011],
    'neonatal_vaccination_rate': 0.8,  # 80% of newborns vaccinated
    'description': 'Demographics with 80% neonatal vaccination (e.g., BCG, HepB)',
}

# Demographic parameter presets registry
DEMOGRAPHIC_PRESETS = {
    'none': DEMOGRAPHIC_PARAMS_NONE,
    'default': DEMOGRAPHIC_PARAMS_DEFAULT,
    'high_income': DEMOGRAPHIC_PARAMS_HIGH_INCOME,
    'low_income': DEMOGRAPHIC_PARAMS_LOW_INCOME,
    'equilibrium': DEMOGRAPHIC_PARAMS_EQUILIBRIUM,
    'neonatal_vax': DEMOGRAPHIC_PARAMS_NEONATAL_VAX,
}


# =======================================================================
# 9.3 Policy Intervention Templates
# =======================================================================
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
    }
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

SCENARIO_LATE_ONSET_MILD_WAVE = {
    'name': 'Late-Onset Mild Wave',
    'description': 'Mild outbreak with late seasonal peak and moderate interventions',
    'beta_base': TRANSMISSION_PRESETS['very_mild']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_NONE,
    'vaccination': VACCINATION_STRATEGIES['none'],
    'VE': VACCINE_EFFICACY_PRESETS['moderate'],
    'Tmax': 300,
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
    'demographic_params': DEMOGRAPHIC_PARAMS_EQUILIBRIUM,  # Open population for endemic
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

SCENARIO_NOVEL_PATHOGEN = {
    'name': 'Novel Pathogen Emergence',
    'description': 'Unknown pathogen with high uncertainty - aggressive response, no prior immunity',
    'beta_base': TRANSMISSION_PRESETS['high']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,  # Unknown, assume no waning initially
    'interventions': INTERVENTION_EARLY_STRONG,  # Aggressive response to unknown threat
    'vaccination': VACCINATION_STRATEGIES['none'],  # No vaccine available initially
    'vaccine_profile': None,  # No vaccine
    'VE': 0.0,
    'Tmax': 200,
    'notes': 'Simulates first wave of novel pathogen before vaccines available',
}

SCENARIO_VACCINE_ROLLOUT_PHASED = {
    'name': 'Phased Vaccine Rollout',
    'description': 'Realistic vaccine campaign: elderly first, then middle-age, then young',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_MILD,
    'waning_params': WANING_MODERATE,
    'interventions': INTERVENTION_SUSTAINED_MODERATE,  # Moderate restrictions during rollout
    'vaccination': {
        'coverage': [0.1, 0.2, 0.5],  # Initial coverage (elderly prioritized)
        'description': 'Phased rollout - elderly priority',
    },
    'vaccine_profile': 'mrna_original',
    'vaccination_rate': [0.002, 0.003, 0.005],  # Higher rate for elderly
    'Tmax': 365,
    'notes': 'Models realistic phased vaccination campaign over one year',
}

SCENARIO_VARIANT_EMERGENCE = {
    'name': 'Variant Emergence Mid-Outbreak',
    'description': 'New variant with immune escape emerges after initial wave',
    'beta_base': TRANSMISSION_PRESETS['severe']['beta_base'],  # More transmissible variant
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_MODERATE,
    'waning_params': WANING_VERY_FAST,  # Immune escape causes faster effective waning
    'interventions': INTERVENTION_TIERED_ESCALATING,
    'vaccination': VACCINATION_STRATEGIES['herd_immunity_target'],
    'vaccine_profile': 'mrna_omicron',  # Reduced efficacy against variant
    'initial_conditions': INITIAL_CONDITIONS_PRESETS['post_wave'],  # Starting with immunity
    'Tmax': 300,
    'notes': 'Simulates immune-evasive variant after population has prior immunity',
}

SCENARIO_CAPACITY_COLLAPSE = {
    'name': 'Healthcare Capacity Collapse',
    'description': 'Severe outbreak overwhelming minimal healthcare infrastructure',
    'beta_base': TRANSMISSION_PRESETS['severe']['beta_base'],
    'age_params': AGE_PARAMS_TEACHING,  # Exaggerated severity for clear demonstration
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_RESOURCE_LIMITED,  # Very limited capacity
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_DELAYED_MODERATE,  # Delayed response worsens outcome
    'vaccination': VACCINATION_STRATEGIES['none'],  # No vaccine available
    'vaccine_profile': None,
    'VE': 0.0,
    'Tmax': 200,
    'notes': 'Demonstrates excess mortality from capacity constraints',
}

SCENARIO_ENDEMIC_VACCINATION = {
    'name': 'Endemic with Ongoing Vaccination',
    'description': 'Long-term endemic equilibrium with seasonal vaccination campaigns',
    'beta_base': TRANSMISSION_PRESETS['mild']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_STRONG,  # Strong seasonal pattern
    'waning_params': WANING_MODERATE,  # Natural immunity wanes
    'interventions': INTERVENTION_NONE,  # No interventions in endemic phase
    'vaccination': VACCINATION_STRATEGIES['elderly_priority'],
    'vaccine_profile': 'influenza_typical',
    'vaccination_rate': [0.001, 0.001, 0.002],  # Ongoing vaccination
    'vaccine_waning_params': {'omega_vax': 0.003, 'waning_destination': 'S_vax'},
    'demographic_params': DEMOGRAPHIC_PARAMS_EQUILIBRIUM,  # Open population for endemic
    'Tmax': 730,  # 2 years
    'notes': 'Models endemic dynamics with annual vaccination similar to influenza',
}

SCENARIO_SCHOOL_REOPENING = {
    'name': 'School Reopening Policy Test',
    'description': 'Comparison of transmission dynamics with schools open vs closed',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,  # Schools open (use SCHOOL_CLOSURE for closed)
    'healthcare_system': HEALTHCARE_SYSTEM_SUBURBAN,
    'seasonal_params': SEASONAL_PARAMS_MODERATE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_NONE,  # No additional interventions
    'vaccination': VACCINATION_STRATEGIES['young_priority'],
    'vaccine_profile': 'mrna_original',
    'initial_conditions': INITIAL_CONDITIONS_PRESETS['young_seed'],
    'Tmax': 180,
    'notes': 'Run with CONTACT_MATRIX_DEFAULT (open) and CONTACT_MATRIX_SCHOOL_CLOSURE (closed) to compare',
}

# Sensitivity analysis base scenarios
SCENARIO_SENSITIVITY_TRANSMISSION = {
    'name': 'Transmission Sensitivity Base',
    'description': 'Base scenario for beta sensitivity analysis',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],  # Will be varied
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_NONE,
    'vaccination': VACCINATION_STRATEGIES['balanced_risk'],
    'vaccine_profile': 'mrna_original',
    'Tmax': 200,
}

SCENARIO_SENSITIVITY_CAPACITY = {
    'name': 'Capacity Sensitivity Base',
    'description': 'Base scenario for hospital capacity sensitivity analysis',
    'beta_base': TRANSMISSION_PRESETS['high']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,  # Will be varied
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_NONE,
    'vaccination': VACCINATION_STRATEGIES['balanced_risk'],
    'vaccine_profile': 'mrna_original',
    'Tmax': 200,
}

SCENARIO_POPULATION_DYNAMICS = {
    'name': 'Open Population Dynamics',
    'description': 'Long-term simulation with births and background deaths for endemic analysis',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_MODERATE,
    'waning_params': WANING_MODERATE,
    'interventions': INTERVENTION_NONE,
    'vaccination': VACCINATION_STRATEGIES['balanced_risk'],
    'vaccine_profile': 'mrna_original',
    'demographic_params': DEMOGRAPHIC_PARAMS_EQUILIBRIUM,
    'Tmax': 1095,  # 3 years
    'notes': 'Demonstrates open population dynamics with births/deaths for long-term endemic equilibrium',
}

SCENARIO_NEONATAL_VACCINATION = {
    'name': 'Neonatal Vaccination Program',
    'description': 'Long-term endemic with neonatal vaccination (e.g., HepB model)',
    'beta_base': TRANSMISSION_PRESETS['mild']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_SLOW,
    'interventions': INTERVENTION_NONE,
    'vaccination': VACCINATION_STRATEGIES['none'],  # No adult vaccination
    'demographic_params': DEMOGRAPHIC_PARAMS_NEONATAL_VAX,
    'Tmax': 1095,  # 3 years
    'notes': 'Models vaccination at birth only (e.g., HepB, BCG programs)',
}


# Registry of all scenarios for easy access
SCENARIO_REGISTRY = {
    'baseline': SCENARIO_BASELINE,
    'late_onset': SCENARIO_LATE_ONSET_MILD_WAVE,
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
    'novel_pathogen': SCENARIO_NOVEL_PATHOGEN,
    'vaccine_rollout_phased': SCENARIO_VACCINE_ROLLOUT_PHASED,
    'variant_emergence': SCENARIO_VARIANT_EMERGENCE,
    'capacity_collapse': SCENARIO_CAPACITY_COLLAPSE,
    'endemic_vaccination': SCENARIO_ENDEMIC_VACCINATION,
    'school_reopening': SCENARIO_SCHOOL_REOPENING,
    'sensitivity_transmission': SCENARIO_SENSITIVITY_TRANSMISSION,
    'sensitivity_capacity': SCENARIO_SENSITIVITY_CAPACITY,
    'population_dynamics': SCENARIO_POPULATION_DYNAMICS,
    'neonatal_vaccination': SCENARIO_NEONATAL_VACCINATION,
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

