"""
Scenario data and configuration presets for model simulations.

This module contains all scenario-specific data, parameter presets, and configurations.
These are pure data definitions with no logic. Use scenario_helpers.py for functions
that operate on these configurations.

It's honestly the bread and butter of the model. You can do so much in here.

Configurations are organized hierarchically:
1. CORE PARAMETERS - Fundamental simulation and model parameters
2. TRANSMISSION RATE PRESETS - Beta values for different transmissibility scenarios
3. AGE GROUP DEFINITIONS - Labels and counts for age-structured modeling
4. AGE-SPECIFIC DISEASE PARAMETERS - Epidemiological rates by age group  
5. DIFFERENTIAL MORTALITY PARAMETERS - Mortality multipliers when care is denied
6. CONTACT MATRICES - Mixing patterns between age groups
7. HEALTHCARE SYSTEM CONFIGURATIONS - Bundled capacity + population presets
8. INITIAL CONDITIONS - Various outbreak seeding scenarios
9. TIME-VARYING PARAMETERS - Seasonality, interventions, waning immunity
10. VACCINATION STRATEGIES - Coverage allocation patterns
11. COMPLETE SCENARIO BUNDLES - Ready-to-use simulation configurations
12. SENSITIVITY ANALYSIS RANGES - Parameter ranges for sensitivity studies
"""

import numpy as np
from typing import List, Optional
from model_types import (
    SimParams,
    CapacityParams,
    AgeParams,
    ContactMatrix,
    VaccineEfficacyParams,
    VaccineWaningParams,
    SeasonalParams,
    Intervention,
    DemographicParams,
    DifferentialMortalityParams
)


# ============================================================================
# SECTION 1: CORE SIMULATION PARAMETERS
# ============================================================================

DEFAULT_SIM_PARAMS: SimParams = {
    'Tmax': 200,          # simulation duration in days
    'time_step': 0.1,     # Euler integration time step
    'theta_X': 0.5,       # relative infectiousness of X compartment
    'theta_H': 0.3,       # relative infectiousness of H (ward + ICU) compartment
    'VE': 0.7,            # vaccine efficacy (leaky model, legacy - use VACCINE_EFFICACY_PARAMS for three-factor)
    'theta_vax': 0.5,     # relative infectiousness of vaccinated infected individuals (breakthrough)
}

DEFAULT_CAPACITY_PARAMS: CapacityParams = {
    'ward_capacity': 80,
    'icu_capacity': 20,
    'total_capacity': 100,
    'hill_coef_ward': 4,
    'hill_coef_icu': 4
}

AGE_POPS_DEFAULT = [3000, 5000, 2000]  # Original default (10,000 total)


# ============================================================================
# SECTION 1.1: THREE-FACTOR VACCINE MODEL PARAMETERS
# ============================================================================

VACCINE_EFFICACY_PARAMS: VaccineEfficacyParams = {
    'VE_infection': 0.6,    # 60% reduction in susceptibility to infection
    'VE_severe': 0.8,       # 80% reduction in progression to severe disease
    'VE_death': 0.9,        # 90% reduction in mortality
    'theta_vax': 0.5,       # breakthrough infections are 50% as infectious as unvaccinated
}

VACCINATION_RATE_PARAMS = {
    'vaccination_rate': 0.0,          # daily rate of vaccination (fraction of S → S_vax)
    'vaccination_rate_by_age': None,  # age-specific rates [young, middle, elderly] or None for uniform
}

VACCINE_WANING_PARAMS: VaccineWaningParams = {
    'omega_vax': 0.0,                 # vaccine immunity waning rate (1/days), 0 = no waning
    'omega_vax_by_age': None,         # age-specific waning [young, middle, elderly] or None for uniform
    'waning_destination': 'S_vax',    # 'S' = return to vaccination susceptibility, 'S_vax' = partial protection
}

VACCINE_PROFILES = {
    'mrna_original': {
        'description': 'mRNA vaccine (Pfizer/Moderna) vs original strain',
        'VE_infection': 0.80,
        'VE_severe': 0.90,
        'VE_death': 0.95,
        'theta_vax': 0.3,
        'omega_vax': 0.002,
    },
    'mrna_omicron': {
        'description': 'mRNA vaccine vs Omicron variant (immune escape)',
        'VE_infection': 0.30,
        'VE_severe': 0.70,
        'VE_death': 0.85,
        'theta_vax': 0.6,
        'omega_vax': 0.004,
    },
    'adenovirus': {
        'description': 'Adenovirus vaccine (AZ/J&J)',
        'VE_infection': 0.65,
        'VE_severe': 0.85,
        'VE_death': 0.90,
        'theta_vax': 0.4,
        'omega_vax': 0.003,
    },
    'inactivated': {
        'description': 'Inactivated virus vaccine (Sinovac/Sinopharm)',
        'VE_infection': 0.50,
        'VE_severe': 0.70,
        'VE_death': 0.80,
        'theta_vax': 0.5,
        'omega_vax': 0.004,
    },
    'influenza_typical': {
        'description': 'Typical seasonal influenza vaccine',
        'VE_infection': 0.40,
        'VE_severe': 0.60,
        'VE_death': 0.75,
        'theta_vax': 0.6,
        'omega_vax': 0.003,
    },
    'ideal': {
        'description': 'Ideal/sterilizing vaccine',
        'VE_infection': 0.95,
        'VE_severe': 0.99,
        'VE_death': 0.99,
        'theta_vax': 0.1,
        'omega_vax': 0.0,
    },
    'minimal': {
        'description': 'Minimal/suboptimal vaccine',
        'VE_infection': 0.30,
        'VE_severe': 0.50,
        'VE_death': 0.60,
        'theta_vax': 0.7,
        'omega_vax': 0.005,
    },
}

SIM_DURATION_PRESETS = {
    'short': 100,
    'standard': 200,
    'extended': 365,
    'endemic': 730,
    'long_term': 1095,
}


# ============================================================================
# SECTION 2: TRANSMISSION RATE (BETA) PRESETS
# ============================================================================

TRANSMISSION_PRESETS = {
    'very_mild': {'beta_base': 0.12, 'approx_R0': 1.2, 'description': 'Subcritical or barely spreading'},
    'mild': {'beta_base': 0.18, 'approx_R0': 1.5, 'description': 'Low transmission (seasonal influenza)'},
    'moderate': {'beta_base': 0.28, 'approx_R0': 2.5, 'description': 'Moderate transmission (early COVID-19)'},
    'high': {'beta_base': 0.38, 'approx_R0': 3.5, 'description': 'High transmission (Delta variant)'},
    'severe': {'beta_base': 0.45, 'approx_R0': 4.5, 'description': 'Very high transmission (Omicron)'},
    'extreme': {'beta_base': 0.55, 'approx_R0': 6.0, 'description': 'Extreme transmission (measles)'},
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

# Empirical Parameters (COVID-calibrated)
YOUNG_PARAMS_EMPIRICAL: AgeParams = {
    'alpha': 0.2, 'sigma': 0.02, 'eta': 0.05, 'eta_icu': 0.02,
    'gamma_I': 0.14, 'gamma_X': 0.2, 'gamma_X_admit': 0.5, 'gamma_H': 0.2, 'gamma_ward': 0.2, 'gamma_icu': 0.1,
    'mu_I': 0.0001, 'mu_X': 0.002, 'mu_X_untreated': 0.006, 'mu_H': 0.005, 'mu_ward': 0.001, 'mu_ward_denied_icu': 0.02, 'mu_icu': 0.005,
}

MIDDLE_PARAMS_EMPIRICAL: AgeParams = {
    'alpha': 0.2, 'sigma': 0.08, 'eta': 0.15, 'eta_icu': 0.10,
    'gamma_I': 0.12, 'gamma_X': 0.15, 'gamma_X_admit': 0.5, 'gamma_H': 0.15, 'gamma_ward': 0.14, 'gamma_icu': 0.08,
    'mu_I': 0.001, 'mu_X': 0.008, 'mu_X_untreated': 0.024, 'mu_H': 0.02, 'mu_ward': 0.005, 'mu_ward_denied_icu': 0.06, 'mu_icu': 0.02,
}

ELDERLY_PARAMS_EMPIRICAL: AgeParams = {
    'alpha': 0.18, 'sigma': 0.15, 'eta': 0.35, 'eta_icu': 0.25,
    'gamma_I': 0.10, 'gamma_X': 0.10, 'gamma_X_admit': 0.5, 'gamma_H': 0.10, 'gamma_ward': 0.10, 'gamma_icu': 0.05,
    'mu_I': 0.005, 'mu_X': 0.025, 'mu_X_untreated': 0.10, 'mu_H': 0.05, 'mu_ward': 0.015, 'mu_ward_denied_icu': 0.12, 'mu_icu': 0.04,
}

AGE_PARAMS_EMPIRICAL: List[AgeParams] = [YOUNG_PARAMS_EMPIRICAL, MIDDLE_PARAMS_EMPIRICAL, ELDERLY_PARAMS_EMPIRICAL]

# Teaching Parameters (exaggerated effects)
YOUNG_PARAMS_TEACHING: AgeParams = {
    'alpha': 0.2, 'sigma': 0.1, 'eta': 0.2, 'eta_icu': 0.05,
    'gamma_I': 0.12, 'gamma_X': 0.18, 'gamma_X_admit': 0.5, 'gamma_H': 0.2, 'gamma_ward': 0.25, 'gamma_icu': 0.15,
    'mu_I': 0.001, 'mu_X': 0.01, 'mu_X_untreated': 0.015, 'mu_H': 0.02, 'mu_ward': 0.003, 'mu_ward_denied_icu': 0.025, 'mu_icu': 0.008,
}

MIDDLE_PARAMS_TEACHING: AgeParams = {
    'alpha': 0.2, 'sigma': 0.2, 'eta': 0.3, 'eta_icu': 0.15,
    'gamma_I': 0.1, 'gamma_X': 0.15, 'gamma_X_admit': 0.5, 'gamma_H': 0.18, 'gamma_ward': 0.2, 'gamma_icu': 0.12,
    'mu_I': 0.01, 'mu_X': 0.02, 'mu_X_untreated': 0.06, 'mu_H': 0.02, 'mu_ward': 0.01, 'mu_ward_denied_icu': 0.12, 'mu_icu': 0.04,
}

ELDERLY_PARAMS_TEACHING: AgeParams = {
    'alpha': 0.18, 'sigma': 0.3, 'eta': 0.5, 'eta_icu': 0.3,
    'gamma_I': 0.08, 'gamma_X': 0.12, 'gamma_X_admit': 0.5, 'gamma_H': 0.10, 'gamma_ward': 0.15, 'gamma_icu': 0.08,
    'mu_I': 0.03, 'mu_X': 0.15, 'mu_X_untreated': 0.45, 'mu_H': 0.08, 'mu_ward': 0.04, 'mu_ward_denied_icu': 0.35, 'mu_icu': 0.12,
}

AGE_PARAMS_TEACHING: List[AgeParams] = [YOUNG_PARAMS_TEACHING, MIDDLE_PARAMS_TEACHING, ELDERLY_PARAMS_TEACHING]

USE_EMPIRICAL_PARAMS = True
AGE_PARAMS_DEFAULT = AGE_PARAMS_EMPIRICAL if USE_EMPIRICAL_PARAMS else AGE_PARAMS_TEACHING
YOUNG_PARAMS_ACTIVE = YOUNG_PARAMS_EMPIRICAL if USE_EMPIRICAL_PARAMS else YOUNG_PARAMS_TEACHING
MIDDLE_PARAMS_ACTIVE = MIDDLE_PARAMS_EMPIRICAL if USE_EMPIRICAL_PARAMS else MIDDLE_PARAMS_TEACHING
ELDERLY_PARAMS_ACTIVE = ELDERLY_PARAMS_EMPIRICAL if USE_EMPIRICAL_PARAMS else ELDERLY_PARAMS_TEACHING


# ============================================================================
# SECTION 5: DIFFERENTIAL MORTALITY PARAMETERS
# ============================================================================

DIFFERENTIAL_MORTALITY_PARAMS: DifferentialMortalityParams = {
    'mu_X_untreated_multiplier': 2.0,
    'mu_ward_denied_icu_multiplier': 1.5,
    'mu_X_untreated_multiplier_young': 1.5,
    'mu_X_untreated_multiplier_middle': 2.0,
    'mu_X_untreated_multiplier_elderly': 3.0,
    'mu_ward_denied_icu_multiplier_young': 1.3,
    'mu_ward_denied_icu_multiplier_middle': 1.5,
    'mu_ward_denied_icu_multiplier_elderly': 2.0
}


# ============================================================================
# SECTION 6: CONTACT MATRICES
# ============================================================================

CONTACT_MATRIX_DEFAULT: ContactMatrix = np.array([
    [10.0, 3.0, 1.0], [3.0, 8.0, 2.0], [1.0, 2.0, 4.0]
])

CONTACT_MATRIX_HOMOGENEOUS: ContactMatrix = np.array([
    [8.0, 8.0, 8.0], [8.0, 8.0, 8.0], [8.0, 8.0, 8.0]
])

CONTACT_MATRIX_ASSORTATIVE: ContactMatrix = np.array([
    [15.0, 1.0, 0.5], [1.0, 12.0, 1.0], [0.5, 1.0, 6.0]
])

CONTACT_MATRIX_SCHOOL_CLOSURE: ContactMatrix = np.array([
    [4.0, 1.5, 0.5], [1.5, 8.0, 2.0], [0.5, 2.0, 4.0]
])

CONTACT_MATRIX_WORK_FROM_HOME: ContactMatrix = np.array([
    [10.0, 1.5, 1.0], [1.5, 4.0, 1.0], [1.0, 1.0, 4.0]
])

CONTACT_MATRIX_ELDERLY_SHIELDING: ContactMatrix = np.array([
    [10.0, 3.0, 0.3], [3.0, 8.0, 0.6], [0.3, 0.6, 2.0]
])


# ============================================================================
# SECTION 7: HEALTHCARE SYSTEM CONFIGURATIONS
# ============================================================================

HEALTHCARE_SYSTEM_SMALL = {
    'name': 'Small Community Hospital', 'description': 'Single small hospital serving rural community',
    'ward_capacity': 40, 'icu_capacity': 8, 'hill_coef_ward': 4, 'hill_coef_icu': 4,
    'age_pops': [12000, 20000, 8000], 'beds_per_1000': 1.2,
}

HEALTHCARE_SYSTEM_RURAL = {
    'name': 'Rural Hospital Network', 'description': 'Regional rural health system with limited resources',
    'ward_capacity': 80, 'icu_capacity': 20, 'hill_coef_ward': 4, 'hill_coef_icu': 4,
    'age_pops': [30000, 50000, 20000], 'beds_per_1000': 1.0,
}

HEALTHCARE_SYSTEM_SUBURBAN = {
    'name': 'Suburban Hospital', 'description': 'Mid-sized suburban hospital system',
    'ward_capacity': 800, 'icu_capacity': 200, 'hill_coef_ward': 4, 'hill_coef_icu': 4,
    'age_pops': [60000, 100000, 40000], 'beds_per_1000': 1.0,
}

HEALTHCARE_SYSTEM_URBAN = {
    'name': 'Urban Medical Center', 'description': 'Major urban hospital with regional referral capacity',
    'ward_capacity': 800, 'icu_capacity': 200, 'hill_coef_ward': 4, 'hill_coef_icu': 4,
    'age_pops': [120000, 200000, 80000], 'beds_per_1000': 2.5,
}

HEALTHCARE_SYSTEM_METROPOLITAN = {
    'name': 'Metropolitan Health System', 'description': 'Large metropolitan area with multiple hospitals',
    'ward_capacity': 1600, 'icu_capacity': 400, 'hill_coef_ward': 4, 'hill_coef_icu': 4,
    'age_pops': [600000, 1000000, 400000], 'beds_per_1000': 1.0,
}

HEALTHCARE_SYSTEM_WELL_RESOURCED = {
    'name': 'Well-Resourced System', 'description': 'High-income region with abundant healthcare resources',
    'ward_capacity': 2400, 'icu_capacity': 600, 'hill_coef_ward': 4, 'hill_coef_icu': 4,
    'age_pops': [600000, 1000000, 400000], 'beds_per_1000': 1.5,
}

HEALTHCARE_SYSTEM_RESOURCE_LIMITED = {
    'name': 'Resource-Limited System', 'description': 'Low-resource setting with constrained healthcare',
    'ward_capacity': 200, 'icu_capacity': 25, 'hill_coef_ward': 6, 'hill_coef_icu': 6,
    'age_pops': [600000, 1000000, 400000], 'beds_per_1000': 0.11,
}

HEALTHCARE_SYSTEM_SURGE_MILD = {
    'name': 'Mild Surge Capacity', 'description': 'Urban system with 25% surge capacity activated',
    'ward_capacity': 500, 'icu_capacity': 125, 'hill_coef_ward': 4, 'hill_coef_icu': 4,
    'age_pops': [120000, 200000, 80000], 'beds_per_1000': 1.56,
}

HEALTHCARE_SYSTEM_SURGE_MAJOR = {
    'name': 'Major Surge Capacity', 'description': 'Urban system with 50% surge capacity (field hospitals)',
    'ward_capacity': 600, 'icu_capacity': 150, 'hill_coef_ward': 5, 'hill_coef_icu': 5,
    'age_pops': [120000, 200000, 80000], 'beds_per_1000': 1.88,
}


# ============================================================================
# SECTION 8: INITIAL CONDITIONS
# ============================================================================

DEFAULT_INITIAL_CONDITIONS = {
    'I_single': 10, 'E_single': 0, 'X_single': 0, 'H_single': 0, 'R_single': 0, 'D_single': 0,
    'E_by_age': [0, 0, 0], 'I_by_age': [10, 0, 0], 'X_by_age': [0, 0, 0],
    'H_ward_by_age': [0, 0, 0], 'H_icu_by_age': [0, 0, 0], 'R_by_age': [0, 0, 0], 'D_by_age': [0, 0, 0],
    'S_vax_by_age': [0, 0, 0], 'E_vax_by_age': [0, 0, 0], 'I_vax_by_age': [0, 0, 0], 'X_vax_by_age': [0, 0, 0],
    'H_ward_vax_by_age': [0, 0, 0], 'H_icu_vax_by_age': [0, 0, 0], 'R_vax_by_age': [0, 0, 0], 'D_vax_by_age': [0, 0, 0],
}

INITIAL_CONDITIONS_PRESETS = {
    'young_seed': {'description': 'Outbreak starting in young population (school)', 'E_by_age': [5, 0, 0], 'I_by_age': [10, 0, 0]},
    'middle_seed': {'description': 'Outbreak starting in working-age adults', 'E_by_age': [0, 5, 0], 'I_by_age': [0, 10, 0]},
    'elderly_seed': {'description': 'Outbreak starting in elderly (care home)', 'E_by_age': [0, 0, 5], 'I_by_age': [0, 0, 10]},
    'multi_source': {'description': 'Multiple simultaneous introductions', 'E_by_age': [3, 3, 2], 'I_by_age': [5, 5, 3]},
    'established_outbreak': {
        'description': 'Simulation starting mid-outbreak',
        'E_by_age': [50, 80, 30], 'I_by_age': [100, 150, 60], 'X_by_age': [5, 20, 15],
        'H_ward_by_age': [1, 8, 6], 'H_icu_by_age': [0, 2, 3], 'R_by_age': [200, 300, 100],
    },
    'post_wave': {'description': 'Starting after first wave with immunity', 'E_by_age': [2, 3, 1], 'I_by_age': [5, 8, 3], 'R_by_age': [3000, 5000, 2000]},
}


# ============================================================================
# SECTION 9: TIME-VARYING PARAMETERS
# ============================================================================

# 9.1 Seasonal Transmission Patterns
SEASONAL_PARAMS_NONE: SeasonalParams = {'amplitude': 0.0, 'period': 365, 'peak_day': 0, 'description': 'No seasonal variation'}
SEASONAL_PARAMS_MILD: SeasonalParams = {'amplitude': 0.15, 'period': 365, 'peak_day': 0, 'description': 'Mild winter peak (subtropical)'}
SEASONAL_PARAMS_MODERATE: SeasonalParams = {'amplitude': 0.25, 'period': 365, 'peak_day': 0, 'description': 'Moderate seasonality (temperate)'}
SEASONAL_PARAMS_STRONG: SeasonalParams = {'amplitude': 0.40, 'period': 365, 'peak_day': 0, 'description': 'Strong winter peak (continental)'}

# 9.2 Waning Immunity Presets
WANING_NONE = {'omega': 0.0, 'description': 'No waning immunity (permanent protection)', 'mean_duration_days': float('inf')}
WANING_SLOW = {'omega': 0.001, 'description': 'Slow waning (~3 years)', 'mean_duration_days': 1000}
WANING_MODERATE = {'omega': 0.003, 'description': 'Moderate waning (~1 year)', 'mean_duration_days': 333}
WANING_FAST = {'omega': 0.005, 'description': 'Fast waning (~6 months)', 'mean_duration_days': 200}
WANING_VERY_FAST = {'omega': 0.01, 'description': 'Very fast waning (~3 months)', 'mean_duration_days': 100}
WANING_AGE_DIFFERENTIAL = {'omega_young': 0.002, 'omega_middle': 0.003, 'omega_elderly': 0.006, 'description': 'Age-dependent waning'}

# 9.3 Demographic Parameters
DEMOGRAPHIC_PARAMS_NONE: Optional[DemographicParams] = None

DEMOGRAPHIC_PARAMS_DEFAULT: DemographicParams = {
    'birth_rate': 0.000049, 'birth_age_distribution': [1.0, 0.0, 0.0],
    'mu_background': [0.0000014, 0.0000082, 0.00011], 'neonatal_vaccination_rate': 0.0,
    'description': 'Default global average demographics',
}

DEMOGRAPHIC_PARAMS_HIGH_INCOME: DemographicParams = {
    'birth_rate': 0.000030, 'birth_age_distribution': [1.0, 0.0, 0.0],
    'mu_background': [0.0000008, 0.0000055, 0.00012], 'neonatal_vaccination_rate': 0.0,
    'description': 'High-income country demographics',
}

DEMOGRAPHIC_PARAMS_LOW_INCOME: DemographicParams = {
    'birth_rate': 0.000082, 'birth_age_distribution': [1.0, 0.0, 0.0],
    'mu_background': [0.0000055, 0.00011, 0.00022], 'neonatal_vaccination_rate': 0.0,
    'description': 'Low-income country demographics',
}

DEMOGRAPHIC_PARAMS_EQUILIBRIUM: DemographicParams = {
    'birth_rate': 0.000049, 'birth_age_distribution': [1.0, 0.0, 0.0],
    'mu_background': [0.0000014, 0.0000082, 0.00011], 'neonatal_vaccination_rate': 0.0,
    'description': 'Demographics balanced for stable population',
}

DEMOGRAPHIC_PARAMS_NEONATAL_VAX: DemographicParams = {
    'birth_rate': 0.000049, 'birth_age_distribution': [1.0, 0.0, 0.0],
    'mu_background': [0.0000014, 0.0000082, 0.00011], 'neonatal_vaccination_rate': 0.8,
    'description': 'Demographics with 80% neonatal vaccination',
}

DEMOGRAPHIC_PRESETS = {
    'none': DEMOGRAPHIC_PARAMS_NONE, 'default': DEMOGRAPHIC_PARAMS_DEFAULT, 'high_income': DEMOGRAPHIC_PARAMS_HIGH_INCOME,
    'low_income': DEMOGRAPHIC_PARAMS_LOW_INCOME, 'equilibrium': DEMOGRAPHIC_PARAMS_EQUILIBRIUM, 'neonatal_vax': DEMOGRAPHIC_PARAMS_NEONATAL_VAX,
}

# 9.4 Policy Intervention Templates
INTERVENTION_NONE: List[Intervention] = []
INTERVENTION_EARLY_STRONG: List[Intervention] = [{'start_day': 14, 'end_day': 60, 'transmission_reduction': 0.6}]
INTERVENTION_EARLY_MODERATE: List[Intervention] = [{'start_day': 21, 'end_day': 75, 'transmission_reduction': 0.4}]
INTERVENTION_DELAYED_STRONG: List[Intervention] = [{'start_day': 45, 'end_day': 105, 'transmission_reduction': 0.6}]
INTERVENTION_DELAYED_MODERATE: List[Intervention] = [{'start_day': 60, 'end_day': 120, 'transmission_reduction': 0.4}]

INTERVENTION_TIERED_ESCALATING: List[Intervention] = [
    {'start_day': 21, 'end_day': 35, 'transmission_reduction': 0.2},
    {'start_day': 35, 'end_day': 56, 'transmission_reduction': 0.4},
    {'start_day': 56, 'end_day': 84, 'transmission_reduction': 0.6},
]

INTERVENTION_TIERED_DEESCALATING = [
    {'start_day': 30, 'end_day': 60, 'transmission_reduction': 0.6},
    {'start_day': 60, 'end_day': 90, 'transmission_reduction': 0.4},
    {'start_day': 90, 'end_day': 120, 'transmission_reduction': 0.2},
]

INTERVENTION_CYCLICAL = [
    {'start_day': 30, 'end_day': 51, 'transmission_reduction': 0.5},
    {'start_day': 72, 'end_day': 93, 'transmission_reduction': 0.5},
    {'start_day': 114, 'end_day': 135, 'transmission_reduction': 0.5},
    {'start_day': 156, 'end_day': 177, 'transmission_reduction': 0.5},
]

INTERVENTION_SUSTAINED_MODERATE = [{'start_day': 30, 'end_day': 150, 'transmission_reduction': 0.3}]
INTERVENTION_SHORT_SHARP = [{'start_day': 25, 'end_day': 39, 'transmission_reduction': 0.7}]

INTERVENTION_MULTI_WAVE = [
    {'start_day': 30, 'end_day': 60, 'transmission_reduction': 0.5},
    {'start_day': 120, 'end_day': 150, 'transmission_reduction': 0.4},
    {'start_day': 210, 'end_day': 240, 'transmission_reduction': 0.35},
]

# Legacy aliases
LOCKDOWN_SCENARIO = [{'start_day': 50, 'end_day': 100, 'transmission_reduction': 0.6}]
MULTIPLE_WAVES_SCENARIO = [
    {'start_day': 50, 'end_day': 80, 'transmission_reduction': 0.5},
    {'start_day': 150, 'end_day': 180, 'transmission_reduction': 0.4}
]


# ============================================================================
# SECTION 10: VACCINATION STRATEGIES
# ============================================================================

VACCINATION_STRATEGIES = {
    'none': {'coverage': [0.0, 0.0, 0.0], 'description': 'No vaccination'},
    'uniform_low': {'coverage': [0.2, 0.2, 0.2], 'description': 'Low uniform coverage (20%)'},
    'uniform_moderate': {'coverage': [0.4, 0.4, 0.4], 'description': 'Moderate uniform coverage (40%)'},
    'uniform_high': {'coverage': [0.7, 0.7, 0.7], 'description': 'High uniform coverage (70%)'},
    'elderly_priority': {'coverage': [0.1, 0.3, 0.8], 'description': 'Prioritize elderly (high-risk)'},
    'elderly_only': {'coverage': [0.0, 0.1, 0.9], 'description': 'Focus on elderly only'},
    'working_age_priority': {'coverage': [0.2, 0.7, 0.5], 'description': 'Prioritize working-age adults'},
    'young_priority': {'coverage': [0.7, 0.3, 0.4], 'description': 'Prioritize young (school/transmission)'},
    'balanced_risk': {'coverage': [0.3, 0.5, 0.8], 'description': 'Balanced by mortality risk'},
    'herd_immunity_target': {'coverage': [0.6, 0.75, 0.85], 'description': 'High coverage targeting herd immunity'},
}

VACCINE_EFFICACY_PRESETS = {
    'low': 0.50, 'moderate': 0.65, 'high': 0.80, 'very_high': 0.90, 'sterilizing': 0.95,
}


# ============================================================================
# SECTION 11: COMPLETE SCENARIO BUNDLES
# ============================================================================

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
    'description': 'Mild outbreak with late seasonal peak',
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
    'interventions': INTERVENTION_NONE,
    'vaccination': VACCINATION_STRATEGIES['balanced_risk'],
    'VE': VACCINE_EFFICACY_PRESETS['moderate'],
    'Tmax': 365,
}

SCENARIO_COVID_OMICRON = {
    'name': 'COVID-19 Omicron Wave',
    'description': 'Very high transmission but lower severity, high immunity',
    'beta_base': TRANSMISSION_PRESETS['severe']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_MODERATE,
    'waning_params': WANING_VERY_FAST,
    'interventions': INTERVENTION_EARLY_MODERATE,
    'vaccination': VACCINATION_STRATEGIES['herd_immunity_target'],
    'VE': VACCINE_EFFICACY_PRESETS['low'],
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
    'demographic_params': DEMOGRAPHIC_PARAMS_EQUILIBRIUM,
    'Tmax': 730,
}

SCENARIO_HOSPITAL_STRESS_TEST = {
    'name': 'Hospital Stress Test',
    'description': 'Severe outbreak overwhelming limited healthcare capacity',
    'beta_base': TRANSMISSION_PRESETS['high']['beta_base'],
    'age_params': AGE_PARAMS_TEACHING,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_RURAL,
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
    'contact_matrix': CONTACT_MATRIX_SCHOOL_CLOSURE,
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
    'description': 'Unknown pathogen with high uncertainty',
    'beta_base': TRANSMISSION_PRESETS['high']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_EARLY_STRONG,
    'vaccination': VACCINATION_STRATEGIES['none'],
    'vaccine_profile': None,
    'VE': 0.0,
    'Tmax': 200,
}

SCENARIO_VACCINE_ROLLOUT_PHASED = {
    'name': 'Phased Vaccine Rollout',
    'description': 'Realistic vaccine campaign: elderly first',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_MILD,
    'waning_params': WANING_MODERATE,
    'interventions': INTERVENTION_SUSTAINED_MODERATE,
    'vaccination': {'coverage': [0.1, 0.2, 0.5], 'description': 'Phased rollout'},
    'vaccine_profile': 'mrna_original',
    'vaccination_rate': [0.002, 0.003, 0.005],
    'Tmax': 365,
}

SCENARIO_VARIANT_EMERGENCE = {
    'name': 'Variant Emergence Mid-Outbreak',
    'description': 'New variant with immune escape',
    'beta_base': TRANSMISSION_PRESETS['severe']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_MODERATE,
    'waning_params': WANING_VERY_FAST,
    'interventions': INTERVENTION_TIERED_ESCALATING,
    'vaccination': VACCINATION_STRATEGIES['herd_immunity_target'],
    'vaccine_profile': 'mrna_omicron',
    'initial_conditions': INITIAL_CONDITIONS_PRESETS['post_wave'],
    'Tmax': 300,
}

SCENARIO_CAPACITY_COLLAPSE = {
    'name': 'Healthcare Capacity Collapse',
    'description': 'Severe outbreak overwhelming minimal healthcare',
    'beta_base': TRANSMISSION_PRESETS['severe']['beta_base'],
    'age_params': AGE_PARAMS_TEACHING,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_RESOURCE_LIMITED,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_DELAYED_MODERATE,
    'vaccination': VACCINATION_STRATEGIES['none'],
    'vaccine_profile': None,
    'VE': 0.0,
    'Tmax': 200,
}

SCENARIO_ENDEMIC_VACCINATION = {
    'name': 'Endemic with Ongoing Vaccination',
    'description': 'Long-term endemic with seasonal vaccination',
    'beta_base': TRANSMISSION_PRESETS['mild']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_STRONG,
    'waning_params': WANING_MODERATE,
    'interventions': INTERVENTION_NONE,
    'vaccination': VACCINATION_STRATEGIES['elderly_priority'],
    'vaccine_profile': 'influenza_typical',
    'vaccination_rate': [0.001, 0.001, 0.002],
    'vaccine_waning_params': {'omega_vax': 0.003, 'waning_destination': 'S_vax'},
    'demographic_params': DEMOGRAPHIC_PARAMS_EQUILIBRIUM, 'Tmax': 730,
}

SCENARIO_SCHOOL_REOPENING = {
    'name': 'School Reopening Policy Test',
    'description': 'Comparison of schools open vs closed',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_SUBURBAN,
    'seasonal_params': SEASONAL_PARAMS_MODERATE,
    'waning_params': WANING_NONE,
    'interventions': INTERVENTION_NONE,
    'vaccination': VACCINATION_STRATEGIES['young_priority'],
    'vaccine_profile': 'mrna_original',
    'initial_conditions': INITIAL_CONDITIONS_PRESETS['young_seed'],
    'Tmax': 180,
}

SCENARIO_SENSITIVITY_TRANSMISSION = {
    'name': 'Transmission Sensitivity Base',
    'description': 'Base scenario for beta sensitivity analysis',
    'beta_base': TRANSMISSION_PRESETS['moderate']['beta_base'],
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
    'description': 'Base scenario for capacity sensitivity analysis',
    'beta_base': TRANSMISSION_PRESETS['high']['beta_base'],
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

SCENARIO_POPULATION_DYNAMICS = {
    'name': 'Open Population Dynamics',
    'description': 'Long-term simulation with births and deaths',
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
    'Tmax': 1095,
}

SCENARIO_NEONATAL_VACCINATION = {
    'name': 'Neonatal Vaccination Program',
    'description': 'Long-term endemic with neonatal vaccination',
    'beta_base': TRANSMISSION_PRESETS['mild']['beta_base'],
    'age_params': AGE_PARAMS_EMPIRICAL,
    'contact_matrix': CONTACT_MATRIX_DEFAULT,
    'healthcare_system': HEALTHCARE_SYSTEM_URBAN,
    'seasonal_params': SEASONAL_PARAMS_NONE,
    'waning_params': WANING_SLOW,
    'interventions': INTERVENTION_NONE,
    'vaccination': VACCINATION_STRATEGIES['none'],
    'demographic_params': DEMOGRAPHIC_PARAMS_NEONATAL_VAX,
    'Tmax': 1095,
}

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

PARAMETER_RANGES = {
    'beta_base': {'min': 0.1, 'max': 0.6, 'default': 0.28, 'description': 'Baseline transmission rate'},
    'ward_capacity': {'min': 20, 'max': 500, 'default': 100, 'description': 'General ward bed capacity'},
    'icu_capacity': {'min': 5, 'max': 125, 'default': 25, 'description': 'ICU bed capacity'},
    'VE': {'min': 0.3, 'max': 0.95, 'default': 0.7, 'description': 'Vaccine efficacy'},
    'coverage_elderly': {'min': 0.0, 'max': 0.95, 'default': 0.5, 'description': 'Elderly vaccination coverage'},
    'seasonal_amplitude': {'min': 0.0, 'max': 0.5, 'default': 0.25, 'description': 'Seasonal transmission amplitude'},
    'omega': {'min': 0.0, 'max': 0.02, 'default': 0.003, 'description': 'Immunity waning rate'},
    'hill_coef': {'min': 2, 'max': 8, 'default': 4, 'description': 'Hill coefficient (admission gating steepness)'},
}
