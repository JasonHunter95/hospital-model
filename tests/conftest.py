"""
Shared pytest fixtures for hospital model tests.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    AGE_PARAMS_DEFAULT,
    CONTACT_MATRIX_DEFAULT,
    DEFAULT_SIM_PARAMS,
    DEFAULT_CAPACITY_PARAMS,
    DEFAULT_INITIAL_CONDITIONS,
    DIFFERENTIAL_MORTALITY_PARAMS,
    AGE_POPS_DEFAULT,
    DEMOGRAPHIC_PARAMS_DEFAULT,
    DEMOGRAPHIC_PARAMS_EQUILIBRIUM,
    DEMOGRAPHIC_PARAMS_NEONATAL_VAX,
)


# ========================================
# Basic Fixtures
# ========================================

@pytest.fixture
def minimal_inputs():
    """Minimal valid inputs for simulate_master_hospital_model."""
    return {
        'beta_base': 0.3,
        'age_params': AGE_PARAMS_DEFAULT,
        'contact_matrix': CONTACT_MATRIX_DEFAULT,
        'age_pops': [3000, 5000, 2000],
    }


@pytest.fixture
def minimal_inputs_single_age():
    """Minimal inputs with a single age group."""
    single_age_params = [{
        'alpha': 0.2,
        'sigma': 0.1,
        'eta': 0.2,
        'eta_icu': 0.1,
        'gamma_I': 0.1,
        'mu_I': 0.01,
        'gamma_X': 0.15,
        'mu_X': 0.02,
        'gamma_ward': 0.2,
        'mu_ward': 0.01,
        'gamma_icu': 0.1,
        'mu_icu': 0.03,
        'gamma_H': 0.2,
        'mu_H': 0.02,
    }]
    return {
        'beta_base': 0.3,
        'age_params': single_age_params,
        'contact_matrix': np.array([[8.0]]),
        'age_pops': [10000],
    }


@pytest.fixture
def high_capacity_inputs(minimal_inputs):
    """Inputs with very high capacity (no overflow)."""
    return {
        **minimal_inputs,
        'ward_capacity': 10000,
        'icu_capacity': 5000,
    }


@pytest.fixture
def low_capacity_inputs(minimal_inputs):
    """Inputs with very low capacity (maximum constraint)."""
    return {
        **minimal_inputs,
        'ward_capacity': 5,
        'icu_capacity': 2,
    }


@pytest.fixture
def zero_capacity_inputs(minimal_inputs):
    """Inputs with zero capacity (all admissions denied)."""
    return {
        **minimal_inputs,
        'ward_capacity': 0,
        'icu_capacity': 0,
    }


# ========================================
# Time-Varying Fixtures
# ========================================

@pytest.fixture
def seasonal_inputs(minimal_inputs):
    """Inputs with seasonal forcing enabled."""
    return {
        **minimal_inputs,
        'seasonal_params': {
            'amplitude': 0.3,
            'period': 365,
            'peak_day': 0,
        },
        'Tmax': 365,
    }


@pytest.fixture
def intervention_inputs(minimal_inputs):
    """Inputs with a single intervention period."""
    return {
        **minimal_inputs,
        'interventions': [
            {
                'start_day': 30,
                'end_day': 60,
                'transmission_reduction': 0.5,
            }
        ],
        'Tmax': 100,
    }


@pytest.fixture
def waning_inputs(minimal_inputs):
    """Inputs with waning immunity enabled."""
    return {
        **minimal_inputs,
        'waning_params': {
            'omega': 0.01,  # ~100 day immunity duration
        },
        'Tmax': 365,
    }


# ========================================
# Vaccination Fixtures
# ========================================

@pytest.fixture
def full_vaccination_inputs(minimal_inputs):
    """Inputs with 100% vaccine coverage and efficacy."""
    return {
        **minimal_inputs,
        'coverage': 1.0,
        'VE': 1.0,
    }


@pytest.fixture
def partial_vaccination_inputs(minimal_inputs):
    """Inputs with age-specific vaccination coverage."""
    return {
        **minimal_inputs,
        'coverage': [0.2, 0.4, 0.8],  # elderly priority
        'VE': 0.7,
    }


# ========================================
# Edge Case Fixtures
# ========================================

@pytest.fixture
def no_infection_inputs(minimal_inputs):
    """Inputs with no initial infections."""
    return {
        **minimal_inputs,
        'initial_conditions': {
            'I_by_age': [0, 0, 0],
            'E_by_age': [0, 0, 0],
        },
    }


@pytest.fixture
def short_simulation_inputs(minimal_inputs):
    """Inputs for a very short simulation."""
    return {
        **minimal_inputs,
        'Tmax': 10,
        'time_step': 0.1,
    }


@pytest.fixture
def long_simulation_inputs(minimal_inputs):
    """Inputs for a long simulation with waning immunity."""
    return {
        **minimal_inputs,
        'Tmax': 730,  # 2 years
        'time_step': 0.5,  # larger step for speed
        'waning_params': {'omega': 0.005},
    }


# ========================================
# Reference Data Fixtures
# ========================================

@pytest.fixture
def config_defaults():
    """Access to all config defaults for testing."""
    return {
        'sim_params': DEFAULT_SIM_PARAMS,
        'capacity_params': DEFAULT_CAPACITY_PARAMS,
        'initial_conditions': DEFAULT_INITIAL_CONDITIONS,
        'differential_mortality': DIFFERENTIAL_MORTALITY_PARAMS,
    }


@pytest.fixture
def n_ages():
    """Default number of age groups."""
    return 3


@pytest.fixture
def total_population():
    """Default total population."""
    return sum(AGE_POPS_DEFAULT)


# ========================================
# Demographic Fixtures
# ========================================

@pytest.fixture
def demographic_inputs(minimal_inputs):
    """Inputs with demographic parameters (births and background deaths)."""
    return {
        **minimal_inputs,
        'demographic_params': DEMOGRAPHIC_PARAMS_DEFAULT,
        'Tmax': 365,  # 1 year for meaningful demographic effects
    }


@pytest.fixture
def demographic_equilibrium_inputs(minimal_inputs):
    """Inputs with balanced demographic parameters for stable population."""
    return {
        **minimal_inputs,
        'demographic_params': DEMOGRAPHIC_PARAMS_EQUILIBRIUM,
        'Tmax': 730,  # 2 years
    }


@pytest.fixture
def neonatal_vaccination_inputs(minimal_inputs):
    """Inputs with neonatal vaccination enabled."""
    return {
        **minimal_inputs,
        'demographic_params': DEMOGRAPHIC_PARAMS_NEONATAL_VAX,
        'VE_infection': 0.8,
        'VE_severe': 0.9,
        'VE_death': 0.95,
        'Tmax': 365,
    }


@pytest.fixture
def high_birth_rate_inputs(minimal_inputs):
    """Inputs with high birth rate for testing population growth."""
    return {
        **minimal_inputs,
        'demographic_params': {
            'birth_rate': 0.0001,  # ~36.5/1000/year (high)
            'birth_age_distribution': [1.0, 0.0, 0.0],
            'mu_background': [0.0, 0.0, 0.0],  # No background deaths
            'neonatal_vaccination_rate': 0.0,
        },
        'Tmax': 365,
    }


@pytest.fixture
def high_mortality_inputs(minimal_inputs):
    """Inputs with high background mortality for testing population decline."""
    return {
        **minimal_inputs,
        'demographic_params': {
            'birth_rate': 0.0,  # No births
            'birth_age_distribution': [1.0, 0.0, 0.0],
            'mu_background': [0.0001, 0.0001, 0.0001],  # Uniform high mortality
            'neonatal_vaccination_rate': 0.0,
        },
        'Tmax': 365,
    }


@pytest.fixture
def zero_demographic_inputs(minimal_inputs):
    """Inputs with zero demographic rates (closed population via explicit params)."""
    return {
        **minimal_inputs,
        'demographic_params': {
            'birth_rate': 0.0,
            'birth_age_distribution': [1.0, 0.0, 0.0],
            'mu_background': [0.0, 0.0, 0.0],
            'neonatal_vaccination_rate': 0.0,
        },
        'Tmax': 200,
    }
