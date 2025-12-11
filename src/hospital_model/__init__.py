"""
Hospital Model - SEIXHRD Epidemic Simulation Package

An age-structured compartmental epidemic model with:
- Split-X architecture (queued vs admitted severe cases)
- Three-Factor vaccination model (VE_infection, VE_severe, VE_death)
- Hill function capacity gating for ward and ICU
- Differential mortality tracking
- Time-varying parameters (seasonality, NPI interventions)
- Open population dynamics (births, background mortality)
"""

from hospital_model.simulate_model import simulate_model
from hospital_model.scenario_helpers import (
    get_scenario_params,
    list_scenarios,
    list_vaccine_profiles,
    describe_scenario,
    describe_vaccine_profile,
    get_vaccine_profile,
    compare_vaccine_profiles,
    run_scenario_with_overrides,
)
from hospital_model.scenarios import (
    AGE_PARAMS_EMPIRICAL,
    AGE_PARAMS_TEACHING,
    CONTACT_MATRIX_DEFAULT,
    AGE_POPS_DEFAULT,
    SCENARIO_REGISTRY,
    VACCINE_PROFILES,
)

__version__ = "0.1.0"

__all__ = [
    # Core simulation
    "simulate_model",
    # Scenario helpers
    "get_scenario_params",
    "list_scenarios",
    "list_vaccine_profiles",
    "describe_scenario",
    "describe_vaccine_profile",
    "get_vaccine_profile",
    "compare_vaccine_profiles",
    "run_scenario_with_overrides",
    # Data
    "AGE_PARAMS_EMPIRICAL",
    "AGE_PARAMS_TEACHING",
    "CONTACT_MATRIX_DEFAULT",
    "AGE_POPS_DEFAULT",
    "SCENARIO_REGISTRY",
    "VACCINE_PROFILES",
]
