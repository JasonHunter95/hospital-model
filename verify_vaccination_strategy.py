"""
Verification script for Experiment 9: vaccination strategy comparison.

Runs SCENARIO_COVID_DELTA with two allocation strategies:
- Elderly Priority (high-risk focused)
- Proportional (uniform coverage with the same total doses)

Vaccine efficacy is set to VE_death=0.9, VE_infection=0.5 to reflect
reduced protection against infection but strong protection against death.
"""

from copy import deepcopy

import config
from config_helpers import get_scenario_params
from master_hospital_model import simulate_master_hospital_model


def _proportional_coverage(reference_coverage, age_pops):
    """Return uniform coverage that uses the same total doses as reference_coverage."""
    total_doses = sum(c * pop for c, pop in zip(reference_coverage, age_pops))
    uniform_fraction = total_doses / sum(age_pops)
    return [uniform_fraction] * len(age_pops)


def _run_strategy(name, coverage, base_params):
    """Run the model for a given coverage vector and return total deaths."""
    params = deepcopy(base_params)
    params.pop('VE', None)  # Avoid legacy VE when using three-factor inputs
    params.update({
        'coverage': coverage,
        'VE_infection': 0.5,
        'VE_death': 0.9,
        'VE_severe': config.VACCINE_EFFICACY_PARAMS['VE_severe'],
    })
    
    result = simulate_master_hospital_model(**params)
    return result['D_total'][-1]


def main():
    base_params = get_scenario_params('covid_delta')
    age_pops = base_params['age_pops']
    
    elderly_priority_cov = config.VACCINATION_STRATEGIES['elderly_priority']['coverage']
    proportional_cov = _proportional_coverage(elderly_priority_cov, age_pops)
    
    deaths_elderly = _run_strategy('elderly_priority', elderly_priority_cov, base_params)
    deaths_proportional = _run_strategy('proportional', proportional_cov, base_params)
    
    print("Vaccination strategy comparison (SCENARIO_COVID_DELTA)")
    print("Assumptions: VE_infection=0.5, VE_death=0.9")
    print(f"  Elderly Priority coverage {elderly_priority_cov} -> Total deaths: {deaths_elderly:.0f}")
    print(f"  Proportional coverage {[round(c, 3) for c in proportional_cov]} -> Total deaths: {deaths_proportional:.0f}")
    
    if deaths_elderly < deaths_proportional:
        print("Result: Elderly Priority yields fewer deaths (expected with corrected FOI).")
    else:
        print("Result: Proportional outperformed Elderly Priority (unexpected — investigate).")


if __name__ == "__main__":
    main()
