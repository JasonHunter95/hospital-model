import warnings as _warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional
import numpy as np

from config import HEALTHCARE_SYSTEM_SMALL, SCENARIO_REGISTRY, VACCINATION_STRATEGIES, \
    AGE_LABELS_SHORT, AGE_PARAMS_DEFAULT, CONTACT_MATRIX_DEFAULT, \
    SEASONAL_PARAMS_NONE, WANING_NONE, INTERVENTION_NONE, \
    HEALTHCARE_SYSTEM_RURAL, HEALTHCARE_SYSTEM_SUBURBAN, HEALTHCARE_SYSTEM_URBAN, \
    HEALTHCARE_SYSTEM_METROPOLITAN, HEALTHCARE_SYSTEM_WELL_RESOURCED, \
    HEALTHCARE_SYSTEM_RESOURCE_LIMITED, HEALTHCARE_SYSTEM_SURGE_MILD, \
    HEALTHCARE_SYSTEM_SURGE_MAJOR, VACCINE_PROFILES, DEFAULT_SIM_PARAMS

# ============================================================================
# HELPER FUNCTIONS FOR CONFIGURATION AND SCENARIO MANAGEMENT
# ============================================================================

def list_scenarios() -> List[str]:
    """Return list of available scenario names."""
    return list(SCENARIO_REGISTRY.keys())

def list_vaccine_profiles() -> List[str]:
    """Return list of available vaccine profile names."""
    return list(VACCINE_PROFILES.keys())

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

def describe_vaccine_profile(profile_name: str) -> str:
    """Return detailed description of a vaccine profile."""
    if profile_name not in VACCINE_PROFILES:
        raise ValueError(f"Unknown vaccine profile: {profile_name}")
    
    p = VACCINE_PROFILES[profile_name]
    omega_days = f"{1/p['omega_vax']:.0f}" if p['omega_vax'] > 0 else "∞"
    
    lines = [
        f"=== {profile_name} ===",
        f"Description: {p['description']}",
        f"",
        f"Three-Factor Efficacy:",
        f"  VE_infection: {p['VE_infection']:.0%} (against infection)",
        f"  VE_severe:    {p['VE_severe']:.0%} (against severe disease)",
        f"  VE_death:     {p['VE_death']:.0%} (against death)",
        f"",
        f"Breakthrough Dynamics:",
        f"  θ_vax: {p['theta_vax']:.0%} (relative infectiousness)",
        f"  Immunity duration: ~{omega_days} days",
    ]
    return '\n'.join(lines)

## ===========================================================================
## Getters for predefined configurations
## ==========================================================================

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

def get_vaccine_profile(profile_name: str) -> Dict[str, Any]:
    """
    Get vaccine efficacy parameters for a named vaccine profile.
    
    Args:
        profile_name: Key from VACCINE_PROFILES (e.g., 'mrna_original', 'influenza_typical')
        
    Returns:
        Dictionary containing VE_infection, VE_severe, VE_death, theta_vax, omega_vax
        
    Example:
        profile = get_vaccine_profile('mrna_original')
        results = simulate_master_hospital_model(
            ...,
            VE_infection=profile['VE_infection'],
            VE_severe=profile['VE_severe'],
            VE_death=profile['VE_death'],
        )
    """
    if profile_name not in VACCINE_PROFILES:
        available = list(VACCINE_PROFILES.keys())
        raise ValueError(f"Unknown vaccine profile '{profile_name}'. Available: {available}")
    
    return deepcopy(VACCINE_PROFILES[profile_name])

def get_all_scenario_names() -> List[str]:
    """Return all available scenario names, sorted alphabetically."""
    return sorted(list(SCENARIO_REGISTRY.keys()))

## ============================================================================
## This is a central function to extract and prepare scenario parameters for 
## simulate_master_hospital_model()
## ============================================================================
def get_scenario_params(scenario_name: str, validate: bool = True) -> Dict[str, Any]:
    """
    Extract parameters from a scenario bundle for simulate_master_hospital_model().
    
    - Applies vaccine profiles from three-factor model
    - Handles vaccination_rate for dynamic vaccination
    - Validates parameters before returning
    
    Args:
        scenario_name: Key from SCENARIO_REGISTRY
        validate: If True, run validation on extracted params
        
    Returns:
        Dictionary ready to unpack into simulate_master_hospital_model(**params)
    """
    if scenario_name not in SCENARIO_REGISTRY:
        available = list(SCENARIO_REGISTRY.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")
    
    scenario = deepcopy(SCENARIO_REGISTRY[scenario_name])
    healthcare = scenario.pop('healthcare_system')
    vaccination = scenario.pop('vaccination')
    seasonal = scenario.pop('seasonal_params', {})
    waning = scenario.pop('waning_params', {})
    vaccine_profile = scenario.pop('vaccine_profile', None)
    
    # Handle vaccination - can be dict with 'coverage' key or direct list
    if isinstance(vaccination, dict):
        coverage = vaccination.get('coverage', [0.0, 0.0, 0.0])
    else:
        coverage = vaccination
    
    # Build parameter dict with nested configs
    params = {
        'beta_base': scenario['beta_base'],
        'age_params': scenario['age_params'],
        'contact_matrix': scenario.get('contact_matrix', CONTACT_MATRIX_DEFAULT),
        'age_pops': healthcare['age_pops'],
        'sim_config': {
            'Tmax': scenario.get('Tmax', 200),
        },
        'capacity_config': {
            'ward_capacity': healthcare['ward_capacity'],
            'icu_capacity': healthcare['icu_capacity'],
            'hill_coef_ward': healthcare.get('hill_coef_ward', 4),
            'hill_coef_icu': healthcare.get('hill_coef_icu', 4),
            'theta_X': DEFAULT_SIM_PARAMS.get('theta_X', 0.5),
            'theta_H': DEFAULT_SIM_PARAMS.get('theta_H', 0.3),
        },
        'vaccine_config': {
            'coverage': coverage,
        },
        'intervention_config': scenario.get('interventions', []),
    }
    
    # Apply vaccine profile if specified (three-factor model)
    if vaccine_profile is not None:
        params = apply_vaccine_profile_to_params(params, vaccine_profile)
    
    # Handle dynamic vaccination rate
    if 'vaccination_rate' in scenario:
        params['vaccine_config']['vaccination_rate'] = scenario['vaccination_rate']
    
    # Handle vaccine waning params from scenario
    if 'vaccine_waning_params' in scenario:
        if 'vaccine_waning_config' not in params:
            params['vaccine_waning_config'] = {}
        params['vaccine_waning_config'].update(scenario['vaccine_waning_params'])
    
    # Add seasonal parameters
    if seasonal and seasonal.get('amplitude', 0) > 0:
        params['seasonal_config'] = {
            'amplitude': seasonal['amplitude'],
            'period': seasonal.get('period', 365),
            'peak_day': seasonal.get('peak_day', 0),
        }
    
    # Add waning immunity
    if waning:
        omega = waning.get('omega', 0.0)
        if omega > 0:
            params['waning_config'] = {'omega': omega}
        elif 'omega_young' in waning:
            params['waning_config'] = {
                'omega_young': waning['omega_young'],
                'omega_middle': waning['omega_middle'],
                'omega_elderly': waning['omega_elderly'],
            }
    
    # Add initial conditions if specified
    if 'initial_conditions' in scenario:
        ic = scenario['initial_conditions']
        if 'E_by_age' in ic:
            params['initial_conditions'] = params.get('initial_conditions', {})
            params['initial_conditions']['E_by_age'] = ic['E_by_age']
        if 'I_by_age' in ic:
            params['initial_conditions'] = params.get('initial_conditions', {})
            params['initial_conditions']['I_by_age'] = ic['I_by_age']
        if 'R_by_age' in ic:
            params['initial_conditions'] = params.get('initial_conditions', {})
            params['initial_conditions']['R_by_age'] = ic['R_by_age']
            
    # Handle demographic config
    if 'demographic_params' in scenario:
        params['demographic_config'] = scenario['demographic_params']
    elif 'demographic_config' in scenario:
        params['demographic_config'] = scenario['demographic_config']
    
    # Validate if requested
    if validate:
        params = validate_scenario_params(params, strict=False)
    
    return params

## ===========================================================================
## Validation Functions for scenario parameters
## ===========================================================================
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

def validate_scenario_params(params: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
    """
    Validate parameters for simulate_master_hospital_model() before simulation.
    
    Checks for required parameters, valid ranges, and internal consistency.
    Returns validated params dict.
    
    Args:
        params: Dictionary of simulation parameters
        strict: If True, raise errors for missing required params.
                If False, only warn and use defaults where possible.
    
    Returns:
        Validated parameter dictionary.
        
    Raises:
        ValueError: If required parameters are missing (when strict=True)
    """
    validated = deepcopy(params)
    issues = []
    warnings_list = []
    
    # ========================================
    # Required Parameters
    # ========================================
    required = ['beta_base', 'age_params', 'contact_matrix', 'age_pops']
    for key in required:
        if key not in validated or validated[key] is None:
            issues.append(f"Missing required parameter: '{key}'")
    
    if issues and strict:
        raise ValueError("Validation failed:\n  " + "\n  ".join(issues))
    
    # ========================================
    # Numerical Range Validation
    # ========================================
    if 'beta_base' in validated:
        beta = validated['beta_base']
        if not (0 < beta < 2.0):
            warnings_list.append(f"beta_base={beta} outside typical range (0, 2.0)")
    
    # Vaccine Config Validation
    if 'vaccine_config' in validated:
        vax_config = validated['vaccine_config']
        
        for ve_key in ['VE_infection', 'VE_severe', 'VE_death']:
            if ve_key in vax_config and vax_config[ve_key] is not None:
                ve = vax_config[ve_key]
                if not (0 <= ve <= 1):
                    issues.append(f"vaccine_config['{ve_key}']={ve} must be in [0, 1]")
        
        # Coverage validation
        if 'coverage' in vax_config:
            coverage = vax_config['coverage']
            if isinstance(coverage, list):
                for i, c in enumerate(coverage):
                    if not (0 <= c <= 1):
                        issues.append(f"coverage[{i}]={c} must be in [0, 1]")
            elif isinstance(coverage, (int, float)):
                if not (0 <= coverage <= 1):
                    issues.append(f"coverage={coverage} must be in [0, 1]")
                    
        # Vaccination rate validation
        if 'vaccination_rate' in vax_config:
            rate = vax_config['vaccination_rate']
            if isinstance(rate, (int, float)):
                if rate < 0:
                    issues.append(f"vaccination_rate={rate} must be non-negative")
            # If list/array, assume valid for now or add more complex validation

    # Capacity Config Validation
    if 'capacity_config' in validated:
        cap_config = validated['capacity_config']
        for cap_key in ['ward_capacity', 'icu_capacity']:
            if cap_key in cap_config and cap_config[cap_key] is not None:
                cap = cap_config[cap_key]
                if cap <= 0:
                    issues.append(f"{cap_key}={cap} must be positive")
    
    # ========================================
    # Age Params Validation
    # ========================================
    if 'age_params' in validated and validated['age_params'] is not None:
        try:
            validate_age_params(validated['age_params'])
        except ValueError as e:
            issues.append(str(e))
    
    # ========================================
    # Contact Matrix Validation
    # ========================================
    if 'contact_matrix' in validated and validated['contact_matrix'] is not None:
        cm = np.asarray(validated['contact_matrix'])
        if cm.ndim != 2:
            issues.append(f"contact_matrix must be 2D, got {cm.ndim}D")
        elif cm.shape[0] != cm.shape[1]:
            issues.append(f"contact_matrix must be square, got shape {cm.shape}")
        elif 'age_params' in validated and validated['age_params'] is not None:
            n_ages = len(validated['age_params'])
            if cm.shape[0] != n_ages:
                issues.append(
                    f"contact_matrix shape {cm.shape} doesn't match "
                    f"number of age groups ({n_ages})"
                )
        if np.any(cm < 0):
            issues.append("contact_matrix contains negative values")
    
    # ========================================
    # Intervention Validation
    # ========================================
    if 'intervention_config' in validated and validated['intervention_config']:
        for i, intv in enumerate(validated['intervention_config']):
            required_keys = ['start_day', 'end_day', 'transmission_reduction']
            for key in required_keys:
                if key not in intv:
                    issues.append(f"intervention_config[{i}] missing '{key}'")
            if 'transmission_reduction' in intv:
                tr = intv['transmission_reduction']
                if not (0 <= tr <= 1):
                    issues.append(
                        f"intervention_config[{i}]['transmission_reduction']={tr} "
                        "must be in [0, 1]"
                    )
            if 'start_day' in intv and 'end_day' in intv:
                if intv['start_day'] >= intv['end_day']:
                    issues.append(
                        f"intervention_config[{i}]: start_day must be < end_day"
                    )
    
    # ========================================
    # Final Validation Check
    # ========================================
    if issues and strict:
        raise ValueError("Validation failed:\n  " + "\n  ".join(issues))
    
    for w in warnings_list:
        _warnings.warn(w, UserWarning, stacklevel=2)
    
    return validated

def apply_vaccine_profile_to_params(params: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
    """
    Internal helper to apply a vaccine profile to scenario parameters.
    
    Adds VE_infection, VE_severe, VE_death, theta_vax, and omega_vax
    from the specified vaccine profile to the params dict.
    """
    if profile_name is None:
        return params
    
    profile = get_vaccine_profile(profile_name)
    
    if 'vaccine_config' not in params:
        params['vaccine_config'] = {}
        
    params['vaccine_config']['VE_infection'] = profile['VE_infection']
    params['vaccine_config']['VE_severe'] = profile['VE_severe']
    params['vaccine_config']['VE_death'] = profile['VE_death']
    params['vaccine_config']['theta_vax'] = profile['theta_vax']
    
    # Handle vaccine waning from profile
    if profile.get('omega_vax', 0) > 0:
        if 'vaccine_waning_config' not in params or params['vaccine_waning_config'] is None:
            params['vaccine_waning_config'] = {}
        params['vaccine_waning_config']['omega_vax'] = profile['omega_vax']
    
    return params

# ============================================================================
# SECTION 17: MASTER MODEL HELPER FUNCTIONS
# ============================================================================
# Functions for running comparisons, parameter sweeps, and scenario analysis
# using the master_hospital_model.

def run_scenario_with_overrides(
    scenario_name: str,
    overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Get scenario parameters with optional overrides applied.
    
    Useful for sensitivity analysis where you want to modify specific
    parameters from a base scenario.
    
    Args:
        scenario_name: Base scenario from SCENARIO_REGISTRY
        overrides: Dictionary of parameters to override
        validate: Whether to validate final parameters
        
    Returns:
        Parameter dictionary ready for simulate_master_hospital_model()
        
    Example:
        # Run covid_delta with higher transmission
        params = run_scenario_with_overrides(
            'covid_delta',
            overrides={'beta_base': 0.5}
        )
        results = simulate_master_hospital_model(**params)
    """
    params = get_scenario_params(scenario_name, validate=False)
    
    if overrides:
        # Deep merge for nested dicts
        for key, value in overrides.items():
            if isinstance(value, dict) and key in params and isinstance(params[key], dict):
                params[key].update(value)
            else:
                params[key] = value
    
    if validate:
        params = validate_scenario_params(params, strict=False)
    
    return params

def compare_vaccine_profiles(
    base_scenario: str,
    profile_names: Optional[List[str]] = None,
    include_no_vaccine: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Generate parameter sets to compare different vaccine profiles.
    
    Args:
        base_scenario: Scenario name from SCENARIO_REGISTRY to use as base
        profile_names: List of vaccine profile names. If None, uses all profiles.
        include_no_vaccine: If True, includes a no-vaccination scenario
        
    Returns:
        Dictionary mapping profile names to parameter dictionaries
        
    Example:
        scenarios = compare_vaccine_profiles('covid_delta', 
                                             ['mrna_original', 'inactivated'])
        for name, params in scenarios.items():
            results[name] = simulate_master_hospital_model(**params)
    """
    if profile_names is None:
        profile_names = list_vaccine_profiles()
    
    results = {}
    
    if include_no_vaccine:
        no_vax_params = get_scenario_params(base_scenario, validate=False)
        no_vax_params['vaccine_config']['coverage'] = [0.0, 0.0, 0.0]
        no_vax_params['vaccine_config']['VE_infection'] = 0.0
        no_vax_params['vaccine_config']['VE_severe'] = 0.0
        no_vax_params['vaccine_config']['VE_death'] = 0.0
        results['no_vaccine'] = validate_scenario_params(no_vax_params, strict=False)
    
    for profile_name in profile_names:
        params = get_scenario_params(base_scenario, validate=False)
        params = apply_vaccine_profile_to_params(params, profile_name)
        results[profile_name] = validate_scenario_params(params, strict=False)
    
    return results

def compare_healthcare_systems(
    base_scenario: str,
    system_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Generate parameter sets comparing same epidemic across different healthcare systems.
    
    Args:
        base_scenario: Scenario name from SCENARIO_REGISTRY
        system_names: List of healthcare system names. If None, uses subset.
        
    Returns:
        Dictionary mapping system names to parameter dictionaries
    """
    if system_names is None:
        system_names = ['rural', 'suburban', 'urban', 'well_resourced', 'resource_limited']
    
    systems = get_healthcare_systems()
    results = {}
    
    for sys_name in system_names:
        if sys_name not in systems:
            _warnings.warn(f"Unknown healthcare system: {sys_name}", UserWarning)
            continue
        
        system = systems[sys_name]
        params = get_scenario_params(base_scenario, validate=False)
        
        # Override healthcare system parameters
        params['age_pops'] = system['age_pops']
        params['capacity_config']['ward_capacity'] = system['ward_capacity']
        params['capacity_config']['icu_capacity'] = system['icu_capacity']
        params['capacity_config']['hill_coef_ward'] = system.get('hill_coef_ward', 4)
        params['capacity_config']['hill_coef_icu'] = system.get('hill_coef_icu', 4)
        
        results[sys_name] = validate_scenario_params(params, strict=False)
    
    return results

def create_sensitivity_variants(
    base_scenario: str,
    parameter: str,
    values: List[Any],
    labels: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Generate scenario variants for parameter sensitivity analysis.
    
    Args:
        base_scenario: Scenario name from SCENARIO_REGISTRY
        parameter: Parameter name to vary (e.g., 'beta_base', 'ward_capacity')
        values: List of values to test
        labels: Optional labels for each variant. If None, uses str(value).
        
    Returns:
        Dictionary mapping variant labels to parameter dictionaries
        
    Example:
        # Beta sensitivity
        variants = create_sensitivity_variants(
            'baseline',
            'beta_base',
            [0.2, 0.3, 0.4, 0.5],
            labels=['R0~1.8', 'R0~2.7', 'R0~3.6', 'R0~4.5']
        )
    """
    if labels is None:
        labels = [f"{parameter}={v}" for v in values]
    
    if len(labels) != len(values):
        raise ValueError("Length of labels must match length of values")
    
    results = {}
    
    for label, value in zip(labels, values):
        params = get_scenario_params(base_scenario, validate=False)
        
        # Handle nested parameters (e.g., 'capacity_config.ward_capacity')
        if '.' in parameter:
            parts = parameter.split('.')
            target = params
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        else:
            params[parameter] = value
        
        results[label] = validate_scenario_params(params, strict=False)
    
    return results

def create_intervention_comparison(
    base_scenario: str,
    intervention_sets: Dict[str, List[Dict]]
) -> Dict[str, Dict[str, Any]]:
    """
    Generate scenarios comparing different intervention strategies.
    
    Args:
        base_scenario: Scenario name from SCENARIO_REGISTRY
        intervention_sets: Dict mapping strategy names to intervention lists
        
    Returns:
        Dictionary mapping strategy names to parameter dictionaries
        
    Example:
        interventions = {
            'No intervention': [],
            'Early lockdown': INTERVENTION_EARLY_STRONG,
            'Delayed lockdown': INTERVENTION_DELAYED_STRONG,
            'Cyclical': INTERVENTION_CYCLICAL,
        }
        variants = create_intervention_comparison('baseline', interventions)
    """
    results = {}
    
    for name, interventions in intervention_sets.items():
        params = get_scenario_params(base_scenario, validate=False)
        params['intervention_config'] = interventions
        results[name] = validate_scenario_params(params, strict=False)
    
    return results

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

def summarize_scenarios() -> str:
    """Return a formatted summary of all available scenarios."""
    lines = ["=" * 70]
    lines.append("AVAILABLE SCENARIOS")
    lines.append("=" * 70)
    
    for name in sorted(SCENARIO_REGISTRY.keys()):
        scenario = SCENARIO_REGISTRY[name]
        desc = scenario.get('description', 'No description')
        lines.append(f"\n{name}:")
        lines.append(f"  {desc}")
        lines.append(f"  β_base = {scenario['beta_base']:.2f}, Tmax = {scenario.get('Tmax', 200)}")
        
        if 'vaccine_profile' in scenario and scenario['vaccine_profile']:
            lines.append(f"  Vaccine: {scenario['vaccine_profile']}")
    
    return '\n'.join(lines)