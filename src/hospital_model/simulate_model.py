"""
SEIXHRD model with Three-Factor Vaccination Compartments, age-structured compartments, time-varying parameters, and NPIs.

This model includes:
- Age-structured compartments (S, E, I, X_queued, X_admitted, H_ward, H_icu, R, D)
- Vaccinated compartments (S_vax, E_vax, I_vax, X_queued_vax, X_admitted_vax, H_ward_vax, H_icu_vax, R_vax, D_vax)
- Split X compartment for mathematically rigorous differential mortality:
  * X_queued: Severe cases waiting for ward admission (experience untreated mortality)
  * X_admitted: Severe cases that secured a ward spot (experience treated mortality)
- Exposed (E) compartment for latent period between infection and infectiousness
- Three-Factor Vaccination Model:
  * VE_infection: Efficacy against infection (reduces force of infection for S_vax)
  * VE_severe: Efficacy against severe disease (reduces I→X progression for vaccinated)
  * VE_death: Efficacy against death (reduces mortality for vaccinated)
- Dynamic vaccination with age-specific rates (S → S_vax)
- Vaccine immunity waning (R_vax → S or S_vax)
- Breakthrough infection tracking
- Infectiousness modifiers for compartments X and H (and reduced for vaccinated: theta_vax)
- Separate ward and ICU capacity constraints with Hill function gating
- Differential mortality tracking (treated vs untreated deaths)
- Seasonal forcing of transmission
- Policy interventions (lockdowns/relaxations)
- Waning natural immunity

Numerical Integration:
- Uses scipy.integrate.odeint (LSODA) by default for improved numerical stability
- Supports scipy.integrate.solve_ivp with configurable methods (RK45, BDF, Radau, etc.)
- Uses tcrit parameter to handle intervention discontinuities accurately
- Vectorized force of infection calculation using numpy matrix operations
- Post-integration clipping with warning threshold for non-negativity enforcement
"""

import warnings
import numpy as np
from scipy.integrate import odeint, solve_ivp
from . import scenarios
from .utils import (
    validate_age_structured_inputs,
    coerce_initial_vector,
    pack_state,
)
from .derivatives import (
    master_deriv_solve_ivp,
    master_deriv,
)
from .demographics_helpers import validate_demographic_params
from .time_varying_helpers import seasonal_forcing, policy_multiplier
from .model_types import (
    ODEParams, SimParams, CapacityParams, VaccineEfficacyParams, 
    VaccineWaningParams, DemographicParams, SeasonalParams, 
    Intervention, AgeParams, ContactMatrix
)
from typing import List, Dict, Optional, Any, Union


def simulate_model(
    # Core Data (Required)
    beta_base: float,
    age_params: List[AgeParams],
    contact_matrix: ContactMatrix,
    age_pops: List[float],
    
    # Grouped Configurations (Optional)
    sim_config: Optional[SimParams] = None,
    capacity_config: Optional[CapacityParams] = None,
    vaccine_config: Optional[Dict[str, Any]] = None,  # Can be simple dict or VaccineEfficacyParams + rate
    vaccine_waning_config: Optional[VaccineWaningParams] = None,
    demographic_config: Optional[DemographicParams] = None,
    seasonal_config: Optional[SeasonalParams] = None,
    waning_config: Optional[Dict[str, float]] = None,
    intervention_config: Optional[List[Intervention]] = None,
    
    # Initial Conditions
    initial_conditions: Optional[Dict[str, List[float]]] = None,
    
    # Simulation Control
    solver: str = 'odeint',
    solver_method: str = 'LSODA',
    rtol: float = 1e-6,
    atol: float = 1e-9,
    
    # Tracking
    track_differential_mortality: bool = True,
    track_compartment_flows: bool = False,
):
    """
    Simulate the SEIXHRD hospital model with Three-Factor Vaccination and
    age-structured compartments. The model also supports time-varying parameters
    such as seasonality as well as NPIs like lockdowns and other types of interventions.
    
    This model implements a comprehensive age-structured compartmental model with
    explicit vaccinated compartments using a three-factor vaccine efficacy model:
    - VE_infection: Protection against infection (reduces susceptibility)
    - VE_severe: Protection against severe disease (reduces I→X progression)
    - VE_death: Protection against death (reduces mortality rates)
    
    Parameters
    ----------
    beta_base : float
        Baseline transmission rate (modified by seasonality and interventions).
    age_params : list of dict
        Age-specific parameters for each group. Each dict should contain:
        - Latent period: 'alpha' (E → I rate, 1/latent_period)
        - Core rates: 'sigma', 'eta', 'eta_icu', 'gamma_I', 'mu_I', 'gamma_X', 'mu_X'
        - Ward/ICU: 'gamma_ward', 'mu_ward', 'gamma_icu', 'mu_icu'
        - Differential mortality: 'mu_X_untreated', 'mu_ward_denied_icu' (optional)
    contact_matrix : ndarray
        Contact rates C[a,b] between age groups (infector a, infectee b).
        Shape: (n_ages, n_ages).
    age_pops : list
        Population size for each age group. Required.
    
    sim_config : dict, optional
        Simulation configuration parameters:
        - 'Tmax': simulation duration in days
        - 'time_step': integration time step
        - 'theta_X': relative infectiousness of X compartment
        - 'theta_H': relative infectiousness of H compartment
        - 'theta_vax': relative infectiousness of vaccinated infected individuals
        Default: uses scenarios.DEFAULT_SIM_PARAMS
    
    capacity_config : dict, optional
        Healthcare capacity configuration:
        - 'ward_capacity': total general ward capacity
        - 'icu_capacity': total ICU capacity
        - 'hill_coef_ward': Hill coefficient for ward admission gating
        - 'hill_coef_icu': Hill coefficient for ICU admission gating
        Default: uses scenarios.DEFAULT_CAPACITY_PARAMS
    
    vaccine_config : dict, optional
        Vaccine efficacy and coverage configuration:
        - 'coverage': initial vaccine coverage (float or list for age-specific)
        - 'VE_infection': efficacy against infection (0-1)
        - 'VE_severe': efficacy against severe disease (0-1)
        - 'VE_death': efficacy against death (0-1)
        - 'theta_vax': relative infectiousness of vaccinated infected
        - 'vaccination_rate': daily vaccination rate (float or list)
        Default: uses scenarios.VACCINE_EFFICACY_PARAMS
    
    vaccine_waning_config : dict, optional
        Vaccine immunity waning parameters:
        - 'omega_vax': vaccine waning rate (1/days)
        - 'omega_vax_by_age': age-specific waning rates [young, middle, elderly]
        - 'waning_destination': 'S' (fully susceptible) or 'S_vax' (partial protection)
        Default: no waning
    
    demographic_config : dict, optional
        Demographic (vital dynamics) parameters:
        - 'birth_rate': per-capita birth rate (births per person per day)
        - 'mu_background': age-specific background mortality rate
        - 'birth_age_distribution': fraction of births entering each age group
        - 'neonatal_vaccination_rate': fraction of newborns vaccinated at birth
        Default: None (closed population)
    
    seasonal_config : dict, optional
        Seasonal forcing parameters:
        - 'amplitude': seasonal amplitude (0-1)
        - 'period': period in days
        - 'peak_day': day of peak transmission
        Default: no seasonality
    
    waning_config : dict, optional
        Natural immunity waning parameters (R → S):
        - 'omega': uniform waning rate (1/days), OR
        - 'omega_young', 'omega_middle', 'omega_elderly': age-specific rates
        Default: no waning
    
    intervention_config : list of dict, optional
        Policy interventions, each with:
        - 'start_day': intervention start
        - 'end_day': intervention end  
        - 'transmission_reduction': fraction reduction (0-1)
        Default: no interventions
    
    initial_conditions : dict, optional
        Override default initial conditions. Supports both unvaccinated
        (*_by_age) and vaccinated (*_vax_by_age) compartment keys.
    
    solver : str, optional
        ODE solver to use: 'odeint' or 'solve_ivp'. Default: 'odeint'
    
    solver_method : str, optional
        Method for solve_ivp: 'LSODA', 'RK45', 'BDF', 'Radau', etc.
        Default: 'LSODA'
    
    rtol : float, optional
        Relative tolerance for ODE solver. Default: 1e-6
    
    atol : float, optional
        Absolute tolerance for ODE solver. Default: 1e-9
    
    track_differential_mortality : bool, optional
        Track deaths by care status. Default: True
    
    track_compartment_flows : bool, optional
        Track daily flows between compartments. Default: False
    
    Returns
    -------
    dict
        Comprehensive results dictionary containing:
        
        Time:
        - 'times': array of time points
        
        Unvaccinated compartments by age (lists of arrays, one per age group):
        - 'S', 'E', 'I', 'X', 'R', 'D': standard compartments
        - 'H_ward', 'H_icu': hospital compartments
        - 'H': combined ward + ICU (backward compatibility)
        
        Vaccinated compartments by age (lists of arrays):
        - 'S_vax', 'E_vax', 'I_vax', 'X_vax', 'R_vax', 'D_vax': vaccinated compartments
        - 'H_ward_vax', 'H_icu_vax': vaccinated hospital compartments
        - 'H_vax': combined vaccinated ward + ICU
        
        Aggregated totals (arrays):
        - 'H_ward_total', 'H_icu_total', 'H_total': hospital occupancy (all)
        - 'H_ward_vax_total', 'H_icu_vax_total', 'H_vax_total': vaccinated hospital
        - 'E_total', 'I_total', 'X_total', 'D_total': infection/death totals
        - 'vaccinated_total': total vaccinated population over time
        - 'breakthrough_infections': cumulative breakthrough infections
        
        Capacity metrics:
        - 'ward_overflow', 'icu_overflow': instantaneous overflow
        - 'cum_ward_overflow', 'cum_icu_overflow': cumulative overflow
        - 'cum_unmet_ward', 'cum_unmet_icu': unmet care by age
        - 'g_ward', 'g_icu': admission gating factors over time
        
        Differential mortality (if track_differential_mortality=True):
        - 'D_treated', 'D_untreated': deaths by care status per age
        - 'D_treated_total', 'D_untreated_total': aggregated
        - 'D_vax_total': total vaccinated deaths over time
        
        Demographic metrics (if demographic_config provided):
        - 'cum_births': cumulative births by age group
        - 'cum_births_total': total cumulative births
        - 'cum_background_deaths': cumulative background deaths by age group
        - 'cum_background_deaths_total': total cumulative background deaths
        - 'live_population': total living population over time
        - 'population_change': net population change from demographics
        
        Time-varying parameters:
        - 'beta_t': effective transmission rate over time
        - 'seasonal_factor': seasonal multiplier
        - 'policy_mult': policy intervention multiplier
        
        Compartment flows (if track_compartment_flows=True):
        - 'new_infections', 'ward_admissions', 'icu_admissions': daily flows
        - 'new_vaccinations': daily new vaccinations by age
        - 'breakthrough_infections_daily': daily breakthrough infections
        
        Metadata:
        - 'ward_capacity', 'icu_capacity', 'age_pops'
        - 'parameters': dict of all input parameters
    
    Notes
    -----
    Compartment structure for each age group:
    
    Unvaccinated pathway:
        S_a → E_a → I_a → X_a → H_ward_a → H_icu_a → R_a or D_a
                     ↓           ↓            ↓
                    R_a         R_a          R_a
                                 ↓            ↓
                                D_a          D_a
    
    Vaccinated pathway (breakthrough infections):
        S_vax_a → E_vax_a → I_vax_a → X_vax_a → H_ward_vax_a → H_icu_vax_a → R_vax_a or D_vax_a
                            ↓              ↓                ↓
                           R_vax_a       R_vax_a          R_vax_a
    
    Vaccination flow: S_a → S_vax_a (at vaccination_rate)
    
    Waning flows:
        Natural immunity: R_a → S_a (at omega rate)
        Vaccine immunity: R_vax_a → S_a or S_vax_a (at omega_vax rate, destination configurable)
    
    Three-Factor Vaccine Efficacy:
        - VE_infection: λ_vax = (1 - VE_infection) * λ
        - VE_severe: σ_vax = (1 - VE_severe) * σ
        - VE_death: μ_vax = (1 - VE_death) * μ
    
    Time-varying transmission:
        beta(t) = beta_base * seasonal_factor(t) * policy_multiplier(t)
    
    Differential mortality:
    - D_treated: deaths occurring with appropriate care
    - D_untreated: excess deaths from capacity constraints
    """
    # ========================================
    # Parameter Setup with Defaults
    # ========================================
    
    # Unpack grouped configurations with defaults from scenarios module
    
    # Simulation parameters
    if sim_config is None:
        sim_config = {}
    Tmax = sim_config.get('Tmax', scenarios.DEFAULT_SIM_PARAMS['Tmax'])
    time_step = sim_config.get('time_step', scenarios.DEFAULT_SIM_PARAMS['time_step'])
    theta_X = sim_config.get('theta_X', scenarios.DEFAULT_SIM_PARAMS['theta_X'])
    theta_H = sim_config.get('theta_H', scenarios.DEFAULT_SIM_PARAMS['theta_H'])
    clip_warning_threshold = sim_config.get('clip_warning_threshold', 1e-6)
    
    # Capacity parameters
    if capacity_config is None:
        capacity_config = {}
    K_ward = capacity_config.get('ward_capacity', scenarios.DEFAULT_CAPACITY_PARAMS['ward_capacity'])
    K_icu = capacity_config.get('icu_capacity', scenarios.DEFAULT_CAPACITY_PARAMS['icu_capacity'])
    n_ward = capacity_config.get('hill_coef_ward', scenarios.DEFAULT_CAPACITY_PARAMS['hill_coef_ward'])
    n_icu = capacity_config.get('hill_coef_icu', scenarios.DEFAULT_CAPACITY_PARAMS['hill_coef_icu'])
    
    # Vaccine efficacy parameters
    if vaccine_config is None:
        vaccine_config = {}
    
    # Handle coverage - default to 0.0 if not provided
    coverage = vaccine_config.get('coverage', 0.0)
    
    # Three-factor vaccine efficacy
    vaccine_params = scenarios.VACCINE_EFFICACY_PARAMS
    VE_infection = vaccine_config.get('VE_infection', vaccine_params['VE_infection'])
    VE_severe = vaccine_config.get('VE_severe', vaccine_params['VE_severe'])
    VE_death = vaccine_config.get('VE_death', vaccine_params['VE_death'])
    
    # Theta_vax can come from vaccine_config or sim_config
    if 'theta_vax' in vaccine_config:
        theta_vax = vaccine_config['theta_vax']
    elif 'theta_vax' in sim_config:
        theta_vax = sim_config['theta_vax']
    else:
        theta_vax = scenarios.DEFAULT_SIM_PARAMS.get('theta_vax', 0.5)
    
    # Vaccination rate
    vaccination_rate = vaccine_config.get('vaccination_rate', 0.0)
    
    # Vaccine waning parameters
    if vaccine_waning_config is not None:
        vaccine_waning_params = vaccine_waning_config
    else:
        vaccine_waning_params = None
    
    # Seasonal parameters
    if seasonal_config is not None:
        seasonal_params = seasonal_config
    else:
        seasonal_params = {'amplitude': 0.0, 'period': 365, 'peak_day': 0}
    
    # Waning immunity parameters
    if waning_config is not None:
        waning_params = waning_config
    else:
        waning_params = None
    
    # Intervention parameters
    if intervention_config is not None:
        interventions = intervention_config
    else:
        interventions = []
    
    # Demographic parameters
    if demographic_config is not None:
        demographic_params = demographic_config
    else:
        demographic_params = None
    
    # Validate age_pops is provided
    n_ages = len(age_pops)
    
    dt = time_step
    
    # Validate inputs
    validate_age_structured_inputs(age_params, contact_matrix, age_pops, coverage)
    
    # Handle coverage - convert to list if scalar
    if not isinstance(coverage, list):
        coverage = [coverage] * n_ages
    
    # ========================================
    # Three-Factor Vaccine Model Setup
    # ========================================
    # Use default vaccine efficacy parameters if not provided
    
    if VE_infection is None:
        VE_infection = vaccine_params['VE_infection'] if any(c > 0 for c in coverage) else 0.0
    if VE_severe is None:
        VE_severe = vaccine_params['VE_severe'] if any(c > 0 for c in coverage) else 0.0
    if VE_death is None:
        VE_death = vaccine_params['VE_death'] if any(c > 0 for c in coverage) else 0.0
    
    # Vaccination rate - convert to list if scalar
    if vaccination_rate is None:
        vaccination_rate = [0.0] * n_ages
    elif not isinstance(vaccination_rate, list):
        vaccination_rate = [vaccination_rate] * n_ages
    else:
        # Extend if shorter than n_ages
        while len(vaccination_rate) < n_ages:
            vaccination_rate.append(vaccination_rate[-1] if vaccination_rate else 0.0)
        vaccination_rate = vaccination_rate[:n_ages]
    
    # Vaccine waning parameters
    if vaccine_waning_params is None:
        omega_vax = [0.0] * n_ages
        vax_waning_destination = 'S'
    else:
        # Get waning destination (default to 'S' if not specified)
        vax_waning_destination = vaccine_waning_params.get('waning_destination', 'S')
        if 'omega_vax' in vaccine_waning_params:
            omega_vax = [vaccine_waning_params['omega_vax']] * n_ages
        elif 'omega_vax_by_age' in vaccine_waning_params and vaccine_waning_params['omega_vax_by_age'] is not None:
            omega_vax = list(vaccine_waning_params['omega_vax_by_age'])
            while len(omega_vax) < n_ages:
                omega_vax.append(omega_vax[-1] if omega_vax else 0.0)
            omega_vax = omega_vax[:n_ages]
        else:
            omega_vax = [0.0] * n_ages
    
    # ========================================
    # Time-Varying Parameter Setup
    # ========================================
    
    # Waning immunity rates
    if waning_params is None:
        omega = [0.0] * n_ages
    elif 'omega' in waning_params:
        omega = [waning_params['omega']] * n_ages
    else:
        omega = [
            waning_params.get('omega_young', 0.0),
            waning_params.get('omega_middle', 0.0),
            waning_params.get('omega_elderly', 0.0)
        ]
        # Extend if more age groups
        while len(omega) < n_ages:
            omega.append(omega[-1] if omega else 0.0)
    
    # ========================================
    # Initialize Compartments
    # ========================================
    ic_defaults = scenarios.DEFAULT_INITIAL_CONDITIONS if initial_conditions is None else {
        **scenarios.DEFAULT_INITIAL_CONDITIONS, **initial_conditions
    }
    
    # Unvaccinated compartments
    I = coerce_initial_vector(ic_defaults, 'I_by_age', n_ages, scenarios.DEFAULT_INITIAL_CONDITIONS['I_by_age'])
    E = coerce_initial_vector(ic_defaults, 'E_by_age', n_ages, scenarios.DEFAULT_INITIAL_CONDITIONS['E_by_age'])
    
    # Handle X compartments (split into queued and admitted)
    X_queued_default = scenarios.DEFAULT_INITIAL_CONDITIONS.get('X_queued_by_age', [0, 0, 0])
    X_admitted_default = scenarios.DEFAULT_INITIAL_CONDITIONS.get('X_admitted_by_age', [0, 0, 0])
    
    X_queued = coerce_initial_vector(ic_defaults, 'X_queued_by_age', n_ages, X_queued_default)
    X_admitted = coerce_initial_vector(ic_defaults, 'X_admitted_by_age', n_ages, X_admitted_default)
    
    H_ward = coerce_initial_vector(ic_defaults, 'H_ward_by_age', n_ages, scenarios.DEFAULT_INITIAL_CONDITIONS['H_ward_by_age'])
    H_icu = coerce_initial_vector(ic_defaults, 'H_icu_by_age', n_ages, scenarios.DEFAULT_INITIAL_CONDITIONS['H_icu_by_age'])
    R = coerce_initial_vector(ic_defaults, 'R_by_age', n_ages, scenarios.DEFAULT_INITIAL_CONDITIONS['R_by_age'])
    D = coerce_initial_vector(ic_defaults, 'D_by_age', n_ages, scenarios.DEFAULT_INITIAL_CONDITIONS['D_by_age'])
    
    # Vaccinated compartments - handle X split
    E_vax = coerce_initial_vector(ic_defaults, 'E_vax_by_age', n_ages, scenarios.DEFAULT_INITIAL_CONDITIONS['E_vax_by_age'])
    I_vax = coerce_initial_vector(ic_defaults, 'I_vax_by_age', n_ages, scenarios.DEFAULT_INITIAL_CONDITIONS['I_vax_by_age'])
    
    X_queued_vax_default = scenarios.DEFAULT_INITIAL_CONDITIONS.get('X_queued_vax_by_age', [0, 0, 0])
    X_admitted_vax_default = scenarios.DEFAULT_INITIAL_CONDITIONS.get('X_admitted_vax_by_age', [0, 0, 0])
    
    X_queued_vax = coerce_initial_vector(ic_defaults, 'X_queued_vax_by_age', n_ages, X_queued_vax_default)
    X_admitted_vax = coerce_initial_vector(ic_defaults, 'X_admitted_vax_by_age', n_ages, X_admitted_vax_default)
    
    H_ward_vax = coerce_initial_vector(ic_defaults, 'H_ward_vax_by_age', n_ages, scenarios.DEFAULT_INITIAL_CONDITIONS['H_ward_vax_by_age'])
    H_icu_vax = coerce_initial_vector(ic_defaults, 'H_icu_vax_by_age', n_ages, scenarios.DEFAULT_INITIAL_CONDITIONS['H_icu_vax_by_age'])
    R_vax = coerce_initial_vector(ic_defaults, 'R_vax_by_age', n_ages, scenarios.DEFAULT_INITIAL_CONDITIONS['R_vax_by_age'])
    D_vax = coerce_initial_vector(ic_defaults, 'D_vax_by_age', n_ages, scenarios.DEFAULT_INITIAL_CONDITIONS['D_vax_by_age'])
    
    # Calculate S_vax based on coverage (initial vaccinated susceptible population)
    # If S_vax_by_age is explicitly provided in initial_conditions, use it
    # Otherwise, distribute based on coverage fraction of remaining susceptible pool
    # Note: X_total = X_queued + X_admitted for each age group
    X_total_init = [X_queued[a] + X_admitted[a] for a in range(n_ages)]
    X_vax_total_init = [X_queued_vax[a] + X_admitted_vax[a] for a in range(n_ages)]
    
    if 'S_vax_by_age' in ic_defaults and ic_defaults['S_vax_by_age'] != scenarios.DEFAULT_INITIAL_CONDITIONS.get('S_vax_by_age', [0, 0, 0]):
        S_vax = coerce_initial_vector(ic_defaults, 'S_vax_by_age', n_ages, [0, 0, 0])
        # Calculate remaining S after all compartments
        S = [max(0, age_pops[a] - E[a] - I[a] - X_total_init[a] - H_ward[a] - H_icu[a] - R[a] - D[a]
                 - S_vax[a] - E_vax[a] - I_vax[a] - X_vax_total_init[a] - H_ward_vax[a] - H_icu_vax[a] - R_vax[a] - D_vax[a]) 
             for a in range(n_ages)]
    else:
        # Calculate total non-S population for each age group
        non_S_total = [E[a] + I[a] + X_total_init[a] + H_ward[a] + H_icu[a] + R[a] + D[a] +
                       E_vax[a] + I_vax[a] + X_vax_total_init[a] + H_ward_vax[a] + H_icu_vax[a] + R_vax[a] + D_vax[a]
                       for a in range(n_ages)]
        # Pool of susceptibles (both vaccinated and unvaccinated)
        S_pool = [max(0, age_pops[a] - non_S_total[a]) for a in range(n_ages)]
        # Distribute according to coverage
        S_vax = [S_pool[a] * coverage[a] for a in range(n_ages)]
        S = [S_pool[a] * (1 - coverage[a]) for a in range(n_ages)]
    
    # Differential mortality tracking (unvaccinated)
    D_treated = [0.0] * n_ages
    D_untreated = [0.0] * n_ages
    
    # Differential mortality tracking (vaccinated)
    D_vax_treated = [0.0] * n_ages
    D_vax_untreated = [0.0] * n_ages
    
    # Breakthrough infection tracking
    cum_breakthrough = [0.0] * n_ages
    
    # Demographic tracking (births and background deaths)
    cum_births = [0.0] * n_ages
    cum_background_deaths = [0.0] * n_ages
    
    # ========================================
    # Validate and Prepare Demographic Parameters
    # ========================================
    validated_demo_params = validate_demographic_params(demographic_params, n_ages)
    
    # Warn about population drift for long simulations with demographics
    if Tmax > 365 and validated_demo_params['birth_rate'] > 0:
        total_pop = sum(age_pops)
        expected_births_per_year = validated_demo_params['birth_rate'] * total_pop * 365
        expected_bg_deaths_per_year = np.sum(validated_demo_params['mu_background'] * np.array(age_pops)) * 365
        net_change_per_year = expected_births_per_year - expected_bg_deaths_per_year
        pct_change = abs(net_change_per_year) / total_pop * 100
        if pct_change > 1.0:  # More than 1% annual change
            warnings.warn(
                f"Long simulation ({Tmax} days) with demographic imbalance detected. "
                f"Expected annual births: {expected_births_per_year:.0f}, "
                f"expected annual background deaths: {expected_bg_deaths_per_year:.0f}. "
                f"Net population change: {net_change_per_year:+.0f} ({pct_change:.1f}%/year). "
                f"Consider adjusting birth_rate or mu_background for population stability.",
                UserWarning
            )
    
    # ========================================
    # Build ODE Parameters Dictionary
    # ========================================
    ode_params = {
        'n_ages': n_ages,
        'beta_base': beta_base,
        'contact_matrix': np.asarray(contact_matrix),
        'age_params': age_params,
        'age_pops': np.array(age_pops),
        'K_ward': K_ward,
        'K_icu': K_icu,
        'n_ward': n_ward,
        'n_icu': n_icu,
        'VE_infection': VE_infection,
        'VE_severe': VE_severe,
        'VE_death': VE_death,
        'theta_X': theta_X,
        'theta_H': theta_H,
        'theta_vax': theta_vax,
        'omega': np.array(omega),
        'omega_vax': np.array(omega_vax),
        'vax_waning_destination': vax_waning_destination,
        'vaccination_rate': np.array(vaccination_rate),
        'seasonal_params': seasonal_params,
        'interventions': interventions,
        'dm_params': scenarios.DIFFERENTIAL_MORTALITY_PARAMS,
        'demographic_params': validated_demo_params,
    }
        
    # Convert ode_params dict to the typed ODEParams dataclass expected by ResultProcessor
    ode_params_obj = ODEParams(**ode_params)
    
    # ========================================
    # Pack Initial State
    # ========================================
    initial_state = {
        'S': np.array(S), 'E': np.array(E), 'I': np.array(I),
        'X_queued': np.array(X_queued), 'X_admitted': np.array(X_admitted),
        'H_ward': np.array(H_ward), 'H_icu': np.array(H_icu), 'R': np.array(R), 'D': np.array(D),
        'S_vax': np.array(S_vax), 'E_vax': np.array(E_vax), 'I_vax': np.array(I_vax),
        'X_queued_vax': np.array(X_queued_vax), 'X_admitted_vax': np.array(X_admitted_vax),
        'H_ward_vax': np.array(H_ward_vax), 'H_icu_vax': np.array(H_icu_vax), 'R_vax': np.array(R_vax), 'D_vax': np.array(D_vax),
        'D_treated': np.array(D_treated), 'D_untreated': np.array(D_untreated),
        'D_vax_treated': np.array(D_vax_treated), 'D_vax_untreated': np.array(D_vax_untreated),
        'cum_breakthrough': np.array(cum_breakthrough),
        'cum_births': np.array(cum_births),
        'cum_background_deaths': np.array(cum_background_deaths),
    }
    y0 = pack_state(initial_state, n_ages)
    
    # ========================================
    # Time Points for Integration
    # ========================================
    n_steps = int(Tmax / time_step) + 1
    t_eval = np.linspace(0, Tmax, n_steps)
    
    # Build tcrit array for intervention discontinuities
    tcrit_points = set()
    for intervention in interventions:
        tcrit_points.add(intervention.get('start_day', 0))
        tcrit_points.add(intervention.get('end_day', Tmax))
    tcrit = sorted([t for t in tcrit_points if 0 < t < Tmax])
    
    # ========================================
    # Run ODE Solver
    # ========================================
    if solver == 'odeint':
        # Use scipy.integrate.odeint (LSODA)
        solution = odeint(
            master_deriv,
            y0,
            t_eval,
            args=(ode_params,),
            tcrit=tcrit if tcrit else None,
            rtol=rtol,
            atol=atol,
            full_output=False
        )
    elif solver == 'solve_ivp':
        # Use scipy.integrate.solve_ivp with configurable method
        sol = solve_ivp(
            master_deriv_solve_ivp,
            (0, Tmax),
            y0,
            method=solver_method,
            t_eval=t_eval,
            args=(ode_params,),
            rtol=rtol,
            atol=atol,
            dense_output=False
        )
        if not sol.success:
            warnings.warn(f"solve_ivp did not converge: {sol.message}")
        solution = sol.y.T  # Transpose to match odeint output shape (n_times, n_states)
    else:
        raise ValueError(f"Unknown solver: {solver}. Use 'odeint' or 'solve_ivp'.")
    
    # ========================================
    # Post-Integration Clipping and NaN Check
    # ========================================
    # Check for NaN/inf values
    if np.any(np.isnan(solution)) or np.any(np.isinf(solution)):
        warnings.warn("ODE solver produced NaN or Inf values. Results may be unreliable.")
        solution = np.nan_to_num(solution, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip negative values with warning threshold
    min_val = np.min(solution)
    if min_val < -clip_warning_threshold:
        warnings.warn(
            f"ODE solver produced significantly negative values (min={min_val:.2e}). "
            f"Clipping to zero. Consider reducing time_step or using a stiffer solver method."
        )
    solution = np.clip(solution, 0, None)
    
    
    # ========================================
    # Process Results
    # ========================================
    from .result_processor import ResultProcessor



    processor = ResultProcessor(
        solution=solution,
        times=list(t_eval),
        ode_params=ode_params_obj,
        coverage=coverage,
        vaccination_rate=vaccination_rate,
        vaccine_waning_params=vaccine_waning_params,
        seasonal_params=seasonal_params,
        waning_params=waning_params,
        interventions=interventions,
        demographic_params=validated_demo_params,
        track_differential_mortality=track_differential_mortality,
        track_compartment_flows=track_compartment_flows,
        solver=solver,
        solver_method=solver_method,
        rtol=rtol,
        atol=atol,
        Tmax=Tmax,
        time_step=time_step,
    )
    
    return processor.process()
# End of simulate_model.py
# More features that I want to add later:
# - Integration with real-world data for parameter fitting
# - Export results to common data formats (CSV, JSON, etc.)
# - User-friendly interface for setting up simulations (maybe a web interface)
# - Maybe look at some performance optimizations for larger scale simulations
# - Interactive dashboards for exploring simulation results