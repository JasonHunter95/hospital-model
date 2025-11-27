"""
Master SEIXHRD hospital model with Three-Factor Vaccination Compartments

This model includes:
- Age-structured compartments (S, E, I, X, H_ward, H_icu, R, D)
- Vaccinated compartments (S_vax, E_vax, I_vax, X_vax, H_ward_vax, H_icu_vax, R_vax, D_vax)
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
"""

import config
from hospital_models import hill_gate, _validate_age_structured_inputs, _coerce_initial_vector
from time_varying_helpers import seasonal_forcing, policy_multiplier


def simulate_master_hospital_model(
    beta_base,
    age_params,
    contact_matrix,
    # Capacity parameters
    ward_capacity=None,
    icu_capacity=None,
    hill_coef_ward=None,
    hill_coef_icu=None,
    # Vaccination parameters (legacy leaky model - for backward compatibility)
    coverage=0.0,
    VE=None,
    # Three-Factor Vaccination Model parameters
    VE_infection=None,
    VE_severe=None,
    VE_death=None,
    vaccination_rate=None,
    theta_vax=None,
    vaccine_waning_params=None,
    # Population
    age_pops=None,
    # Infectiousness modifiers
    theta_X=None,
    theta_H=None,
    # Time-varying parameters
    seasonal_params=None,
    waning_params=None,
    interventions=None,
    # Simulation control
    Tmax=None,
    time_step=None,
    # Tracking options
    track_differential_mortality=True,
    track_compartment_flows=False,
    # Initial conditions
    initial_conditions=None
):
    """
    Simulate the Master SEIXHRD hospital model with Three-Factor Vaccination.
    
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
        - Legacy aliases: 'gamma_H', 'mu_H' (used as fallback)
    contact_matrix : ndarray
        Contact rates C[a,b] between age groups (infector a, infectee b).
        Shape: (n_ages, n_ages).
    ward_capacity : float, optional
        Total general ward capacity. Default from config.
    icu_capacity : float, optional
        Total ICU capacity. Default from config.
    hill_coef_ward : float, optional
        Hill coefficient for ward admission gating. Default from config.
    hill_coef_icu : float, optional
        Hill coefficient for ICU admission gating. Default from config.
    coverage : float or list, optional
        Initial vaccine coverage (fraction already vaccinated at t=0).
        Float for uniform, list for age-specific. Default 0.0.
        When using Three-Factor Model, this determines initial S_vax population.
    VE : float, optional
        Legacy vaccine efficacy (0-1) for backward compatibility.
        If VE_infection/VE_severe/VE_death are not provided, VE is used for all three.
    VE_infection : float, optional
        Three-Factor Model: Efficacy against infection (0-1).
        Reduces force of infection for vaccinated susceptibles.
        Default from config.VACCINE_EFFICACY_PARAMS.
    VE_severe : float, optional
        Three-Factor Model: Efficacy against severe disease (0-1).
        Reduces sigma (I→X progression) for vaccinated individuals.
        Default from config.VACCINE_EFFICACY_PARAMS.
    VE_death : float, optional
        Three-Factor Model: Efficacy against death (0-1).
        Reduces mortality rates (mu_*) for vaccinated individuals.
        Default from config.VACCINE_EFFICACY_PARAMS.
    vaccination_rate : float or list, optional
        Daily vaccination rate (fraction of S → S_vax per day).
        Float for uniform rate, list for age-specific [young, middle, elderly].
        Default 0.0 (no ongoing vaccination, only initial coverage).
    theta_vax : float, optional
        Relative infectiousness of vaccinated infectious individuals.
        Default from config.
    vaccine_waning_params : dict, optional
        Vaccine immunity waning parameters:
        - 'omega_vax': vaccine waning rate (1/days)
        - 'omega_vax_by_age': age-specific waning rates [young, middle, elderly]
        - 'waning_destination': 'S' (fully susceptible) or 'S_vax' (partial protection)
    age_pops : list
        Population size for each age group. Required.
    theta_X : float, optional
        Relative infectiousness of X compartment. Default from config.
    theta_H : float, optional
        Relative infectiousness of hospitalized compartments. Default from config.
    seasonal_params : dict, optional
        Seasonal forcing parameters:
        - 'amplitude': seasonal amplitude (0-1), default 0
        - 'period': period in days, default 365
        - 'peak_day': day of peak transmission, default 0
    waning_params : dict, optional
        Natural immunity waning parameters (R → S):
        - 'omega': uniform waning rate (1/days), OR
        - 'omega_young', 'omega_middle', 'omega_elderly': age-specific rates
    interventions : list of dict, optional
        Policy interventions, each with:
        - 'start_day': intervention start
        - 'end_day': intervention end  
        - 'transmission_reduction': fraction reduction (0-1)
    Tmax : float, optional
        Simulation duration in days. Default from config.
    time_step : float, optional
        Integration time step. Default from config.
    track_differential_mortality : bool, optional
        Track deaths by care status. Default True.
    track_compartment_flows : bool, optional
        Track daily flows between compartments. Default False.
    initial_conditions : dict, optional
        Override default initial conditions. Supports both unvaccinated
        (*_by_age) and vaccinated (*_vax_by_age) compartment keys.
    
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
    
    Examples
    --------
    >>> from config import (AGE_PARAMS_DEFAULT, CONTACT_MATRIX_DEFAULT,
    ...                     get_vaccine_profile)
    >>> 
    >>> # Three-Factor Vaccination simulation
    >>> vaccine = get_vaccine_profile('mrna_original')
    >>> results = simulate_master_hospital_model(
    ...     beta_base=0.3,
    ...     age_params=AGE_PARAMS_DEFAULT,
    ...     contact_matrix=CONTACT_MATRIX_DEFAULT,
    ...     ward_capacity=80,
    ...     icu_capacity=20,
    ...     coverage=[0.2, 0.3, 0.7],  # initial vaccinated population
    ...     VE_infection=vaccine['VE_infection'],
    ...     VE_severe=vaccine['VE_severe'],
    ...     VE_death=vaccine['VE_death'],
    ...     theta_vax=vaccine['theta_vax'],
    ...     vaccination_rate=0.005,  # ongoing vaccination
    ...     vaccine_waning_params={'omega_vax': vaccine['omega_vax']},
    ...     age_pops=[3000, 5000, 2000],
    ...     Tmax=365
    ... )
    >>> 
    >>> print(f"Peak ICU: {max(results['H_icu_total']):.0f}")
    >>> print(f"Total deaths: {results['D_total'][-1]:.0f}")
    >>> print(f"Vaccinated deaths: {results['D_vax_total'][-1]:.0f}")
    >>> print(f"Breakthrough infections: {results['breakthrough_infections'][-1]:.0f}")
    """
    # ========================================
    # Parameter Setup with Defaults
    # ========================================
    if age_pops is None:
        raise ValueError("age_pops must be provided.")
    
    n_ages = len(age_pops)
    
    # Load defaults from config
    theta_X = config.DEFAULT_SIM_PARAMS['theta_X'] if theta_X is None else theta_X
    theta_H = config.DEFAULT_SIM_PARAMS['theta_H'] if theta_H is None else theta_H
    Tmax = config.DEFAULT_SIM_PARAMS['Tmax'] if Tmax is None else Tmax
    time_step = config.DEFAULT_SIM_PARAMS['time_step'] if time_step is None else time_step
    VE = config.DEFAULT_SIM_PARAMS['VE'] if VE is None else VE
    
    # Capacity parameters
    K_ward = config.DEFAULT_CAPACITY_PARAMS['ward_capacity'] if ward_capacity is None else ward_capacity
    K_icu = config.DEFAULT_CAPACITY_PARAMS['icu_capacity'] if icu_capacity is None else icu_capacity
    n_ward = config.DEFAULT_CAPACITY_PARAMS['hill_coef_ward'] if hill_coef_ward is None else hill_coef_ward
    n_icu = config.DEFAULT_CAPACITY_PARAMS['hill_coef_icu'] if hill_coef_icu is None else hill_coef_icu
    
    dt = time_step
    
    # Validate inputs
    _validate_age_structured_inputs(age_params, contact_matrix, age_pops, coverage)
    
    # Handle coverage - convert to list if scalar
    if not isinstance(coverage, list):
        coverage = [coverage] * n_ages
    
    # ========================================
    # Three-Factor Vaccine Model Setup
    # ========================================
    # If VE_infection/VE_severe/VE_death are not provided, use legacy VE for backward compatibility
    # This maps the old leaky model to the three-factor model
    vaccine_params = config.VACCINE_EFFICACY_PARAMS
    
    if VE_infection is None:
        VE_infection = vaccine_params['VE_infection'] if any(c > 0 for c in coverage) else 0.0
    if VE_severe is None:
        VE_severe = vaccine_params['VE_severe'] if any(c > 0 for c in coverage) else 0.0
    if VE_death is None:
        VE_death = vaccine_params['VE_death'] if any(c > 0 for c in coverage) else 0.0
    
    # Vaccinated infectiousness modifier
    theta_vax = config.DEFAULT_SIM_PARAMS.get('theta_vax', 0.5) if theta_vax is None else theta_vax
    
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
    
    # Seasonal parameters
    if seasonal_params is None:
        seasonal_params = {'amplitude': 0.0, 'period': 365, 'peak_day': 0}
    
    # Interventions
    if interventions is None:
        interventions = []
    
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
    ic_defaults = config.DEFAULT_INITIAL_CONDITIONS if initial_conditions is None else {
        **config.DEFAULT_INITIAL_CONDITIONS, **initial_conditions
    }
    
    # Unvaccinated compartments
    I = _coerce_initial_vector(ic_defaults, 'I_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['I_by_age'])
    E = _coerce_initial_vector(ic_defaults, 'E_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['E_by_age'])
    X = _coerce_initial_vector(ic_defaults, 'X_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['X_by_age'])
    H_ward = _coerce_initial_vector(ic_defaults, 'H_ward_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_ward_by_age'])
    H_icu = _coerce_initial_vector(ic_defaults, 'H_icu_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_icu_by_age'])
    R = _coerce_initial_vector(ic_defaults, 'R_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['R_by_age'])
    D = _coerce_initial_vector(ic_defaults, 'D_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['D_by_age'])
    
    # Vaccinated compartments
    E_vax = _coerce_initial_vector(ic_defaults, 'E_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['E_vax_by_age'])
    I_vax = _coerce_initial_vector(ic_defaults, 'I_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['I_vax_by_age'])
    X_vax = _coerce_initial_vector(ic_defaults, 'X_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['X_vax_by_age'])
    H_ward_vax = _coerce_initial_vector(ic_defaults, 'H_ward_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_ward_vax_by_age'])
    H_icu_vax = _coerce_initial_vector(ic_defaults, 'H_icu_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_icu_vax_by_age'])
    R_vax = _coerce_initial_vector(ic_defaults, 'R_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['R_vax_by_age'])
    D_vax = _coerce_initial_vector(ic_defaults, 'D_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['D_vax_by_age'])
    
    # Calculate S_vax based on coverage (initial vaccinated susceptible population)
    # If S_vax_by_age is explicitly provided in initial_conditions, use it
    # Otherwise, distribute based on coverage fraction of remaining susceptible pool
    if 'S_vax_by_age' in ic_defaults and ic_defaults['S_vax_by_age'] != config.DEFAULT_INITIAL_CONDITIONS.get('S_vax_by_age', [0, 0, 0]):
        S_vax = _coerce_initial_vector(ic_defaults, 'S_vax_by_age', n_ages, [0, 0, 0])
        # Calculate remaining S after all compartments
        S = [max(0, age_pops[a] - E[a] - I[a] - X[a] - H_ward[a] - H_icu[a] - R[a] - D[a]
                 - S_vax[a] - E_vax[a] - I_vax[a] - X_vax[a] - H_ward_vax[a] - H_icu_vax[a] - R_vax[a] - D_vax[a]) 
             for a in range(n_ages)]
    else:
        # Calculate total non-S population for each age group
        non_S_total = [E[a] + I[a] + X[a] + H_ward[a] + H_icu[a] + R[a] + D[a] +
                       E_vax[a] + I_vax[a] + X_vax[a] + H_ward_vax[a] + H_icu_vax[a] + R_vax[a] + D_vax[a]
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
    
    # ========================================
    # Storage Arrays
    # ========================================
    times = []
    
    # Per-age compartment histories (unvaccinated)
    S_history = [[] for _ in range(n_ages)]
    E_history = [[] for _ in range(n_ages)]
    I_history = [[] for _ in range(n_ages)]
    X_history = [[] for _ in range(n_ages)]
    H_ward_history = [[] for _ in range(n_ages)]
    H_icu_history = [[] for _ in range(n_ages)]
    R_history = [[] for _ in range(n_ages)]
    D_history = [[] for _ in range(n_ages)]
    D_treated_history = [[] for _ in range(n_ages)]
    D_untreated_history = [[] for _ in range(n_ages)]
    
    # Per-age compartment histories (vaccinated)
    S_vax_history = [[] for _ in range(n_ages)]
    E_vax_history = [[] for _ in range(n_ages)]
    I_vax_history = [[] for _ in range(n_ages)]
    X_vax_history = [[] for _ in range(n_ages)]
    H_ward_vax_history = [[] for _ in range(n_ages)]
    H_icu_vax_history = [[] for _ in range(n_ages)]
    R_vax_history = [[] for _ in range(n_ages)]
    D_vax_history = [[] for _ in range(n_ages)]
    D_vax_treated_history = [[] for _ in range(n_ages)]
    D_vax_untreated_history = [[] for _ in range(n_ages)]
    
    # Aggregated histories (unvaccinated)
    H_ward_total_history = []
    H_icu_total_history = []
    H_total_history = []
    E_total_history = []
    I_total_history = []
    X_total_history = []
    D_total_history = []
    D_treated_total_history = []
    D_untreated_total_history = []
    
    # Aggregated histories (vaccinated)
    H_ward_vax_total_history = []
    H_icu_vax_total_history = []
    H_vax_total_history = []
    E_vax_total_history = []
    I_vax_total_history = []
    X_vax_total_history = []
    D_vax_total_history = []
    vaccinated_total_history = []
    breakthrough_infections_history = []
    
    # Capacity metrics
    ward_overflow_history = []
    icu_overflow_history = []
    g_ward_history = []
    g_icu_history = []
    
    # Time-varying parameter tracking
    beta_t_history = []
    seasonal_factor_history = []
    policy_mult_history = []
    
    # Flow tracking (optional) - always create lists so .append() is safe;
    # they will only be included in final results if track_compartment_flows is True.
    new_infections_history = []
    ward_admissions_history = []
    icu_admissions_history = []
    new_vaccinations_history = []
    breakthrough_infections_daily_history = []
    
    # Cumulative metrics
    cum_ward_overflow = 0
    cum_icu_overflow = 0
    cum_unmet_ward = [0.0] * n_ages
    cum_unmet_icu = [0.0] * n_ages
    
    # ========================================
    # Main Simulation Loop
    # ========================================
    t = 0
    while t <= Tmax:
        times.append(t)
        
        # Store current state (unvaccinated)
        for a in range(n_ages):
            S_history[a].append(S[a])
            E_history[a].append(E[a])
            I_history[a].append(I[a])
            X_history[a].append(X[a])
            H_ward_history[a].append(H_ward[a])
            H_icu_history[a].append(H_icu[a])
            R_history[a].append(R[a])
            D_history[a].append(D[a])
            D_treated_history[a].append(D_treated[a])
            D_untreated_history[a].append(D_untreated[a])
            
            # Store vaccinated compartments
            S_vax_history[a].append(S_vax[a])
            E_vax_history[a].append(E_vax[a])
            I_vax_history[a].append(I_vax[a])
            X_vax_history[a].append(X_vax[a])
            H_ward_vax_history[a].append(H_ward_vax[a])
            H_icu_vax_history[a].append(H_icu_vax[a])
            R_vax_history[a].append(R_vax[a])
            D_vax_history[a].append(D_vax[a])
            D_vax_treated_history[a].append(D_vax_treated[a])
            D_vax_untreated_history[a].append(D_vax_untreated[a])
        
        # Aggregated totals (combined vaccinated + unvaccinated for hospital capacity)
        H_ward_total = sum(H_ward) + sum(H_ward_vax)
        H_icu_total = sum(H_icu) + sum(H_icu_vax)
        H_total = H_ward_total + H_icu_total
        E_total = sum(E) + sum(E_vax)
        I_total = sum(I) + sum(I_vax)
        X_total = sum(X) + sum(X_vax)
        D_total = sum(D) + sum(D_vax)
        
        # Vaccinated totals
        H_ward_vax_total = sum(H_ward_vax)
        H_icu_vax_total = sum(H_icu_vax)
        H_vax_total = H_ward_vax_total + H_icu_vax_total
        vaccinated_total = sum(S_vax) + sum(E_vax) + sum(I_vax) + sum(X_vax) + H_vax_total + sum(R_vax) + sum(D_vax)
        
        H_ward_total_history.append(H_ward_total)
        H_icu_total_history.append(H_icu_total)
        H_total_history.append(H_total)
        E_total_history.append(E_total)
        I_total_history.append(I_total)
        X_total_history.append(X_total)
        D_total_history.append(D_total)
        D_treated_total_history.append(sum(D_treated) + sum(D_vax_treated))
        D_untreated_total_history.append(sum(D_untreated) + sum(D_vax_untreated))
        
        # Vaccinated aggregates
        H_ward_vax_total_history.append(H_ward_vax_total)
        H_icu_vax_total_history.append(H_icu_vax_total)
        H_vax_total_history.append(H_vax_total)
        E_vax_total_history.append(sum(E_vax))
        I_vax_total_history.append(sum(I_vax))
        X_vax_total_history.append(sum(X_vax))
        D_vax_total_history.append(sum(D_vax))
        vaccinated_total_history.append(vaccinated_total)
        breakthrough_infections_history.append(sum(cum_breakthrough))
        
        # ========================================
        # Time-Varying Transmission
        # ========================================
        seasonal_factor = seasonal_forcing(
            t, 1.0,
            amplitude=seasonal_params.get('amplitude', 0.0),
            period=seasonal_params.get('period', 365),
            peak_day=seasonal_params.get('peak_day', 0)
        )
        
        policy_mult = policy_multiplier(t, interventions)
        
        beta_t = beta_base * seasonal_factor * policy_mult
        
        beta_t_history.append(beta_t)
        seasonal_factor_history.append(seasonal_factor)
        policy_mult_history.append(policy_mult)
        
        # ========================================
        # Capacity Gating (combined vaccinated + unvaccinated)
        # ========================================
        g_ward = hill_gate(H_ward_total, K_ward, n_ward)
        g_icu = hill_gate(H_icu_total, K_icu, n_icu)
        
        g_ward_history.append(g_ward)
        g_icu_history.append(g_icu)
        
        # ========================================
        # Force of Infection (includes both vaccinated and unvaccinated infectious)
        # ========================================
        # lambda_foi is the base force of infection affecting unvaccinated susceptibles
        # lambda_foi_vax is the reduced force affecting vaccinated susceptibles (multiplied by 1-VE_infection)
        lambda_foi = [0.0] * n_ages
        for a in range(n_ages):
            for b in range(n_ages):
                # Hospital contribution (both vaccinated and unvaccinated)
                H_contrib = H_ward[b] + H_icu[b] + H_ward_vax[b] + H_icu_vax[b]
                
                # Infectious contribution:
                # - Unvaccinated I, X have full infectiousness (modified by theta_X)
                # - Vaccinated I_vax, X_vax have reduced infectiousness (theta_vax modifier)
                infectious_unvax = I[b] + theta_X * X[b]
                infectious_vax = theta_vax * (I_vax[b] + theta_X * X_vax[b])
                
                # Total live population for age group b
                live_pop_b = (S[b] + E[b] + I[b] + X[b] + H_ward[b] + H_icu[b] + R[b] +
                              S_vax[b] + E_vax[b] + I_vax[b] + X_vax[b] + H_ward_vax[b] + H_icu_vax[b] + R_vax[b])
                
                if live_pop_b > 0:
                    lambda_foi[a] += (beta_t * contact_matrix[a][b] *
                                     (infectious_unvax + infectious_vax + theta_H * H_contrib) / live_pop_b)
        
        # Force of infection for vaccinated (reduced by VE_infection)
        lambda_foi_vax = [(1 - VE_infection) * lambda_foi[a] for a in range(n_ages)]
        
        # ========================================
        # Transitions (Unvaccinated)
        # ========================================
        # S -> E: new exposures (latent period begins)
        new_exposed = [lambda_foi[a] * S[a] for a in range(n_ages)]
        
        # E -> I: becoming infectious (latent period ends)
        becoming_infectious = [age_params[a].get('alpha', 0.2) * E[a] for a in range(n_ages)]
        
        # Ward admissions from X (unvaccinated)
        admit_ward = [age_params[a]['eta'] * X[a] * g_ward for a in range(n_ages)]
        
        # ICU admissions from ward (patients needing escalation)
        need_icu = [age_params[a].get('eta_icu', 0.1) * H_ward[a] for a in range(n_ages)]
        admit_icu = [need_icu[a] * g_icu for a in range(n_ages)]
        
        # Waning natural immunity flow (R -> S)
        waning_flow = [omega[a] * R[a] for a in range(n_ages)]
        
        # Vaccination flow (S -> S_vax)
        new_vaccinations = [vaccination_rate[a] * S[a] for a in range(n_ages)]
        
        # ========================================
        # Transitions (Vaccinated - Breakthrough Infections)
        # ========================================
        # S_vax -> E_vax: breakthrough exposures
        new_exposed_vax = [lambda_foi_vax[a] * S_vax[a] for a in range(n_ages)]
        
        # Track breakthrough infections
        for a in range(n_ages):
            cum_breakthrough[a] += new_exposed_vax[a] * dt
        
        # E_vax -> I_vax: becoming infectious
        becoming_infectious_vax = [age_params[a].get('alpha', 0.2) * E_vax[a] for a in range(n_ages)]
        
        # I_vax -> X_vax: progression to severe (reduced by VE_severe)
        sigma_vax = [(1 - VE_severe) * age_params[a]['sigma'] for a in range(n_ages)]
        
        # Ward admissions from X_vax (vaccinated)
        admit_ward_vax = [age_params[a]['eta'] * X_vax[a] * g_ward for a in range(n_ages)]
        
        # ICU admissions from ward (vaccinated)
        need_icu_vax = [age_params[a].get('eta_icu', 0.1) * H_ward_vax[a] for a in range(n_ages)]
        admit_icu_vax = [need_icu_vax[a] * g_icu for a in range(n_ages)]
        
        # Waning vaccine immunity flow (R_vax -> S or S_vax)
        waning_flow_vax = [omega_vax[a] * R_vax[a] for a in range(n_ages)]
        
        # Unmet care calculations (combined)
        unmet_ward = [max(0, age_params[a]['eta'] * (X[a] + X_vax[a]) - admit_ward[a] - admit_ward_vax[a]) 
                      for a in range(n_ages)]
        unmet_icu = [max(0, need_icu[a] + need_icu_vax[a] - admit_icu[a] - admit_icu_vax[a]) 
                     for a in range(n_ages)]
        
        # Track flows if requested
        if track_compartment_flows:
            new_infections_history.append(list(becoming_infectious))
            ward_admissions_history.append([admit_ward[a] + admit_ward_vax[a] for a in range(n_ages)])
            icu_admissions_history.append([admit_icu[a] + admit_icu_vax[a] for a in range(n_ages)])
            new_vaccinations_history.append(list(new_vaccinations))
            breakthrough_infections_daily_history.append(list(new_exposed_vax))
        
        # ========================================
        # ODEs - Unvaccinated Compartments
        # ========================================
        # dS/dt = -lambda*S + omega*R - vaccination_rate*S
        dS = [-new_exposed[a] + waning_flow[a] - new_vaccinations[a] for a in range(n_ages)]
        
        # Add vaccine waning contribution if destination is S (fully susceptible)
        if vax_waning_destination == 'S':
            dS = [dS[a] + waning_flow_vax[a] for a in range(n_ages)]
        
        # dE/dt = lambda*S - alpha*E
        dE = [new_exposed[a] - becoming_infectious[a] for a in range(n_ages)]
        
        # dI/dt = alpha*E - (gamma_I + mu_I + sigma)*I
        dI = [becoming_infectious[a] - (age_params[a]['gamma_I'] + age_params[a]['mu_I'] +
              age_params[a]['sigma']) * I[a] for a in range(n_ages)]
        
        dH_ward = []
        dH_icu = []
        dX = []
        dR = []
        dD = []
        dD_treated = []
        dD_untreated = []
        
        # Vaccinated derivatives
        dS_vax = []
        dE_vax = []
        dI_vax = []
        dX_vax = []
        dH_ward_vax = []
        dH_icu_vax = []
        dR_vax = []
        dD_vax = []
        dD_vax_treated = []
        dD_vax_untreated = []
        
        for a in range(n_ages):
            # Get ward/ICU parameters with fallbacks
            gamma_w = age_params[a].get('gamma_ward', age_params[a].get('gamma_H', 0.2))
            mu_w = age_params[a].get('mu_ward', age_params[a].get('mu_H', 0.02) * 0.5)
            gamma_i = age_params[a].get('gamma_icu', age_params[a].get('gamma_H', 0.2) * 0.6)
            mu_i = age_params[a].get('mu_icu', age_params[a].get('mu_H', 0.02) * 2.0)
            
            # ========================================
            # Differential Mortality Rates (Unvaccinated)
            # ========================================
            mu_X_treated = age_params[a]['mu_X']
            dm_params = config.DIFFERENTIAL_MORTALITY_PARAMS
            
            # Age-specific multipliers
            age_keys = ['young', 'middle', 'elderly']
            age_key = age_keys[a] if a < len(age_keys) else None
            
            mu_X_multiplier = dm_params.get(
                f'mu_X_untreated_multiplier_{age_key}', dm_params['mu_X_untreated_multiplier']
            ) if age_key else dm_params['mu_X_untreated_multiplier']
            
            mu_ward_multiplier = dm_params.get(
                f'mu_ward_denied_icu_multiplier_{age_key}', dm_params['mu_ward_denied_icu_multiplier']
            ) if age_key else dm_params['mu_ward_denied_icu_multiplier']
            
            # Use explicit params if provided, else calculate from multipliers
            mu_X_untreated = age_params[a].get('mu_X_untreated', mu_X_treated * mu_X_multiplier)
            mu_ward_denied = age_params[a].get('mu_ward_denied_icu', mu_w * mu_ward_multiplier)
            
            # Effective mortality rates based on care availability (unvaccinated)
            fraction_X_denied = 1.0 - g_ward if X[a] > 0 else 0.0
            effective_mu_X = mu_X_treated * g_ward + mu_X_untreated * fraction_X_denied
            
            fraction_icu_denied = 1.0 - g_icu if H_ward[a] > 0 and need_icu[a] > 0 else 0.0
            eta_icu_a = age_params[a].get('eta_icu', 0.1)
            effective_mu_ward = mu_w + (mu_ward_denied - mu_w) * eta_icu_a * fraction_icu_denied
            
            # ========================================
            # Death Flows for Tracking (Unvaccinated)
            # ========================================
            deaths_I = age_params[a]['mu_I'] * I[a]
            deaths_X_treated = mu_X_treated * g_ward * X[a]
            deaths_X_untreated = mu_X_untreated * fraction_X_denied * X[a]
            deaths_ward_baseline = mu_w * H_ward[a]
            deaths_ward_icu_denied = (mu_ward_denied - mu_w) * eta_icu_a * fraction_icu_denied * H_ward[a]
            deaths_icu = mu_i * H_icu[a]
            
            # ========================================
            # Vaccinated Mortality Rates (reduced by VE_death)
            # ========================================
            mu_I_vax = (1 - VE_death) * age_params[a]['mu_I']
            mu_X_treated_vax = (1 - VE_death) * mu_X_treated
            mu_X_untreated_vax = (1 - VE_death) * mu_X_untreated
            mu_w_vax = (1 - VE_death) * mu_w
            mu_ward_denied_vax = (1 - VE_death) * mu_ward_denied
            mu_i_vax = (1 - VE_death) * mu_i
            
            # Effective mortality for vaccinated based on care availability
            fraction_X_vax_denied = 1.0 - g_ward if X_vax[a] > 0 else 0.0
            effective_mu_X_vax = mu_X_treated_vax * g_ward + mu_X_untreated_vax * fraction_X_vax_denied
            
            fraction_icu_vax_denied = 1.0 - g_icu if H_ward_vax[a] > 0 and need_icu_vax[a] > 0 else 0.0
            effective_mu_ward_vax = mu_w_vax + (mu_ward_denied_vax - mu_w_vax) * eta_icu_a * fraction_icu_vax_denied
            
            # ========================================
            # Death Flows for Tracking (Vaccinated)
            # ========================================
            deaths_I_vax = mu_I_vax * I_vax[a]
            deaths_X_treated_vax = mu_X_treated_vax * g_ward * X_vax[a]
            deaths_X_untreated_vax = mu_X_untreated_vax * fraction_X_vax_denied * X_vax[a]
            deaths_ward_baseline_vax = mu_w_vax * H_ward_vax[a]
            deaths_ward_icu_denied_vax = (mu_ward_denied_vax - mu_w_vax) * eta_icu_a * fraction_icu_vax_denied * H_ward_vax[a]
            deaths_icu_vax = mu_i_vax * H_icu_vax[a]
            
            # ========================================
            # Compartment Derivatives (Unvaccinated)
            # ========================================
            dX.append(age_params[a]['sigma'] * I[a] - 
                     (age_params[a]['gamma_X'] + effective_mu_X) * X[a] - admit_ward[a])
            
            dH_ward.append(admit_ward[a] - (gamma_w + effective_mu_ward) * H_ward[a] - admit_icu[a])
            
            dH_icu.append(admit_icu[a] - (gamma_i + mu_i) * H_icu[a])
            
            dR.append(age_params[a]['gamma_I'] * I[a] + 
                     age_params[a]['gamma_X'] * X[a] +
                     gamma_w * H_ward[a] + 
                     gamma_i * H_icu[a] - 
                     waning_flow[a])
            
            total_deaths = (deaths_I + deaths_X_treated + deaths_X_untreated +
                          deaths_ward_baseline + deaths_ward_icu_denied + deaths_icu)
            dD.append(total_deaths)
            
            # Differential mortality tracking (unvaccinated)
            dD_treated.append(deaths_I + deaths_X_treated + deaths_ward_baseline + deaths_icu)
            dD_untreated.append(deaths_X_untreated + deaths_ward_icu_denied)
            
            # ========================================
            # Compartment Derivatives (Vaccinated)
            # ========================================
            # dS_vax/dt = vaccination_rate*S - lambda_vax*S_vax [+ omega_vax*R_vax if waning to S_vax]
            dS_vax_a = new_vaccinations[a] - new_exposed_vax[a]
            if vax_waning_destination == 'S_vax':
                dS_vax_a += waning_flow_vax[a]
            dS_vax.append(dS_vax_a)
            
            # dE_vax/dt = lambda_vax*S_vax - alpha*E_vax
            dE_vax.append(new_exposed_vax[a] - becoming_infectious_vax[a])
            
            # dI_vax/dt = alpha*E_vax - (gamma_I + mu_I_vax + sigma_vax)*I_vax
            dI_vax.append(becoming_infectious_vax[a] - (age_params[a]['gamma_I'] + mu_I_vax +
                          sigma_vax[a]) * I_vax[a])
            
            # dX_vax/dt = sigma_vax*I_vax - (gamma_X + effective_mu_X_vax)*X_vax - admit_ward_vax
            dX_vax.append(sigma_vax[a] * I_vax[a] - 
                         (age_params[a]['gamma_X'] + effective_mu_X_vax) * X_vax[a] - admit_ward_vax[a])
            
            # dH_ward_vax/dt
            dH_ward_vax.append(admit_ward_vax[a] - (gamma_w + effective_mu_ward_vax) * H_ward_vax[a] - admit_icu_vax[a])
            
            # dH_icu_vax/dt
            dH_icu_vax.append(admit_icu_vax[a] - (gamma_i + mu_i_vax) * H_icu_vax[a])
            
            # dR_vax/dt
            dR_vax.append(age_params[a]['gamma_I'] * I_vax[a] + 
                         age_params[a]['gamma_X'] * X_vax[a] +
                         gamma_w * H_ward_vax[a] + 
                         gamma_i * H_icu_vax[a] - 
                         waning_flow_vax[a])
            
            total_deaths_vax = (deaths_I_vax + deaths_X_treated_vax + deaths_X_untreated_vax +
                               deaths_ward_baseline_vax + deaths_ward_icu_denied_vax + deaths_icu_vax)
            dD_vax.append(total_deaths_vax)
            
            # Differential mortality tracking (vaccinated)
            dD_vax_treated.append(deaths_I_vax + deaths_X_treated_vax + deaths_ward_baseline_vax + deaths_icu_vax)
            dD_vax_untreated.append(deaths_X_untreated_vax + deaths_ward_icu_denied_vax)
        
        # ========================================
        # Euler Update
        # ========================================
        for a in range(n_ages):
            # Unvaccinated
            S[a] = max(0, S[a] + dS[a] * dt)
            E[a] = max(0, E[a] + dE[a] * dt)
            I[a] = max(0, I[a] + dI[a] * dt)
            X[a] = max(0, X[a] + dX[a] * dt)
            H_ward[a] = max(0, H_ward[a] + dH_ward[a] * dt)
            H_icu[a] = max(0, H_icu[a] + dH_icu[a] * dt)
            R[a] = max(0, R[a] + dR[a] * dt)
            D[a] = max(0, D[a] + dD[a] * dt)
            D_treated[a] = max(0, D_treated[a] + dD_treated[a] * dt)
            D_untreated[a] = max(0, D_untreated[a] + dD_untreated[a] * dt)
            
            # Vaccinated
            S_vax[a] = max(0, S_vax[a] + dS_vax[a] * dt)
            E_vax[a] = max(0, E_vax[a] + dE_vax[a] * dt)
            I_vax[a] = max(0, I_vax[a] + dI_vax[a] * dt)
            X_vax[a] = max(0, X_vax[a] + dX_vax[a] * dt)
            H_ward_vax[a] = max(0, H_ward_vax[a] + dH_ward_vax[a] * dt)
            H_icu_vax[a] = max(0, H_icu_vax[a] + dH_icu_vax[a] * dt)
            R_vax[a] = max(0, R_vax[a] + dR_vax[a] * dt)
            D_vax[a] = max(0, D_vax[a] + dD_vax[a] * dt)
            D_vax_treated[a] = max(0, D_vax_treated[a] + dD_vax_treated[a] * dt)
            D_vax_untreated[a] = max(0, D_vax_untreated[a] + dD_vax_untreated[a] * dt)
            
            # Track unmet care
            cum_unmet_ward[a] += unmet_ward[a] * dt
            cum_unmet_icu[a] += unmet_icu[a] * dt
        
        # Track overflow
        ward_overflow = max(0, H_ward_total - K_ward)
        icu_overflow = max(0, H_icu_total - K_icu)
        cum_ward_overflow += ward_overflow * dt
        cum_icu_overflow += icu_overflow * dt
        ward_overflow_history.append(ward_overflow)
        icu_overflow_history.append(icu_overflow)
        
        t += dt
    
    # ========================================
    # Build Results Dictionary
    # ========================================
    results = {
        # Time
        'times': times,
        
        # Per-age compartments (unvaccinated)
        'S': S_history,
        'E': E_history,
        'I': I_history,
        'X': X_history,
        'H_ward': H_ward_history,
        'H_icu': H_icu_history,
        'H': [[(H_ward_history[a][t] + H_icu_history[a][t]) 
               for t in range(len(times))] for a in range(n_ages)],
        'R': R_history,
        'D': D_history,
        
        # Per-age compartments (vaccinated)
        'S_vax': S_vax_history,
        'E_vax': E_vax_history,
        'I_vax': I_vax_history,
        'X_vax': X_vax_history,
        'H_ward_vax': H_ward_vax_history,
        'H_icu_vax': H_icu_vax_history,
        'H_vax': [[(H_ward_vax_history[a][t] + H_icu_vax_history[a][t]) 
                   for t in range(len(times))] for a in range(n_ages)],
        'R_vax': R_vax_history,
        'D_vax': D_vax_history,
        
        # Aggregated totals (combined unvaccinated + vaccinated)
        'H_ward_total': H_ward_total_history,
        'H_icu_total': H_icu_total_history,
        'H_total': H_total_history,
        'E_total': E_total_history,
        'I_total': I_total_history,
        'X_total': X_total_history,
        'D_total': D_total_history,
        
        # Aggregated totals (vaccinated only)
        'H_ward_vax_total': H_ward_vax_total_history,
        'H_icu_vax_total': H_icu_vax_total_history,
        'H_vax_total': H_vax_total_history,
        'E_vax_total': E_vax_total_history,
        'I_vax_total': I_vax_total_history,
        'X_vax_total': X_vax_total_history,
        'D_vax_total': D_vax_total_history,
        'vaccinated_total': vaccinated_total_history,
        'breakthrough_infections': breakthrough_infections_history,
        
        # Capacity metrics
        'ward_overflow': ward_overflow_history,
        'icu_overflow': icu_overflow_history,
        'cum_ward_overflow': cum_ward_overflow,
        'cum_icu_overflow': cum_icu_overflow,
        'cum_overflow': cum_ward_overflow + cum_icu_overflow,
        'cum_unmet_ward': cum_unmet_ward,
        'cum_unmet_icu': cum_unmet_icu,
        'cum_unmet': [cum_unmet_ward[a] + cum_unmet_icu[a] for a in range(n_ages)],
        'g_ward': g_ward_history,
        'g_icu': g_icu_history,
        
        # Time-varying parameters
        'beta_t': beta_t_history,
        'seasonal_factor': seasonal_factor_history,
        'policy_mult': policy_mult_history,
        
        # Metadata
        'ward_capacity': K_ward,
        'icu_capacity': K_icu,
        'age_pops': age_pops,
        
        # Parameters for reproducibility
        'parameters': {
            'beta_base': beta_base,
            'coverage': coverage,
            'VE': VE,
            'VE_infection': VE_infection,
            'VE_severe': VE_severe,
            'VE_death': VE_death,
            'vaccination_rate': vaccination_rate,
            'theta_vax': theta_vax,
            'vaccine_waning_params': vaccine_waning_params,
            'theta_X': theta_X,
            'theta_H': theta_H,
            'seasonal_params': seasonal_params,
            'waning_params': waning_params,
            'interventions': interventions,
            'Tmax': Tmax,
            'time_step': time_step,
            'track_differential_mortality': track_differential_mortality,
            'track_compartment_flows': track_compartment_flows,
            'age_params': age_params,
            'contact_matrix': contact_matrix
        }
    }
    
    # Differential mortality results
    if track_differential_mortality:
        results.update({
            'D_treated': D_treated_history,
            'D_untreated': D_untreated_history,
            'D_treated_total': D_treated_total_history,
            'D_untreated_total': D_untreated_total_history,
            # Vaccinated differential mortality
            'D_vax_treated': D_vax_treated_history,
            'D_vax_untreated': D_vax_untreated_history,
        })
    
    # Compartment flows
    if track_compartment_flows:
        results.update({
            'new_infections': new_infections_history,
            'ward_admissions': ward_admissions_history,
            'icu_admissions': icu_admissions_history,
            'new_vaccinations': new_vaccinations_history,
            'breakthrough_infections_daily': breakthrough_infections_daily_history,
        })
    
    return results

# End of master_hospital_model.py
# Vaccination compartments have been implemented using Three-Factor Model!
# More features that I want to add later:
# - Booster doses and waning of vaccine-induced immunity (DONE: omega_vax)
# - Variants with different transmissibility and severity
# - Integration with real-world data for parameter fitting
# - Visualization utilities for results analysis
# - Sensitivity analysis tools
# - Export results to common data formats (CSV, JSON, etc.)
# - User-friendly interface for setting up simulations
# - Unit tests for model validation
# - Performance optimizations for large populations
# - Integration with GIS data for spatial modeling
# - Customizable output metrics for specific research questions
# - Interactive dashboards for exploring simulation results
# - Machine learning integration for parameter estimation