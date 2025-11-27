"""
Master SIXHRD hospital model

This model includes:
- Age-structured compartments (S, I, X, H_ward, H_icu, R, D)
- Vaccination with age-specific coverage and efficacy
- Infectiousness modifiers for compartments X and H
- Separate ward and ICU capacity constraints with Hill function gating
- Differential mortality tracking (treated vs untreated deaths)
- Seasonal forcing of transmission
- Policy interventions (lockdowns/relaxations)
- Waning immunity
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
    # Vaccination parameters
    coverage=0.0,
    VE=None,
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
    Parameters
    ----------
    beta_base : float
        Baseline transmission rate (modified by seasonality and interventions).
    age_params : list of dict
        Age-specific parameters for each group. Each dict should contain:
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
        Vaccine coverage. Float for uniform, list for age-specific. Default 0.0.
    VE : float, optional
        Vaccine efficacy (0-1). Default from config.
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
        Waning immunity parameters:
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
        Override default initial conditions.
    
    Returns
    -------
    dict
        Comprehensive results dictionary containing:
        
        Time:
        - 'times': array of time points
        
        Compartments by age (lists of arrays, one per age group):
        - 'S', 'I', 'X', 'R', 'D': standard compartments
        - 'H_ward', 'H_icu': hospital compartments
        - 'H': combined ward + ICU (backward compatibility)
        
        Aggregated totals (arrays):
        - 'H_ward_total', 'H_icu_total', 'H_total': hospital occupancy
        - 'I_total', 'X_total', 'D_total': infection/death totals
        
        Capacity metrics:
        - 'ward_overflow', 'icu_overflow': instantaneous overflow
        - 'cum_ward_overflow', 'cum_icu_overflow': cumulative overflow
        - 'cum_unmet_ward', 'cum_unmet_icu': unmet care by age
        - 'g_ward', 'g_icu': admission gating factors over time
        
        Differential mortality (if track_differential_mortality=True):
        - 'D_treated', 'D_untreated': deaths by care status per age
        - 'D_treated_total', 'D_untreated_total': aggregated
        
        Time-varying parameters:
        - 'beta_t': effective transmission rate over time
        - 'seasonal_factor': seasonal multiplier
        - 'policy_mult': policy intervention multiplier
        
        Compartment flows (if track_compartment_flows=True):
        - 'new_infections', 'ward_admissions', 'icu_admissions': daily flows
        
        Metadata:
        - 'ward_capacity', 'icu_capacity', 'age_pops'
        - 'parameters': dict of all input parameters
    
    Notes
    -----
    Compartment flow for each age group:
        S_a → I_a → X_a → H_ward_a → H_icu_a → R_a or D_a
                ↓           ↓            ↓
               R_a         R_a          R_a
                            ↓            ↓
                           D_a          D_a
    
    With waning immunity: R_a → S_a
    
    Time-varying transmission:
        beta(t) = beta_base * seasonal_factor(t) * policy_multiplier(t)
    
    Differential mortality:
    - D_treated: deaths occurring with appropriate care
    - D_untreated: excess deaths from capacity constraints
    
    Examples
    --------
    >>> from config import (AGE_PARAMS_DEFAULT, CONTACT_MATRIX_DEFAULT,
    ...                     SEASONAL_PARAMS, LOCKDOWN_SCENARIO, WANING_PARAMS)
    >>> 
    >>> # Full-featured simulation
    >>> results = simulate_master_model(
    ...     beta_base=0.3,
    ...     age_params=AGE_PARAMS_DEFAULT,
    ...     contact_matrix=CONTACT_MATRIX_DEFAULT,
    ...     ward_capacity=80,
    ...     icu_capacity=20,
    ...     coverage=[0.2, 0.3, 0.7],
    ...     VE=0.7,
    ...     age_pops=[3000, 5000, 2000],
    ...     seasonal_params={'amplitude': 0.2, 'period': 365, 'peak_day': 0},
    ...     waning_params={'omega': 0.005},
    ...     interventions=[{'start_day': 50, 'end_day': 80, 'transmission_reduction': 0.5}],
    ...     Tmax=365
    ... )
    >>> 
    >>> print(f"Peak ICU: {max(results['H_icu_total']):.0f}")
    >>> print(f"Treated deaths: {results['D_treated_total'][-1]:.0f}")
    >>> print(f"Untreated deaths: {results['D_untreated_total'][-1]:.0f}")
    >>> print(f"Final R0 multiplier: {results['seasonal_factor'][-1]:.2f}")
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
    
    I = _coerce_initial_vector(ic_defaults, 'I_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['I_by_age'])
    X = _coerce_initial_vector(ic_defaults, 'X_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['X_by_age'])
    H_ward = _coerce_initial_vector(ic_defaults, 'H_ward_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_ward_by_age'])
    H_icu = _coerce_initial_vector(ic_defaults, 'H_icu_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_icu_by_age'])
    R = _coerce_initial_vector(ic_defaults, 'R_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['R_by_age'])
    D = _coerce_initial_vector(ic_defaults, 'D_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['D_by_age'])
    S = [max(0, age_pops[a] - I[a] - X[a] - H_ward[a] - H_icu[a] - R[a] - D[a]) for a in range(n_ages)]
    
    # Differential mortality tracking
    D_treated = [0.0] * n_ages
    D_untreated = [0.0] * n_ages
    
    # ========================================
    # Storage Arrays
    # ========================================
    times = []
    
    # Per-age compartment histories
    S_history = [[] for _ in range(n_ages)]
    I_history = [[] for _ in range(n_ages)]
    X_history = [[] for _ in range(n_ages)]
    H_ward_history = [[] for _ in range(n_ages)]
    H_icu_history = [[] for _ in range(n_ages)]
    R_history = [[] for _ in range(n_ages)]
    D_history = [[] for _ in range(n_ages)]
    D_treated_history = [[] for _ in range(n_ages)]
    D_untreated_history = [[] for _ in range(n_ages)]
    
    # Aggregated histories
    H_ward_total_history = []
    H_icu_total_history = []
    H_total_history = []
    I_total_history = []
    X_total_history = []
    D_total_history = []
    D_treated_total_history = []
    D_untreated_total_history = []
    
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
    
    # Cumulative metrics
    cum_ward_overflow = 0
    cum_icu_overflow = 0
    cum_unmet_ward = [0.0] * n_ages
    cum_unmet_icu = [0.0] * n_ages
    
    # Age-specific vaccine-adjusted beta (before time-varying effects)
    eff_beta_vax = [beta_base * (1 - coverage[a] * VE) for a in range(n_ages)]
    
    # ========================================
    # Main Simulation Loop
    # ========================================
    t = 0
    while t <= Tmax:
        times.append(t)
        
        # Store current state
        for a in range(n_ages):
            S_history[a].append(S[a])
            I_history[a].append(I[a])
            X_history[a].append(X[a])
            H_ward_history[a].append(H_ward[a])
            H_icu_history[a].append(H_icu[a])
            R_history[a].append(R[a])
            D_history[a].append(D[a])
            D_treated_history[a].append(D_treated[a])
            D_untreated_history[a].append(D_untreated[a])
        
        # Aggregated totals
        H_ward_total = sum(H_ward)
        H_icu_total = sum(H_icu)
        H_total = H_ward_total + H_icu_total
        I_total = sum(I)
        X_total = sum(X)
        D_total = sum(D)
        
        H_ward_total_history.append(H_ward_total)
        H_icu_total_history.append(H_icu_total)
        H_total_history.append(H_total)
        I_total_history.append(I_total)
        X_total_history.append(X_total)
        D_total_history.append(D_total)
        D_treated_total_history.append(sum(D_treated))
        D_untreated_total_history.append(sum(D_untreated))
        
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
        
        # Age-specific effective beta (vaccine + time-varying)
        eff_beta = [eff_beta_vax[a] * seasonal_factor * policy_mult for a in range(n_ages)]
        
        # ========================================
        # Capacity Gating
        # ========================================
        g_ward = hill_gate(H_ward_total, K_ward, n_ward)
        g_icu = hill_gate(H_icu_total, K_icu, n_icu)
        
        g_ward_history.append(g_ward)
        g_icu_history.append(g_icu)
        
        # ========================================
        # Force of Infection
        # ========================================
        lambda_foi = [0.0] * n_ages
        for a in range(n_ages):
            for b in range(n_ages):
                H_contrib = H_ward[b] + H_icu[b]
                live_pop_b = S[b] + I[b] + X[b] + H_contrib + R[b]
                if live_pop_b > 0:
                    lambda_foi[a] += (eff_beta[a] * contact_matrix[a][b] *
                                     (I[b] + theta_X * X[b] + theta_H * H_contrib) / live_pop_b)
        
        # ========================================
        # Transitions
        # ========================================
        new_inf = [lambda_foi[a] * S[a] for a in range(n_ages)]
        
        # Ward admissions from X
        admit_ward = [age_params[a]['eta'] * X[a] * g_ward for a in range(n_ages)]
        
        # ICU admissions from ward (patients needing escalation)
        need_icu = [age_params[a].get('eta_icu', 0.1) * H_ward[a] for a in range(n_ages)]
        admit_icu = [need_icu[a] * g_icu for a in range(n_ages)]
        
        # Waning immunity flow
        waning_flow = [omega[a] * R[a] for a in range(n_ages)]
        
        # Unmet care calculations
        unmet_ward = [max(0, age_params[a]['eta'] * X[a] - admit_ward[a]) for a in range(n_ages)]
        unmet_icu = [max(0, need_icu[a] - admit_icu[a]) for a in range(n_ages)]
        
        # Track flows if requested
        if track_compartment_flows:
            new_infections_history.append(list(new_inf))
            ward_admissions_history.append(list(admit_ward))
            icu_admissions_history.append(list(admit_icu))
        
        # ========================================
        # ODEs with Differential Mortality
        # ========================================
        dS = [-new_inf[a] + waning_flow[a] for a in range(n_ages)]
        dI = [new_inf[a] - (age_params[a]['gamma_I'] + age_params[a]['mu_I'] +
              age_params[a]['sigma']) * I[a] for a in range(n_ages)]
        
        dH_ward = []
        dH_icu = []
        dX = []
        dR = []
        dD = []
        dD_treated = []
        dD_untreated = []
        
        for a in range(n_ages):
            # Get ward/ICU parameters with fallbacks
            gamma_w = age_params[a].get('gamma_ward', age_params[a].get('gamma_H', 0.2))
            mu_w = age_params[a].get('mu_ward', age_params[a].get('mu_H', 0.02) * 0.5)
            gamma_i = age_params[a].get('gamma_icu', age_params[a].get('gamma_H', 0.2) * 0.6)
            mu_i = age_params[a].get('mu_icu', age_params[a].get('mu_H', 0.02) * 2.0)
            
            # ========================================
            # Differential Mortality Rates
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
            
            # Effective mortality rates based on care availability
            fraction_X_denied = 1.0 - g_ward if X[a] > 0 else 0.0
            effective_mu_X = mu_X_treated * g_ward + mu_X_untreated * fraction_X_denied
            
            fraction_icu_denied = 1.0 - g_icu if H_ward[a] > 0 and need_icu[a] > 0 else 0.0
            eta_icu_a = age_params[a].get('eta_icu', 0.1)
            effective_mu_ward = mu_w + (mu_ward_denied - mu_w) * eta_icu_a * fraction_icu_denied
            
            # ========================================
            # Death Flows for Tracking
            # ========================================
            deaths_I = age_params[a]['mu_I'] * I[a]
            deaths_X_treated = mu_X_treated * g_ward * X[a]
            deaths_X_untreated = mu_X_untreated * fraction_X_denied * X[a]
            deaths_ward_baseline = mu_w * H_ward[a]
            deaths_ward_icu_denied = (mu_ward_denied - mu_w) * eta_icu_a * fraction_icu_denied * H_ward[a]
            deaths_icu = mu_i * H_icu[a]
            
            # ========================================
            # Compartment Derivatives
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
            
            # Differential mortality tracking
            dD_treated.append(deaths_I + deaths_X_treated + deaths_ward_baseline + deaths_icu)
            dD_untreated.append(deaths_X_untreated + deaths_ward_icu_denied)
        
        # ========================================
        # Euler Update
        # ========================================
        for a in range(n_ages):
            S[a] = max(0, S[a] + dS[a] * dt)
            I[a] = max(0, I[a] + dI[a] * dt)
            X[a] = max(0, X[a] + dX[a] * dt)
            H_ward[a] = max(0, H_ward[a] + dH_ward[a] * dt)
            H_icu[a] = max(0, H_icu[a] + dH_icu[a] * dt)
            R[a] = max(0, R[a] + dR[a] * dt)
            D[a] = max(0, D[a] + dD[a] * dt)
            D_treated[a] = max(0, D_treated[a] + dD_treated[a] * dt)
            D_untreated[a] = max(0, D_untreated[a] + dD_untreated[a] * dt)
            
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
        
        # Per-age compartments
        'S': S_history,
        'I': I_history,
        'X': X_history,
        'H_ward': H_ward_history,
        'H_icu': H_icu_history,
        'H': [[(H_ward_history[a][t] + H_icu_history[a][t]) 
               for t in range(len(times))] for a in range(n_ages)],
        'R': R_history,
        'D': D_history,
        
        # Aggregated totals
        'H_ward_total': H_ward_total_history,
        'H_icu_total': H_icu_total_history,
        'H_total': H_total_history,
        'I_total': I_total_history,
        'X_total': X_total_history,
        'D_total': D_total_history,
        
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
            'D_untreated_total': D_untreated_total_history
        })
    # Compartment flows
    if track_compartment_flows:
        results.update({
            'new_infections': new_infections_history,
            'ward_admissions': ward_admissions_history,
            'icu_admissions': icu_admissions_history
        })
    return results

# End of master_hospital_model.py
# More features that I want to add later:
# - Vaccination compartments (S_vax, I_vax, etc.)
# - Booster doses and waning of vaccine-induced immunity
# - Variants with different transmissibility and severity
# - Integration with real-world data for parameter fitting
# - Visualization utilities for results analysis
# - Sensitivity analysis tools
# - Export results to common data formats (CSV, JSON, etc.)
# - User-friendly interface for setting up simulations
# - Documentation and examples for all features
# - Unit tests for model validation
# - Performance optimizations for large populations
# - Integration with GIS data for spatial modeling
# - Customizable output metrics for specific research questions
# - Interactive dashboards for exploring simulation results
# - Machine learning integration for parameter estimation