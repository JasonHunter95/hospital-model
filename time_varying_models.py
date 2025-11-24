"""
Time-varying extensions to the SIXHRD hospital model.

This module provides models with time-varying transmission rates including:
- Seasonal forcing
- Policy interventions (lockdowns/relaxations)
- Waning immunity
"""

import numpy as np
from config import seasonal_forcing, policy_multiplier


def simulate_age_structured_time_varying(beta_base, age_params, contact_matrix, hosp_capacity,
                                         hill_coef, coverage, VE, age_pops,
                                         seasonal_params=None, waning_params=None,
                                         interventions=None, theta_X=0.5, theta_H=0.3,
                                         Tmax=200, time_step=0.1):
    """
    Simulate age-structured SIXHRD model with time-varying parameters.
    
    Extends the basic age-structured model to include:
    - Seasonal variation in transmission: beta(t) = beta_base * (1 + amp*cos(...))
    - Policy interventions: step functions reducing transmission
    - Waning immunity: flow from R back to S
    
    Parameters
    ----------
    beta_base : float
        Baseline transmission rate (modified by seasonality and interventions).
    age_params : list of dict
        Age-specific disease parameters for each group.
    contact_matrix : ndarray
        Contact rates between age groups [infector, infectee].
    hosp_capacity : float
        Total hospital capacity.
    hill_coef : float
        Hill coefficient for admission gating.
    coverage : float or list
        Vaccine coverage for each age group.
    VE : float
        Vaccine efficacy (0-1).
    age_pops : list
        Population size for each age group.
    seasonal_params : dict, optional
        Seasonal forcing parameters with keys:
        - 'amplitude': seasonal amplitude (0-1)
        - 'period': period in days (default 365)
        - 'peak_day': day of peak transmission
    waning_params : dict, optional
        Waning immunity parameters with keys:
        - 'omega': uniform waning rate (1/days), or
        - 'omega_young', 'omega_middle', 'omega_elderly': age-specific rates
    interventions : list of dict, optional
        Policy interventions, each with:
        - 'start_day': intervention start
        - 'end_day': intervention end
        - 'transmission_reduction': fraction reduction (0-1)
    theta_X : float, optional
        Relative infectiousness of X compartment (default: 0.5).
    theta_H : float, optional
        Relative infectiousness of H compartment (default: 0.3).
    Tmax : float, optional
        Simulation duration in days (default: 200).
    time_step : float, optional
        Integration time step (default: 0.1).
    
    Returns
    -------
    dict
        Results dictionary with additional time series:
        - 'beta_t': time-varying transmission rate
        - 'policy_mult': policy multiplier over time
        - All standard compartment time series (S, I, X, H, R, D by age)
    
    Notes
    -----
    Time-varying beta:
        beta(t) = beta_base * seasonal_factor(t) * policy_multiplier(t)
    
    Waning immunity:
        dS/dt includes +omega * R (immunity loss)
        dR/dt includes -omega * R
    
    Examples
    --------
    >>> from config import SEASONAL_PARAMS, LOCKDOWN_SCENARIO, WANING_PARAMS
    >>> results = simulate_age_structured_time_varying(
    ...     beta_base=0.3,
    ...     age_params=age_params,
    ...     contact_matrix=contact_matrix,
    ...     hosp_capacity=100,
    ...     hill_coef=4,
    ...     coverage=[0.2, 0.3, 0.7],
    ...     VE=0.7,
    ...     age_pops=[3000, 5000, 2000],
    ...     seasonal_params=SEASONAL_PARAMS,
    ...     waning_params={'omega': 0.005},
    ...     interventions=LOCKDOWN_SCENARIO
    ... )
    """
    n_ages = len(age_pops)
    K = hosp_capacity
    n = hill_coef
    dt = time_step
    
    # handle coverage - convert to list if scalar
    if not isinstance(coverage, list):
        coverage = [coverage] * n_ages
    
    # setup waning rates
    if waning_params is None:
        omega = [0.0] * n_ages
    elif 'omega' in waning_params:
        # uniform waning across all ages
        omega = [waning_params['omega']] * n_ages
    else:
        # age-specific waning
        omega = [
            waning_params.get('omega_young', 0.0),
            waning_params.get('omega_middle', 0.0),
            waning_params.get('omega_elderly', 0.0)
        ]
    
    # setup seasonal parameters
    if seasonal_params is None:
        seasonal_params = {'amplitude': 0.0, 'period': 365, 'peak_day': 0}
    
    # setup interventions
    if interventions is None:
        interventions = []
    
    # initialize compartments
    S = [age_pops[a] - 10 if a == 0 else age_pops[a] for a in range(n_ages)]
    I = [10 if a == 0 else 0 for a in range(n_ages)]
    X = [0.0] * n_ages
    H = [0.0] * n_ages
    R = [0.0] * n_ages
    D = [0.0] * n_ages
    
    # storage arrays
    times = []
    S_history = [[] for _ in range(n_ages)]
    I_history = [[] for _ in range(n_ages)]
    X_history = [[] for _ in range(n_ages)]
    H_history = [[] for _ in range(n_ages)]
    R_history = [[] for _ in range(n_ages)]
    D_history = [[] for _ in range(n_ages)]
    H_total_history = []
    overflow_history = []
    beta_t_history = []
    policy_mult_history = []
    
    cum_overflow = 0
    cum_unmet = [0.0] * n_ages
    
    # age-specific effective beta (vaccine effect only, not time-varying)
    eff_beta_vax = [beta_base * (1 - coverage[a] * VE) for a in range(n_ages)]
    
    t = 0
    while t <= Tmax:
        times.append(t)
        
        # store current state
        for a in range(n_ages):
            S_history[a].append(S[a])
            I_history[a].append(I[a])
            X_history[a].append(X[a])
            H_history[a].append(H[a])
            R_history[a].append(R[a])
            D_history[a].append(D[a])
        
        H_total = sum(H)
        H_total_history.append(H_total)
        
        # calculate time-varying beta
        seasonal_factor = seasonal_forcing(
            t, 1.0,  # factor applied to base
            amplitude=seasonal_params.get('amplitude', 0.0),
            period=seasonal_params.get('period', 365),
            peak_day=seasonal_params.get('peak_day', 0)
        )
        
        policy_mult = policy_multiplier(t, interventions)
        
        # combine all time-varying effects
        beta_t = beta_base * seasonal_factor * policy_mult
        beta_t_history.append(beta_t)
        policy_mult_history.append(policy_mult)
        
        # apply time-varying beta to vaccine-adjusted rates
        eff_beta = [eff_beta_vax[a] * seasonal_factor * policy_mult for a in range(n_ages)]
        
        # hospital admission gating
        if K > 0:
            g = 1 / (1 + (H_total/K)**n)
        else:
            g = 0
        
        # force of infection for each age group
        lambda_foi = [0] * n_ages
        for a in range(n_ages):
            for b in range(n_ages):
                if age_pops[b] > 0:
                    lambda_foi[a] += (eff_beta[a] * contact_matrix[a][b] *
                                     (I[b] + theta_X*X[b] + theta_H*H[b]) / age_pops[b])
        
        # new infections and admissions
        new_inf = [lambda_foi[a] * S[a] for a in range(n_ages)]
        admit = [age_params[a]['eta'] * X[a] * g for a in range(n_ages)]
        
        # waning immunity flow
        waning_flow = [omega[a] * R[a] for a in range(n_ages)]
        
        # update compartments (with waning immunity)
        dS = [-new_inf[a] + waning_flow[a] for a in range(n_ages)]
        dI = [new_inf[a] - (age_params[a]['gamma_I'] + age_params[a]['mu_I'] +
              age_params[a]['sigma']) * I[a] for a in range(n_ages)]
        dX = [age_params[a]['sigma'] * I[a] - (age_params[a]['gamma_X'] +
              age_params[a]['mu_X']) * X[a] - admit[a] for a in range(n_ages)]
        dH = [admit[a] - (age_params[a]['gamma_H'] + age_params[a]['mu_H']) * H[a]
              for a in range(n_ages)]
        dR = [age_params[a]['gamma_I'] * I[a] + age_params[a]['gamma_X'] * X[a] +
              age_params[a]['gamma_H'] * H[a] - waning_flow[a] for a in range(n_ages)]
        dD = [age_params[a]['mu_I'] * I[a] + age_params[a]['mu_X'] * X[a] +
              age_params[a]['mu_H'] * H[a] for a in range(n_ages)]
        
        # euler update
        for a in range(n_ages):
            S[a] = max(0, S[a] + dS[a] * dt)
            I[a] = max(0, I[a] + dI[a] * dt)
            X[a] = max(0, X[a] + dX[a] * dt)
            H[a] = max(0, H[a] + dH[a] * dt)
            R[a] = max(0, R[a] + dR[a] * dt)
            D[a] = max(0, D[a] + dD[a] * dt)
            
            # track unmet care
            unmet = age_params[a]['eta'] * X[a] - admit[a]
            cum_unmet[a] += max(0, unmet) * dt
        
        # track overflow
        overflow = max(0, H_total - K)
        cum_overflow += overflow * dt
        overflow_history.append(overflow)
        
        t += dt
    
    return {
        'times': times,
        'S': S_history,
        'I': I_history,
        'X': X_history,
        'H': H_history,
        'R': R_history,
        'D': D_history,
        'H_total': H_total_history,
        'overflow': overflow_history,
        'cum_overflow': cum_overflow,
        'cum_unmet': cum_unmet,
        'age_pops': age_pops,
        'beta_t': beta_t_history,
        'policy_mult': policy_mult_history
    }
