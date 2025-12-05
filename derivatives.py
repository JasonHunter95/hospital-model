"""
Derivative functions for the hospital model ODE system.

This module handles:
- Force of infection calculations
- Unified derivative computation for both vaccinated and unvaccinated compartments
- Main ODE derivative function (master_deriv) and solve_ivp wrapper
"""

import numpy as np
from typing import Dict, Tuple, List

from model_types import ODEParams
from utils import pack_state, unpack_state
from capacity_helpers import hill_gate_vectorized
from demographics_helpers import compute_birth_rate, compute_background_death_rate
from time_varying_helpers import seasonal_forcing, policy_multiplier


def compute_force_of_infection(
    state: Dict[str, np.ndarray],
    params: ODEParams,
    beta_t: float,
    theta_X: float,
    theta_H: float,
    theta_vax: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate the force of infection for both unvaccinated and vaccinated populations.
    
    Parameters
    ----------
    state : dict
        Unpacked state compartments.
    params : dict
        Model parameters.
    beta_t : float
        Time-varying transmission rate.
    theta_X : float
        Infectiousness modifier for X compartments.
    theta_H : float
        Infectiousness modifier for H compartments.
    theta_vax : float
        Infectiousness modifier for vaccinated individuals.
    
    Returns
    -------
    tuple
        (lambda_foi, lambda_foi_vax, live_pop, total_live_pop)
        - lambda_foi: force of infection for unvaccinated
        - lambda_foi_vax: force of infection for vaccinated (reduced by VE_infection)
        - live_pop: live population per age group
        - total_live_pop: total live population

    Mathematical Mapping
    --------------------
    - beta_t -> beta(t) (Transmission rate)
    - contact_matrix -> C_{ba} (Contact matrix, note transpose usage)
    - I -> I (Infected)
    - X_queued, X_admitted -> X_{queued}, X_{admitted}
    - H_ward, H_icu -> H_{ward}, H_{icu}
    - theta_X -> theta_X (Severe infectiousness)
    - theta_H -> theta_H (Hospital infectiousness)
    - theta_vax -> theta_{vax} (Vaccinated infectiousness)
    - VE_infection -> VE_I (Vaccine efficacy against infection)
    """
    n_ages = params['n_ages']
    contact_matrix = params['contact_matrix']
    VE_infection = params['VE_infection']
    
    # Extract compartments
    S = state['S']
    E = state['E']
    I = state['I']
    X_queued = state['X_queued']
    X_admitted = state['X_admitted']
    H_ward = state['H_ward']
    H_icu = state['H_icu']
    R = state['R']
    S_vax = state['S_vax']
    E_vax = state['E_vax']
    I_vax = state['I_vax']
    X_queued_vax = state['X_queued_vax']
    X_admitted_vax = state['X_admitted_vax']
    H_ward_vax = state['H_ward_vax']
    H_icu_vax = state['H_icu_vax']
    R_vax = state['R_vax']
    
    # Live population per age group (exclude dead)
    X_total = X_queued + X_admitted
    X_vax_total = X_queued_vax + X_admitted_vax
    live_pop = (S + E + I + X_total + H_ward + H_icu + R +
                S_vax + E_vax + I_vax + X_vax_total + H_ward_vax + H_icu_vax + R_vax)
    total_live_pop = np.sum(live_pop)
    
    # Avoid division by zero
    live_pop_safe = np.maximum(live_pop, 1e-10)
    
    # Infectious contributions (unvaccinated and vaccinated)
    H_contrib = H_ward + H_icu + H_ward_vax + H_icu_vax
    infectious_unvax = I + theta_X * X_total
    infectious_vax = theta_vax * (I_vax + theta_X * X_vax_total)
    
    # Total infectious proportion per age group
    infectious_fraction = (infectious_unvax + infectious_vax + theta_H * H_contrib) / live_pop_safe
    
    # Calculate absolute effective infectious population (weighted by infectiousness)
    I_eff_absolute = infectious_fraction * live_pop_safe
    
    # Vectorized FOI: lambda_j = beta_t * sum_i(C_ij * I_eff_i) / N_j
    lambda_foi = beta_t * (contact_matrix.T @ I_eff_absolute) / live_pop_safe
    
    # Force of infection for vaccinated (reduced by VE_infection)
    lambda_foi_vax = (1 - VE_infection) * lambda_foi
    
    return lambda_foi, lambda_foi_vax, live_pop, total_live_pop


def compute_generic_derivatives(
    state_subset: Dict[str, np.ndarray],
    params: ODEParams,
    lambda_foi: np.ndarray,
    g_ward: float,
    g_icu: float,
    births_inflow: np.ndarray,
    waning_in: np.ndarray,
    waning_out: np.ndarray,
    bg_deaths_dict: Dict[str, np.ndarray],
    is_vaccinated: bool,
    new_vaccinations: np.ndarray = None
) -> Dict[str, np.ndarray]:
    """
    Unified derivative computation for both vaccinated and unvaccinated compartments.
    
    Parameters
    ----------
    state_subset : dict
        Compartments for this population (e.g., S, E, I, X_queued, etc. or S_vax, E_vax, etc.).
    params : ODEParams
        Model parameters.
    lambda_foi : np.ndarray
        Force of infection for this population.
    g_ward : float
        Ward capacity gating factor.
    g_icu : float
        ICU capacity gating factor.
    births_inflow : np.ndarray
        Birth inflow to S compartment.
    waning_in : np.ndarray
        Waning inflow from the other R compartment (for S destination).
    waning_out : np.ndarray
        Waning outflow from this R compartment.
    bg_deaths_dict : dict
        Background death flows for all compartments.
    is_vaccinated : bool
        True for vaccinated population, applies VE modifiers.
    new_vaccinations : np.ndarray, optional
        Vaccination flow (S -> S_vax). Only used for unvaccinated (outflow) and vaccinated (inflow).
    
    Returns
    -------
    dict
        Dictionary containing compartment derivatives and death flows.
    """
    n_ages = params['n_ages']
    age_params = params['age_params']
    omega = params['omega']
    vax_waning_destination = params['vax_waning_destination']
    dm_params = params['dm_params']
    
    # Get VE modifiers
    VE_severe = params['VE_severe'] if is_vaccinated else 0.0
    VE_death = params['VE_death'] if is_vaccinated else 0.0
    
    # Compartment prefix
    prefix = '_vax' if is_vaccinated else ''
    
    # Extract compartments
    S = state_subset[f'S{prefix}']
    E = state_subset[f'E{prefix}']
    I = state_subset[f'I{prefix}']
    X_queued = state_subset[f'X_queued{prefix}']
    X_admitted = state_subset[f'X_admitted{prefix}']
    H_ward = state_subset[f'H_ward{prefix}']
    H_icu = state_subset[f'H_icu{prefix}']
    R = state_subset[f'R{prefix}']
    
    # Extract age-specific parameters as arrays
    alpha = np.array([ap.get('alpha', 0.2) for ap in age_params])
    sigma = np.array([ap['sigma'] for ap in age_params])
    eta = np.array([ap['eta'] for ap in age_params])
    eta_icu = np.array([ap.get('eta_icu', 0.1) for ap in age_params])
    gamma_I = np.array([ap['gamma_I'] for ap in age_params])
    mu_I = np.array([ap['mu_I'] for ap in age_params])
    gamma_X = np.array([ap['gamma_X'] for ap in age_params])
    mu_X = np.array([ap['mu_X'] for ap in age_params])
    
    # Ward/ICU parameters with fallbacks
    gamma_ward = np.array([ap.get('gamma_ward') or ap.get('gamma_H') or 0.2 for ap in age_params])
    mu_ward = np.array([ap.get('mu_ward') or (ap.get('mu_H') or 0.02) * 0.5 for ap in age_params])
    gamma_icu = np.array([ap.get('gamma_icu') or (ap.get('gamma_H') or 0.2) * 0.6 for ap in age_params])
    mu_icu = np.array([ap.get('mu_icu') or (ap.get('mu_H') or 0.02) * 2.0 for ap in age_params])
    
    # Differential mortality parameters
    age_keys = ['young', 'middle', 'elderly']
    mu_X_untreated = np.zeros(n_ages)
    mu_ward_denied = np.zeros(n_ages)
    for a in range(n_ages):
        age_key = age_keys[a] if a < len(age_keys) else None
        mu_X_mult = dm_params.get(
            f'mu_X_untreated_multiplier_{age_key}', dm_params['mu_X_untreated_multiplier']
        ) if age_key else dm_params['mu_X_untreated_multiplier']
        mu_ward_mult = dm_params.get(
            f'mu_ward_denied_icu_multiplier_{age_key}', dm_params['mu_ward_denied_icu_multiplier']
        ) if age_key else dm_params['mu_ward_denied_icu_multiplier']
        mu_X_untreated[a] = age_params[a].get('mu_X_untreated', mu_X[a] * mu_X_mult)
        mu_ward_denied[a] = age_params[a].get('mu_ward_denied_icu', mu_ward[a] * mu_ward_mult)
    
    # Apply VE modifiers for vaccinated
    if is_vaccinated:
        sigma = (1 - VE_severe) * sigma
        gamma_I = gamma_I + (np.array([ap['sigma'] for ap in age_params]) - sigma)  # Preserve total I exit
        mu_I = (1 - VE_death) * mu_I
        mu_X = (1 - VE_death) * mu_X
        mu_X_untreated = (1 - VE_death) * mu_X_untreated
        mu_ward = (1 - VE_death) * mu_ward
        mu_ward_denied = (1 - VE_death) * mu_ward_denied
        mu_icu = (1 - VE_death) * mu_icu
    
    # Transitions
    new_exposed = lambda_foi * S
    becoming_infectious = alpha * E
    
    # X_queued -> X_admitted flow (gated by ward capacity)
    admit_to_X_admitted = eta * X_queued * g_ward
    
    # X_admitted -> H_ward flow
    gamma_X_admit = np.array([ap.get('gamma_X_admit', ap['eta']) for ap in age_params])
    admit_ward = gamma_X_admit * X_admitted
    
    need_icu = eta_icu * H_ward
    admit_icu = need_icu * g_icu
    
    # Differential mortality
    fraction_icu_denied = np.where((H_ward > 0) & (need_icu > 0), 1.0 - g_icu, 0.0)
    effective_mu_ward = mu_ward + (mu_ward_denied - mu_ward) * eta_icu * fraction_icu_denied
    
    # Death flows
    deaths_I = mu_I * I
    deaths_X_queued = mu_X_untreated * X_queued
    deaths_X_admitted = mu_X * X_admitted
    deaths_ward_baseline = mu_ward * H_ward
    deaths_ward_icu_denied = (mu_ward_denied - mu_ward) * eta_icu * fraction_icu_denied * H_ward
    deaths_icu = mu_icu * H_icu
    
    # Get background death keys
    bg_S = bg_deaths_dict[f'S{prefix}']
    bg_E = bg_deaths_dict[f'E{prefix}']
    bg_I = bg_deaths_dict[f'I{prefix}']
    bg_X_queued = bg_deaths_dict[f'X_queued{prefix}']
    bg_X_admitted = bg_deaths_dict[f'X_admitted{prefix}']
    bg_H_ward = bg_deaths_dict[f'H_ward{prefix}']
    bg_H_icu = bg_deaths_dict[f'H_icu{prefix}']
    bg_R = bg_deaths_dict[f'R{prefix}']
    
    # Compartment derivatives
    dS = births_inflow - new_exposed - bg_S
    
    # Handle waning and vaccination flows
    if is_vaccinated:
        # Vaccinated S gets inflow from vaccinations
        if new_vaccinations is not None:
            dS = dS + new_vaccinations
        # Waning destination check
        if vax_waning_destination == 'S_vax':
            dS = dS + waning_out  # waning_out is omega_vax * R_vax for vax pop
    else:
        # Unvaccinated S loses to vaccinations
        if new_vaccinations is not None:
            dS = dS - new_vaccinations
        # Natural immunity waning from R -> S
        dS = dS + omega * R
        # If vax waning destination is S, add waning_in (from R_vax)
        if vax_waning_destination == 'S':
            dS = dS + waning_in
    
    dE = new_exposed - becoming_infectious - bg_E
    dI = becoming_infectious - (gamma_I + mu_I + sigma) * I - bg_I
    
    dX_queued = sigma * I - (gamma_X + mu_X_untreated) * X_queued - admit_to_X_admitted - bg_X_queued
    dX_admitted = admit_to_X_admitted - (gamma_X + mu_X) * X_admitted - admit_ward - bg_X_admitted
    
    dH_ward = admit_ward - (gamma_ward + effective_mu_ward) * H_ward - admit_icu - bg_H_ward
    dH_icu = admit_icu - (gamma_icu + mu_icu) * H_icu - bg_H_icu
    
    # Recovery
    dR = gamma_I * I + gamma_X * (X_queued + X_admitted) + gamma_ward * H_ward + gamma_icu * H_icu - bg_R
    if not is_vaccinated:
        dR = dR - omega * R  # Natural waning out
    else:
        dR = dR - waning_out  # Vaccine waning out
    
    total_deaths = (deaths_I + deaths_X_queued + deaths_X_admitted +
                    deaths_ward_baseline + deaths_ward_icu_denied + deaths_icu)
    dD = total_deaths
    
    # Treated vs untreated deaths
    dD_treated = deaths_I + deaths_X_admitted + deaths_ward_baseline + deaths_icu
    dD_untreated = deaths_X_queued + deaths_ward_icu_denied
    
    return {
        'dS': dS, 'dE': dE, 'dI': dI,
        'dX_queued': dX_queued, 'dX_admitted': dX_admitted,
        'dH_ward': dH_ward, 'dH_icu': dH_icu, 'dR': dR, 'dD': dD,
        'dD_treated': dD_treated, 'dD_untreated': dD_untreated,
        'new_exposed': new_exposed
    }


def master_deriv(y: np.ndarray, t: float, params: ODEParams) -> np.ndarray:
    """
    Compute derivatives for the master SEIXHRD model with vaccination and demographics.
    
    This function computes dy/dt for all compartments using vectorized numpy
    operations for the force of infection calculation.
    
    Parameters
    ----------
    y : np.ndarray
        Current state vector (flattened compartments).
    t : float
        Current time.
    params : dict
        Model parameters.
    
    Returns
    -------
    np.ndarray
        Derivative vector dy/dt.
    """
    n_ages = params['n_ages']
    
    # Unpack state
    state = unpack_state(y, n_ages)
    
    # Extract parameters
    beta_base = params['beta_base']
    theta_X = params['theta_X']
    theta_H = params['theta_H']
    theta_vax = params['theta_vax']
    omega_vax = params['omega_vax']
    K_ward = params['K_ward']
    K_icu = params['K_icu']
    n_ward = params['n_ward']
    n_icu = params['n_icu']
    seasonal_params = params['seasonal_params']
    interventions = params.get('interventions', [])
    vaccination_rate = params['vaccination_rate']
    
    # Extract demographic parameters
    demo_params = params.get('demographic_params', {})
    birth_rate = demo_params.get('birth_rate', 0.0)
    mu_background = demo_params.get('mu_background', np.zeros(n_ages))
    birth_age_dist = demo_params.get('birth_age_distribution', None)
    neonatal_vax_rate = demo_params.get('neonatal_vaccination_rate', 0.0)
    age_pops = params.get('age_pops', np.ones(n_ages))
    
    # Time-Varying Transmission
    seasonal_factor = seasonal_forcing(
        t, 1.0,
        amplitude=seasonal_params.get('amplitude', 0.0),
        period=seasonal_params.get('period', 365),
        peak_day=seasonal_params.get('peak_day', 0),
    )
    policy_mult = policy_multiplier(t, interventions)
    beta_t = beta_base * seasonal_factor * policy_mult
    
    # Capacity Gating
    H_ward_total = (np.sum(state['H_ward']) + np.sum(state['H_ward_vax']) +
                    np.sum(state['X_admitted']) + np.sum(state['X_admitted_vax']))
    H_icu_total = np.sum(state['H_icu']) + np.sum(state['H_icu_vax'])
    g_ward = hill_gate_vectorized(H_ward_total, K_ward, n_ward)
    g_icu = hill_gate_vectorized(H_icu_total, K_icu, n_icu)
    
    # Force of Infection
    lambda_foi, lambda_foi_vax, live_pop, total_live_pop = compute_force_of_infection(
        state, params, beta_t, theta_X, theta_H, theta_vax
    )
    
    # Demographic Flows
    births_total = compute_birth_rate(total_live_pop, birth_rate, age_pops, birth_age_dist)
    births_to_S = births_total * (1.0 - neonatal_vax_rate)
    births_to_S_vax = births_total * neonatal_vax_rate
    
    # Background deaths
    bg_deaths_dict = {}
    for comp_name in ['S', 'E', 'I', 'X_queued', 'X_admitted', 'H_ward', 'H_icu', 'R',
                      'S_vax', 'E_vax', 'I_vax', 'X_queued_vax', 'X_admitted_vax', 
                      'H_ward_vax', 'H_icu_vax', 'R_vax']:
        bg_deaths_dict[comp_name] = compute_background_death_rate(state[comp_name], mu_background)
    
    total_bg_deaths = sum(bg_deaths_dict.values())
    
    # Waning flows
    omega = params['omega']
    waning_flow_vax = omega_vax * state['R_vax']
    new_vaccinations = vaccination_rate * state['S']
    
    # Compute derivatives using unified function
    unvax_results = compute_generic_derivatives(
        state_subset=state,
        params=params,
        lambda_foi=lambda_foi,
        g_ward=g_ward,
        g_icu=g_icu,
        births_inflow=births_to_S,
        waning_in=waning_flow_vax,
        waning_out=omega * state['R'],
        bg_deaths_dict=bg_deaths_dict,
        is_vaccinated=False,
        new_vaccinations=new_vaccinations
    )
    
    vax_results = compute_generic_derivatives(
        state_subset=state,
        params=params,
        lambda_foi=lambda_foi_vax,
        g_ward=g_ward,
        g_icu=g_icu,
        births_inflow=births_to_S_vax,
        waning_in=np.zeros(n_ages),  # Not used for vax
        waning_out=waning_flow_vax,
        bg_deaths_dict=bg_deaths_dict,
        is_vaccinated=True,
        new_vaccinations=new_vaccinations
    )
    
    # Tracked Variable Derivatives
    d_cum_breakthrough = vax_results['new_exposed']
    d_cum_births = births_total
    d_cum_background_deaths = total_bg_deaths
    
    # Pack Derivatives
    derivs = {
        'S': unvax_results['dS'], 
        'E': unvax_results['dE'], 
        'I': unvax_results['dI'], 
        'X_queued': unvax_results['dX_queued'], 
        'X_admitted': unvax_results['dX_admitted'],
        'H_ward': unvax_results['dH_ward'], 
        'H_icu': unvax_results['dH_icu'], 
        'R': unvax_results['dR'], 
        'D': unvax_results['dD'],
        'S_vax': vax_results['dS'], 
        'E_vax': vax_results['dE'], 
        'I_vax': vax_results['dI'],
        'X_queued_vax': vax_results['dX_queued'], 
        'X_admitted_vax': vax_results['dX_admitted'],
        'H_ward_vax': vax_results['dH_ward'], 
        'H_icu_vax': vax_results['dH_icu'], 
        'R_vax': vax_results['dR'], 
        'D_vax': vax_results['dD'],
        'D_treated': unvax_results['dD_treated'], 
        'D_untreated': unvax_results['dD_untreated'],
        'D_vax_treated': vax_results['dD_treated'], 
        'D_vax_untreated': vax_results['dD_untreated'],
        'cum_breakthrough': d_cum_breakthrough,
        'cum_births': d_cum_births,
        'cum_background_deaths': d_cum_background_deaths
    }
    
    return pack_state(derivs, n_ages)


def master_deriv_solve_ivp(t: float, y: np.ndarray, params: ODEParams) -> np.ndarray:
    """
    Wrapper for master_deriv with solve_ivp argument order (t, y).
    """
    return master_deriv(y, t, params)
