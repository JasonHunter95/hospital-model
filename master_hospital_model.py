"""
Master SEIXHRD hospital model with Three-Factor Vaccination Compartments

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
import config
from simulation_helpers import hill_gate, _validate_age_structured_inputs, _coerce_initial_vector
from time_varying_helpers import seasonal_forcing, policy_multiplier


# ============================================================================
# STATE VECTOR PACKING/UNPACKING
# ============================================================================
# Order of compartments in the flattened state vector.
# For n_ages age groups, the state vector has 20 * n_ages elements.
# Layout: [S_0..S_{n-1}, E_0..E_{n-1}, ..., D_vax_0..D_vax_{n-1}]
#
# X compartment is split into X_queued (waiting for admission, untreated mortality)
# and X_admitted (secured ward spot, treated mortality, can recover or flow to H_ward)

STATE_ORDER = [
    'S', 'E', 'I', 'X_queued', 'X_admitted', 'H_ward', 'H_icu', 'R', 'D',
    'S_vax', 'E_vax', 'I_vax', 'X_queued_vax', 'X_admitted_vax', 'H_ward_vax', 'H_icu_vax', 'R_vax', 'D_vax'
]
NUM_COMPARTMENTS = len(STATE_ORDER)  # 20

# Additional tracked variables (differential mortality) - these are integrated
# separately as part of the state vector for accumulation
TRACKED_ORDER = ['D_treated', 'D_untreated', 'D_vax_treated', 'D_vax_untreated', 'cum_breakthrough']
NUM_TRACKED = len(TRACKED_ORDER)  # 5


def _pack_state(compartments, n_ages):
    """
    Pack compartment dictionaries into a flat 1D numpy array for ODE solvers.
    
    Parameters
    ----------
    compartments : dict
        Dictionary mapping compartment names to lists of values per age group.
        Keys should match STATE_ORDER and TRACKED_ORDER.
    n_ages : int
        Number of age groups.
    
    Returns
    -------
    np.ndarray
        Flattened state vector of shape ((NUM_COMPARTMENTS + NUM_TRACKED) * n_ages,)
    
    Notes
    -----
    State vector layout:
    [S_0, S_1, ..., S_{n-1}, E_0, E_1, ..., D_vax_{n-1}, 
     D_treated_0, ..., cum_breakthrough_{n-1}]
    """
    y = np.zeros((NUM_COMPARTMENTS + NUM_TRACKED) * n_ages)
    
    # Pack main compartments
    for i, name in enumerate(STATE_ORDER):
        start = i * n_ages
        y[start:start + n_ages] = compartments[name]
    
    # Pack tracked variables
    base = NUM_COMPARTMENTS * n_ages
    for i, name in enumerate(TRACKED_ORDER):
        start = base + i * n_ages
        y[start:start + n_ages] = compartments[name]
    
    return y


def _unpack_state(y, n_ages):
    """
    Unpack flat state vector into compartment dictionary.
    
    Parameters
    ----------
    y : np.ndarray
        Flattened state vector of shape ((NUM_COMPARTMENTS + NUM_TRACKED) * n_ages,)
    n_ages : int
        Number of age groups.
    
    Returns
    -------
    dict
        Dictionary mapping compartment names to numpy arrays of values per age group.
    """
    compartments = {}
    
    # Unpack main compartments
    for i, name in enumerate(STATE_ORDER):
        start = i * n_ages
        compartments[name] = y[start:start + n_ages]
    
    # Unpack tracked variables
    base = NUM_COMPARTMENTS * n_ages
    for i, name in enumerate(TRACKED_ORDER):
        start = base + i * n_ages
        compartments[name] = y[start:start + n_ages]
    
    return compartments


def _hill_gate_vectorized(occupancy, capacity, hill_coef):
    """
    Vectorized Hill function gating with numerical stability.
    
    This is an internal function called from within the ODE solver.
    Input validation is performed in the outer simulate function.
    Uses log-domain computation for large hill_coef to prevent overflow.
    
    Parameters
    ----------
    occupancy : float
        Current total occupancy. Assumed non-negative.
    capacity : float
        Maximum capacity.
    hill_coef : float
        Hill coefficient. Assumed non-negative.
    
    Returns
    -------
    float
        Gating factor between 0 and 1.
    """
    # Edge cases
    if capacity <= 0:
        return 0.0
    if occupancy <= 0:
        return 1.0
    if hill_coef == 0:
        return 0.5
    
    ratio = occupancy / capacity
    
    # Use log-domain for numerical stability with large hill_coef
    if ratio >= 1.0:
        # For ratio >= 1, result is <= 0.5
        log_ratio = np.log(ratio)
        exponent = hill_coef * log_ratio
        if exponent > 700:  # exp(700) ≈ 1e304, near float max
            return 0.0  # Overwhelmed capacity
        power = np.exp(exponent)
    else:
        # For ratio < 1, (ratio)^n approaches 0, safe to compute directly
        power = ratio ** hill_coef
    
    return 1.0 / (1.0 + power)


# ============================================================================
# DERIVATIVE FUNCTION FOR ODE SOLVERS
# ============================================================================

def _master_deriv(y, t, params):
    """
    Compute derivatives for the master SEIXHRD model with vaccination.
    
    This function computes dy/dt for all compartments using vectorized numpy
    operations for the force of infection calculation.
    
    Parameters
    ----------
    y : np.ndarray
        Current state vector (flattened compartments).
    t : float
        Current time.
    params : dict
        Model parameters including:
        - n_ages: number of age groups
        - beta_base: baseline transmission rate
        - contact_matrix: age-structured contact matrix
        - age_params: list of age-specific parameter dicts
        - K_ward, K_icu: ward and ICU capacity
        - n_ward, n_icu: Hill coefficients
        - VE_infection, VE_severe, VE_death: vaccine efficacies
        - theta_X, theta_H, theta_vax: infectiousness modifiers
        - omega: natural immunity waning rates
        - omega_vax: vaccine immunity waning rates
        - vax_waning_destination: 'S' or 'S_vax'
        - vaccination_rate: vaccination rates by age
        - seasonal_params: seasonal forcing parameters
        - interventions: list of intervention dicts
    
    Returns
    -------
    np.ndarray
        Derivative vector dy/dt.
    """
    n_ages = params['n_ages']
    
    # Unpack state
    state = _unpack_state(y, n_ages)
    S = state['S']
    E = state['E']
    I = state['I']
    X_queued = state['X_queued']
    X_admitted = state['X_admitted']
    H_ward = state['H_ward']
    H_icu = state['H_icu']
    R = state['R']
    D = state['D']
    S_vax = state['S_vax']
    E_vax = state['E_vax']
    I_vax = state['I_vax']
    X_queued_vax = state['X_queued_vax']
    X_admitted_vax = state['X_admitted_vax']
    H_ward_vax = state['H_ward_vax']
    H_icu_vax = state['H_icu_vax']
    R_vax = state['R_vax']
    D_vax = state['D_vax']
    # Tracked accumulators (will have their rates computed)
    # D_treated, D_untreated, D_vax_treated, D_vax_untreated, cum_breakthrough
    
    # Extract parameters
    beta_base = params['beta_base']
    contact_matrix = params['contact_matrix']
    age_params = params['age_params']
    K_ward = params['K_ward']
    K_icu = params['K_icu']
    n_ward = params['n_ward']
    n_icu = params['n_icu']
    VE_infection = params['VE_infection']
    VE_severe = params['VE_severe']
    VE_death = params['VE_death']
    theta_X = params['theta_X']
    theta_H = params['theta_H']
    theta_vax = params['theta_vax']
    omega = params['omega']
    omega_vax = params['omega_vax']
    vax_waning_destination = params['vax_waning_destination']
    vaccination_rate = params['vaccination_rate']
    seasonal_params = params['seasonal_params']
    interventions = params['interventions']
    dm_params = params['dm_params']
    
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
    
    # ========================================
    # Capacity Gating
    # ========================================
    # Count admitted-but-not-yet-moved patients toward capacity to close the "ghost ward" gap
    H_ward_total = (np.sum(H_ward) + np.sum(H_ward_vax) +
                    np.sum(X_admitted) + np.sum(X_admitted_vax))
    H_icu_total = np.sum(H_icu) + np.sum(H_icu_vax)
    g_ward = _hill_gate_vectorized(H_ward_total, K_ward, n_ward)
    g_icu = _hill_gate_vectorized(H_icu_total, K_icu, n_icu)
    
    # ========================================
    # Force of Infection (Vectorized)
    # ========================================
    # Live population per age group (exclude dead)
    # X_total = X_queued + X_admitted for live population count
    X_total = X_queued + X_admitted
    X_vax_total = X_queued_vax + X_admitted_vax
    live_pop = (S + E + I + X_total + H_ward + H_icu + R +
                S_vax + E_vax + I_vax + X_vax_total + H_ward_vax + H_icu_vax + R_vax)
    
    # Avoid division by zero
    live_pop_safe = np.maximum(live_pop, 1e-10)
    
    # Infectious contributions (unvaccinated and vaccinated)
    # Both X_queued and X_admitted contribute to infectiousness
    H_contrib = H_ward + H_icu + H_ward_vax + H_icu_vax
    infectious_unvax = I + theta_X * X_total
    infectious_vax = theta_vax * (I_vax + theta_X * X_vax_total)
    
    # Total infectious proportion per age group
    infectious_fraction = (infectious_unvax + infectious_vax + theta_H * H_contrib) / live_pop_safe
    
    # Vectorized FOI: lambda_a = beta_t * sum_b(C_ab * infectious_fraction_b)
    # contact_matrix[a,b] = contacts from age a to age b
    lambda_foi = beta_t * (contact_matrix @ infectious_fraction)
    
    # Force of infection for vaccinated (reduced by VE_infection)
    lambda_foi_vax = (1 - VE_infection) * lambda_foi
    
    # ========================================
    # Extract Age-Specific Parameters as Arrays
    # ========================================
    alpha = np.array([ap.get('alpha', 0.2) for ap in age_params])
    sigma = np.array([ap['sigma'] for ap in age_params])
    eta = np.array([ap['eta'] for ap in age_params])
    eta_icu = np.array([ap.get('eta_icu', 0.1) for ap in age_params])
    gamma_I = np.array([ap['gamma_I'] for ap in age_params])
    mu_I = np.array([ap['mu_I'] for ap in age_params])
    gamma_X = np.array([ap['gamma_X'] for ap in age_params])
    mu_X = np.array([ap['mu_X'] for ap in age_params])
    
    # Ward/ICU parameters with fallbacks
    gamma_ward = np.array([ap.get('gamma_ward', ap.get('gamma_H', 0.2)) for ap in age_params])
    mu_ward = np.array([ap.get('mu_ward', ap.get('mu_H', 0.02) * 0.5) for ap in age_params])
    gamma_icu = np.array([ap.get('gamma_icu', ap.get('gamma_H', 0.2) * 0.6) for ap in age_params])
    mu_icu = np.array([ap.get('mu_icu', ap.get('mu_H', 0.02) * 2.0) for ap in age_params])
    
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
    
    # ========================================
    # Transitions (Unvaccinated)
    # ========================================
    new_exposed = lambda_foi * S
    becoming_infectious = alpha * E
    
    # X_queued -> X_admitted flow (gated by ward capacity)
    admit_to_X_admitted = eta * X_queued * g_ward
    
    # X_admitted -> H_ward flow (rate gamma_X_admit, i.e., actual ward admission from X_admitted)
    gamma_X_admit = np.array([ap.get('gamma_X_admit', ap['eta']) for ap in age_params])
    admit_ward = gamma_X_admit * X_admitted
    
    need_icu = eta_icu * H_ward
    admit_icu = need_icu * g_icu
    waning_flow = omega * R
    new_vaccinations = vaccination_rate * S
    
    # ========================================
    # Transitions (Vaccinated)
    # ========================================
    new_exposed_vax = lambda_foi_vax * S_vax
    becoming_infectious_vax = alpha * E_vax
    sigma_vax = (1 - VE_severe) * sigma
    # Preserve/accelerate total I exit when sigma is reduced by vaccination
    gamma_I_vax = gamma_I + (sigma - sigma_vax)
    
    # X_queued_vax -> X_admitted_vax flow (gated by ward capacity)
    admit_to_X_admitted_vax = eta * X_queued_vax * g_ward
    
    # X_admitted_vax -> H_ward_vax flow
    admit_ward_vax = gamma_X_admit * X_admitted_vax
    
    need_icu_vax = eta_icu * H_ward_vax
    admit_icu_vax = need_icu_vax * g_icu
    waning_flow_vax = omega_vax * R_vax
    
    # ========================================
    # Differential Mortality (Unvaccinated)
    # ========================================
    # With X_queued/X_admitted split:
    # - X_queued deaths are ALL untreated (waiting for admission)
    # - X_admitted deaths are ALL treated (secured ward spot)
    # This eliminates the "instantaneous mortality fallacy"
    
    # Fraction of ward patients denied ICU (this logic remains)
    fraction_icu_denied = np.where((H_ward > 0) & (need_icu > 0), 1.0 - g_icu, 0.0)
    effective_mu_ward = mu_ward + (mu_ward_denied - mu_ward) * eta_icu * fraction_icu_denied
    
    # Death flows (unvaccinated)
    deaths_I = mu_I * I
    deaths_X_queued = mu_X_untreated * X_queued  # All X_queued deaths are untreated
    deaths_X_admitted = mu_X * X_admitted  # All X_admitted deaths are treated
    deaths_ward_baseline = mu_ward * H_ward
    deaths_ward_icu_denied = (mu_ward_denied - mu_ward) * eta_icu * fraction_icu_denied * H_ward
    deaths_icu = mu_icu * H_icu
    
    # ========================================
    # Differential Mortality (Vaccinated)
    # ========================================
    mu_I_vax = (1 - VE_death) * mu_I
    mu_X_vax = (1 - VE_death) * mu_X
    mu_X_untreated_vax = (1 - VE_death) * mu_X_untreated
    mu_ward_vax = (1 - VE_death) * mu_ward
    mu_ward_denied_vax = (1 - VE_death) * mu_ward_denied
    mu_icu_vax = (1 - VE_death) * mu_icu
    
    fraction_icu_vax_denied = np.where((H_ward_vax > 0) & (need_icu_vax > 0), 1.0 - g_icu, 0.0)
    effective_mu_ward_vax = mu_ward_vax + (mu_ward_denied_vax - mu_ward_vax) * eta_icu * fraction_icu_vax_denied
    
    # Death flows (vaccinated)
    deaths_I_vax = mu_I_vax * I_vax
    deaths_X_queued_vax = mu_X_untreated_vax * X_queued_vax  # All X_queued_vax deaths are untreated
    deaths_X_admitted_vax = mu_X_vax * X_admitted_vax  # All X_admitted_vax deaths are treated
    deaths_ward_baseline_vax = mu_ward_vax * H_ward_vax
    deaths_ward_icu_denied_vax = (mu_ward_denied_vax - mu_ward_vax) * eta_icu * fraction_icu_vax_denied * H_ward_vax
    deaths_icu_vax = mu_icu_vax * H_icu_vax
    
    # ========================================
    # Compartment Derivatives (Unvaccinated)
    # ========================================
    dS = -new_exposed + waning_flow - new_vaccinations
    if vax_waning_destination == 'S':
        dS = dS + waning_flow_vax
    
    dE = new_exposed - becoming_infectious
    dI = becoming_infectious - (gamma_I + mu_I + sigma) * I
    
    # X_queued: inflow from I, outflow to X_admitted (gated), recovery, and untreated deaths
    dX_queued = sigma * I - (gamma_X + mu_X_untreated) * X_queued - admit_to_X_admitted
    
    # X_admitted: inflow from X_queued (gated), outflow to H_ward, recovery, and treated deaths
    dX_admitted = admit_to_X_admitted - (gamma_X + mu_X) * X_admitted - admit_ward
    
    dH_ward = admit_ward - (gamma_ward + effective_mu_ward) * H_ward - admit_icu
    dH_icu = admit_icu - (gamma_icu + mu_icu) * H_icu
    
    # Recovery from X_queued and X_admitted both contribute to R
    dR = gamma_I * I + gamma_X * (X_queued + X_admitted) + gamma_ward * H_ward + gamma_icu * H_icu - waning_flow
    
    total_deaths = (deaths_I + deaths_X_queued + deaths_X_admitted +
                    deaths_ward_baseline + deaths_ward_icu_denied + deaths_icu)
    dD = total_deaths
    
    # ========================================
    # Compartment Derivatives (Vaccinated)
    # ========================================
    dS_vax = new_vaccinations - new_exposed_vax
    if vax_waning_destination == 'S_vax':
        dS_vax = dS_vax + waning_flow_vax
    
    dE_vax = new_exposed_vax - becoming_infectious_vax
    dI_vax = becoming_infectious_vax - (gamma_I_vax + mu_I_vax + sigma_vax) * I_vax
    
    # X_queued_vax: inflow from I_vax, outflow to X_admitted_vax (gated), recovery, untreated deaths
    dX_queued_vax = sigma_vax * I_vax - (gamma_X + mu_X_untreated_vax) * X_queued_vax - admit_to_X_admitted_vax
    
    # X_admitted_vax: inflow from X_queued_vax (gated), outflow to H_ward_vax, recovery, treated deaths
    dX_admitted_vax = admit_to_X_admitted_vax - (gamma_X + mu_X_vax) * X_admitted_vax - admit_ward_vax
    
    dH_ward_vax = admit_ward_vax - (gamma_ward + effective_mu_ward_vax) * H_ward_vax - admit_icu_vax
    dH_icu_vax = admit_icu_vax - (gamma_icu + mu_icu_vax) * H_icu_vax
    
    # Recovery from X_queued_vax and X_admitted_vax both contribute to R_vax
    dR_vax = (gamma_I_vax * I_vax + gamma_X * (X_queued_vax + X_admitted_vax) + gamma_ward * H_ward_vax + 
              gamma_icu * H_icu_vax - waning_flow_vax)
    
    total_deaths_vax = (deaths_I_vax + deaths_X_queued_vax + deaths_X_admitted_vax +
                        deaths_ward_baseline_vax + deaths_ward_icu_denied_vax + deaths_icu_vax)
    dD_vax = total_deaths_vax
    
    # ========================================
    # Tracked Variable Derivatives
    # ========================================
    # Treated deaths: I, X_admitted, ward baseline, ICU
    dD_treated = deaths_I + deaths_X_admitted + deaths_ward_baseline + deaths_icu
    # Untreated deaths: X_queued, ward patients denied ICU
    dD_untreated = deaths_X_queued + deaths_ward_icu_denied
    dD_vax_treated = deaths_I_vax + deaths_X_admitted_vax + deaths_ward_baseline_vax + deaths_icu_vax
    dD_vax_untreated = deaths_X_queued_vax + deaths_ward_icu_denied_vax
    d_cum_breakthrough = new_exposed_vax  # Rate of breakthrough infections
    
    # ========================================
    # Pack Derivatives
    # ========================================
    derivs = {
        'S': dS, 'E': dE, 'I': dI, 
        'X_queued': dX_queued, 'X_admitted': dX_admitted,
        'H_ward': dH_ward, 'H_icu': dH_icu, 'R': dR, 'D': dD,
        'S_vax': dS_vax, 'E_vax': dE_vax, 'I_vax': dI_vax,
        'X_queued_vax': dX_queued_vax, 'X_admitted_vax': dX_admitted_vax,
        'H_ward_vax': dH_ward_vax, 'H_icu_vax': dH_icu_vax, 'R_vax': dR_vax, 'D_vax': dD_vax,
        'D_treated': dD_treated, 'D_untreated': dD_untreated,
        'D_vax_treated': dD_vax_treated, 'D_vax_untreated': dD_vax_untreated,
        'cum_breakthrough': d_cum_breakthrough
    }
    
    return _pack_state(derivs, n_ages)


def _master_deriv_solve_ivp(t, y, params):
    """
    Wrapper for _master_deriv with solve_ivp argument order (t, y).
    """
    return _master_deriv(y, t, params)


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
    initial_conditions=None,
    # Solver options (new)
    solver='odeint',
    solver_method='LSODA',
    rtol=1e-6,
    atol=1e-9,
    clip_warning_threshold=1e-6
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
    
    # Handle X compartments: support both legacy 'X_by_age' and new split compartments
    # For backward compatibility: if only X_by_age is provided, put all in X_queued (none admitted yet)
    X_queued_default = config.DEFAULT_INITIAL_CONDITIONS.get('X_queued_by_age', config.DEFAULT_INITIAL_CONDITIONS.get('X_by_age', [0, 0, 0]))
    X_admitted_default = config.DEFAULT_INITIAL_CONDITIONS.get('X_admitted_by_age', [0, 0, 0])
    
    if 'X_queued_by_age' in ic_defaults or 'X_admitted_by_age' in ic_defaults:
        # Use new split compartments if provided
        X_queued = _coerce_initial_vector(ic_defaults, 'X_queued_by_age', n_ages, X_queued_default)
        X_admitted = _coerce_initial_vector(ic_defaults, 'X_admitted_by_age', n_ages, X_admitted_default)
    elif 'X_by_age' in ic_defaults:
        # Legacy: put all X in X_queued (waiting for admission)
        X_queued = _coerce_initial_vector(ic_defaults, 'X_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS.get('X_by_age', [0, 0, 0]))
        X_admitted = [0.0] * n_ages
    else:
        X_queued = _coerce_initial_vector(ic_defaults, 'X_queued_by_age', n_ages, X_queued_default)
        X_admitted = [0.0] * n_ages
    
    H_ward = _coerce_initial_vector(ic_defaults, 'H_ward_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_ward_by_age'])
    H_icu = _coerce_initial_vector(ic_defaults, 'H_icu_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_icu_by_age'])
    R = _coerce_initial_vector(ic_defaults, 'R_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['R_by_age'])
    D = _coerce_initial_vector(ic_defaults, 'D_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['D_by_age'])
    
    # Vaccinated compartments - handle X split similarly
    E_vax = _coerce_initial_vector(ic_defaults, 'E_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['E_vax_by_age'])
    I_vax = _coerce_initial_vector(ic_defaults, 'I_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['I_vax_by_age'])
    
    X_queued_vax_default = config.DEFAULT_INITIAL_CONDITIONS.get('X_queued_vax_by_age', config.DEFAULT_INITIAL_CONDITIONS.get('X_vax_by_age', [0, 0, 0]))
    X_admitted_vax_default = config.DEFAULT_INITIAL_CONDITIONS.get('X_admitted_vax_by_age', [0, 0, 0])
    
    if 'X_queued_vax_by_age' in ic_defaults or 'X_admitted_vax_by_age' in ic_defaults:
        X_queued_vax = _coerce_initial_vector(ic_defaults, 'X_queued_vax_by_age', n_ages, X_queued_vax_default)
        X_admitted_vax = _coerce_initial_vector(ic_defaults, 'X_admitted_vax_by_age', n_ages, X_admitted_vax_default)
    elif 'X_vax_by_age' in ic_defaults:
        X_queued_vax = _coerce_initial_vector(ic_defaults, 'X_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS.get('X_vax_by_age', [0, 0, 0]))
        X_admitted_vax = [0.0] * n_ages
    else:
        X_queued_vax = _coerce_initial_vector(ic_defaults, 'X_queued_vax_by_age', n_ages, X_queued_vax_default)
        X_admitted_vax = [0.0] * n_ages
    
    H_ward_vax = _coerce_initial_vector(ic_defaults, 'H_ward_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_ward_vax_by_age'])
    H_icu_vax = _coerce_initial_vector(ic_defaults, 'H_icu_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_icu_vax_by_age'])
    R_vax = _coerce_initial_vector(ic_defaults, 'R_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['R_vax_by_age'])
    D_vax = _coerce_initial_vector(ic_defaults, 'D_vax_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['D_vax_by_age'])
    
    # Calculate S_vax based on coverage (initial vaccinated susceptible population)
    # If S_vax_by_age is explicitly provided in initial_conditions, use it
    # Otherwise, distribute based on coverage fraction of remaining susceptible pool
    # Note: X_total = X_queued + X_admitted for each age group
    X_total_init = [X_queued[a] + X_admitted[a] for a in range(n_ages)]
    X_vax_total_init = [X_queued_vax[a] + X_admitted_vax[a] for a in range(n_ages)]
    
    if 'S_vax_by_age' in ic_defaults and ic_defaults['S_vax_by_age'] != config.DEFAULT_INITIAL_CONDITIONS.get('S_vax_by_age', [0, 0, 0]):
        S_vax = _coerce_initial_vector(ic_defaults, 'S_vax_by_age', n_ages, [0, 0, 0])
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
    
    # ========================================
    # Build ODE Parameters Dictionary
    # ========================================
    ode_params = {
        'n_ages': n_ages,
        'beta_base': beta_base,
        'contact_matrix': np.asarray(contact_matrix),
        'age_params': age_params,
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
        'dm_params': config.DIFFERENTIAL_MORTALITY_PARAMS,
    }
    
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
    }
    y0 = _pack_state(initial_state, n_ages)
    
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
            _master_deriv,
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
            _master_deriv_solve_ivp,
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
    # Unpack Solution and Build History Arrays
    # ========================================
    times = list(t_eval)
    n_times = len(times)
    
    # Per-age compartment histories (unvaccinated)
    S_history = [[] for _ in range(n_ages)]
    E_history = [[] for _ in range(n_ages)]
    I_history = [[] for _ in range(n_ages)]
    X_queued_history = [[] for _ in range(n_ages)]
    X_admitted_history = [[] for _ in range(n_ages)]
    X_history = [[] for _ in range(n_ages)]  # Combined X for backward compatibility
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
    X_queued_vax_history = [[] for _ in range(n_ages)]
    X_admitted_vax_history = [[] for _ in range(n_ages)]
    X_vax_history = [[] for _ in range(n_ages)]  # Combined X_vax for backward compatibility
    H_ward_vax_history = [[] for _ in range(n_ages)]
    H_icu_vax_history = [[] for _ in range(n_ages)]
    R_vax_history = [[] for _ in range(n_ages)]
    D_vax_history = [[] for _ in range(n_ages)]
    D_vax_treated_history = [[] for _ in range(n_ages)]
    D_vax_untreated_history = [[] for _ in range(n_ages)]
    
    # Extract per-age histories from solution
    for t_idx in range(n_times):
        state = _unpack_state(solution[t_idx], n_ages)
        for a in range(n_ages):
            S_history[a].append(state['S'][a])
            E_history[a].append(state['E'][a])
            I_history[a].append(state['I'][a])
            X_queued_history[a].append(state['X_queued'][a])
            X_admitted_history[a].append(state['X_admitted'][a])
            X_history[a].append(state['X_queued'][a] + state['X_admitted'][a])  # Backward compat
            H_ward_history[a].append(state['H_ward'][a])
            H_icu_history[a].append(state['H_icu'][a])
            R_history[a].append(state['R'][a])
            D_history[a].append(state['D'][a])
            D_treated_history[a].append(state['D_treated'][a])
            D_untreated_history[a].append(state['D_untreated'][a])
            
            S_vax_history[a].append(state['S_vax'][a])
            E_vax_history[a].append(state['E_vax'][a])
            I_vax_history[a].append(state['I_vax'][a])
            X_queued_vax_history[a].append(state['X_queued_vax'][a])
            X_admitted_vax_history[a].append(state['X_admitted_vax'][a])
            X_vax_history[a].append(state['X_queued_vax'][a] + state['X_admitted_vax'][a])  # Backward compat
            H_ward_vax_history[a].append(state['H_ward_vax'][a])
            H_icu_vax_history[a].append(state['H_icu_vax'][a])
            R_vax_history[a].append(state['R_vax'][a])
            D_vax_history[a].append(state['D_vax'][a])
            D_vax_treated_history[a].append(state['D_vax_treated'][a])
            D_vax_untreated_history[a].append(state['D_vax_untreated'][a])
    
    # ========================================
    # Post-Processing: Compute Auxiliary Metrics
    # ========================================
    # Aggregated histories
    H_ward_total_history = []
    H_icu_total_history = []
    H_total_history = []
    E_total_history = []
    I_total_history = []
    X_total_history = []
    D_total_history = []
    D_treated_total_history = []
    D_untreated_total_history = []
    
    # Vaccinated aggregates
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
    
    # Flow tracking (computed from state differences)
    new_infections_history = []
    ward_admissions_history = []
    icu_admissions_history = []
    new_vaccinations_history = []
    breakthrough_infections_daily_history = []
    
    # Cumulative overflow (computed via trapezoidal integration)
    cum_ward_overflow = 0.0
    cum_icu_overflow = 0.0
    cum_unmet_ward = [0.0] * n_ages
    cum_unmet_icu = [0.0] * n_ages
    
    # Iterate through solution to compute auxiliary metrics
    for t_idx in range(n_times):
        t = times[t_idx]
        state = _unpack_state(solution[t_idx], n_ages)
        
        # Extract compartments
        S_t = state['S']
        E_t = state['E']
        I_t = state['I']
        # Compute combined X from X_queued and X_admitted
        X_t = state['X_queued'] + state['X_admitted']
        H_ward_t = state['H_ward']
        H_icu_t = state['H_icu']
        R_t = state['R']
        D_t = state['D']
        S_vax_t = state['S_vax']
        E_vax_t = state['E_vax']
        I_vax_t = state['I_vax']
        # Compute combined X_vax from X_queued_vax and X_admitted_vax
        X_vax_t = state['X_queued_vax'] + state['X_admitted_vax']
        H_ward_vax_t = state['H_ward_vax']
        H_icu_vax_t = state['H_icu_vax']
        R_vax_t = state['R_vax']
        D_vax_t = state['D_vax']
        D_treated_t = state['D_treated']
        D_untreated_t = state['D_untreated']
        D_vax_treated_t = state['D_vax_treated']
        D_vax_untreated_t = state['D_vax_untreated']
        cum_breakthrough_t = state['cum_breakthrough']
        
        # Compute aggregates
        H_ward_total = np.sum(H_ward_t) + np.sum(H_ward_vax_t)
        H_icu_total = np.sum(H_icu_t) + np.sum(H_icu_vax_t)
        H_total = H_ward_total + H_icu_total
        E_total = np.sum(E_t) + np.sum(E_vax_t)
        I_total = np.sum(I_t) + np.sum(I_vax_t)
        X_total = np.sum(X_t) + np.sum(X_vax_t)
        D_total = np.sum(D_t) + np.sum(D_vax_t)
        
        H_ward_vax_total = np.sum(H_ward_vax_t)
        H_icu_vax_total = np.sum(H_icu_vax_t)
        H_vax_total = H_ward_vax_total + H_icu_vax_total
        vaccinated_total = (np.sum(S_vax_t) + np.sum(E_vax_t) + np.sum(I_vax_t) + np.sum(X_vax_t) + 
                           H_vax_total + np.sum(R_vax_t) + np.sum(D_vax_t))
        
        H_ward_total_history.append(H_ward_total)
        H_icu_total_history.append(H_icu_total)
        H_total_history.append(H_total)
        E_total_history.append(E_total)
        I_total_history.append(I_total)
        X_total_history.append(X_total)
        D_total_history.append(D_total)
        D_treated_total_history.append(np.sum(D_treated_t) + np.sum(D_vax_treated_t))
        D_untreated_total_history.append(np.sum(D_untreated_t) + np.sum(D_vax_untreated_t))
        
        H_ward_vax_total_history.append(H_ward_vax_total)
        H_icu_vax_total_history.append(H_icu_vax_total)
        H_vax_total_history.append(H_vax_total)
        E_vax_total_history.append(np.sum(E_vax_t))
        I_vax_total_history.append(np.sum(I_vax_t))
        X_vax_total_history.append(np.sum(X_vax_t))
        D_vax_total_history.append(np.sum(D_vax_t))
        vaccinated_total_history.append(vaccinated_total)
        breakthrough_infections_history.append(np.sum(cum_breakthrough_t))
        
        # Time-varying parameters
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
        
        # Capacity gating
        g_ward = hill_gate(H_ward_total, K_ward, n_ward)
        g_icu = hill_gate(H_icu_total, K_icu, n_icu)
        g_ward_history.append(g_ward)
        g_icu_history.append(g_icu)
        
        # Overflow
        ward_overflow = max(0, H_ward_total - K_ward)
        icu_overflow = max(0, H_icu_total - K_icu)
        ward_overflow_history.append(ward_overflow)
        icu_overflow_history.append(icu_overflow)
        
        # Cumulative overflow (trapezoidal integration)
        if t_idx > 0:
            dt_local = times[t_idx] - times[t_idx - 1]
            cum_ward_overflow += 0.5 * (ward_overflow_history[-1] + ward_overflow_history[-2]) * dt_local
            cum_icu_overflow += 0.5 * (icu_overflow_history[-1] + icu_overflow_history[-2]) * dt_local
            
            # Compute unmet care per age group
            for a in range(n_ages):
                eta_a = age_params[a]['eta']
                eta_icu_a = age_params[a].get('eta_icu', 0.1)
                desired_ward = eta_a * (X_t[a] + X_vax_t[a])
                actual_ward = desired_ward * g_ward
                unmet_ward_a = max(0, desired_ward - actual_ward)
                
                desired_icu = eta_icu_a * (H_ward_t[a] + H_ward_vax_t[a])
                actual_icu = desired_icu * g_icu
                unmet_icu_a = max(0, desired_icu - actual_icu)
                
                cum_unmet_ward[a] += unmet_ward_a * dt_local
                cum_unmet_icu[a] += unmet_icu_a * dt_local
        
        # Flow tracking from state differences
        if track_compartment_flows and t_idx > 0:
            state_prev = _unpack_state(solution[t_idx - 1], n_ages)
            dt_local = times[t_idx] - times[t_idx - 1]
            
            # Approximate flows from compartment changes
            # new_infections ~ alpha * E (rate of E -> I)
            alpha_arr = np.array([age_params[a].get('alpha', 0.2) for a in range(n_ages)])
            new_inf = alpha_arr * state_prev['E']
            new_infections_history.append(list(new_inf))
            
            # Ward admissions come from X_admitted (already past the admission gate)
            # ward_admissions ~ eta * X_admitted * g_ward
            eta_arr = np.array([age_params[a]['eta'] for a in range(n_ages)])
            X_prev = state_prev['X_queued'] + state_prev['X_admitted']
            X_vax_prev = state_prev['X_queued_vax'] + state_prev['X_admitted_vax']
            ward_adm = eta_arr * (X_prev + X_vax_prev) * g_ward_history[-2]
            ward_admissions_history.append(list(ward_adm))
            
            # ICU admissions ~ eta_icu * H_ward * g_icu
            eta_icu_arr = np.array([age_params[a].get('eta_icu', 0.1) for a in range(n_ages)])
            icu_adm = eta_icu_arr * (state_prev['H_ward'] + state_prev['H_ward_vax']) * g_icu_history[-2]
            icu_admissions_history.append(list(icu_adm))
            
            # New vaccinations ~ vaccination_rate * S
            new_vax = vaccination_rate * state_prev['S']
            new_vaccinations_history.append(list(new_vax))
            
            # Breakthrough infections (rate of change)
            breakthrough_rate = (cum_breakthrough_t - state_prev['cum_breakthrough']) / dt_local
            breakthrough_infections_daily_history.append(list(breakthrough_rate))
    
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
        'X': X_history,  # Combined X = X_queued + X_admitted for backward compatibility
        'X_queued': X_queued_history,  # Waiting for ward admission
        'X_admitted': X_admitted_history,  # Admitted, receiving treatment
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
        'X_vax': X_vax_history,  # Combined X_vax for backward compatibility
        'X_queued_vax': X_queued_vax_history,
        'X_admitted_vax': X_admitted_vax_history,
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
            'vaccination_rate': list(vaccination_rate) if isinstance(vaccination_rate, np.ndarray) else vaccination_rate,
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
            'contact_matrix': contact_matrix.tolist() if isinstance(contact_matrix, np.ndarray) else contact_matrix,
            # Solver parameters
            'solver': solver,
            'solver_method': solver_method,
            'rtol': rtol,
            'atol': atol,
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
# More features that I want to add later:
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
