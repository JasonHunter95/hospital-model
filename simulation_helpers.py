import warnings
import numpy as np
from time_varying_helpers import seasonal_forcing, policy_multiplier
from model_types import ODEParams
from typing import Dict, Tuple


# ============================================================================
# DEMOGRAPHIC HELPER FUNCTIONS
# ============================================================================

def compute_birth_rate(total_live_pop, birth_rate, age_pops, birth_age_distribution=None):
    """
    Compute age-distributed birth inflow to susceptible compartments.
    
    Births are proportional to the total live population and distributed
    according to birth_age_distribution (defaults to newborns entering youngest age group).
    
    Parameters
    ----------
    total_live_pop : float
        Total current live population (sum across all compartments except D).
    birth_rate : float
        Per-capita birth rate (births per person per day).
        Typical value: ~0.00003 (≈12 births per 1000 per year).
    age_pops : array-like
        Initial population sizes by age group (used for scaling).
    birth_age_distribution : array-like, optional
        Fraction of births entering each age group. Default: [1, 0, 0, ...]
        (all births enter youngest age group).
    
    Returns
    -------
    np.ndarray
        Birth inflow rate per age group (persons per day).
    
    Notes
    -----
    Total births = birth_rate × total_live_pop
    Births per age group = total_births × birth_age_distribution
    
    The model assumes births enter the S (unvaccinated susceptible) compartment.
    For neonatal vaccination, use the neonatal_vaccination_rate parameter in
    simulate_master_hospital_model() to route a fraction of births to S_vax.
    """
    n_ages = len(age_pops)
    
    if birth_rate <= 0:
        return np.zeros(n_ages)
    
    total_births = birth_rate * total_live_pop
    
    if birth_age_distribution is None:
        # Default: all births enter youngest age group
        birth_age_distribution = np.zeros(n_ages)
        birth_age_distribution[0] = 1.0
    else:
        birth_age_distribution = np.array(birth_age_distribution)
        # Normalize to sum to 1
        if birth_age_distribution.sum() > 0:
            birth_age_distribution = birth_age_distribution / birth_age_distribution.sum()
    
    return total_births * birth_age_distribution


def compute_background_death_rate(compartment_pop, mu_background):
    """
    Compute background (non-disease) mortality outflow from a compartment.
    
    Parameters
    ----------
    compartment_pop : np.ndarray
        Current population in compartment by age group.
    mu_background : np.ndarray
        Age-specific background mortality rate (deaths per person per day).
        Typical values: [0.00001, 0.00005, 0.0003] for young, middle, elderly.
    
    Returns
    -------
    np.ndarray
        Background death outflow rate per age group (persons per day).
    
    Notes
    -----
    Background mortality applies to all living compartments:
    S, E, I, X_queued, X_admitted, H_ward, H_icu, R, and their vaccinated counterparts.
    
    This represents deaths from causes other than the modeled disease
    (accidents, other diseases, aging, etc.).
    
    Age-specific rates allow realistic modeling where elderly have higher
    background mortality than young individuals.
    """
    return mu_background * compartment_pop


def validate_demographic_params(demographic_params, n_ages):
    """
    Validate demographic parameters for simulation.
    
    Parameters
    ----------
    demographic_params : dict or None
        Dictionary containing:
        - 'birth_rate': float, per-capita birth rate
        - 'mu_background': float or list, background mortality rate(s)
        - 'birth_age_distribution': list, optional
        - 'neonatal_vaccination_rate': float, optional
    n_ages : int
        Number of age groups.
    
    Returns
    -------
    dict
        Validated and normalized demographic parameters.
    
    Raises
    ------
    ValueError
        If parameters are invalid or inconsistent.
    """
    if demographic_params is None:
        # Return defaults (no demographics)
        return {
            'birth_rate': 0.0,
            'mu_background': np.zeros(n_ages),
            'birth_age_distribution': None,
            'neonatal_vaccination_rate': 0.0,
        }
    
    validated = {}
    
    # Birth rate
    birth_rate = demographic_params.get('birth_rate', 0.0)
    if birth_rate < 0:
        raise ValueError(f"birth_rate must be non-negative, got {birth_rate}")
    validated['birth_rate'] = birth_rate
    
    # Background mortality (age-specific)
    mu_background = demographic_params.get('mu_background', 0.0)
    if isinstance(mu_background, (int, float)):
        # Uniform rate across all ages
        validated['mu_background'] = np.full(n_ages, float(mu_background))
    else:
        mu_background = np.array(mu_background)
        if len(mu_background) != n_ages:
            raise ValueError(
                f"mu_background length ({len(mu_background)}) must match n_ages ({n_ages})"
            )
        if np.any(mu_background < 0):
            raise ValueError("mu_background values must be non-negative")
        validated['mu_background'] = mu_background
    
    # Birth age distribution
    birth_age_dist = demographic_params.get('birth_age_distribution', None)
    if birth_age_dist is not None:
        birth_age_dist = np.array(birth_age_dist)
        if len(birth_age_dist) != n_ages:
            raise ValueError(
                f"birth_age_distribution length ({len(birth_age_dist)}) must match n_ages ({n_ages})"
            )
        if np.any(birth_age_dist < 0):
            raise ValueError("birth_age_distribution values must be non-negative")
        # Normalize
        if birth_age_dist.sum() > 0:
            birth_age_dist = birth_age_dist / birth_age_dist.sum()
        validated['birth_age_distribution'] = birth_age_dist
    else:
        validated['birth_age_distribution'] = None
    
    # Neonatal vaccination rate
    neonatal_vax_rate = demographic_params.get('neonatal_vaccination_rate', 0.0)
    if not (0.0 <= neonatal_vax_rate <= 1.0):
        raise ValueError(
            f"neonatal_vaccination_rate must be between 0 and 1, got {neonatal_vax_rate}"
        )
    validated['neonatal_vaccination_rate'] = neonatal_vax_rate
    
    return validated


def validate_age_structured_inputs(age_params, contact_matrix, age_pops, coverage):
    """Basic shape validation to catch common configuration mistakes early."""
    n_ages = len(age_pops)
    if len(age_params) != n_ages:
        raise ValueError(f"age_params length ({len(age_params)}) must match age_pops ({n_ages}).")
    if contact_matrix.shape != (n_ages, n_ages):
        raise ValueError(f"contact_matrix must be shape {(n_ages, n_ages)}, got {contact_matrix.shape}.")
    if isinstance(coverage, list) and len(coverage) != n_ages:
        raise ValueError(f"coverage length ({len(coverage)}) must match age_pops ({n_ages}).")


def coerce_initial_vector(initial_conditions, key, n_ages, fallback):
    """Return an initial-condition vector of length n_ages."""
    values = initial_conditions.get(key, fallback)
    # ensure list copy to avoid side effects
    values = list(values)
    if len(values) < n_ages:
        values.extend([0] * (n_ages - len(values)))
    return values[:n_ages]


def hill_gate(occupancy, capacity, hill_coef):
    """
    Calculate Hill function gating factor for capacity-constrained admissions.
    
    Parameters
    ----------
    occupancy : float
        Current occupancy level. Must be non-negative.
    capacity : float
        Maximum capacity. Must be positive.
    hill_coef : float
        Hill coefficient controlling steepness of gating. Must be non-negative.
    
    Returns
    -------
    float
        Gating factor between 0 and 1.
    
    Raises
    ------
    ValueError
        If occupancy is negative, capacity is non-positive, or hill_coef is negative.
    
    Notes
    -----
    g(H) = 1 / (1 + (H/K)^n)
    - When H << K: g ≈ 1 (admissions unrestricted)
    - When H = K: g = 0.5 (admissions halved)
    - When H >> K: g → 0 (admissions severely limited)
    
    Uses log-domain computation for large hill_coef values to prevent overflow.

    Mathematical Mapping
    --------------------
    - occupancy -> H
    - capacity -> K
    - hill_coef -> n
    - Formula: g(H) = 1 / (1 + (H/K)^n)
    """
    # Input validation
    if occupancy < 0:
        raise ValueError(f"occupancy must be non-negative, got {occupancy}")
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    if hill_coef < 0:
        raise ValueError(f"hill_coef must be non-negative, got {hill_coef}")
    
    # Edge cases
    if occupancy == 0:
        return 1.0
    if hill_coef == 0:
        return 0.5
    
    import numpy as np
    
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

def hill_gate_vectorized(occupancy, capacity, hill_coef):
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

# Additional tracked variables (differential mortality, demographics) - these are integrated
# separately as part of the state vector for accumulation
TRACKED_ORDER = ['D_treated', 'D_untreated', 'D_vax_treated', 'D_vax_untreated', 
                 'cum_breakthrough', 'cum_births', 'cum_background_deaths']
NUM_TRACKED = len(TRACKED_ORDER)  # 7


def pack_state(compartments, n_ages):
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


def unpack_state(y, n_ages):
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





# ============================================================================
# DERIVATIVE FUNCTION FOR ODE SOLVERS
# ============================================================================

def compute_force_of_infection(state: Dict[str, np.ndarray], params: ODEParams, beta_t: float, theta_X: float, theta_H: float, theta_vax: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
    # X_total = X_queued + X_admitted for live population count
    X_total = X_queued + X_admitted
    X_vax_total = X_queued_vax + X_admitted_vax
    live_pop = (S + E + I + X_total + H_ward + H_icu + R +
                S_vax + E_vax + I_vax + X_vax_total + H_ward_vax + H_icu_vax + R_vax)
    total_live_pop = np.sum(live_pop)
    
    # Avoid division by zero
    live_pop_safe = np.maximum(live_pop, 1e-10)
    
    # Infectious contributions (unvaccinated and vaccinated)
    # Both X_queued and X_admitted contribute to infectiousness
    H_contrib = H_ward + H_icu + H_ward_vax + H_icu_vax
    infectious_unvax = I + theta_X * X_total
    infectious_vax = theta_vax * (I_vax + theta_X * X_vax_total)
    
    # Total infectious proportion per age group
    infectious_fraction = (infectious_unvax + infectious_vax + theta_H * H_contrib) / live_pop_safe
    
    # Calculate absolute effective infectious population (weighted by infectiousness)
    I_eff_absolute = infectious_fraction * live_pop_safe
    
    # Vectorized FOI: lambda_j = beta_t * sum_i(C_ij * I_eff_i) / N_j
    # Assumes contact_matrix[i,j] is contacts per person in i directed at j
    # 1. Total infectious contacts from i to j = C_ij * I_eff_i
    # 2. Sum over i to get total contacts hitting group j
    # 3. Divide by N_j to get contacts per susceptible person
    lambda_foi = beta_t * (contact_matrix.T @ I_eff_absolute) / live_pop_safe
    
    # Force of infection for vaccinated (reduced by VE_infection)
    lambda_foi_vax = (1 - VE_infection) * lambda_foi
    
    return lambda_foi, lambda_foi_vax, live_pop, total_live_pop


def compute_unvax_derivatives(state: Dict[str, np.ndarray], params: ODEParams, lambda_foi: np.ndarray, g_ward: float, g_icu: float, 
                                births_to_S: np.ndarray, waning_flow_vax: np.ndarray, bg_deaths_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate derivatives for unvaccinated compartments.
    
    Parameters
    ----------
    state : dict
        Unpacked state compartments.
    params : dict
        Model parameters.
    lambda_foi : np.ndarray
        Force of infection for unvaccinated.
    g_ward : float
        Ward capacity gating factor.
    g_icu : float
        ICU capacity gating factor.
    births_to_S : np.ndarray
        Birth inflow to S compartment.
    waning_flow_vax : np.ndarray
        Waning flow from R_vax (if destination is 'S').
    bg_deaths_dict : dict
        Background death flows for all compartments.
    
    Returns
    -------
    dict
        Dictionary containing unvaccinated compartment derivatives and death flows.

    Mathematical Mapping
    --------------------
    - lambda_foi -> lambda (Force of infection)
    - alpha -> alpha (E -> I progression)
    - sigma -> sigma (I -> X progression)
    - gamma_I -> gamma_I (Recovery from I)
    - mu_I -> mu_I (Mortality in I)
    - eta -> eta (Ward admission attempt rate)
    - gamma_X -> gamma_X (Recovery from X)
    - mu_X_untreated -> mu_{X,untreated}
    - mu_X -> mu_X (Treated mortality)
    - gamma_X_admit -> gamma_{X,admit} (X -> Ward transfer)
    - gamma_ward -> gamma_{ward}
    - mu_ward -> mu_{ward}
    - eta_icu -> eta_{icu} (ICU admission attempt)
    - gamma_icu -> gamma_{icu}
    - mu_icu -> mu_{icu}
    - omega -> omega (Waning immunity)
    - vaccination_rate -> v
    """
    n_ages = params['n_ages']
    age_params = params['age_params']
    omega = params['omega']
    vaccination_rate = params['vaccination_rate']
    vax_waning_destination = params['vax_waning_destination']
    dm_params = params['dm_params']
    
    # Extract compartments
    S = state['S']
    E = state['E']
    I = state['I']
    X_queued = state['X_queued']
    X_admitted = state['X_admitted']
    H_ward = state['H_ward']
    H_icu = state['H_icu']
    R = state['R']
    
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
    
    # Transitions
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
    
    # Differential mortality
    # Fraction of ward patients denied ICU
    fraction_icu_denied = np.where((H_ward > 0) & (need_icu > 0), 1.0 - g_icu, 0.0)
    effective_mu_ward = mu_ward + (mu_ward_denied - mu_ward) * eta_icu * fraction_icu_denied
    
    # Death flows
    deaths_I = mu_I * I
    deaths_X_queued = mu_X_untreated * X_queued  # All X_queued deaths are untreated
    deaths_X_admitted = mu_X * X_admitted  # All X_admitted deaths are treated
    deaths_ward_baseline = mu_ward * H_ward
    deaths_ward_icu_denied = (mu_ward_denied - mu_ward) * eta_icu * fraction_icu_denied * H_ward
    deaths_icu = mu_icu * H_icu
    
    # Compartment derivatives
    dS = births_to_S - new_exposed + waning_flow - new_vaccinations - bg_deaths_dict['S']
    if vax_waning_destination == 'S':
        dS = dS + waning_flow_vax
    
    dE = new_exposed - becoming_infectious - bg_deaths_dict['E']
    dI = becoming_infectious - (gamma_I + mu_I + sigma) * I - bg_deaths_dict['I']
    
    # X_queued: inflow from I, outflow to X_admitted (gated), recovery, untreated deaths, bg deaths
    dX_queued = sigma * I - (gamma_X + mu_X_untreated) * X_queued - admit_to_X_admitted - bg_deaths_dict['X_queued']
    
    # X_admitted: inflow from X_queued (gated), outflow to H_ward, recovery, treated deaths, bg deaths
    dX_admitted = admit_to_X_admitted - (gamma_X + mu_X) * X_admitted - admit_ward - bg_deaths_dict['X_admitted']
    
    dH_ward = admit_ward - (gamma_ward + effective_mu_ward) * H_ward - admit_icu - bg_deaths_dict['H_ward']
    dH_icu = admit_icu - (gamma_icu + mu_icu) * H_icu - bg_deaths_dict['H_icu']
    
    # Recovery from X_queued and X_admitted both contribute to R
    dR = gamma_I * I + gamma_X * (X_queued + X_admitted) + gamma_ward * H_ward + gamma_icu * H_icu - waning_flow - bg_deaths_dict['R']
    
    total_deaths = (deaths_I + deaths_X_queued + deaths_X_admitted +
                    deaths_ward_baseline + deaths_ward_icu_denied + deaths_icu)
    dD = total_deaths
    
    # Treated deaths: I, X_admitted, ward baseline, ICU
    dD_treated = deaths_I + deaths_X_admitted + deaths_ward_baseline + deaths_icu
    # Untreated deaths: X_queued, ward patients denied ICU
    dD_untreated = deaths_X_queued + deaths_ward_icu_denied
    
    return {
        'dS': dS, 'dE': dE, 'dI': dI,
        'dX_queued': dX_queued, 'dX_admitted': dX_admitted,
        'dH_ward': dH_ward, 'dH_icu': dH_icu, 'dR': dR, 'dD': dD,
        'dD_treated': dD_treated, 'dD_untreated': dD_untreated,
        'new_exposed_vax_for_tracking': None  # Placeholder, will be filled by vax function
    }


def compute_vax_derivatives(state: Dict[str, np.ndarray], params: ODEParams, lambda_foi_vax: np.ndarray, g_ward: float, g_icu: float,
                              births_to_S_vax: np.ndarray, new_vaccinations: np.ndarray, waning_flow_vax: np.ndarray, bg_deaths_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate derivatives for vaccinated compartments.
    
    Parameters
    ----------
    state : dict
        Unpacked state compartments.
    params : dict
        Model parameters.
    lambda_foi_vax : np.ndarray
        Force of infection for vaccinated (reduced by VE_infection).
    g_ward : float
        Ward capacity gating factor.
    g_icu : float
        ICU capacity gating factor.
    births_to_S_vax : np.ndarray
        Birth inflow to S_vax compartment (neonatal vaccination).
    new_vaccinations : np.ndarray
        Vaccination flow from S to S_vax.
    waning_flow_vax : np.ndarray
        Waning flow from R_vax.
    bg_deaths_dict : dict
        Background death flows for all compartments.
    
    Returns
    -------
    dict
        Dictionary containing vaccinated compartment derivatives and death flows.

    Mathematical Mapping
    --------------------
    - lambda_foi_vax -> lambda_{vax} (Force of infection for vaccinated)
    - alpha -> alpha (E -> I progression)
    - sigma -> sigma (I -> X progression)
    - gamma_I -> gamma_I (Recovery from I)
    - mu_I -> mu_I (Mortality in I)
    - eta -> eta (Ward admission attempt rate)
    - gamma_X -> gamma_X (Recovery from X)
    - mu_X_untreated -> mu_{X,untreated}
    - mu_X -> mu_X (Treated mortality)
    - gamma_X_admit -> gamma_{X,admit} (X -> Ward transfer)
    - gamma_ward -> gamma_{ward}
    - mu_ward -> mu_{ward}
    - eta_icu -> eta_{icu} (ICU admission attempt)
    - gamma_icu -> gamma_{icu}
    - mu_icu -> mu_{icu}
    - omega -> omega (Waning immunity)
    - vaccination_rate -> v
    """
    n_ages = params['n_ages']
    age_params = params['age_params']
    VE_severe = params['VE_severe']
    VE_death = params['VE_death']
    vax_waning_destination = params['vax_waning_destination']
    dm_params = params['dm_params']
    
    # Extract compartments
    S_vax = state['S_vax']
    E_vax = state['E_vax']
    I_vax = state['I_vax']
    X_queued_vax = state['X_queued_vax']
    X_admitted_vax = state['X_admitted_vax']
    H_ward_vax = state['H_ward_vax']
    H_icu_vax = state['H_icu_vax']
    R_vax = state['R_vax']
    
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
    
    # Transitions
    new_exposed_vax = lambda_foi_vax * S_vax
    becoming_infectious_vax = alpha * E_vax
    sigma_vax = (1 - VE_severe) * sigma
    # Preserve/accelerate total I exit when sigma is reduced by vaccination
    gamma_I_vax = gamma_I + (sigma - sigma_vax)
    
    # X_queued_vax -> X_admitted_vax flow (gated by ward capacity)
    admit_to_X_admitted_vax = eta * X_queued_vax * g_ward
    
    # X_admitted_vax -> H_ward_vax flow
    gamma_X_admit = np.array([ap.get('gamma_X_admit', ap['eta']) for ap in age_params])
    admit_ward_vax = gamma_X_admit * X_admitted_vax
    
    need_icu_vax = eta_icu * H_ward_vax
    admit_icu_vax = need_icu_vax * g_icu
    
    # Differential mortality (vaccinated)
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
    
    # Compartment derivatives
    dS_vax = births_to_S_vax + new_vaccinations - new_exposed_vax - bg_deaths_dict['S_vax']
    if vax_waning_destination == 'S_vax':
        dS_vax = dS_vax + waning_flow_vax
    
    dE_vax = new_exposed_vax - becoming_infectious_vax - bg_deaths_dict['E_vax']
    dI_vax = becoming_infectious_vax - (gamma_I_vax + mu_I_vax + sigma_vax) * I_vax - bg_deaths_dict['I_vax']
    
    # X_queued_vax: inflow from I_vax, outflow to X_admitted_vax (gated), recovery, untreated deaths, bg deaths
    dX_queued_vax = sigma_vax * I_vax - (gamma_X + mu_X_untreated_vax) * X_queued_vax - admit_to_X_admitted_vax - bg_deaths_dict['X_queued_vax']
    
    # X_admitted_vax: inflow from X_queued_vax (gated), outflow to H_ward_vax, recovery, treated deaths, bg deaths
    dX_admitted_vax = admit_to_X_admitted_vax - (gamma_X + mu_X_vax) * X_admitted_vax - admit_ward_vax - bg_deaths_dict['X_admitted_vax']
    
    dH_ward_vax = admit_ward_vax - (gamma_ward + effective_mu_ward_vax) * H_ward_vax - admit_icu_vax - bg_deaths_dict['H_ward_vax']
    dH_icu_vax = admit_icu_vax - (gamma_icu + mu_icu_vax) * H_icu_vax - bg_deaths_dict['H_icu_vax']
    
    # Recovery from X_queued_vax and X_admitted_vax both contribute to R_vax
    dR_vax = (gamma_I_vax * I_vax + gamma_X * (X_queued_vax + X_admitted_vax) + gamma_ward * H_ward_vax + 
              gamma_icu * H_icu_vax - waning_flow_vax - bg_deaths_dict['R_vax'])
    
    total_deaths_vax = (deaths_I_vax + deaths_X_queued_vax + deaths_X_admitted_vax +
                        deaths_ward_baseline_vax + deaths_ward_icu_denied_vax + deaths_icu_vax)
    dD_vax = total_deaths_vax
    
    # Treated and untreated deaths
    dD_vax_treated = deaths_I_vax + deaths_X_admitted_vax + deaths_ward_baseline_vax + deaths_icu_vax
    dD_vax_untreated = deaths_X_queued_vax + deaths_ward_icu_denied_vax
    
    return {
        'dS_vax': dS_vax, 'dE_vax': dE_vax, 'dI_vax': dI_vax,
        'dX_queued_vax': dX_queued_vax, 'dX_admitted_vax': dX_admitted_vax,
        'dH_ward_vax': dH_ward_vax, 'dH_icu_vax': dH_icu_vax, 'dR_vax': dR_vax, 'dD_vax': dD_vax,
        'dD_vax_treated': dD_vax_treated, 'dD_vax_untreated': dD_vax_untreated,
        'new_exposed_vax': new_exposed_vax
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
        - demographic_params: dict with birth_rate, mu_background, 
          birth_age_distribution, neonatal_vaccination_rate
    
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
    interventions = params['interventions']
    vaccination_rate = params['vaccination_rate']
    
    # Extract demographic parameters
    demo_params = params.get('demographic_params', {})
    birth_rate = demo_params.get('birth_rate', 0.0)
    mu_background = demo_params.get('mu_background', np.zeros(n_ages))
    birth_age_dist = demo_params.get('birth_age_distribution', None)
    neonatal_vax_rate = demo_params.get('neonatal_vaccination_rate', 0.0)
    age_pops = params.get('age_pops', np.ones(n_ages))  # For birth scaling
    
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
    H_ward_total = (np.sum(state['H_ward']) + np.sum(state['H_ward_vax']) +
                    np.sum(state['X_admitted']) + np.sum(state['X_admitted_vax']))
    H_icu_total = np.sum(state['H_icu']) + np.sum(state['H_icu_vax'])
    g_ward = hill_gate_vectorized(H_ward_total, K_ward, n_ward)
    g_icu = hill_gate_vectorized(H_icu_total, K_icu, n_icu)
    
    # ========================================
    # Force of Infection
    # ========================================
    lambda_foi, lambda_foi_vax, live_pop, total_live_pop = compute_force_of_infection(
        state, params, beta_t, theta_X, theta_H, theta_vax
    )
    
    # ========================================
    # Demographic Flows (Births and Background Deaths)
    # ========================================
    # Births: enter S (unvaccinated) by default, with optional neonatal vaccination to S_vax
    births_total = compute_birth_rate(total_live_pop, birth_rate, age_pops, birth_age_dist)
    births_to_S = births_total * (1.0 - neonatal_vax_rate)
    births_to_S_vax = births_total * neonatal_vax_rate
    
    # Background deaths: age-specific mortality applied to all living compartments
    bg_deaths_dict = {}
    for comp_name in ['S', 'E', 'I', 'X_queued', 'X_admitted', 'H_ward', 'H_icu', 'R',
                      'S_vax', 'E_vax', 'I_vax', 'X_queued_vax', 'X_admitted_vax', 
                      'H_ward_vax', 'H_icu_vax', 'R_vax']:
        bg_deaths_dict[comp_name] = compute_background_death_rate(state[comp_name], mu_background)
    
    # Total background deaths (for tracking)
    total_bg_deaths = sum(bg_deaths_dict.values())
    
    # ========================================
    # Compute Waning Flow (needed by both unvax and vax derivatives)
    # ========================================
    omega = params['omega']
    waning_flow_vax = omega_vax * state['R_vax']
    new_vaccinations = vaccination_rate * state['S']
    
    # ========================================
    # Compute Derivatives Using Helper Functions
    # ========================================
    unvax_results = compute_unvax_derivatives(
        state, params, lambda_foi, g_ward, g_icu, 
        births_to_S, waning_flow_vax, bg_deaths_dict
    )
    
    vax_results = compute_vax_derivatives(
        state, params, lambda_foi_vax, g_ward, g_icu,
        births_to_S_vax, new_vaccinations, waning_flow_vax, bg_deaths_dict
    )
    
    # ========================================
    # Tracked Variable Derivatives
    # ========================================
    d_cum_breakthrough = vax_results['new_exposed_vax']  # Rate of breakthrough infections
    d_cum_births = births_total  # Rate of new births
    d_cum_background_deaths = total_bg_deaths  # Rate of background deaths
    
    # ========================================
    # Pack Derivatives
    # ========================================
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
        'S_vax': vax_results['dS_vax'], 
        'E_vax': vax_results['dE_vax'], 
        'I_vax': vax_results['dI_vax'],
        'X_queued_vax': vax_results['dX_queued_vax'], 
        'X_admitted_vax': vax_results['dX_admitted_vax'],
        'H_ward_vax': vax_results['dH_ward_vax'], 
        'H_icu_vax': vax_results['dH_icu_vax'], 
        'R_vax': vax_results['dR_vax'], 
        'D_vax': vax_results['dD_vax'],
        'D_treated': unvax_results['dD_treated'], 
        'D_untreated': unvax_results['dD_untreated'],
        'D_vax_treated': vax_results['dD_vax_treated'], 
        'D_vax_untreated': vax_results['dD_vax_untreated'],
        'cum_breakthrough': d_cum_breakthrough,
        'cum_births': d_cum_births,
        'cum_background_deaths': d_cum_background_deaths
    }
    
    return pack_state(derivs, n_ages)


def master_deriv_solve_ivp(t, y, params):
    """
    Wrapper for master_deriv with solve_ivp argument order (t, y).
    """
    return master_deriv(y, t, params)
