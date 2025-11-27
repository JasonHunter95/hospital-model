import config
from time_varying_helpers import seasonal_forcing, policy_multiplier


def _validate_age_structured_inputs(age_params, contact_matrix, age_pops, coverage):
    """Basic shape validation to catch common configuration mistakes early."""
    n_ages = len(age_pops)
    if len(age_params) != n_ages:
        raise ValueError(f"age_params length ({len(age_params)}) must match age_pops ({n_ages}).")
    if contact_matrix.shape != (n_ages, n_ages):
        raise ValueError(f"contact_matrix must be shape {(n_ages, n_ages)}, got {contact_matrix.shape}.")
    if isinstance(coverage, list) and len(coverage) != n_ages:
        raise ValueError(f"coverage length ({len(coverage)}) must match age_pops ({n_ages}).")


def _coerce_initial_vector(initial_conditions, key, n_ages, fallback):
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
        Current occupancy level.
    capacity : float
        Maximum capacity.
    hill_coef : float
        Hill coefficient controlling steepness of gating.
    
    Returns
    -------
    float
        Gating factor between 0 and 1.
    
    Notes
    -----
    g(H) = 1 / (1 + (H/K)^n)
    - When H << K: g ≈ 1 (admissions unrestricted)
    - When H = K: g = 0.5 (admissions halved)
    - When H >> K: g → 0 (admissions severely limited)
    """
    if capacity <= 0:
        return 0.0
    return 1.0 / (1.0 + (occupancy / capacity) ** hill_coef)


def simulate_basic_hospital_model(
    beta,
    sigma,
    eta,
    gamma_I,
    mu_I,
    gamma_X,
    mu_X,
    gamma_H,
    mu_H,
    theta_X=None,
    theta_H=None,
    hosp_capacity=None,
    hill_coef=None,
    coverage=0.0,
    VE=None,
    N=None,
    S=None,
    I=None,
    X=None,
    H=None,
    R=None,
    D=None,
    Tmax=None,
    time_step=None
):
    """
    Simulate SIXHRD hospital model with capacity constraints (legacy single-H version).
    
    This model simulates a compartmental epidemic model with six compartments:
    S (susceptible), I (infected), X (severe cases needing hospital care),
    H (hospitalized), R (recovered), and D (dead). Hospital capacity is
    enforced using a smooth Hill function to gate admissions.
    
    Parameters
    ----------
    beta : float
        Base transmission rate (contacts per day * probability of transmission).
    sigma : float
        Progression rate from I to X (1/days).
    eta : float
        Fraction of X cases that need hospitalization.
    gamma_I : float
        Recovery rate from I compartment (1/days).
    mu_I : float
        Mortality rate from I compartment (1/days).
    gamma_X : float
        Recovery rate from X compartment (1/days).
    mu_X : float
        Mortality rate from X compartment (1/days).
    gamma_H : float
        Recovery rate from H compartment (1/days).
    mu_H : float
        Mortality rate from H compartment (1/days).
    theta_X : float
        Relative infectiousness of X compartment compared to I.
    theta_H : float
        Relative infectiousness of H compartment compared to I.
    hosp_capacity : int
        Maximum hospital bed capacity (K).
    hill_coef : float
        Hill coefficient (n) for admission gating function steepness.
    coverage : float
        Vaccine coverage as fraction of population (0-1).
    VE : float
        Vaccine efficacy using leaky model (0-1).
    N : int
        Total population size.
    S : float, optional
        Initial susceptible population (default: N - 10).
    I : float, optional
        Initial infected population (default: 10).
    X : float, optional
        Initial severe cases (default: 0).
    H : float, optional
        Initial hospitalized (default: 0).
    R : float, optional
        Initial recovered (default: 0).
    D : float, optional
        Initial dead (default: 0).
    Tmax : float, optional
        Simulation duration in days (default: 200).
    time_step : float, optional
        Integration time step for Euler method in days (default: 0.1).
    
    Returns
    -------
    tuple
        (times, S_vals, I_vals, X_vals, H_vals, R_vals, D_vals, 
         overflow_vals, cum_overflow, cum_unmet, unmet_vals)
        All time series arrays and cumulative metrics.
    
    Notes
    -----
    The admission gating function is: g(H) = 1 / (1 + (H/K)^n)
    When H << K, g ≈ 1 (admissions unrestricted).
    When H >> K, g → 0 (admissions severely limited).
    
    Vaccine effect: beta_eff = beta * (1 - coverage * VE)
    
    See Also
    --------
    simulate_hospital_model_icu : Extended model with separate ward and ICU compartments.
    
    Examples
    --------
    >>> times, S, I, X, H, R, D, overflow, cum_overflow, cum_unmet, unmet = \\
    ...     simulate_basic_hospital_model(
    ...         beta=0.3, sigma=0.2, eta=0.3, gamma_I=0.1, mu_I=0.01,
    ...         gamma_X=0.15, mu_X=0.05, gamma_H=0.2, mu_H=0.02,
    ...         theta_X=0.5, theta_H=0.3, hosp_capacity=100, hill_coef=4,
    ...         coverage=0.1, VE=0.7, N=10000
    ...     )
    >>> print(f"Final deaths: {D[-1]:.0f}")
    >>> print(f"Peak hospital occupancy: {max(H):.0f}")
    """
    theta_X = config.DEFAULT_SIM_PARAMS['theta_X'] if theta_X is None else theta_X
    theta_H = config.DEFAULT_SIM_PARAMS['theta_H'] if theta_H is None else theta_H
    hill_coef = config.DEFAULT_SIM_PARAMS['hill_coef'] if hill_coef is None else hill_coef
    Tmax = config.DEFAULT_SIM_PARAMS['Tmax'] if Tmax is None else Tmax
    time_step = config.DEFAULT_SIM_PARAMS['time_step'] if time_step is None else time_step
    VE = config.DEFAULT_SIM_PARAMS['VE'] if VE is None else VE
    if hosp_capacity is None:
        hosp_capacity = config.DEFAULT_CAPACITY_PARAMS['total_capacity']
    K = hosp_capacity
    n = hill_coef
    dt = time_step
    
    if N is None:
        raise ValueError("Total population N must be provided.")
    defaults = config.DEFAULT_INITIAL_CONDITIONS
    # Initial conditions - use parameters if provided, else defaults
    if S is None:
        S = N - defaults['I_single']
    if I is None:
        I = defaults['I_single']
    if X is None:
        X = defaults['X_single']
    if H is None:
        H = defaults['H_single']
    if R is None:
        R = defaults['R_single']
    if D is None:
        D = defaults['D_single']
    
    # storage arrays
    times = []
    S_vals, I_vals, X_vals, H_vals, R_vals, D_vals = [], [], [], [], [], []
    overflow_vals = []
    unmet_vals = []
    cum_overflow = 0
    cum_unmet = 0
    
    # vaccine effect
    eff_beta = beta * (1 - coverage * VE)
    
    # loop across time
    t = 0
    while t <= Tmax:
        # store values at each time step
        times.append(t)
        S_vals.append(S)
        I_vals.append(I)
        X_vals.append(X)
        H_vals.append(H)
        R_vals.append(R)
        D_vals.append(D)
        
        # Force of infection
        lambda_foi = eff_beta * (I + theta_X*X + theta_H*H) / N
        new_inf = lambda_foi * S
        
        # Capacity gate (smooth Hill function)
        g = hill_gate(H, K, n)
        admit = eta * X * g
        
        # The ODEs
        dS = -new_inf
        dI = new_inf - (gamma_I + mu_I + sigma) * I
        dX = sigma*I - (gamma_X + mu_X) * X - admit
        dH = admit - (gamma_H + mu_H) * H
        dR = gamma_I*I + gamma_X*X + gamma_H*H
        dD = mu_I*I + mu_X*X + mu_H*H
        
        # Euler update
        S += dS * dt
        I += dI * dt
        X += dX * dt
        H += dH * dt
        R += dR * dt
        D += dD * dt
        
        # Prevent negative values
        S = max(0, S)
        I = max(0, I)
        X = max(0, X)
        H = max(0, H)
        R = max(0, R)
        D = max(0, D)
        
        # Track metrics
        overflow = max(0, H - K)
        cum_overflow += overflow * dt
        overflow_vals.append(overflow)
        
        unmet_care = eta*X - admit
        unmet_vals.append(unmet_care)
        cum_unmet += max(0, unmet_care) * dt
        
        t += dt
        
    return times, S_vals, I_vals, X_vals, H_vals, R_vals, D_vals, overflow_vals, cum_overflow, cum_unmet, unmet_vals


def simulate_age_structured_hospital_model(
    beta,
    age_params,
    contact_matrix,
    hosp_capacity=None,
    hill_coef=None,
    coverage=0.0,
    VE=None,
    age_pops=None,
    theta_X=None,
    theta_H=None,
    Tmax=None,
    time_step=None,
    initial_conditions=None
):
    """
    Simulate age-structured SIXHRD model with shared hospital capacity (legacy version).
    
    This extends the basic SIXHRD model to include age structure with separate
    compartments for each age group. Age groups have distinct disease parameters
    (severity, mortality rates) and interact via a contact matrix. Hospital
    capacity is shared across all age groups with a single admission gating function.
    
    Parameters
    ----------
    beta : float
        Base transmission rate (contacts per day * probability of transmission).
    age_params : list of dict
        Age-specific parameters for each group. Each dict must contain:
        'sigma', 'eta', 'gamma_I', 'mu_I', 'gamma_X', 'mu_X', 'gamma_H', 'mu_H'.
    contact_matrix : ndarray
        Contact rates C[a,b] between age groups (infector a, infectee b).
        Shape: (n_ages, n_ages).
    hosp_capacity : float
        Total hospital capacity (shared across all age groups).
    hill_coef : float
        Hill coefficient for admission gating function steepness.
    coverage : float or list
        Vaccine coverage for each age group. If float, applied uniformly.
        If list, specifies age-specific coverage [young, middle, elderly].
    VE : float
        Vaccine efficacy using leaky model (0-1).
    age_pops : list
        Population size for each age group (e.g., [3000, 5000, 2000]).
    theta_X : float, optional
        Relative infectiousness of X compartment (default: 0.5).
    theta_H : float, optional
        Relative infectiousness of H compartment (default: 0.3).
    Tmax : float, optional
        Simulation duration in days (default: 200).
    time_step : float, optional
        Integration time step for Euler method in days (default: 0.1).
    
    Returns
    -------
    dict
        Results dictionary containing:
        - 'times': array of time points
        - 'S', 'I', 'X', 'H', 'R', 'D': lists of arrays for each age group
        - 'H_total': total hospitalized across all ages
        - 'overflow': overflow time series
        - 'cum_overflow': cumulative overflow (patient-days)
        - 'cum_unmet': list of cumulative unmet care by age group
        - 'age_pops': population sizes (echoed back)
    
    Notes
    -----
    Force of infection for age group a:
        λ_a = β_eff * Σ_b C[a,b] * (I_b + θ_X * X_b + θ_H * H_b) / N_b
    
    Admission gating uses total hospitalized: g = 1 / (1 + (H_total/K)^n)
    
    Vaccine effect: β_eff[a] = β * (1 - coverage[a] * VE)
    
    See Also
    --------
    simulate_age_structured_model_icu : Extended model with separate ward and ICU.
    
    Examples
    --------
    >>> from config import AGE_PARAMS_DEFAULT, CONTACT_MATRIX_DEFAULT
    >>> results = simulate_age_structured_hospital_model(
    ...     beta=0.3,
    ...     age_params=AGE_PARAMS_DEFAULT,
    ...     contact_matrix=CONTACT_MATRIX_DEFAULT,
    ...     hosp_capacity=100,
    ...     hill_coef=4,
    ...     coverage=[0.2, 0.3, 0.7],
    ...     VE=0.7,
    ...     age_pops=[3000, 5000, 2000]
    ... )
    >>> total_deaths = sum([results['D'][a][-1] for a in range(3)])
    >>> print(f"Total deaths: {total_deaths:.0f}")
    """
    if age_pops is None:
        raise ValueError("age_pops must be provided.")
    theta_X = config.DEFAULT_SIM_PARAMS['theta_X'] if theta_X is None else theta_X
    theta_H = config.DEFAULT_SIM_PARAMS['theta_H'] if theta_H is None else theta_H
    hill_coef = config.DEFAULT_SIM_PARAMS['hill_coef'] if hill_coef is None else hill_coef
    Tmax = config.DEFAULT_SIM_PARAMS['Tmax'] if Tmax is None else Tmax
    time_step = config.DEFAULT_SIM_PARAMS['time_step'] if time_step is None else time_step
    VE = config.DEFAULT_SIM_PARAMS['VE'] if VE is None else VE
    if hosp_capacity is None:
        hosp_capacity = config.DEFAULT_CAPACITY_PARAMS['total_capacity']
    n_ages = len(age_pops)
    K = hosp_capacity
    n = hill_coef
    dt = time_step
    
    _validate_age_structured_inputs(age_params, contact_matrix, age_pops, coverage)
    
    # Handle coverage - convert to list if scalar
    if not isinstance(coverage, list):
        coverage = [coverage] * n_ages
    
    # Initialize compartments for each age group
    ic_defaults = config.DEFAULT_INITIAL_CONDITIONS if initial_conditions is None else {**config.DEFAULT_INITIAL_CONDITIONS, **initial_conditions}
    I = _coerce_initial_vector(ic_defaults, 'I_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['I_by_age'])
    X = _coerce_initial_vector(ic_defaults, 'X_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['X_by_age'])
    H = _coerce_initial_vector(ic_defaults, 'H_ward_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_ward_by_age'])
    R = _coerce_initial_vector(ic_defaults, 'R_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['R_by_age'])
    D = _coerce_initial_vector(ic_defaults, 'D_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['D_by_age'])
    S = [max(0, age_pops[a] - I[a] - X[a] - H[a] - R[a] - D[a]) for a in range(n_ages)]
    
    # Storage
    times = []
    S_history = [[] for _ in range(n_ages)]
    I_history = [[] for _ in range(n_ages)]
    X_history = [[] for _ in range(n_ages)]
    H_history = [[] for _ in range(n_ages)]
    R_history = [[] for _ in range(n_ages)]
    D_history = [[] for _ in range(n_ages)]
    H_total_history = []
    overflow_history = []
    
    cum_overflow = 0
    cum_unmet = [0] * n_ages
    
    # Age-specific effective beta
    eff_beta = [beta * (1 - coverage[a] * VE) for a in range(n_ages)]
    
    t = 0
    while t <= Tmax:
        times.append(t)
        
        # Store current state
        for a in range(n_ages):
            S_history[a].append(S[a])
            I_history[a].append(I[a])
            X_history[a].append(X[a])
            H_history[a].append(H[a])
            R_history[a].append(R[a])
            D_history[a].append(D[a])
        
        H_total = sum(H)
        H_total_history.append(H_total)
        
        # Calculate total hospital load for admission gating
        g = hill_gate(H_total, K, n)
        
        # Calculate force of infection for each age group
        lambda_foi = [0] * n_ages
        for a in range(n_ages):
            for b in range(n_ages):
                live_pop_b = S[b] + I[b] + X[b] + H[b] + R[b]
                if live_pop_b > 0:
                    lambda_foi[a] += (eff_beta[a] * contact_matrix[a][b] * 
                                     (I[b] + theta_X*X[b] + theta_H*H[b]) / live_pop_b)
        
        # Calculate new infections and transitions for each age group
        new_inf = [lambda_foi[a] * S[a] for a in range(n_ages)]
        admit = [age_params[a]['eta'] * X[a] * g for a in range(n_ages)]
        
        # Update compartments
        dS = [-new_inf[a] for a in range(n_ages)]
        dI = [new_inf[a] - (age_params[a]['gamma_I'] + age_params[a]['mu_I'] + 
              age_params[a]['sigma']) * I[a] for a in range(n_ages)]
        dX = [age_params[a]['sigma'] * I[a] - (age_params[a]['gamma_X'] + 
              age_params[a]['mu_X']) * X[a] - admit[a] for a in range(n_ages)]
        dH = [admit[a] - (age_params[a]['gamma_H'] + age_params[a]['mu_H']) * H[a] 
              for a in range(n_ages)]
        dR = [age_params[a]['gamma_I'] * I[a] + age_params[a]['gamma_X'] * X[a] + 
              age_params[a]['gamma_H'] * H[a] for a in range(n_ages)]
        dD = [age_params[a]['mu_I'] * I[a] + age_params[a]['mu_X'] * X[a] + 
              age_params[a]['mu_H'] * H[a] for a in range(n_ages)]
        
        # Euler update
        for a in range(n_ages):
            S[a] = max(0, S[a] + dS[a] * dt)
            I[a] = max(0, I[a] + dI[a] * dt)
            X[a] = max(0, X[a] + dX[a] * dt)
            H[a] = max(0, H[a] + dH[a] * dt)
            R[a] = max(0, R[a] + dR[a] * dt)
            D[a] = max(0, D[a] + dD[a] * dt)
            
            # Track unmet care
            unmet = age_params[a]['eta'] * X[a] - admit[a]
            cum_unmet[a] += max(0, unmet) * dt
        
        # Track overflow
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
        'age_pops': age_pops
    }


def simulate_age_structured_hospital_model_with_icu_ward_split(
    beta,
    age_params,
    contact_matrix,
    ward_capacity=None,
    icu_capacity=None,
    hill_coef_ward=None,
    hill_coef_icu=None,
    coverage=0.0,
    VE=None,
    age_pops=None,
    theta_X=None,
    theta_H=None,
    Tmax=None,
    time_step=None,
    track_differential_mortality=True,
    initial_conditions=None
):
    """
    Simulate age-structured SIXHRD model with separate ward and ICU capacity.
    
    This extends the age-structured model to include separate general ward (H_ward)
    and ICU (H_icu) compartments for each age group, with independent capacity
    constraints. ICU overflow has different mortality implications.
    
    Parameters
    ----------
    beta : float
        Base transmission rate (contacts per day * probability of transmission).
    age_params : list of dict
        Age-specific parameters for each group. Each dict must contain:
        'sigma', 'eta', 'eta_icu', 'gamma_I', 'mu_I', 'gamma_X', 'mu_X',
        'gamma_ward', 'mu_ward', 'gamma_icu', 'mu_icu'.
        For differential mortality, can also include:
        'mu_X_untreated', 'mu_ward_denied_icu'.
    contact_matrix : ndarray
        Contact rates C[a,b] between age groups (infector a, infectee b).
    ward_capacity : float
        Total general ward capacity (shared across all age groups).
    icu_capacity : float
        Total ICU capacity (shared across all age groups).
    hill_coef_ward : float
        Hill coefficient for ward admission gating.
    hill_coef_icu : float
        Hill coefficient for ICU admission gating.
    coverage : float or list
        Vaccine coverage for each age group.
    VE : float
        Vaccine efficacy using leaky model (0-1).
    age_pops : list
        Population size for each age group.
    theta_X : float, optional
        Relative infectiousness of X compartment (default: 0.5).
    theta_H : float, optional
        Relative infectiousness of hospitalized compartments (default: 0.3).
    Tmax : float, optional
        Simulation duration in days (default: 200).
    time_step : float, optional
        Integration time step (default: 0.1).
    track_differential_mortality : bool, optional
        Whether to track deaths by care status (default: True).
        If True, tracks D_treated vs D_untreated separately.
    
    Returns
    -------
    dict
        Results dictionary containing:
        - 'times': array of time points
        - 'S', 'I', 'X', 'R', 'D': lists of arrays for each age group
        - 'H_ward', 'H_icu': ward and ICU by age group
        - 'H_ward_total', 'H_icu_total', 'H_total': aggregated totals
        - 'ward_overflow', 'icu_overflow': overflow time series
        - 'cum_ward_overflow', 'cum_icu_overflow': cumulative overflow
        - 'D_treated', 'D_untreated': deaths by care status (if tracking enabled)
        - 'D_treated_total', 'D_untreated_total': total treated/untreated deaths
        - Capacity values and age populations
    
    Notes
    -----
    Compartment flow for each age group:
        S_a → I_a → X_a → H_ward_a → H_icu_a → R_a or D_a
    
    Both ward and ICU capacities are shared across all age groups.
    
    Differential mortality tracks:
    - D_treated: Deaths in I (baseline), H_ward, H_icu compartments
    - D_untreated: Deaths in X when care denied (higher mu_X_untreated rate),
                   plus excess deaths in ward when ICU is denied
    
    Examples
    --------
    >>> from config import AGE_PARAMS_DEFAULT, CONTACT_MATRIX_DEFAULT
    >>> results = simulate_age_structured_hospital_model_with_icu_ward_split(
    ...     beta=0.3,
    ...     age_params=AGE_PARAMS_DEFAULT,
    ...     contact_matrix=CONTACT_MATRIX_DEFAULT,
    ...     ward_capacity=80, icu_capacity=20,
    ...     hill_coef_ward=4, hill_coef_icu=4,
    ...     coverage=[0.2, 0.3, 0.7], VE=0.7,
    ...     age_pops=[3000, 5000, 2000]
    ... )
    >>> print(f"Peak ICU: {max(results['H_icu_total']):.0f}")
    >>> print(f"Treated deaths: {results['D_treated_total'][-1]:.0f}")
    >>> print(f"Untreated deaths: {results['D_untreated_total'][-1]:.0f}")
    """
    if age_pops is None:
        raise ValueError("age_pops must be provided.")
    theta_X = config.DEFAULT_SIM_PARAMS['theta_X'] if theta_X is None else theta_X
    theta_H = config.DEFAULT_SIM_PARAMS['theta_H'] if theta_H is None else theta_H
    hill_coef_ward = config.DEFAULT_CAPACITY_PARAMS['hill_coef_ward'] if hill_coef_ward is None else hill_coef_ward
    hill_coef_icu = config.DEFAULT_CAPACITY_PARAMS['hill_coef_icu'] if hill_coef_icu is None else hill_coef_icu
    Tmax = config.DEFAULT_SIM_PARAMS['Tmax'] if Tmax is None else Tmax
    time_step = config.DEFAULT_SIM_PARAMS['time_step'] if time_step is None else time_step
    VE = config.DEFAULT_SIM_PARAMS['VE'] if VE is None else VE
    if ward_capacity is None:
        ward_capacity = config.DEFAULT_CAPACITY_PARAMS['ward_capacity']
    if icu_capacity is None:
        icu_capacity = config.DEFAULT_CAPACITY_PARAMS['icu_capacity']
    n_ages = len(age_pops)
    K_ward = ward_capacity
    K_icu = icu_capacity
    n_ward = hill_coef_ward
    n_icu = hill_coef_icu
    dt = time_step
    
    _validate_age_structured_inputs(age_params, contact_matrix, age_pops, coverage)
    
    # Handle coverage
    if not isinstance(coverage, list):
        coverage = [coverage] * n_ages
    
    # Initialize compartments for each age group
    ic_defaults = config.DEFAULT_INITIAL_CONDITIONS if initial_conditions is None else {**config.DEFAULT_INITIAL_CONDITIONS, **initial_conditions}
    I = _coerce_initial_vector(ic_defaults, 'I_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['I_by_age'])
    X = _coerce_initial_vector(ic_defaults, 'X_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['X_by_age'])
    H_ward = _coerce_initial_vector(ic_defaults, 'H_ward_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_ward_by_age'])
    H_icu = _coerce_initial_vector(ic_defaults, 'H_icu_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_icu_by_age'])
    R = _coerce_initial_vector(ic_defaults, 'R_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['R_by_age'])
    D = _coerce_initial_vector(ic_defaults, 'D_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['D_by_age'])
    S = [max(0, age_pops[a] - I[a] - X[a] - H_ward[a] - H_icu[a] - R[a] - D[a]) for a in range(n_ages)]
    
    # Differential mortality tracking
    D_treated = [0.0] * n_ages      # deaths with care (I baseline, ward, ICU)
    D_untreated = [0.0] * n_ages    # deaths without care (X overflow, ward denied ICU)
    
    # Storage
    times = []
    S_history = [[] for _ in range(n_ages)]
    I_history = [[] for _ in range(n_ages)]
    X_history = [[] for _ in range(n_ages)]
    H_ward_history = [[] for _ in range(n_ages)]
    H_icu_history = [[] for _ in range(n_ages)]
    R_history = [[] for _ in range(n_ages)]
    D_history = [[] for _ in range(n_ages)]
    D_treated_history = [[] for _ in range(n_ages)]
    D_untreated_history = [[] for _ in range(n_ages)]
    
    H_ward_total_history = []
    H_icu_total_history = []
    H_total_history = []
    ward_overflow_history = []
    icu_overflow_history = []
    D_treated_total_history = []
    D_untreated_total_history = []
    
    cum_ward_overflow = 0
    cum_icu_overflow = 0
    cum_unmet_ward = [0.0] * n_ages
    cum_unmet_icu = [0.0] * n_ages
    
    # Age-specific effective beta
    eff_beta = [beta * (1 - coverage[a] * VE) for a in range(n_ages)]
    
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
        
        H_ward_total = sum(H_ward)
        H_icu_total = sum(H_icu)
        H_total = H_ward_total + H_icu_total
        
        H_ward_total_history.append(H_ward_total)
        H_icu_total_history.append(H_icu_total)
        H_total_history.append(H_total)
        D_treated_total_history.append(sum(D_treated))
        D_untreated_total_history.append(sum(D_untreated))
        
        # Capacity gating (shared across age groups)
        g_ward = hill_gate(H_ward_total, K_ward, n_ward)
        g_icu = hill_gate(H_icu_total, K_icu, n_icu)
        
        # Force of infection for each age group
        lambda_foi = [0.0] * n_ages
        for a in range(n_ages):
            for b in range(n_ages):
                H_contrib = H_ward[b] + H_icu[b]
                live_pop_b = S[b] + I[b] + X[b] + H_contrib + R[b]
                if live_pop_b > 0:
                    lambda_foi[a] += (eff_beta[a] * contact_matrix[a][b] *
                                     (I[b] + theta_X * X[b] + theta_H * H_contrib) / live_pop_b)
        
        # Transitions for each age group
        new_inf = [lambda_foi[a] * S[a] for a in range(n_ages)]
        
        # Ward admissions from X
        admit_ward = [age_params[a]['eta'] * X[a] * g_ward for a in range(n_ages)]
        
        # ICU admissions from ward (patients needing escalation)
        need_icu = [age_params[a].get('eta_icu', 0.1) * H_ward[a] for a in range(n_ages)]
        admit_icu = [need_icu[a] * g_icu for a in range(n_ages)]
        
        # Calculate unmet care (for differential mortality)
        unmet_ward = [max(0, age_params[a]['eta'] * X[a] - admit_ward[a]) for a in range(n_ages)]
        unmet_icu = [max(0, need_icu[a] - admit_icu[a]) for a in range(n_ages)]
        
        # ODEs with differential mortality
        dS = [-new_inf[a] for a in range(n_ages)]
        dI = [new_inf[a] - (age_params[a]['gamma_I'] + age_params[a]['mu_I'] +
              age_params[a]['sigma']) * I[a] for a in range(n_ages)]
        
        # Use ward/icu specific parameters if available, fall back to gamma_H/mu_H
        dH_ward = []
        dH_icu = []
        dR = []
        dD = []
        dD_treated = []
        dD_untreated = []
        dX = []
        
        for a in range(n_ages):
            gamma_w = age_params[a].get('gamma_ward', age_params[a]['gamma_H'])
            mu_w = age_params[a].get('mu_ward', age_params[a]['mu_H'] * 0.5)
            gamma_i = age_params[a].get('gamma_icu', age_params[a]['gamma_H'] * 0.6)
            mu_i = age_params[a].get('mu_icu', age_params[a]['mu_H'] * 2.0)
            
            # Differential mortality rates
            mu_X_treated = age_params[a]['mu_X']
            dm_params = config.DIFFERENTIAL_MORTALITY_PARAMS
            age_keys = ['young', 'middle', 'elderly']
            age_key = age_keys[a] if a < len(age_keys) else None
            mu_X_multiplier_age = dm_params.get(
                f'mu_X_untreated_multiplier_{age_key}', dm_params['mu_X_untreated_multiplier']
            ) if age_key else dm_params['mu_X_untreated_multiplier']
            mu_ward_multiplier_age = dm_params.get(
                f'mu_ward_denied_icu_multiplier_{age_key}', dm_params['mu_ward_denied_icu_multiplier']
            ) if age_key else dm_params['mu_ward_denied_icu_multiplier']
            mu_X_untreated = age_params[a].get('mu_X_untreated', mu_X_treated * mu_X_multiplier_age)
            mu_ward_denied = age_params[a].get('mu_ward_denied_icu', mu_w * mu_ward_multiplier_age)
            
            # Calculate X compartment mortality with differential rates
            # Patients who get admitted use mu_X, those denied use mu_X_untreated
            # The fraction denied is proportional to (1 - g_ward)
            fraction_X_denied = 1.0 - g_ward if X[a] > 0 else 0.0
            effective_mu_X = mu_X_treated * g_ward + mu_X_untreated * fraction_X_denied
            
            # Ward mortality with ICU denial effect
            # Patients who need ICU but can't get it have elevated mortality
            fraction_icu_denied = 1.0 - g_icu if H_ward[a] > 0 and need_icu[a] > 0 else 0.0
            eta_icu_a = age_params[a].get('eta_icu', 0.1)
            # Only the fraction needing ICU is affected by denial
            effective_mu_ward = mu_w + (mu_ward_denied - mu_w) * eta_icu_a * fraction_icu_denied
            
            # Calculate death flows for differential tracking
            deaths_I = age_params[a]['mu_I'] * I[a]
            deaths_X_treated = mu_X_treated * g_ward * X[a]
            deaths_X_untreated = mu_X_untreated * fraction_X_denied * X[a]
            deaths_ward_baseline = mu_w * H_ward[a]
            deaths_ward_icu_denied = (mu_ward_denied - mu_w) * eta_icu_a * fraction_icu_denied * H_ward[a]
            deaths_icu = mu_i * H_icu[a]
            
            # X compartment dynamics (using effective mortality)
            dX.append(age_params[a]['sigma'] * I[a] - (age_params[a]['gamma_X'] +
                      effective_mu_X) * X[a] - admit_ward[a])
            
            dH_ward.append(admit_ward[a] - (gamma_w + effective_mu_ward) * H_ward[a] - admit_icu[a])
            dH_icu.append(admit_icu[a] - (gamma_i + mu_i) * H_icu[a])
            dR.append(age_params[a]['gamma_I'] * I[a] + age_params[a]['gamma_X'] * X[a] +
                     gamma_w * H_ward[a] + gamma_i * H_icu[a])
            
            # Total deaths
            total_deaths = deaths_I + deaths_X_treated + deaths_X_untreated + \
                          deaths_ward_baseline + deaths_ward_icu_denied + deaths_icu
            dD.append(total_deaths)
            
            # Differential mortality tracking
            # Treated: deaths in I (baseline disease), ward (with care), ICU (with care), X (admitted fraction)
            dD_treated.append(deaths_I + deaths_X_treated + deaths_ward_baseline + deaths_icu)
            # Untreated: deaths in X due to denial, excess ward deaths from ICU denial
            dD_untreated.append(deaths_X_untreated + deaths_ward_icu_denied)
        
        # Euler update
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
    
    results = {
        'times': times,
        'S': S_history,
        'I': I_history,
        'X': X_history,
        'H_ward': H_ward_history,
        'H_icu': H_icu_history,
        'H': [[(H_ward_history[a][t] + H_icu_history[a][t]) 
               for t in range(len(times))] for a in range(n_ages)],  # backward compat
        'R': R_history,
        'D': D_history,
        'H_ward_total': H_ward_total_history,
        'H_icu_total': H_icu_total_history,
        'H_total': H_total_history,
        'ward_overflow': ward_overflow_history,
        'icu_overflow': icu_overflow_history,
        'cum_ward_overflow': cum_ward_overflow,
        'cum_icu_overflow': cum_icu_overflow,
        'cum_overflow': cum_ward_overflow + cum_icu_overflow,  # backward compat
        'cum_unmet_ward': cum_unmet_ward,
        'cum_unmet_icu': cum_unmet_icu,
        'cum_unmet': [cum_unmet_ward[a] + cum_unmet_icu[a] for a in range(n_ages)],
        'ward_capacity': K_ward,
        'icu_capacity': K_icu,
        'age_pops': age_pops
    }
    
    # Add differential mortality data
    if track_differential_mortality:
        results['D_treated'] = D_treated_history
        results['D_untreated'] = D_untreated_history
        results['D_treated_total'] = D_treated_total_history
        results['D_untreated_total'] = D_untreated_total_history
    
    return results
    
    
def simulate_age_structured_hospital_model_with_time_variance(
    beta_base,
    age_params,
    contact_matrix,
    hosp_capacity=None,
    hill_coef=None,
    coverage=0.0,
    VE=None,
    age_pops=None,
    seasonal_params=None,
    waning_params=None,
    interventions=None,
    theta_X=None,
    theta_H=None,
    Tmax=None,
    time_step=None,
    initial_conditions=None
):
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
    >>> results = simulate_age_structured_hospital_model_with_time_variance(
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
    if age_pops is None:
        raise ValueError("age_pops must be provided.")
    theta_X = config.DEFAULT_SIM_PARAMS['theta_X'] if theta_X is None else theta_X
    theta_H = config.DEFAULT_SIM_PARAMS['theta_H'] if theta_H is None else theta_H
    hill_coef = config.DEFAULT_SIM_PARAMS['hill_coef'] if hill_coef is None else hill_coef
    Tmax = config.DEFAULT_SIM_PARAMS['Tmax'] if Tmax is None else Tmax
    time_step = config.DEFAULT_SIM_PARAMS['time_step'] if time_step is None else time_step
    VE = config.DEFAULT_SIM_PARAMS['VE'] if VE is None else VE
    if hosp_capacity is None:
        hosp_capacity = config.DEFAULT_CAPACITY_PARAMS['total_capacity']
    n_ages = len(age_pops)
    K = hosp_capacity
    n = hill_coef
    dt = time_step
    
    _validate_age_structured_inputs(age_params, contact_matrix, age_pops, coverage)
    
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
    ic_defaults = config.DEFAULT_INITIAL_CONDITIONS if initial_conditions is None else {**config.DEFAULT_INITIAL_CONDITIONS, **initial_conditions}
    I = _coerce_initial_vector(ic_defaults, 'I_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['I_by_age'])
    X = _coerce_initial_vector(ic_defaults, 'X_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['X_by_age'])
    H = _coerce_initial_vector(ic_defaults, 'H_ward_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['H_ward_by_age'])
    R = _coerce_initial_vector(ic_defaults, 'R_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['R_by_age'])
    D = _coerce_initial_vector(ic_defaults, 'D_by_age', n_ages, config.DEFAULT_INITIAL_CONDITIONS['D_by_age'])
    S = [max(0, age_pops[a] - I[a] - X[a] - H[a] - R[a] - D[a]) for a in range(n_ages)]
    
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
        g = hill_gate(H_total, K, n)
        
        # force of infection for each age group
        lambda_foi = [0] * n_ages
        for a in range(n_ages):
            for b in range(n_ages):
                live_pop_b = S[b] + I[b] + X[b] + H[b] + R[b]
                if live_pop_b > 0:
                    lambda_foi[a] += (eff_beta[a] * contact_matrix[a][b] *
                                     (I[b] + theta_X*X[b] + theta_H*H[b]) / live_pop_b)
        
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
