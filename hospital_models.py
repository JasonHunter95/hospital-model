import numpy as np
import matplotlib.pyplot as plt

def simulate_hospital_model(beta, sigma, eta, gamma_I, mu_I, gamma_X, mu_X, 
                           gamma_H, mu_H, theta_X, theta_H, hosp_capacity, 
                           hill_coef, coverage, VE, N, S=None, I=None, X=None, 
                           H=None, R=None, D=None, Tmax=200, time_step=0.1):
    """
    Simulate SIXHRD hospital model with capacity constraints.
    
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
    
    Examples
    --------
    >>> times, S, I, X, H, R, D, overflow, cum_overflow, cum_unmet, unmet = \\
    ...     simulate_hospital_model(
    ...         beta=0.3, sigma=0.2, eta=0.3, gamma_I=0.1, mu_I=0.01,
    ...         gamma_X=0.15, mu_X=0.05, gamma_H=0.2, mu_H=0.02,
    ...         theta_X=0.5, theta_H=0.3, hosp_capacity=100, hill_coef=4,
    ...         coverage=0.1, VE=0.7, N=10000
    ...     )
    >>> print(f"Final deaths: {D[-1]:.0f}")
    >>> print(f"Peak hospital occupancy: {max(H):.0f}")
    """
    K = hosp_capacity
    n = hill_coef
    dt = time_step
    
    # Initial conditions - use parameters if provided, else defaults
    if S is None:
        S = N - 10
    if I is None:
        I = 10
    if X is None:
        X = 0
    if H is None:
        H = 0
    if R is None:
        R = 0
    if D is None:
        D = 0
    
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
        if K > 0:
            g = 1 / (1 + (H/K)**n)
        else:
            g = 0
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

def simulate_age_structured_model(beta, age_params, contact_matrix, hosp_capacity, 
                                  hill_coef, coverage, VE, age_pops, theta_X=0.5, 
                                  theta_H=0.3, Tmax=200, time_step=0.1):
    """
    Simulate age-structured SIXHRD model with shared hospital capacity.
    
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
    
    Examples
    --------
    >>> from config import AGE_PARAMS_DEFAULT, CONTACT_MATRIX_DEFAULT
    >>> results = simulate_age_structured_model(
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
    n_ages = len(age_pops)
    K = hosp_capacity
    n = hill_coef
    dt = time_step
    N_total = sum(age_pops)
    
    # Handle coverage - convert to list if scalar
    if not isinstance(coverage, list):
        coverage = [coverage] * n_ages
    
    # Initialize compartments for each age group
    S = [age_pops[a] - 10 if a == 0 else age_pops[a] for a in range(n_ages)]
    I = [10 if a == 0 else 0 for a in range(n_ages)]
    X = [0.0] * n_ages
    H = [0.0] * n_ages
    R = [0.0] * n_ages
    D = [0.0] * n_ages
    
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
        if K > 0:
            g = 1 / (1 + (H_total/K)**n)
        else:
            g = 0
        
        # Calculate force of infection for each age group
        lambda_foi = [0] * n_ages
        for a in range(n_ages):
            for b in range(n_ages):
                if age_pops[b] > 0:
                    lambda_foi[a] += (eff_beta[a] * contact_matrix[a][b] * 
                                     (I[b] + theta_X*X[b] + theta_H*H[b]) / age_pops[b])
        
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
    
