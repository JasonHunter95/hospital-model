"""
Helper functions for running repeated simulations and parameter sweeps using legacy hospital models.

This module provides reusable functions for comparing vaccination strategies,
optimizing vaccine allocation, and conducting parameter sweeps to improve
code modularity and reduce duplication.

Supports both legacy single-H models and extended ward/ICU models.
"""

import numpy as np # pyright: ignore[reportMissingImports]
from legacy_hospital_models import simulate_age_structured_hospital_model, simulate_age_structured_hospital_model_with_icu_ward_split


def compare_vaccination_strategies(beta, age_params, contact_matrix, hosp_capacity, 
                                   age_pops, strategies, hill_coef=4, VE=0.7, 
                                   theta_X=0.5, theta_H=0.3, Tmax=200, time_step=0.1,
                                   verbose=False):
    """
    Compare multiple vaccination strategies and return outcome metrics.
    
    This function runs the age-structured model for each vaccination strategy
    and compiles key outcomes for comparison. Eliminates duplicated simulation
    loops across notebook cells.
    
    Parameters
    ----------
    beta : float
        Base transmission rate.
    age_params : list of dict
        Age-specific disease parameters for each age group.
    contact_matrix : ndarray
        Contact rates between age groups [infector, infectee].
    hosp_capacity : int
        Total hospital bed capacity.
    age_pops : list
        Population size for each age group.
    strategies : dict
        Dictionary mapping strategy names to coverage lists [young, middle, elderly].
        Example: {'Elderly priority': [0.1, 0.2, 0.7]}
    hill_coef : float, optional
        Hill coefficient for admission gating (default: 4).
    VE : float, optional
        Vaccine efficacy, 0-1 (default: 0.7).
    theta_X : float, optional
        Relative infectiousness of X compartment (default: 0.5).
    theta_H : float, optional
        Relative infectiousness of H compartment (default: 0.3).
    Tmax : float, optional
        Simulation duration in days (default: 200).
    time_step : float, optional
        Integration time step in days (default: 0.1).
    
    Returns
    -------
    dict
        Results for each strategy with keys:
        - 'total_deaths': final death count across all ages
        - 'deaths_by_age': list of deaths for each age group
        - 'peak_H': peak hospital occupancy
        - 'cum_overflow': cumulative overflow burden (patient-days)
        - 'coverage': coverage list for this strategy
    
    Examples
    --------
    >>> from config import VACCINATION_STRATEGIES, AGE_PARAMS_DEFAULT
    >>> results = compare_vaccination_strategies(
    ...     beta=0.3,
    ...     age_params=AGE_PARAMS_DEFAULT,
    ...     contact_matrix=contact_matrix,
    ...     hosp_capacity=100,
    ...     age_pops=[3000, 5000, 2000],
    ...     strategies=VACCINATION_STRATEGIES
    ... )
    >>> for name, res in results.items():
    ...     print(f"{name}: {res['total_deaths']:.0f} deaths")
    """
    strategy_results = {}
    n_ages = len(age_pops)
    
    for strategy_name, coverage_by_age in strategies.items():
        results = simulate_age_structured_hospital_model(
            beta=beta,
            age_params=age_params,
            contact_matrix=contact_matrix,
            hosp_capacity=hosp_capacity,
            hill_coef=hill_coef,
            coverage=coverage_by_age,
            VE=VE,
            age_pops=age_pops,
            theta_X=theta_X,
            theta_H=theta_H,
            Tmax=Tmax,
            time_step=time_step
        )
        
        total_deaths = sum([results['D'][a][-1] for a in range(n_ages)])
        peak_H = max(results['H_total'])
        
        strategy_results[strategy_name] = {
            'total_deaths': total_deaths,
            'deaths_by_age': [results['D'][a][-1] for a in range(n_ages)],
            'peak_H': peak_H,
            'cum_overflow': results['cum_overflow'],
            'coverage': coverage_by_age
        }
        
        if verbose:
            print(f"\n{strategy_name}:")
            print(f"  Coverage: Young={coverage_by_age[0]:.1f}, "
                  f"Middle={coverage_by_age[1]:.1f}, Elderly={coverage_by_age[2]:.1f}")
            print(f"  Total deaths: {total_deaths:.0f}")
            print(f"  Deaths by age: Young={results['D'][0][-1]:.0f}, "
                  f"Middle={results['D'][1][-1]:.0f}, Elderly={results['D'][2][-1]:.0f}")
            print(f"  Peak hospital: {peak_H:.1f}")
            print(f"  Cumulative overflow: {results['cum_overflow']:.1f}")
    
    return strategy_results


def optimize_vaccine_allocation(beta, age_params, contact_matrix, hosp_capacity,
                                age_pops, total_coverage_target, n_grid=20,
                                hill_coef=4, VE=0.7, theta_X=0.5, theta_H=0.3,
                                Tmax=200, time_step=0.1):
    """
    Grid search to find optimal vaccine allocation across age groups.
    
    Given a fixed total vaccine supply, this function searches over all feasible
    allocations across age groups to find the allocation that minimizes deaths.
    
    Parameters
    ----------
    beta : float
        Base transmission rate.
    age_params : list of dict
        Age-specific disease parameters for each age group.
    contact_matrix : ndarray
        Contact rates between age groups [infector, infectee].
    hosp_capacity : int
        Total hospital bed capacity.
    age_pops : list
        Population size for each age group [young, middle, elderly].
    total_coverage_target : float
        Target coverage as fraction of total population (e.g., 0.3 for 30%).
    n_grid : int, optional
        Grid resolution for young and middle coverage (default: 20).
    hill_coef : float, optional
        Hill coefficient for admission gating (default: 4).
    VE : float, optional
        Vaccine efficacy, 0-1 (default: 0.7).
    theta_X : float, optional
        Relative infectiousness of X compartment (default: 0.5).
    theta_H : float, optional
        Relative infectiousness of H compartment (default: 0.3).
    Tmax : float, optional
        Simulation duration in days (default: 200).
    time_step : float, optional
        Integration time step in days (default: 0.1).
    
    Returns
    -------
    tuple
        (deaths_grid, overflow_grid, young_cov_range, middle_cov_range)
        - deaths_grid: 2D array of total deaths [young_cov, middle_cov]
        - overflow_grid: 2D array of cumulative overflow [young_cov, middle_cov]
        - young_cov_range: array of young coverage values tested
        - middle_cov_range: array of middle coverage values tested
    
    Notes
    -----
    Elderly coverage is determined by the constraint:
    doses_elderly = total_doses - doses_young - doses_middle
    Infeasible allocations (negative or exceeding population) are marked as NaN.
    
    Examples
    --------
    >>> deaths, overflow, young_range, middle_range = optimize_vaccine_allocation(
    ...     beta=0.3,
    ...     age_params=age_params,
    ...     contact_matrix=contact_matrix,
    ...     hosp_capacity=100,
    ...     age_pops=[3000, 5000, 2000],
    ...     total_coverage_target=0.3,
    ...     n_grid=20
    ... )
    >>> optimal_idx = np.nanargmin(deaths)
    >>> print(f"Minimum deaths: {np.nanmin(deaths):.0f}")
    """
    total_pop = sum(age_pops)
    total_doses = total_coverage_target * total_pop
    n_ages = len(age_pops)
    
    young_cov_range = np.linspace(0, 1, n_grid)
    middle_cov_range = np.linspace(0, 1, n_grid)
    
    deaths_allocation_grid = np.zeros((n_grid, n_grid))
    overflow_allocation_grid = np.zeros((n_grid, n_grid))
    
    print(f"Searching for optimal allocation of {total_doses:.0f} vaccine doses...")
    print(f"(Equivalent to {total_coverage_target*100:.0f}% of total population)")
    
    for i, cov_young in enumerate(young_cov_range):
        for j, cov_middle in enumerate(middle_cov_range):
            doses_young = cov_young * age_pops[0]
            doses_middle = cov_middle * age_pops[1]
            doses_elderly = total_doses - doses_young - doses_middle
            
            # check feasibility constraint
            if doses_elderly < 0 or doses_elderly > age_pops[2]:
                deaths_allocation_grid[i, j] = np.nan
                overflow_allocation_grid[i, j] = np.nan
                continue
            
            cov_elderly = doses_elderly / age_pops[2]
            
            # run simulation
            results = simulate_age_structured_hospital_model(
                beta=beta,
                age_params=age_params,
                contact_matrix=contact_matrix,
                hosp_capacity=hosp_capacity,
                hill_coef=hill_coef,
                coverage=[cov_young, cov_middle, cov_elderly],
                VE=VE,
                age_pops=age_pops,
                theta_X=theta_X,
                theta_H=theta_H,
                Tmax=Tmax,
                time_step=time_step
            )
            
            total_deaths = sum([results['D'][a][-1] for a in range(n_ages)])
            deaths_allocation_grid[i, j] = total_deaths
            overflow_allocation_grid[i, j] = results['cum_overflow']
    
    print("Search complete.")
    
    return deaths_allocation_grid, overflow_allocation_grid, young_cov_range, middle_cov_range


def compare_vaccination_strategies_icu(beta, age_params, contact_matrix, ward_capacity,
                                        icu_capacity, age_pops, strategies,
                                        hill_coef_ward=4, hill_coef_icu=4, VE=0.7,
                                        theta_X=0.5, theta_H=0.3, Tmax=200, time_step=0.1,
                                        verbose=False):
    """
    Compare vaccination strategies using the ward/ICU model.
    
    This function runs the age-structured ward/ICU model for each vaccination
    strategy and compiles key outcomes including separate ward and ICU metrics.
    
    Parameters
    ----------
    beta : float
        Base transmission rate.
    age_params : list of dict
        Age-specific parameters including 'eta_icu', 'gamma_ward', 'mu_ward',
        'gamma_icu', 'mu_icu'.
    contact_matrix : ndarray
        Contact rates between age groups.
    ward_capacity : int
        General ward bed capacity.
    icu_capacity : int
        ICU bed capacity.
    age_pops : list
        Population size for each age group.
    strategies : dict
        Dictionary mapping strategy names to coverage lists.
    hill_coef_ward : float, optional
        Hill coefficient for ward admission gating (default: 4).
    hill_coef_icu : float, optional
        Hill coefficient for ICU admission gating (default: 4).
    VE : float, optional
        Vaccine efficacy (default: 0.7).
    theta_X : float, optional
        Relative infectiousness of X compartment (default: 0.5).
    theta_H : float, optional
        Relative infectiousness of hospitalized (default: 0.3).
    Tmax : float, optional
        Simulation duration in days (default: 200).
    time_step : float, optional
        Integration time step in days (default: 0.1).
    
    Returns
    -------
    dict
        Results for each strategy with keys:
        - 'total_deaths': final death count across all ages
        - 'deaths_by_age': list of deaths for each age group
        - 'peak_ward': peak ward occupancy
        - 'peak_icu': peak ICU occupancy
        - 'cum_ward_overflow': cumulative ward overflow
        - 'cum_icu_overflow': cumulative ICU overflow
        - 'coverage': coverage list for this strategy
    
    Examples
    --------
    >>> results = compare_vaccination_strategies_icu(
    ...     beta=0.3,
    ...     age_params=AGE_PARAMS_DEFAULT,
    ...     contact_matrix=CONTACT_MATRIX_DEFAULT,
    ...     ward_capacity=80, icu_capacity=20,
    ...     age_pops=[3000, 5000, 2000],
    ...     strategies=VACCINATION_STRATEGIES
    ... )
    """
    strategy_results = {}
    n_ages = len(age_pops)
    
    for strategy_name, coverage_by_age in strategies.items():
        results = simulate_age_structured_hospital_model_with_icu_ward_split(
            beta=beta,
            age_params=age_params,
            contact_matrix=contact_matrix,
            ward_capacity=ward_capacity,
            icu_capacity=icu_capacity,
            hill_coef_ward=hill_coef_ward,
            hill_coef_icu=hill_coef_icu,
            coverage=coverage_by_age,
            VE=VE,
            age_pops=age_pops,
            theta_X=theta_X,
            theta_H=theta_H,
            Tmax=Tmax,
            time_step=time_step
        )
        
        total_deaths = sum([results['D'][a][-1] for a in range(n_ages)])
        peak_ward = max(results['H_ward_total'])
        peak_icu = max(results['H_icu_total'])
        
        strategy_results[strategy_name] = {
            'total_deaths': total_deaths,
            'deaths_by_age': [results['D'][a][-1] for a in range(n_ages)],
            'peak_ward': peak_ward,
            'peak_icu': peak_icu,
            'peak_H': peak_ward + peak_icu,  # backward compatibility
            'cum_ward_overflow': results['cum_ward_overflow'],
            'cum_icu_overflow': results['cum_icu_overflow'],
            'cum_overflow': results['cum_overflow'],  # backward compatibility
            'coverage': coverage_by_age
        }
        
        if verbose:
            print(f"\n{strategy_name}:")
            print(f"  Coverage: Young={coverage_by_age[0]:.1f}, "
                  f"Middle={coverage_by_age[1]:.1f}, Elderly={coverage_by_age[2]:.1f}")
            print(f"  Total deaths: {total_deaths:.0f}")
            print(f"  Deaths by age: Young={results['D'][0][-1]:.0f}, "
                  f"Middle={results['D'][1][-1]:.0f}, Elderly={results['D'][2][-1]:.0f}")
            print(f"  Peak ward: {peak_ward:.1f} / {ward_capacity}")
            print(f"  Peak ICU: {peak_icu:.1f} / {icu_capacity}")
            print(f"  Cumulative overflow: Ward={results['cum_ward_overflow']:.1f}, "
                  f"ICU={results['cum_icu_overflow']:.1f}")
    
    return strategy_results


def compare_capacity_scenarios(beta, age_params, contact_matrix, age_pops,
                               capacity_scenarios, coverage, VE=0.7,
                               theta_X=0.5, theta_H=0.3, Tmax=200, time_step=0.1,
                               verbose=False):
    """
    Compare different ward/ICU capacity allocation scenarios.
    
    Given a fixed total bed capacity, explore how different ward vs ICU
    splits affect outcomes. Useful for hospital planning.
    
    Parameters
    ----------
    beta : float
        Base transmission rate.
    age_params : list of dict
        Age-specific disease parameters.
    contact_matrix : ndarray
        Contact rates between age groups.
    age_pops : list
        Population size for each age group.
    capacity_scenarios : dict
        Dictionary mapping scenario names to (ward_capacity, icu_capacity) tuples.
        Example: {'Standard': (80, 20), 'More ICU': (60, 40)}
    coverage : list
        Vaccination coverage [young, middle, elderly].
    VE : float, optional
        Vaccine efficacy (default: 0.7).
    theta_X : float, optional
        Relative infectiousness of X compartment (default: 0.5).
    theta_H : float, optional
        Relative infectiousness of hospitalized (default: 0.3).
    Tmax : float, optional
        Simulation duration in days (default: 200).
    time_step : float, optional
        Integration time step in days (default: 0.1).
    
    Returns
    -------
    dict
        Results for each scenario with detailed metrics.
    
    Examples
    --------
    >>> scenarios = {
    ...     'Standard (80/20)': (80, 20),
    ...     'More ICU (60/40)': (60, 40),
    ...     'Fewer ICU (90/10)': (90, 10)
    ... }
    >>> results = compare_capacity_scenarios(
    ...     beta=0.4, age_params=AGE_PARAMS_DEFAULT,
    ...     contact_matrix=CONTACT_MATRIX_DEFAULT,
    ...     age_pops=[3000, 5000, 2000],
    ...     capacity_scenarios=scenarios,
    ...     coverage=[0.1, 0.2, 0.7]
    ... )
    """
    scenario_results = {}
    n_ages = len(age_pops)
    
    for scenario_name, (ward_cap, icu_cap) in capacity_scenarios.items():
        results = simulate_age_structured_hospital_model_with_icu_ward_split(
            beta=beta,
            age_params=age_params,
            contact_matrix=contact_matrix,
            ward_capacity=ward_cap,
            icu_capacity=icu_cap,
            hill_coef_ward=4,
            hill_coef_icu=4,
            coverage=coverage,
            VE=VE,
            age_pops=age_pops,
            theta_X=theta_X,
            theta_H=theta_H,
            Tmax=Tmax,
            time_step=time_step
        )
        
        total_deaths = sum([results['D'][a][-1] for a in range(n_ages)])
        peak_ward = max(results['H_ward_total'])
        peak_icu = max(results['H_icu_total'])
        
        scenario_results[scenario_name] = {
            'total_deaths': total_deaths,
            'deaths_by_age': [results['D'][a][-1] for a in range(n_ages)],
            'peak_ward': peak_ward,
            'peak_icu': peak_icu,
            'ward_capacity': ward_cap,
            'icu_capacity': icu_cap,
            'cum_ward_overflow': results['cum_ward_overflow'],
            'cum_icu_overflow': results['cum_icu_overflow'],
            'ward_utilization': peak_ward / ward_cap if ward_cap > 0 else 0,
            'icu_utilization': peak_icu / icu_cap if icu_cap > 0 else 0,
            'full_results': results  # store full results for detailed plotting
        }
        
        if verbose:
            print(f"\n{scenario_name}:")
            print(f"  Ward: {ward_cap} beds, ICU: {icu_cap} beds (Total: {ward_cap + icu_cap})")
            print(f"  Total deaths: {total_deaths:.0f}")
            print(f"  Peak ward: {peak_ward:.1f} ({peak_ward/ward_cap*100:.0f}% utilization)")
            print(f"  Peak ICU: {peak_icu:.1f} ({peak_icu/icu_cap*100:.0f}% utilization)")
            print(f"  Overflow: Ward={results['cum_ward_overflow']:.1f}, "
                  f"ICU={results['cum_icu_overflow']:.1f} pt-days")
    
    return scenario_results


def sweep_icu_capacity(beta, age_params, contact_matrix, age_pops, total_beds,
                       coverage, n_points=20, VE=0.7, theta_X=0.5, theta_H=0.3,
                       Tmax=200, time_step=0.1, verbose=False):
    """
    Sweep over ward/ICU allocation for fixed total beds.
    
    Explores the full range of ward vs ICU splits to find optimal allocation.
    
    Parameters
    ----------
    beta : float
        Base transmission rate.
    age_params : list of dict
        Age-specific disease parameters.
    contact_matrix : ndarray
        Contact rates between age groups.
    age_pops : list
        Population size for each age group.
    total_beds : int
        Total hospital beds to allocate between ward and ICU.
    coverage : list
        Vaccination coverage [young, middle, elderly].
    n_points : int, optional
        Number of ICU fraction values to test (default: 20).
    VE : float, optional
        Vaccine efficacy (default: 0.7).
    theta_X : float, optional
        Relative infectiousness of X compartment (default: 0.5).
    theta_H : float, optional
        Relative infectiousness of hospitalized (default: 0.3).
    Tmax : float, optional
        Simulation duration in days (default: 200).
    time_step : float, optional
        Integration time step in days (default: 0.1).
    
    Returns
    -------
    dict
        Sweep results containing:
        - 'icu_fractions': array of ICU fractions tested
        - 'deaths': array of total deaths for each fraction
        - 'ward_overflow': array of cumulative ward overflow
        - 'icu_overflow': array of cumulative ICU overflow
        - 'optimal_icu_fraction': ICU fraction minimizing deaths
        - 'optimal_deaths': minimum deaths achieved
    
    Examples
    --------
    >>> results = sweep_icu_capacity(
    ...     beta=0.4, age_params=AGE_PARAMS_DEFAULT,
    ...     contact_matrix=CONTACT_MATRIX_DEFAULT,
    ...     age_pops=[3000, 5000, 2000],
    ...     total_beds=100, coverage=[0.1, 0.2, 0.7]
    ... )
    >>> print(f"Optimal ICU fraction: {results['optimal_icu_fraction']:.1%}")
    """
    n_ages = len(age_pops)
    
    # ICU fraction from 5% to 50% of total beds
    icu_fractions = np.linspace(0.05, 0.5, n_points)
    
    deaths_array = []
    ward_overflow_array = []
    icu_overflow_array = []
    
    if verbose:
        print(f"Sweeping ICU allocation for {total_beds} total beds...")
    
    for icu_frac in icu_fractions:
        icu_cap = int(total_beds * icu_frac)
        ward_cap = total_beds - icu_cap
        
        results = simulate_age_structured_hospital_model_with_icu_ward_split(
            beta=beta,
            age_params=age_params,
            contact_matrix=contact_matrix,
            ward_capacity=ward_cap,
            icu_capacity=icu_cap,
            hill_coef_ward=4,
            hill_coef_icu=4,
            coverage=coverage,
            VE=VE,
            age_pops=age_pops,
            theta_X=theta_X,
            theta_H=theta_H,
            Tmax=Tmax,
            time_step=time_step
        )
        
        total_deaths = sum([results['D'][a][-1] for a in range(n_ages)])
        deaths_array.append(total_deaths)
        ward_overflow_array.append(results['cum_ward_overflow'])
        icu_overflow_array.append(results['cum_icu_overflow'])
    
    deaths_array = np.array(deaths_array)
    ward_overflow_array = np.array(ward_overflow_array)
    icu_overflow_array = np.array(icu_overflow_array)
    
    # Find optimal
    opt_idx = np.argmin(deaths_array)
    
    if verbose:
        print(f"Sweep complete.")
        print(f"Optimal ICU fraction: {icu_fractions[opt_idx]:.1%} "
              f"({int(total_beds * icu_fractions[opt_idx])} ICU beds)")
        print(f"Minimum deaths: {deaths_array[opt_idx]:.0f}")
    
    return {
        'icu_fractions': icu_fractions,
        'deaths': deaths_array,
        'ward_overflow': ward_overflow_array,
        'icu_overflow': icu_overflow_array,
        'optimal_icu_fraction': icu_fractions[opt_idx],
        'optimal_deaths': deaths_array[opt_idx],
        'total_beds': total_beds
    }
