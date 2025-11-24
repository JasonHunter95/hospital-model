"""
Helper functions for running repeated simulations and parameter sweeps.

This module provides reusable functions for comparing vaccination strategies,
optimizing vaccine allocation, and conducting parameter sweeps to improve
code modularity and reduce duplication.
"""

import numpy as np
from hospital_models import simulate_age_structured_model


def compare_vaccination_strategies(beta, age_params, contact_matrix, hosp_capacity, 
                                   age_pops, strategies, hill_coef=4, VE=0.7, 
                                   theta_X=0.5, theta_H=0.3, Tmax=200, time_step=0.1):
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
        results = simulate_age_structured_model(
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
        
        # print summary for each strategy
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
            results = simulate_age_structured_model(
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
