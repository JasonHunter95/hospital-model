"""
Plotting utilities for hospital SIXHRD model visualizations.

This module provides visualization functions for simulation results,
vaccination strategy comparisons, and vaccine allocation optimization.
"""

import numpy as np
import matplotlib.pyplot as plt


# ========================================
# Shared Constants and Defaults
# ========================================

DEFAULT_AGE_LABELS = ['Young', 'Middle', 'Elderly']
DEFAULT_AGE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
DEFAULT_STRATEGY_COLORS = ['gray', 'skyblue', 'green', 'orange', 'purple', 'red', 'brown']


# ========================================
# Internal Helper Functions (DRY)
# ========================================

def _setup_axis(ax, xlabel, ylabel, title, grid=True, legend=True, loc='best'):
    """Configure common axis properties."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc=loc)


def _plot_capacity_line(ax, capacity, color='red', label_prefix='Capacity'):
    """Add a horizontal capacity reference line."""
    ax.axhline(y=capacity, color=color, linestyle='--',
               label=f'{label_prefix} ({capacity})', alpha=0.7)


def _setup_summary_panel(ax, lines, fontsize=9):
    """Set up a monospace text summary panel from a list of lines."""
    ax.axis('off')
    summary_text = '\n'.join(lines)
    ax.text(0.05, 0.5, summary_text, fontsize=fontsize, family='monospace',
            verticalalignment='center', transform=ax.transAxes)


def _print_results_header(title):
    """Print a formatted results header to console."""
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")


# ========================================
# Plotting Functions
# ========================================

def plot_hospital_simulation_stats(times, S_vals, I_vals, X_vals, H_vals, R_vals, 
                                   D_vals, overflow_vals, cum_overflow, cum_unmet, 
                                   hosp_capacity, N):
    """
    Plot comprehensive results from basic SIXHRD model simulation.
    
    Creates a 2x2 grid showing epidemic dynamics, hospital capacity utilization,
    overflow time series, and summary statistics.
    
    Parameters
    ----------
    times : array
        Time points.
    S_vals, I_vals, X_vals, H_vals, R_vals, D_vals : array
        Compartment values over time.
    overflow_vals : array
        Hospital overflow time series.
    cum_overflow : float
        Cumulative overflow burden (patient-days).
    cum_unmet : float
        Cumulative unmet care needs (patient-days).
    hosp_capacity : int
        Hospital bed capacity.
    N : int
        Total population size.
    
    Returns
    -------
    None
        Displays matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compartments over time
    axes[0, 0].plot(times, S_vals, label='S (Susceptible)')
    axes[0, 0].plot(times, I_vals, label='I (Infected)')
    axes[0, 0].plot(times, X_vals, label='X (Needs care)')
    axes[0, 0].plot(times, H_vals, label='H (Hospitalized)')
    axes[0, 0].plot(times, R_vals, label='R (Recovered)')
    axes[0, 0].plot(times, D_vals, label='D (Deaths)')
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Population')
    axes[0, 0].legend()
    axes[0, 0].set_title('Epidemic Dynamics')
    axes[0, 0].grid(True, alpha=0.3)

    # Hospital capacity
    axes[0, 1].plot(times, H_vals, label='Hospitalized', color='red')
    axes[0, 1].axhline(y=hosp_capacity, color='black', linestyle='--', 
                       label=f'Capacity (K={hosp_capacity})')
    axes[0, 1].fill_between(times, 0, hosp_capacity, alpha=0.2, 
                            color='green', label='Available capacity')
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Hospital beds')
    axes[0, 1].legend()
    axes[0, 1].set_title('Hospital Capacity Utilization')
    axes[0, 1].grid(True, alpha=0.3)

    # Overflow
    axes[1, 0].plot(times, overflow_vals, color='red')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Overflow patients')
    axes[1, 0].set_title(f'Hospital Overflow (Cumulative: {cum_overflow:.1f} patient-days)')
    axes[1, 0].grid(True, alpha=0.3)

    # Summary statistics
    axes[1, 1].axis('off')
    attack_rate = (R_vals[-1] + D_vals[-1]) / N * 100
    summary_text = f"""
  Summary Statistics:
  ━━━━━━━━━━━━━━━━━━━━━━
  Peak Hospitalized: {max(H_vals):.1f}
  Total Deaths: {D_vals[-1]:.1f}
  Total Recovered: {R_vals[-1]:.1f}
  Attack Rate: {attack_rate:.1f}%

  Cumulative Overflow: {cum_overflow:.1f} patient-days
  Cumulative Unmet Care: {cum_unmet:.1f} patient-days

  Final Population:
    S: {S_vals[-1]:.1f}
    I: {I_vals[-1]:.1f}
    X: {X_vals[-1]:.1f}
    H: {H_vals[-1]:.1f}
    R: {R_vals[-1]:.1f}
    D: {D_vals[-1]:.1f}
  """

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace', 
                    verticalalignment='center')

    plt.tight_layout()
    plt.show()

    print(f"\nSimulation complete!")
    print(f"Total deaths: {D_vals[-1]:.0f}")
    print(f"Peak hospital occupancy: {max(H_vals):.1f} (Capacity: {hosp_capacity})")
    print(f"Cumulative overflow: {cum_overflow:.1f} patient-days")
    print(f"Cumulative unmet care: {cum_unmet:.1f} patient-days")


def plot_age_structured_results(results, hosp_capacity, age_labels=['Young', 'Middle', 'Elderly']):
    """
    Plot comprehensive results from age-structured SIXHRD model simulation.
    
    Creates a 2x3 grid showing compartment dynamics by age group, hospital
    utilization, cumulative deaths, and summary statistics.
    
    Parameters
    ----------
    results : dict
        Results dictionary from simulate_age_structured_model().
    hosp_capacity : int
        Hospital bed capacity for reference line.
    age_labels : list, optional
        Labels for age groups (default: ['Young', 'Middle', 'Elderly']).
    
    Returns
    -------
    None
        Displays matplotlib figure and prints summary statistics.
    """
    times = results['times']
    n_ages = len(results['S'])
    N_total = sum(results['age_pops'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot S, I, X, H by age group
    for a in range(n_ages):
        axes[0, 0].plot(times, results['S'][a], label=f'S_{age_labels[a]}', 
                       color=colors[a], linestyle='-')
        axes[0, 1].plot(times, results['I'][a], label=f'I_{age_labels[a]}', 
                       color=colors[a], linestyle='-')
        axes[0, 2].plot(times, results['X'][a], label=f'X_{age_labels[a]}', 
                       color=colors[a], linestyle='-')
    
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Susceptible')
    axes[0, 0].legend()
    axes[0, 0].set_title('Susceptible by Age')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Infected')
    axes[0, 1].legend()
    axes[0, 1].set_title('Infected by Age')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_xlabel('Time (days)')
    axes[0, 2].set_ylabel('Severe (need care)')
    axes[0, 2].legend()
    axes[0, 2].set_title('Severe Cases by Age')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Hospital utilization by age + total capacity
    for a in range(n_ages):
        axes[1, 0].plot(times, results['H'][a], label=f'H_{age_labels[a]}', 
                       color=colors[a], linestyle='-')
    axes[1, 0].plot(times, results['H_total'], label='H_total', 
                   color='black', linewidth=2, linestyle='--')
    axes[1, 0].axhline(y=hosp_capacity, color='red', linestyle='--', 
                      label=f'Capacity K={hosp_capacity}')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Hospitalized')
    axes[1, 0].legend()
    axes[1, 0].set_title('Hospital Utilization by Age')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative deaths by age
    for a in range(n_ages):
        axes[1, 1].plot(times, results['D'][a], label=f'D_{age_labels[a]}', 
                       color=colors[a], linewidth=2)
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Cumulative deaths')
    axes[1, 1].legend()
    axes[1, 1].set_title('Deaths by Age Group')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 2].axis('off')
    total_deaths = sum([results['D'][a][-1] for a in range(n_ages)])
    peak_H = max(results['H_total'])
    attack_rates = [(results['R'][a][-1] + results['D'][a][-1]) / results['age_pops'][a] * 100 
                    for a in range(n_ages)]
    
    summary_text = f"""
    Summary Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Peak H_total: {peak_H:.1f} / {hosp_capacity}
    Total Deaths: {total_deaths:.0f}
    Cum. Overflow: {results['cum_overflow']:.1f} patient-days

    Deaths by Age:
    """
    
    for a in range(n_ages):
        summary_text += f"\n  {age_labels[a]}: {results['D'][a][-1]:.0f} ({results['D'][a][-1]/total_deaths*100:.1f}%)"
    
    summary_text += "\n\nAttack Rates:"
    for a in range(n_ages):
        summary_text += f"\n  {age_labels[a]}: {attack_rates[a]:.1f}%"
    
    summary_text += "\n\nUnmet Care (patient-days):"
    for a in range(n_ages):
        summary_text += f"\n  {age_labels[a]}: {results['cum_unmet'][a]:.1f}"
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*50}")
    print(f"Age-Structured Model Results")
    print(f"{'='*50}")
    print(f"Total deaths: {total_deaths:.0f}")
    print(f"Peak hospital occupancy: {peak_H:.1f} (Capacity: {hosp_capacity})")
    print(f"Cumulative overflow: {results['cum_overflow']:.1f} patient-days")
    for a in range(n_ages):
        print(f"  {age_labels[a]}: {results['D'][a][-1]:.0f} deaths, AR={attack_rates[a]:.1f}%")


def plot_age_structured_icu_results(results, age_labels=None, figsize=(18, 14)):
    """
    Plot comprehensive results from age-structured SIXHRD model with ward/ICU.
    
    Creates a 3x3 grid showing:
    - Row 1: Infection dynamics (I, X, D by age)
    - Row 2: Hospital utilization (Ward, ICU, Combined)
    - Row 3: System performance (Overflow, Deaths breakdown, Summary)
    
    Parameters
    ----------
    results : dict
        Results dictionary from simulate_age_structured_model_icu().
    age_labels : list, optional
        Labels for age groups (default: ['Young', 'Middle', 'Elderly']).
    figsize : tuple, optional
        Figure size (default: (18, 14)).
    
    Returns
    -------
    None
        Displays matplotlib figure and prints summary statistics.
    """
    if age_labels is None:
        age_labels = DEFAULT_AGE_LABELS
    
    times = results['times']
    n_ages = len(results['S'])
    ward_capacity = results['ward_capacity']
    icu_capacity = results['icu_capacity']
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    
    # Row 1: Infection dynamics by age
    for a in range(n_ages):
        axes[0, 0].plot(times, results['I'][a], label=age_labels[a],
                       color=DEFAULT_AGE_COLORS[a], linewidth=2)
    _setup_axis(axes[0, 0], 'Time (days)', 'Infected', 'Infected by Age')
    
    for a in range(n_ages):
        axes[0, 1].plot(times, results['X'][a], label=age_labels[a],
                       color=DEFAULT_AGE_COLORS[a], linewidth=2)
    _setup_axis(axes[0, 1], 'Time (days)', 'Severe (need care)', 'Severe Cases by Age')
    
    for a in range(n_ages):
        axes[0, 2].plot(times, results['D'][a], label=age_labels[a],
                       color=DEFAULT_AGE_COLORS[a], linewidth=2)
    _setup_axis(axes[0, 2], 'Time (days)', 'Cumulative Deaths', 'Deaths by Age')
    
    # Row 2: Hospital utilization
    for a in range(n_ages):
        axes[1, 0].plot(times, results['H_ward'][a], label=age_labels[a],
                       color=DEFAULT_AGE_COLORS[a], linewidth=1.5)
    axes[1, 0].plot(times, results['H_ward_total'], label='Total',
                   color='black', linewidth=2, linestyle='--')
    _plot_capacity_line(axes[1, 0], ward_capacity, color='blue', label_prefix='Ward Capacity')
    _setup_axis(axes[1, 0], 'Time (days)', 'Ward Patients', 'General Ward by Age')
    
    for a in range(n_ages):
        axes[1, 1].plot(times, results['H_icu'][a], label=age_labels[a],
                       color=DEFAULT_AGE_COLORS[a], linewidth=1.5)
    axes[1, 1].plot(times, results['H_icu_total'], label='Total',
                   color='black', linewidth=2, linestyle='--')
    _plot_capacity_line(axes[1, 1], icu_capacity, color='red', label_prefix='ICU Capacity')
    _setup_axis(axes[1, 1], 'Time (days)', 'ICU Patients', 'ICU by Age')
    
    # Combined hospital burden (stacked area)
    axes[1, 2].stackplot(times, results['H_ward_total'], results['H_icu_total'],
                         labels=['Ward', 'ICU'], colors=['#3498db', '#e74c3c'], alpha=0.7)
    axes[1, 2].axhline(y=ward_capacity + icu_capacity, color='purple', linestyle='--',
                       label=f'Total Capacity ({ward_capacity + icu_capacity})', alpha=0.7)
    _setup_axis(axes[1, 2], 'Time (days)', 'Total Hospital Burden', 'Combined Ward + ICU')
    
    # Row 3: System performance
    axes[2, 0].plot(times, results['ward_overflow'], label='Ward Overflow',
                   color='blue', linewidth=2)
    axes[2, 0].plot(times, results['icu_overflow'], label='ICU Overflow',
                   color='red', linewidth=2)
    axes[2, 0].fill_between(times, results['ward_overflow'], alpha=0.3, color='blue')
    axes[2, 0].fill_between(times, results['icu_overflow'], alpha=0.3, color='red')
    _setup_axis(axes[2, 0], 'Time (days)', 'Overflow (patients)', 'Capacity Overflow')
    
    # Deaths breakdown bar chart
    x = np.arange(n_ages)
    width = 0.5
    deaths_final = [results['D'][a][-1] for a in range(n_ages)]
    axes[2, 1].bar(x, deaths_final, width, color=DEFAULT_AGE_COLORS[:n_ages])
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(age_labels)
    _setup_axis(axes[2, 1], 'Age Group', 'Final Deaths', 'Deaths by Age Group', legend=False)
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    
    # Summary statistics panel
    total_deaths = sum(deaths_final)
    peak_ward = max(results['H_ward_total'])
    peak_icu = max(results['H_icu_total'])
    
    summary_lines = [
        "SIMULATION SUMMARY",
        "━" * 36,
        "",
        f"Total Deaths: {total_deaths:.0f}",
        "",
        "Peak Occupancy:",
        f"  Ward: {peak_ward:.1f} / {ward_capacity} ({peak_ward/ward_capacity*100:.0f}%)",
        f"  ICU: {peak_icu:.1f} / {icu_capacity} ({peak_icu/icu_capacity*100:.0f}%)",
        "",
        "Cumulative Overflow (patient-days):",
        f"  Ward: {results['cum_ward_overflow']:.1f}",
        f"  ICU: {results['cum_icu_overflow']:.1f}",
        "",
        "Deaths by Age:",
    ]
    for a in range(n_ages):
        pct = deaths_final[a] / total_deaths * 100 if total_deaths > 0 else 0
        summary_lines.append(f"  {age_labels[a]}: {deaths_final[a]:.0f} ({pct:.1f}%)")
    
    summary_lines.extend([
        "",
        "Unmet Care (patient-days):",
    ])
    for a in range(n_ages):
        summary_lines.append(f"  {age_labels[a]}: Ward={results['cum_unmet_ward'][a]:.1f}, ICU={results['cum_unmet_icu'][a]:.1f}")
    
    _setup_summary_panel(axes[2, 2], summary_lines)
    
    plt.tight_layout()
    plt.show()
    
    # Console summary
    _print_results_header("Age-Structured ICU Model Results")
    print(f"Total deaths: {total_deaths:.0f}")
    print(f"Peak ward: {peak_ward:.1f} / {ward_capacity} ({peak_ward/ward_capacity*100:.0f}%)")
    print(f"Peak ICU: {peak_icu:.1f} / {icu_capacity} ({peak_icu/icu_capacity*100:.0f}%)")
    print(f"Cumulative overflow - Ward: {results['cum_ward_overflow']:.1f}, ICU: {results['cum_icu_overflow']:.1f}")
    print("\nDeaths by age:")
    for a in range(n_ages):
        print(f"  {age_labels[a]}: {deaths_final[a]:.0f}")


def plot_strategy_comparison(strategy_results, hosp_capacity=100, 
                            figsize=(16, 5), age_labels=None):
    """
    Visualize vaccination strategy comparison across multiple metrics.
    
    Creates bar charts comparing total deaths, peak hospital occupancy, and
    cumulative overflow across vaccination strategies. Also shows age-specific
    death breakdown for each strategy.
    
    Parameters
    ----------
    strategy_results : dict
        Results from compare_vaccination_strategies() function.
        Keys are strategy names, values are dicts with 'total_deaths',
        'deaths_by_age', 'peak_H', 'cum_overflow'.
    hosp_capacity : int, optional
        Hospital capacity for reference line (default: 100).
    figsize : tuple, optional
        Figure size (default: (16, 5)).
    age_labels : list, optional
        Labels for age groups (default: ['Young', 'Middle', 'Elderly']).
    
    Returns
    -------
    None
        Displays matplotlib figures.
    
    Examples
    --------
    >>> from simulation_helpers import compare_vaccination_strategies
    >>> results = compare_vaccination_strategies(...)
    >>> plot_strategy_comparison(results, hosp_capacity=100)
    """
    if age_labels is None:
        age_labels = ['Young', 'Middle', 'Elderly']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    strategies_list = list(strategy_results.keys())
    total_deaths_list = [strategy_results[s]['total_deaths'] for s in strategies_list]
    peak_H_list = [strategy_results[s]['peak_H'] for s in strategies_list]
    overflow_list = [strategy_results[s]['cum_overflow'] for s in strategies_list]
    
    colors = ['gray', 'skyblue', 'green', 'orange', 'purple']
    
    # total deaths comparison
    axes[0].barh(strategies_list, total_deaths_list, color=colors[:len(strategies_list)])
    axes[0].set_xlabel('Total Deaths')
    axes[0].set_title('Total Deaths by Vaccination Strategy')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # peak hospital occupancy
    axes[1].barh(strategies_list, peak_H_list, color=colors[:len(strategies_list)])
    axes[1].axvline(x=hosp_capacity, color='red', linestyle='--', 
                    label='Capacity', linewidth=2)
    axes[1].set_xlabel('Peak Hospital Occupancy')
    axes[1].set_title('Peak Hospital Load by Strategy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # cumulative overflow
    axes[2].barh(strategies_list, overflow_list, color=colors[:len(strategies_list)])
    axes[2].set_xlabel('Cumulative Overflow (patient-days)')
    axes[2].set_title('Hospital Overflow by Strategy')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # deaths by age group for each strategy
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(strategies_list))
    width = 0.25
    
    for i, age in enumerate(age_labels):
        deaths = [strategy_results[s]['deaths_by_age'][i] for s in strategies_list]
        ax.bar(x + i*width, deaths, width, label=age)
    
    ax.set_xlabel('Vaccination Strategy')
    ax.set_ylabel('Deaths')
    ax.set_title('Deaths by Age Group for Each Vaccination Strategy')
    ax.set_xticks(x + width)
    ax.set_xticklabels(strategies_list, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_optimal_allocation(deaths_grid, overflow_grid, young_range, middle_range,
                           age_pops, total_doses, figsize=(14, 6)):
    """
    Visualize optimal vaccine allocation heatmaps.
    
    Creates contour plots showing total deaths and hospital overflow as functions
    of vaccine allocation across age groups. Identifies and marks the optimal
    allocation that minimizes deaths.
    
    Parameters
    ----------
    deaths_grid : ndarray
        2D grid of total deaths [young_coverage, middle_coverage].
    overflow_grid : ndarray
        2D grid of cumulative overflow [young_coverage, middle_coverage].
    young_range : array
        Young coverage values tested (x-axis for middle).
    middle_range : array
        Middle coverage values tested (y-axis for young).
    age_pops : list
        Population sizes [young, middle, elderly].
    total_doses : float
        Total vaccine doses available.
    figsize : tuple, optional
        Figure size (default: (14, 6)).
    
    Returns
    -------
    tuple
        (opt_young, opt_middle, opt_elderly) - optimal coverage for each age group.
    
    Examples
    --------
    >>> from simulation_helpers import optimize_vaccine_allocation
    >>> deaths, overflow, young_r, middle_r = optimize_vaccine_allocation(...)
    >>> opt = plot_optimal_allocation(deaths, overflow, young_r, middle_r,
    ...                               age_pops=[3000, 5000, 2000],
    ...                               total_doses=3000)
    >>> print(f"Optimal allocation: Young={opt[0]:.1%}, Middle={opt[1]:.1%}, Elderly={opt[2]:.1%}")
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # deaths heatmap
    im1 = axes[0].contourf(middle_range, young_range, deaths_grid, 
                           levels=30, cmap='RdYlGn_r')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Total Deaths')
    
    # find optimum
    min_idx = np.nanargmin(deaths_grid)
    min_i, min_j = np.unravel_index(min_idx, deaths_grid.shape)
    opt_young = young_range[min_i]
    opt_middle = middle_range[min_j]
    opt_elderly = (total_doses - opt_young*age_pops[0] - opt_middle*age_pops[1]) / age_pops[2]
    
    axes[0].scatter(opt_middle, opt_young, color='red', s=200, marker='*', 
                   edgecolors='black', linewidths=2, zorder=10)
    axes[0].text(opt_middle, opt_young + 0.05, 'Optimal', 
                ha='center', fontsize=10, weight='bold', color='red')
    axes[0].set_xlabel('Middle Age Coverage')
    axes[0].set_ylabel('Young Coverage')
    axes[0].set_title(f'Total Deaths vs Vaccine Allocation\n(Total doses = {total_doses:.0f})')
    
    # overflow heatmap
    im2 = axes[1].contourf(middle_range, young_range, overflow_grid, 
                           levels=30, cmap='RdYlGn_r')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Cumulative Overflow (patient-days)')
    axes[1].scatter(opt_middle, opt_young, color='red', s=200, marker='*', 
                   edgecolors='black', linewidths=2, zorder=10)
    axes[1].set_xlabel('Middle Age Coverage')
    axes[1].set_ylabel('Young Coverage')
    axes[1].set_title('Hospital Overflow vs Vaccine Allocation')
    
    plt.tight_layout()
    plt.show()
    
    # print results
    print(f"\n{'='*60}")
    print(f"OPTIMAL ALLOCATION (Fixed total: {total_doses:.0f} doses)")
    print(f"{'='*60}")
    print(f"Young coverage: {opt_young*100:.1f}%")
    print(f"Middle coverage: {opt_middle*100:.1f}%")
    print(f"Elderly coverage: {opt_elderly*100:.1f}%")
    print(f"\nMinimum deaths: {np.nanmin(deaths_grid):.0f}")
    print(f"Overflow at optimum: {overflow_grid[min_i, min_j]:.1f} patient-days")
    
    return opt_young, opt_middle, opt_elderly


def plot_time_varying_results(results, hosp_capacity, age_labels=None,
                              show_beta=True, show_policy=True):
    """
    Plot results from time-varying SIXHRD model with transmission dynamics.
    
    Extends standard age-structured plots to show time-varying transmission
    rate and policy interventions.
    
    Parameters
    ----------
    results : dict
        Results dictionary from simulate_age_structured_time_varying().
    hosp_capacity : int
        Hospital bed capacity for reference line.
    age_labels : list, optional
        Labels for age groups (default: ['Young', 'Middle', 'Elderly']).
    show_beta : bool, optional
        Whether to plot time-varying beta (default: True).
    show_policy : bool, optional
        Whether to plot policy multiplier (default: True).
    
    Returns
    -------
    None
        Displays matplotlib figure.
    """
    if age_labels is None:
        age_labels = ['Young', 'Middle', 'Elderly']
    
    times = results['times']
    n_ages = len(results['S'])
    
    # determine subplot layout
    n_rows = 3 if (show_beta or show_policy) else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # row 1: compartments by age
    for a in range(n_ages):
        axes[0, 0].plot(times, results['I'][a], label=f'I_{age_labels[a]}',
                       color=colors[a], linestyle='-')
        axes[0, 1].plot(times, results['X'][a], label=f'X_{age_labels[a]}',
                       color=colors[a], linestyle='-')
        axes[0, 2].plot(times, results['S'][a], label=f'S_{age_labels[a]}',
                       color=colors[a], linestyle='-', alpha=0.7)
    
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Infected')
    axes[0, 0].legend()
    axes[0, 0].set_title('Infected by Age')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Severe (need care)')
    axes[0, 1].legend()
    axes[0, 1].set_title('Severe Cases by Age')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_xlabel('Time (days)')
    axes[0, 2].set_ylabel('Susceptible')
    axes[0, 2].legend()
    axes[0, 2].set_title('Susceptible by Age (with waning)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # row 2: hospital and deaths
    for a in range(n_ages):
        axes[1, 0].plot(times, results['H'][a], label=f'H_{age_labels[a]}',
                       color=colors[a], linestyle='-')
    axes[1, 0].plot(times, results['H_total'], label='H_total',
                   color='black', linewidth=2, linestyle='--')
    axes[1, 0].axhline(y=hosp_capacity, color='red', linestyle='--',
                      label=f'Capacity K={hosp_capacity}')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Hospitalized')
    axes[1, 0].legend()
    axes[1, 0].set_title('Hospital Utilization by Age')
    axes[1, 0].grid(True, alpha=0.3)
    
    for a in range(n_ages):
        axes[1, 1].plot(times, results['D'][a], label=f'D_{age_labels[a]}',
                       color=colors[a], linewidth=2)
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Cumulative deaths')
    axes[1, 1].legend()
    axes[1, 1].set_title('Deaths by Age Group')
    axes[1, 1].grid(True, alpha=0.3)
    
    # overflow
    axes[1, 2].plot(times, results['overflow'], color='red')
    axes[1, 2].set_xlabel('Time (days)')
    axes[1, 2].set_ylabel('Overflow patients')
    axes[1, 2].set_title(f'Hospital Overflow\n(Cumulative: {results["cum_overflow"]:.1f} patient-days)')
    axes[1, 2].grid(True, alpha=0.3)
    
    # row 3: time-varying parameters (if requested)
    if show_beta or show_policy:
        if show_beta and 'beta_t' in results:
            axes[2, 0].plot(times, results['beta_t'], color='purple', linewidth=2)
            axes[2, 0].set_xlabel('Time (days)')
            axes[2, 0].set_ylabel('Transmission rate β(t)')
            axes[2, 0].set_title('Time-Varying Transmission Rate')
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].axis('off')
        
        if show_policy and 'policy_mult' in results:
            axes[2, 1].plot(times, results['policy_mult'], color='orange', linewidth=2)
            axes[2, 1].set_xlabel('Time (days)')
            axes[2, 1].set_ylabel('Policy multiplier')
            axes[2, 1].set_title('Policy Interventions\n(1.0 = no intervention)')
            axes[2, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
            axes[2, 1].set_ylim([0, 1.1])
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].axis('off')
        
        # summary statistics
        axes[2, 2].axis('off')
        total_deaths = sum([results['D'][a][-1] for a in range(n_ages)])
        peak_H = max(results['H_total'])
        
        # check if any waning occurred
        total_S_final = sum([results['S'][a][-1] for a in range(n_ages)])
        total_S_initial = sum(results['age_pops'])
        waning_occurred = total_S_final > (total_S_initial * 0.01)  # more than 1% of initial
        
        summary_text = f"""Summary Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Peak H_total: {peak_H:.1f} / {hosp_capacity}
Total Deaths: {total_deaths:.0f}
Cum. Overflow: {results['cum_overflow']:.1f} patient-days

Deaths by Age:"""
        
        for a in range(n_ages):
            summary_text += f"\n  {age_labels[a]}: {results['D'][a][-1]:.0f}"
        
        summary_text += f"\n\nSusceptible (final):"
        for a in range(n_ages):
            summary_text += f"\n  {age_labels[a]}: {results['S'][a][-1]:.0f}"
        
        if waning_occurred:
            summary_text += f"\n\n⚠ Waning immunity active"
        
        axes[2, 2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                       verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    
    # print summary
    total_deaths = sum([results['D'][a][-1] for a in range(n_ages)])
    peak_H = max(results['H_total'])
    print(f"\n{'='*50}")
    print(f"Time-Varying Model Results")
    print(f"{'='*50}")
    print(f"Total deaths: {total_deaths:.0f}")
    print(f"Peak hospital occupancy: {peak_H:.1f} (Capacity: {hosp_capacity})")
    print(f"Cumulative overflow: {results['cum_overflow']:.1f} patient-days")
    
    if 'beta_t' in results:
        print(f"Beta range: [{min(results['beta_t']):.3f}, {max(results['beta_t']):.3f}]")
