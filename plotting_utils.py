"""Plotting utilities for hospital SIXHRD model visualizations.

This module provides visualization functions for simulation results,
vaccination strategy comparisons, and vaccine allocation optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# ========================================
# Configuration Classes
# ========================================

@dataclass
class PlotConfig:
    """Configuration for plot styling and behavior.
    
    This class centralizes all plotting configuration options, reducing
    code duplication and making it easy to customize plot appearance.
    
    Parameters
    ----------
    age_labels : list[str]
        Labels for age groups (default: ['Young', 'Middle', 'Elderly']).
    age_colors : list[str]
        Colors for each age group in plots.
    strategy_colors : list[str]
        Colors for vaccination strategy comparisons.
    figsize : tuple[int, int]
        Default figure size (width, height).
    show : bool
        Whether to display figures after creation.
    verbose : bool
        Whether to print summary statistics to console.
    grid_alpha : float
        Transparency for grid lines.
    linewidth : float
        Default line width for plots.
    summary_fontsize : int
        Font size for summary panels.
    
    Examples
    --------
    >>> config = PlotConfig(show=False, verbose=False)
    >>> fig, axes = plot_age_structured_results(results, 100, config=config)
    
    >>> custom_config = PlotConfig(
    ...     age_labels=['0-18', '19-64', '65+'],
    ...     age_colors=['blue', 'green', 'red'],
    ...     figsize=(20, 12)
    ... )
    """
    age_labels: list[str] = field(
        default_factory=lambda: ['Young', 'Middle', 'Elderly']
    )
    age_colors: list[str] = field(
        default_factory=lambda: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    )
    strategy_colors: list[str] = field(
        default_factory=lambda: ['gray', 'skyblue', 'green', 'orange', 'purple', 'red', 'brown']
    )
    figsize: tuple[int, int] = (14, 10)
    show: bool = True
    verbose: bool = True
    grid_alpha: float = 0.3
    linewidth: float = 1.5
    summary_fontsize: int = 10
    
    def get_age_colors(self, n_ages: int) -> list[str]:
        """Get colors for the specified number of age groups."""
        return self.age_colors[:n_ages]
    
    def get_strategy_colors(self, n_strategies: int) -> list[str]:
        """Get colors for the specified number of strategies."""
        return self.strategy_colors[:n_strategies]


# Default configuration instance
DEFAULT_CONFIG = PlotConfig()


# ========================================
# Shared Constants and Defaults (Legacy)
# ========================================
# These are kept for backward compatibility but PlotConfig is preferred

DEFAULT_AGE_LABELS = DEFAULT_CONFIG.age_labels
DEFAULT_AGE_COLORS = DEFAULT_CONFIG.age_colors
DEFAULT_STRATEGY_COLORS = DEFAULT_CONFIG.strategy_colors


# ========================================
# Internal Helper Functions
# ========================================

def _setup_axis(
    ax: Axes,
    xlabel: str,
    ylabel: str,
    title: str,
    grid: bool = True,
    legend: bool = True,
    loc: str = 'best',
    grid_alpha: float = 0.3
) -> None:
    """Configure common axis properties."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if grid:
        ax.grid(True, alpha=grid_alpha)
    if legend:
        ax.legend(loc=loc)


def _plot_capacity_line(
    ax: Axes,
    capacity: float,
    color: str = 'red',
    linestyle: str = '--',
    label_prefix: str = 'Capacity',
    alpha: float = 0.7
) -> None:
    """Add a horizontal capacity reference line."""
    ax.axhline(y=capacity, color=color, linestyle=linestyle,
               label=f'{label_prefix} ({capacity})', alpha=alpha)
    
def _plot_vertical_capacity_line(
    ax: Axes,
    capacity: float,
    color: str = 'red',
    linestyle: str = '--',
    label: str = 'Capacity',
    linewidth: float = 2
) -> None:
    """Add a vertical capacity reference line (for bar charts)."""
    ax.axvline(x=capacity, color=color, linestyle=linestyle,
               label=label, linewidth=linewidth)


def _setup_summary_panel(
    ax: Axes,
    lines: list[str],
    fontsize: int = 9
) -> None:
    """Set up a monospace text summary panel from a list of lines."""
    ax.axis('off')
    summary_text = '\n'.join(lines)
    ax.text(0.05, 0.5, summary_text, fontsize=fontsize, family='monospace',
            verticalalignment='center', transform=ax.transAxes)


def _create_figure(
    nrows: int,
    ncols: int,
    figsize: tuple[int, int] | None = None,
    config: PlotConfig | None = None
) -> tuple[Figure, npt.NDArray[Any]]:
    """Create a figure with subplots using configuration."""
    if config is None:
        config = DEFAULT_CONFIG
    if figsize is None:
        figsize = config.figsize
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


def _finalize_figure(fig: Figure, config: PlotConfig | None = None) -> None:
    """Apply tight layout and optionally display the figure."""
    if config is None:
        config = DEFAULT_CONFIG
    plt.tight_layout()
    if config.show:
        plt.show()


def _print_results_header(title: str) -> None:
    """Print a formatted results header to console."""
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    
def _print_results_summary(
    title: str,
    total_deaths: float,
    peak_H: float,
    hosp_capacity: float,
    cum_overflow: float,
    extra_lines: list[str] | None = None
) -> None:
    """Print standard results summary to console."""
    _print_results_header(title)
    print(f"Total deaths: {total_deaths:.0f}")
    print(f"Peak hospital occupancy: {peak_H:.1f} (Capacity: {hosp_capacity})")
    print(f"Cumulative overflow: {cum_overflow:.1f} patient-days")
    if extra_lines:
        for line in extra_lines:
            print(line)
    
def _plot_by_age(
    ax: Axes,
    times: npt.ArrayLike,
    data_by_age: list[npt.ArrayLike],
    age_labels: list[str],
    ylabel: str,
    title: str,
    colors: list[str] | None = None,
    show_total: bool = False,
    total_data: npt.ArrayLike | None = None,
    linewidth: float = 1.5,
    label_prefix: str = ''
) -> None:
    """Plot data series for each age group on a single axis."""
    if colors is None:
        colors = DEFAULT_AGE_COLORS
    n_ages = len(data_by_age)
    for a in range(n_ages):
        label = f'{label_prefix}{age_labels[a]}' if label_prefix else age_labels[a]
        ax.plot(times, data_by_age[a], label=label,
                color=colors[a], linewidth=linewidth)
    if show_total and total_data is not None:
        ax.plot(times, total_data, label='Total',
                color='black', linewidth=2, linestyle='--')
    _setup_axis(ax, 'Time (days)', ylabel, title)
    
def _plot_compartments_by_age(
    axes_row: npt.NDArray[Any],
    times: npt.ArrayLike,
    results: dict[str, Any],
    age_labels: list[str],
    colors: list[str] | None = None
) -> None:
    """Plot S, I, X compartments by age on a row of 3 axes."""
    if colors is None:
        colors = DEFAULT_AGE_COLORS
    
    compartments = [
        ('S', 'Susceptible', 'Susceptible by Age'),
        ('I', 'Infected', 'Infected by Age'),
        ('X', 'Severe (need care)', 'Severe Cases by Age'),
    ]
    
    for idx, (key, ylabel, title) in enumerate(compartments):
        _plot_by_age(axes_row[idx], times, results[key], age_labels,
                     ylabel, title, colors=colors, label_prefix=f'{key}_')
        
def _plot_hospital_utilization(
    ax: Axes,
    times: npt.ArrayLike,
    H_by_age: list[npt.ArrayLike],
    H_total: npt.ArrayLike,
    hosp_capacity: float,
    age_labels: list[str],
    colors: list[str] | None = None
) -> None:
    """Plot hospital utilization by age with capacity line."""
    if colors is None:
        colors = DEFAULT_AGE_COLORS
    
    n_ages = len(H_by_age)
    for a in range(n_ages):
        ax.plot(times, H_by_age[a], label=f'H_{age_labels[a]}',
                color=colors[a], linestyle='-')
    ax.plot(times, H_total, label='H_total',
            color='black', linewidth=2, linestyle='--')
    _plot_capacity_line(ax, hosp_capacity, color='red', label_prefix='Capacity K')
    _setup_axis(ax, 'Time (days)', 'Hospitalized', 'Hospital Utilization by Age')
    
def _plot_deaths_by_age(
    ax: Axes,
    times: npt.ArrayLike,
    D_by_age: list[npt.ArrayLike],
    age_labels: list[str],
    colors: list[str] | None = None
) -> None:
    """Plot cumulative deaths by age group."""
    if colors is None:
        colors = DEFAULT_AGE_COLORS
    
    n_ages = len(D_by_age)
    for a in range(n_ages):
        ax.plot(times, D_by_age[a], label=f'D_{age_labels[a]}',
                color=colors[a], linewidth=2)
    _setup_axis(ax, 'Time (days)', 'Cumulative deaths', 'Deaths by Age Group')
    
def _plot_overflow(
    ax: Axes,
    times: npt.ArrayLike,
    overflow: npt.ArrayLike,
    cum_overflow: float,
    color: str = 'red'
) -> None:
    """Plot hospital overflow time series."""
    ax.plot(times, overflow, color=color)
    _setup_axis(ax, 'Time (days)', 'Overflow patients',
                f'Hospital Overflow\n(Cumulative: {cum_overflow:.1f} patient-days)',
                legend=False)
    
def _compute_attack_rates(results: dict[str, Any], n_ages: int) -> list[float]:
    """Compute attack rates for each age group."""
    return [(results['R'][a][-1] + results['D'][a][-1]) / results['age_pops'][a] * 100
            for a in range(n_ages)]
    
def _compute_deaths_summary(
    results: dict[str, Any],
    n_ages: int
) -> tuple[list[float], float]:
    """Compute deaths summary statistics."""
    deaths_by_age = [results['D'][a][-1] for a in range(n_ages)]
    total_deaths = sum(deaths_by_age)
    return deaths_by_age, total_deaths

def _build_age_breakdown_lines(
    label: str,
    values: list[float],
    age_labels: list[str],
    format_str: str = "{:.0f}",
    show_pct: bool = False,
    total: float | None = None
) -> list[str]:
    """Build formatted lines for age breakdown statistics."""
    lines = [f"{label}:"]
    for a, age in enumerate(age_labels):
        val_str = format_str.format(values[a])
        if show_pct and total and total > 0:
            pct = values[a] / total * 100
            lines.append(f"  {age}: {val_str} ({pct:.1f}%)")
        else:
            lines.append(f"  {age}: {val_str}")
    return lines


def extract_summary_metrics(
    results: dict[str, Any],
    hosp_capacity: float | None = None
) -> dict[str, Any]:
    """
    Extract common summary metrics from simulation results.
    
    This helper consolidates the repeated pattern of extracting deaths,
    peak hospitalization, and overflow from results dictionaries.
    
    Parameters
    ----------
    results : dict
        Results dictionary from any simulation function.
    hosp_capacity : float, optional
        Hospital capacity for utilization calculation.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'n_ages': number of age groups
        - 'deaths_by_age': list of deaths per age group
        - 'total_deaths': sum of all deaths
        - 'peak_H': peak total hospitalization
        - 'cum_overflow': cumulative overflow
        - 'attack_rates': attack rate per age group (if R available)
        - 'utilization': peak/capacity ratio (if capacity provided)
    
    Examples
    --------
    >>> metrics = extract_summary_metrics(results, hosp_capacity=100)
    >>> print(f"Total deaths: {metrics['total_deaths']:.0f}")
    """
    n_ages = len(results['S'])
    deaths_by_age, total_deaths = _compute_deaths_summary(results, n_ages)
    
    # Handle both single-H and ward/ICU models
    if 'H_total' in results:
        peak_H = max(results['H_total'])
    elif 'H_ward_total' in results and 'H_icu_total' in results:
        peak_H = max(np.array(results['H_ward_total']) + np.array(results['H_icu_total']))
    else:
        peak_H = 0.0
    
    metrics = {
        'n_ages': n_ages,
        'deaths_by_age': deaths_by_age,
        'total_deaths': total_deaths,
        'peak_H': peak_H,
        'cum_overflow': results.get('cum_overflow', 0.0),
    }
    
    # Add attack rates if R is available
    if 'R' in results and 'age_pops' in results:
        metrics['attack_rates'] = _compute_attack_rates(results, n_ages)
    
    # Add utilization if capacity provided
    if hosp_capacity and hosp_capacity > 0:
        metrics['utilization'] = peak_H / hosp_capacity
    
    return metrics


def format_results_table(
    results_dict: dict[str, dict[str, Any]],
    metrics: list[str] | None = None,
    sort_by: str | None = None,
    ascending: bool = True
) -> list[dict[str, Any]]:
    """
    Format multiple simulation results into a sorted table format.
    
    Useful for comparing vaccination strategies or capacity scenarios.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping names to result dictionaries.
    metrics : list[str], optional
        List of metric keys to include. If None, includes all common metrics.
    sort_by : str, optional
        Metric key to sort by. If None, maintains original order.
    ascending : bool, optional
        Sort order (default: True for ascending).
    
    Returns
    -------
    list[dict]
        List of dictionaries with 'name' and metric values, sorted as specified.
    
    Examples
    --------
    >>> table = format_results_table(strategy_results, sort_by='total_deaths')
    >>> for row in table[:3]:
    ...     print(f"{row['name']}: {row['total_deaths']:.0f} deaths")
    """
    if metrics is None:
        metrics = ['total_deaths', 'peak_H', 'cum_overflow']
    
    table = []
    for name, res in results_dict.items():
        row = {'name': name}
        for metric in metrics:
            if metric in res:
                row[metric] = res[metric]
        table.append(row)
    
    if sort_by and sort_by in metrics:
        table.sort(key=lambda x: x.get(sort_by, float('inf')), reverse=not ascending)
    
    return table


def print_comparison_table(
    results_dict: dict[str, dict[str, Any]],
    title: str = "Comparison Results",
    sort_by: str = 'total_deaths',
    top_n: int | None = None
) -> None:
    """
    Print a formatted comparison table to console.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping names to result dictionaries.
    title : str, optional
        Table title.
    sort_by : str, optional
        Metric to sort by (default: 'total_deaths').
    top_n : int, optional
        Only show top N results. If None, shows all.
    
    Examples
    --------
    >>> print_comparison_table(strategy_results, title="Vaccination Strategies")
    """
    _print_results_header(title)
    
    table = format_results_table(results_dict, sort_by=sort_by)
    if top_n:
        table = table[:top_n]
    
    for row in table:
        deaths = row.get('total_deaths', 0)
        peak = row.get('peak_H', 0)
        overflow = row.get('cum_overflow', 0)
        print(f"  {row['name']}: deaths={deaths:.0f}, peak_H={peak:.1f}, overflow={overflow:.1f}")


def build_summary_lines(
    results: dict[str, Any],
    hosp_capacity: float,
    age_labels: list[str],
    title: str = "Summary Statistics",
    include_attack_rates: bool = True,
    include_unmet_care: bool = True
) -> list[str]:
    """
    Build standard summary lines for results panel.
    
    Consolidates the repeated pattern of building summary text blocks.
    
    Parameters
    ----------
    results : dict
        Simulation results dictionary.
    hosp_capacity : float
        Hospital capacity for reference.
    age_labels : list[str]
        Labels for age groups.
    title : str, optional
        Summary title.
    include_attack_rates : bool, optional
        Whether to include attack rates (default: True).
    include_unmet_care : bool, optional
        Whether to include unmet care stats (default: True).
    
    Returns
    -------
    list[str]
        List of formatted summary lines.
    """
    metrics = extract_summary_metrics(results, hosp_capacity)
    n_ages = metrics['n_ages']
    
    lines = [
        title,
        "━" * max(len(title) + 4, 28),
        f"Peak H_total: {metrics['peak_H']:.1f} / {hosp_capacity}",
        f"Total Deaths: {metrics['total_deaths']:.0f}",
        f"Cum. Overflow: {metrics['cum_overflow']:.1f} patient-days",
        "",
    ]
    
    lines.extend(_build_age_breakdown_lines(
        "Deaths by Age", metrics['deaths_by_age'], age_labels,
        show_pct=True, total=metrics['total_deaths']
    ))
    
    if include_attack_rates and 'attack_rates' in metrics:
        lines.append("")
        lines.extend(_build_age_breakdown_lines(
            "Attack Rates", metrics['attack_rates'], age_labels, format_str="{:.1f}%"
        ))
    
    if include_unmet_care and 'cum_unmet' in results:
        lines.append("")
        lines.extend(_build_age_breakdown_lines(
            "Unmet Care (patient-days)", results['cum_unmet'], age_labels, format_str="{:.1f}"
        ))
    
    return lines



# ========================================
# Plotting Functions
# ========================================

def plot_hospital_simulation_stats(
    times: npt.ArrayLike,
    S_vals: npt.ArrayLike,
    I_vals: npt.ArrayLike,
    X_vals: npt.ArrayLike,
    H_vals: npt.ArrayLike,
    R_vals: npt.ArrayLike,
    D_vals: npt.ArrayLike,
    overflow_vals: npt.ArrayLike,
    cum_overflow: float,
    cum_unmet: float,
    hosp_capacity: int,
    N: int,
    show: bool = True,
    verbose: bool = True
) -> tuple[Figure, npt.NDArray[Any]]:
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
    show : bool, optional
        Whether to display the figure (default: True).
    verbose : bool, optional
        Whether to print summary to console (default: True).
    
    Returns
    -------
    tuple
        (fig, axes) matplotlib Figure and Axes array.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compartments over time
    compartments = [
        (S_vals, 'S (Susceptible)'),
        (I_vals, 'I (Infected)'),
        (X_vals, 'X (Needs care)'),
        (H_vals, 'H (Hospitalized)'),
        (R_vals, 'R (Recovered)'),
        (D_vals, 'D (Deaths)'),
    ]
    for vals, label in compartments:
        axes[0, 0].plot(times, vals, label=label)
    _setup_axis(axes[0, 0], 'Time (days)', 'Population', 'Epidemic Dynamics')

    # Hospital capacity
    axes[0, 1].plot(times, H_vals, label='Hospitalized', color='red')
    axes[0, 1].axhline(y=hosp_capacity, color='black', linestyle='--',
                       label=f'Capacity (K={hosp_capacity})')
    axes[0, 1].fill_between(times, 0, hosp_capacity, alpha=0.2,
                            color='green', label='Available capacity')
    _setup_axis(axes[0, 1], 'Time (days)', 'Hospital beds', 'Hospital Capacity Utilization')

    # Overflow
    _plot_overflow(axes[1, 0], times, overflow_vals, cum_overflow)

    # Summary statistics
    attack_rate = (R_vals[-1] + D_vals[-1]) / N * 100
    summary_lines = [
        "Summary Statistics:",
        "━" * 22,
        f"Peak Hospitalized: {max(H_vals):.1f}",
        f"Total Deaths: {D_vals[-1]:.1f}",
        f"Total Recovered: {R_vals[-1]:.1f}",
        f"Attack Rate: {attack_rate:.1f}%",
        "",
        f"Cumulative Overflow: {cum_overflow:.1f} patient-days",
        f"Cumulative Unmet Care: {cum_unmet:.1f} patient-days",
        "",
        "Final Population:",
        f"  S: {S_vals[-1]:.1f}",
        f"  I: {I_vals[-1]:.1f}",
        f"  X: {X_vals[-1]:.1f}",
        f"  H: {H_vals[-1]:.1f}",
        f"  R: {R_vals[-1]:.1f}",
        f"  D: {D_vals[-1]:.1f}",
    ]
    _setup_summary_panel(axes[1, 1], summary_lines, fontsize=11)

    plt.tight_layout()
    if show:
        plt.show()

    if verbose:
        print(f"\nSimulation complete!")
        print(f"Total deaths: {D_vals[-1]:.0f}")
        print(f"Peak hospital occupancy: {max(H_vals):.1f} (Capacity: {hosp_capacity})")
        print(f"Cumulative overflow: {cum_overflow:.1f} patient-days")
        print(f"Cumulative unmet care: {cum_unmet:.1f} patient-days")

    return fig, axes


def plot_age_structured_results(
    results: dict[str, Any],
    hosp_capacity: int,
    age_labels: list[str] | None = None,
    show: bool = True,
    verbose: bool = True,
    config: PlotConfig | None = None
) -> tuple[Figure, npt.NDArray[Any]]:
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
        Labels for age groups. If None, uses config.age_labels.
    show : bool, optional
        Whether to display the figure (default: True).
    verbose : bool, optional
        Whether to print summary to console (default: True).
    config : PlotConfig, optional
        Plot configuration object. If None, uses defaults.
    
    Returns
    -------
    tuple
        (fig, axes) matplotlib Figure and Axes array.
    """
    if config is None:
        config = PlotConfig(show=show, verbose=verbose)
    
    if age_labels is None:
        age_labels = config.age_labels
    
    colors = config.get_age_colors(len(age_labels))
    
    times = results['times']
    n_ages = len(results['S'])
    
    fig, axes = _create_figure(2, 3, figsize=(18, 10), config=config)
    
    # Row 1: S, I, X by age
    _plot_compartments_by_age(axes[0], times, results, age_labels, colors=colors)
    
    # Row 2: Hospital, Deaths, Summary
    _plot_hospital_utilization(axes[1, 0], times, results['H'], results['H_total'],
                                hosp_capacity, age_labels, colors=colors)
    _plot_deaths_by_age(axes[1, 1], times, results['D'], age_labels, colors=colors)
    
    # Summary statistics using helper
    summary_lines = build_summary_lines(
        results, hosp_capacity, age_labels,
        title="Summary Statistics",
        include_attack_rates=True,
        include_unmet_care=True
    )
    
    _setup_summary_panel(axes[1, 2], summary_lines, fontsize=config.summary_fontsize)
    
    _finalize_figure(fig, config)
    
    if config.verbose:
        metrics = extract_summary_metrics(results, hosp_capacity)
        extra_lines = [
            f"  {age_labels[a]}: {metrics['deaths_by_age'][a]:.0f} deaths, AR={metrics['attack_rates'][a]:.1f}%"
            for a in range(metrics['n_ages'])
        ]
        _print_results_summary("Age-Structured Model Results", metrics['total_deaths'], metrics['peak_H'],
                               hosp_capacity, results['cum_overflow'], extra_lines)
    
    return fig, axes


def plot_age_structured_icu_results(
    results: dict[str, Any],
    age_labels: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    show: bool = True,
    verbose: bool = True,
    config: PlotConfig | None = None,
    show_all_compartments: bool = False
) -> tuple[Figure, npt.NDArray[Any]]:
    """
    Plot comprehensive results from age-structured SIXHRD model with ward/ICU.
    
    Creates a grid showing:
    - Row 1: Infection dynamics (I, X, D by age) OR all compartments (S, I, X, R)
    - Row 2: Hospital utilization (Ward, ICU, Combined)
    - Row 3: System performance (Overflow, Deaths breakdown, Summary)
    
    Parameters
    ----------
    results : dict
        Results dictionary from simulate_age_structured_model_icu().
    age_labels : list, optional
        Labels for age groups. If None, uses config.age_labels.
    figsize : tuple, optional
        Figure size. If None, auto-sizes based on show_all_compartments.
    show : bool, optional
        Whether to display the figure (default: True).
    verbose : bool, optional
        Whether to print summary to console (default: True).
    config : PlotConfig, optional
        Plot configuration object. If None, uses defaults.
    show_all_compartments : bool, optional
        If True, shows S, I, X, R in a 4-column first row (default: False).
        If False, shows I, X, D in a 3-column first row.
    
    Returns
    -------
    tuple
        (fig, axes) matplotlib Figure and Axes array.
    """
    # Determine grid layout based on compartment display mode
    if show_all_compartments:
        n_cols = 4
        default_figsize = (22, 14)
    else:
        n_cols = 3
        default_figsize = (18, 14)
    
    if figsize is None:
        figsize = default_figsize
    
    if config is None:
        config = PlotConfig(show=show, verbose=verbose, figsize=figsize)
    
    if age_labels is None:
        age_labels = config.age_labels
    
    colors = config.get_age_colors(len(age_labels))
    
    times = results['times']
    n_ages = len(results['S'])
    ward_capacity = results['ward_capacity']
    icu_capacity = results['icu_capacity']
    
    fig, axes = _create_figure(3, n_cols, figsize=figsize, config=config)
    
    # Row 1: Infection dynamics by age
    if show_all_compartments:
        # 4-column layout: S, I, X, R
        _plot_by_age(axes[0, 0], times, results['S'], age_labels,
                     'Susceptible', 'Susceptible by Age', colors=colors, linewidth=2)
        _plot_by_age(axes[0, 1], times, results['I'], age_labels,
                     'Infected', 'Infected by Age', colors=colors, linewidth=2)
        _plot_by_age(axes[0, 2], times, results['X'], age_labels,
                     'Severe (need care)', 'Severe Cases by Age', colors=colors, linewidth=2)
        _plot_by_age(axes[0, 3], times, results['R'], age_labels,
                     'Recovered', 'Recovered by Age', colors=colors, linewidth=2)
    else:
        # 3-column layout: I, X, D
        _plot_by_age(axes[0, 0], times, results['I'], age_labels,
                     'Infected', 'Infected by Age', colors=colors, linewidth=2)
        _plot_by_age(axes[0, 1], times, results['X'], age_labels,
                     'Severe (need care)', 'Severe Cases by Age', colors=colors, linewidth=2)
        _plot_by_age(axes[0, 2], times, results['D'], age_labels,
                     'Cumulative Deaths', 'Deaths by Age', colors=colors, linewidth=2)
    
    # Row 2: Hospital utilization
    _plot_by_age(axes[1, 0], times, results['H_ward'], age_labels,
                 'Ward Patients', 'General Ward by Age', colors=colors,
                 show_total=True, total_data=results['H_ward_total'])
    _plot_capacity_line(axes[1, 0], ward_capacity, color='blue', label_prefix='Ward Capacity')
    axes[1, 0].legend()
    
    _plot_by_age(axes[1, 1], times, results['H_icu'], age_labels,
                 'ICU Patients', 'ICU by Age', colors=colors,
                 show_total=True, total_data=results['H_icu_total'])
    _plot_capacity_line(axes[1, 1], icu_capacity, color='red', label_prefix='ICU Capacity')
    axes[1, 1].legend()
    
    # Combined hospital burden (stacked area)
    axes[1, 2].stackplot(times, results['H_ward_total'], results['H_icu_total'],
                         labels=['Ward', 'ICU'], colors=['#3498db', '#e74c3c'], alpha=0.7)
    _plot_capacity_line(axes[1, 2], ward_capacity + icu_capacity, color='purple',
                        label_prefix='Total Capacity')
    _setup_axis(axes[1, 2], 'Time (days)', 'Total Hospital Burden', 'Combined Ward + ICU')
    
    # Handle extra column in 4-column mode for row 2
    if show_all_compartments:
        # Plot deaths by age in the 4th column of row 2
        _plot_by_age(axes[1, 3], times, results['D'], age_labels,
                     'Cumulative Deaths', 'Deaths by Age', colors=colors, linewidth=2)
    
    # Row 3: System performance
    axes[2, 0].plot(times, results['ward_overflow'], label='Ward Overflow',
                    color='blue', linewidth=2)
    axes[2, 0].plot(times, results['icu_overflow'], label='ICU Overflow',
                    color='red', linewidth=2)
    axes[2, 0].fill_between(times, results['ward_overflow'], alpha=0.3, color='blue')
    axes[2, 0].fill_between(times, results['icu_overflow'], alpha=0.3, color='red')
    _setup_axis(axes[2, 0], 'Time (days)', 'Overflow (patients)', 'Capacity Overflow')
    
    # Deaths breakdown bar chart
    deaths_final, total_deaths = _compute_deaths_summary(results, n_ages)
    x = np.arange(n_ages)
    axes[2, 1].bar(x, deaths_final, 0.5, color=colors[:n_ages])
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(age_labels)
    _setup_axis(axes[2, 1], 'Age Group', 'Final Deaths', 'Deaths by Age Group', 
                legend=False, grid_alpha=config.grid_alpha)
    axes[2, 1].grid(True, alpha=config.grid_alpha, axis='y')
    
    # Summary statistics panel
    peak_ward = max(results['H_ward_total'])
    peak_icu = max(results['H_icu_total'])
    
    # Compute epidemiological metrics
    total_pop = sum(results['age_pops'])
    S_final = sum(results['S'][a][-1] for a in range(n_ages))
    R_final = sum(results['R'][a][-1] for a in range(n_ages))
    infected_total = total_pop - S_final
    attack_rate = (infected_total / total_pop) * 100 if total_pop > 0 else 0
    ifr = (total_deaths / infected_total) * 100 if infected_total > 0 else 0
    
    summary_lines = [
        "SIMULATION SUMMARY",
        "━" * 36,
        "",
        f"Population: {total_pop:,.0f}",
        f"Attack Rate: {attack_rate:.1f}%",
        f"IFR: {ifr:.2f}%",
        "",
        f"Total Deaths: {total_deaths:.0f}",
        f"Total Recovered: {R_final:.0f}",
        "",
        "Peak Occupancy:",
        f"  Ward: {peak_ward:.1f} / {ward_capacity} ({peak_ward/ward_capacity*100:.0f}%)",
        f"  ICU: {peak_icu:.1f} / {icu_capacity} ({peak_icu/icu_capacity*100:.0f}%)",
        "",
        "Cumulative Overflow (patient-days):",
        f"  Ward: {results['cum_ward_overflow']:.1f}",
        f"  ICU: {results['cum_icu_overflow']:.1f}",
        "",
    ]
    summary_lines.extend(_build_age_breakdown_lines(
        "Deaths by Age", deaths_final, age_labels, show_pct=True, total=total_deaths))
    summary_lines.append("")
    
    # Unmet care with ward/ICU breakdown
    summary_lines.append("Unmet Care (patient-days):")
    for a in range(n_ages):
        summary_lines.append(
            f"  {age_labels[a]}: Ward={results['cum_unmet_ward'][a]:.1f}, "
            f"ICU={results['cum_unmet_icu'][a]:.1f}")
    
    # Handle summary panel placement based on column count
    if show_all_compartments:
        # Merge last two columns for summary in 4-column mode
        axes[2, 2].axis('off')
        _setup_summary_panel(axes[2, 3], summary_lines, fontsize=config.summary_fontsize)
    else:
        _setup_summary_panel(axes[2, 2], summary_lines, fontsize=config.summary_fontsize)
    
    _finalize_figure(fig, config)
    
    # Console summary
    if config.verbose:
        _print_results_header("Age-Structured ICU Model Results")
        print(f"Population: {total_pop:,}")
        print(f"Attack rate: {attack_rate:.1f}%")
        print(f"IFR: {ifr:.2f}%")
        print(f"Total deaths: {total_deaths:.0f}")
        print(f"Peak ward: {peak_ward:.1f} / {ward_capacity} ({peak_ward/ward_capacity*100:.0f}%)")
        print(f"Peak ICU: {peak_icu:.1f} / {icu_capacity} ({peak_icu/icu_capacity*100:.0f}%)")
        print(f"Cumulative overflow - Ward: {results['cum_ward_overflow']:.1f}, "
              f"ICU: {results['cum_icu_overflow']:.1f}")
        print("\nDeaths by age:")
        for a in range(n_ages):
            print(f"  {age_labels[a]}: {deaths_final[a]:.0f}")
    
    return fig, axes


def plot_strategy_comparison(
    strategy_results: dict[str, dict[str, Any]],
    hosp_capacity: int = 100,
    figsize: tuple[int, int] = (16, 5),
    age_labels: list[str] | None = None,
    show: bool = True,
    config: PlotConfig | None = None
) -> tuple[Figure, Figure]:
    """
    Visualize vaccination strategy comparison across multiple metrics.
    
    Creates bar charts comparing total deaths, peak hospital occupancy, and
    cumulative overflow across vaccination strategies. Also shows age-specific
    death breakdown for each strategy.
    
    Parameters
    ----------
    strategy_results : dict
        Results from compare_vaccination_strategies() function.
    hosp_capacity : int, optional
        Hospital capacity for reference line (default: 100).
    figsize : tuple, optional
        Figure size (default: (16, 5)).
    age_labels : list, optional
        Labels for age groups. If None, uses config.age_labels.
    show : bool, optional
        Whether to display the figures (default: True).
    config : PlotConfig, optional
        Plot configuration object. If None, uses defaults.
    
    Returns
    -------
    tuple
        (fig1, fig2) - comparison figure and age breakdown figure.
    """
    if config is None:
        config = PlotConfig(show=show, figsize=figsize)
    
    if age_labels is None:
        age_labels = config.age_labels
    
    strategies_list = list(strategy_results.keys())
    n_strategies = len(strategies_list)
    colors = config.get_strategy_colors(n_strategies)
    
    # Extract metrics
    metrics = {
        'total_deaths': [strategy_results[s]['total_deaths'] for s in strategies_list],
        'peak_H': [strategy_results[s]['peak_H'] for s in strategies_list],
        'cum_overflow': [strategy_results[s]['cum_overflow'] for s in strategies_list],
    }
    
    fig1, axes = plt.subplots(1, 3, figsize=figsize)
    
    chart_configs = [
        ('total_deaths', 'Total Deaths', 'Total Deaths by Vaccination Strategy', None),
        ('peak_H', 'Peak Hospital Occupancy', 'Peak Hospital Load by Strategy', hosp_capacity),
        ('cum_overflow', 'Cumulative Overflow (patient-days)', 'Hospital Overflow by Strategy', None),
    ]
    
    for idx, (key, xlabel, title, capacity) in enumerate(chart_configs):
        axes[idx].barh(strategies_list, metrics[key], color=colors)
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_title(title)
        axes[idx].grid(True, alpha=config.grid_alpha, axis='x')
        if capacity:
            _plot_vertical_capacity_line(axes[idx], capacity)
            axes[idx].legend()
    
    plt.tight_layout()
    
    # Deaths by age group for each strategy
    fig2, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_strategies)
    width = 0.25
    
    age_colors = config.get_age_colors(len(age_labels))
    for i, age in enumerate(age_labels):
        deaths = [strategy_results[s]['deaths_by_age'][i] for s in strategies_list]
        ax.bar(x + i * width, deaths, width, label=age, color=age_colors[i])
    
    ax.set_xlabel('Vaccination Strategy')
    ax.set_ylabel('Deaths')
    ax.set_title('Deaths by Age Group for Each Vaccination Strategy')
    ax.set_xticks(x + width)
    ax.set_xticklabels(strategies_list, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=config.grid_alpha, axis='y')
    plt.tight_layout()
    
    if config.show:
        plt.show()
    
    return fig1, fig2


def plot_optimal_allocation(
    deaths_grid: npt.NDArray[np.floating[Any]],
    overflow_grid: npt.NDArray[np.floating[Any]],
    young_range: npt.ArrayLike,
    middle_range: npt.ArrayLike,
    age_pops: list[float],
    total_doses: float,
    figsize: tuple[int, int] = (14, 6),
    show: bool = True,
    verbose: bool = True
) -> tuple[float, float, float, Figure, npt.NDArray[Any]]:
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
        Young coverage values tested.
    middle_range : array
        Middle coverage values tested.
    age_pops : list
        Population sizes [young, middle, elderly].
    total_doses : float
        Total vaccine doses available.
    figsize : tuple, optional
        Figure size (default: (14, 6)).
    show : bool, optional
        Whether to display the figure (default: True).
    verbose : bool, optional
        Whether to print results to console (default: True).
    
    Returns
    -------
    tuple
        (opt_young, opt_middle, opt_elderly, fig, axes).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Find optimum
    min_idx = np.nanargmin(deaths_grid)
    min_i, min_j = np.unravel_index(min_idx, deaths_grid.shape)
    opt_young = young_range[min_i]
    opt_middle = middle_range[min_j]
    opt_elderly = (total_doses - opt_young * age_pops[0] - opt_middle * age_pops[1]) / age_pops[2]
    
    heatmap_configs = [
        (deaths_grid, 'Total Deaths', f'Total Deaths vs Vaccine Allocation\n(Total doses = {total_doses:.0f})'),
        (overflow_grid, 'Cumulative Overflow (patient-days)', 'Hospital Overflow vs Vaccine Allocation'),
    ]
    
    for idx, (grid, cbar_label, title) in enumerate(heatmap_configs):
        im = axes[idx].contourf(middle_range, young_range, grid, levels=30, cmap='RdYlGn_r')
        cbar = plt.colorbar(im, ax=axes[idx])
        cbar.set_label(cbar_label)
        axes[idx].scatter(opt_middle, opt_young, color='red', s=200, marker='*',
                          edgecolors='black', linewidths=2, zorder=10)
        axes[idx].set_xlabel('Middle Age Coverage')
        axes[idx].set_ylabel('Young Coverage')
        axes[idx].set_title(title)
    
    axes[0].text(opt_middle, opt_young + 0.05, 'Optimal',
                 ha='center', fontsize=10, weight='bold', color='red')
    
    plt.tight_layout()
    if show:
        plt.show()
    
    if verbose:
        _print_results_header(f"OPTIMAL ALLOCATION (Fixed total: {total_doses:.0f} doses)")
        print(f"Young coverage: {opt_young * 100:.1f}%")
        print(f"Middle coverage: {opt_middle * 100:.1f}%")
        print(f"Elderly coverage: {opt_elderly * 100:.1f}%")
        print(f"\nMinimum deaths: {np.nanmin(deaths_grid):.0f}")
        print(f"Overflow at optimum: {overflow_grid[min_i, min_j]:.1f} patient-days")
    
    return opt_young, opt_middle, opt_elderly, fig, axes


def plot_time_varying_results(
    results: dict[str, Any],
    hosp_capacity: int,
    age_labels: list[str] | None = None,
    show_beta: bool = True,
    show_policy: bool = True,
    show: bool = True,
    verbose: bool = True,
    config: PlotConfig | None = None
) -> tuple[Figure, npt.NDArray[Any]]:
    """
    Plot results from time-varying SIXHRD model with transmission dynamics.
    
    Parameters
    ----------
    results : dict
        Results dictionary from simulate_age_structured_time_varying().
    hosp_capacity : int
        Hospital bed capacity for reference line.
    age_labels : list, optional
        Labels for age groups. If None, uses config.age_labels.
    show_beta : bool, optional
        Whether to plot time-varying beta (default: True).
    show_policy : bool, optional
        Whether to plot policy multiplier (default: True).
    show : bool, optional
        Whether to display the figure (default: True).
    verbose : bool, optional
        Whether to print summary to console (default: True).
    config : PlotConfig, optional
        Plot configuration object. If None, uses defaults.
    
    Returns
    -------
    tuple
        (fig, axes) matplotlib Figure and Axes array.
    """
    if config is None:
        config = PlotConfig(show=show, verbose=verbose)
    
    if age_labels is None:
        age_labels = config.age_labels
    
    colors = config.get_age_colors(len(age_labels))
    
    times = results['times']
    n_ages = len(results['S'])
    
    n_rows = 3 if (show_beta or show_policy) else 2
    fig, axes = _create_figure(n_rows, 3, figsize=(18, n_rows * 5), config=config)
    
    # Row 1: I, X, S by age
    _plot_by_age(axes[0, 0], times, results['I'], age_labels,
                 'Infected', 'Infected by Age', colors=colors, label_prefix='I_')
    _plot_by_age(axes[0, 1], times, results['X'], age_labels,
                 'Severe (need care)', 'Severe Cases by Age', colors=colors, label_prefix='X_')
    _plot_by_age(axes[0, 2], times, results['S'], age_labels,
                 'Susceptible', 'Susceptible by Age (with waning)', colors=colors, label_prefix='S_')
    
    # Row 2: Hospital, Deaths, Overflow
    _plot_hospital_utilization(axes[1, 0], times, results['H'], results['H_total'],
                                hosp_capacity, age_labels, colors=colors)
    _plot_deaths_by_age(axes[1, 1], times, results['D'], age_labels, colors=colors)
    _plot_overflow(axes[1, 2], times, results['overflow'], results['cum_overflow'])
    
    # Row 3: Time-varying parameters
    if show_beta or show_policy:
        if show_beta and 'beta_t' in results:
            axes[2, 0].plot(times, results['beta_t'], color='purple', linewidth=2)
            _setup_axis(axes[2, 0], 'Time (days)', 'Transmission rate β(t)',
                        'Time-Varying Transmission Rate', legend=False)
        else:
            axes[2, 0].axis('off')
        
        if show_policy and 'policy_mult' in results:
            axes[2, 1].plot(times, results['policy_mult'], color='orange', linewidth=2)
            axes[2, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
            axes[2, 1].set_ylim([0, 1.1])
            _setup_axis(axes[2, 1], 'Time (days)', 'Policy multiplier',
                        'Policy Interventions\n(1.0 = no intervention)', legend=False)
        else:
            axes[2, 1].axis('off')
        
        # Summary panel
        deaths_by_age, total_deaths = _compute_deaths_summary(results, n_ages)
        peak_H = max(results['H_total'])
        
        total_S_final = sum([results['S'][a][-1] for a in range(n_ages)])
        total_S_initial = sum(results['age_pops'])
        waning_occurred = total_S_final > (total_S_initial * 0.01)
        
        summary_lines = [
            "Summary Statistics:",
            "━" * 28,
            f"Peak H_total: {peak_H:.1f} / {hosp_capacity}",
            f"Total Deaths: {total_deaths:.0f}",
            f"Cum. Overflow: {results['cum_overflow']:.1f} patient-days",
            "",
        ]
        summary_lines.extend(_build_age_breakdown_lines(
            "Deaths by Age", deaths_by_age, age_labels))
        summary_lines.append("")
        
        susceptible_final = [results['S'][a][-1] for a in range(n_ages)]
        summary_lines.extend(_build_age_breakdown_lines(
            "Susceptible (final)", susceptible_final, age_labels))
        
        if waning_occurred:
            summary_lines.extend(["", "⚠ Waning immunity active"])
        
        _setup_summary_panel(axes[2, 2], summary_lines, fontsize=config.summary_fontsize)
    
    _finalize_figure(fig, config)
    
    if config.verbose:
        deaths_by_age, total_deaths = _compute_deaths_summary(results, n_ages)
        peak_H = max(results['H_total'])
        extra_lines = []
        if 'beta_t' in results:
            extra_lines.append(f"Beta range: [{min(results['beta_t']):.3f}, {max(results['beta_t']):.3f}]")
        _print_results_summary("Time-Varying Model Results", total_deaths, peak_H,
                               hosp_capacity, results['cum_overflow'], extra_lines)
    
    return fig, axes

def plot_age_structured_icu_full(
    results: dict[str, Any],
    age_labels: list[str] | None = None,
    figsize: tuple[int, int] = (24, 18),
    show: bool = True,
    verbose: bool = True,
    config: PlotConfig | None = None
) -> tuple[Figure, npt.NDArray[Any]]:
    """
    Plot ALL SIXHRD compartments from age-structured ICU model.
    
    Creates a comprehensive 4x4 grid showing:
    - Row 1: S (Susceptible), I (Infected), X (Severe), H_ward (Ward)
    - Row 2: H_icu (ICU), R (Recovered), D (Deaths), Combined Hospital
    - Row 3: Overflow metrics, Differential mortality, Deaths breakdown, Unmet care
    - Row 4: Summary statistics (spans multiple columns)
    
    Parameters
    ----------
    results : dict
        Results dictionary from simulate_age_structured_model_icu().
    age_labels : list, optional
        Labels for age groups. If None, uses config.age_labels.
    figsize : tuple, optional
        Figure size (default: (24, 18)).
    show : bool, optional
        Whether to display the figure (default: True).
    verbose : bool, optional
        Whether to print summary to console (default: True).
    config : PlotConfig, optional
        Plot configuration object. If None, uses defaults.
    
    Returns
    -------
    tuple
        (fig, axes) matplotlib Figure and Axes array.
    """
    if config is None:
        config = PlotConfig(show=show, verbose=verbose, figsize=figsize)
    
    if age_labels is None:
        age_labels = config.age_labels
    
    colors = config.get_age_colors(len(age_labels))
    
    times = results['times']
    n_ages = len(results['S'])
    ward_capacity = results['ward_capacity']
    icu_capacity = results['icu_capacity']
    total_pop = sum(results['age_pops'])
    
    fig, axes = _create_figure(4, 4, figsize=figsize, config=config)
    
    # ═══════════════════════════════════════════════════════════════
    # Row 1: Core compartments (S, I, X, H_ward)
    # ═══════════════════════════════════════════════════════════════
    
    # S - Susceptible
    _plot_by_age(axes[0, 0], times, results['S'], age_labels,
                 'Susceptible', 'S: Susceptible by Age', colors=colors, linewidth=2)
    # Add total line
    S_total = [sum(results['S'][a][t] for a in range(n_ages)) for t in range(len(times))]
    axes[0, 0].plot(times, S_total, 'k--', linewidth=2, alpha=0.7, label='Total')
    axes[0, 0].legend(fontsize=8)
    
    # I - Infected
    _plot_by_age(axes[0, 1], times, results['I'], age_labels,
                 'Infected', 'I: Infected by Age', colors=colors, linewidth=2)
    I_total = [sum(results['I'][a][t] for a in range(n_ages)) for t in range(len(times))]
    axes[0, 1].plot(times, I_total, 'k--', linewidth=2, alpha=0.7, label='Total')
    axes[0, 1].legend(fontsize=8)
    
    # X - Severe (needs care)
    _plot_by_age(axes[0, 2], times, results['X'], age_labels,
                 'Severe', 'X: Severe Cases by Age', colors=colors, linewidth=2)
    X_total = [sum(results['X'][a][t] for a in range(n_ages)) for t in range(len(times))]
    axes[0, 2].plot(times, X_total, 'k--', linewidth=2, alpha=0.7, label='Total')
    axes[0, 2].legend(fontsize=8)
    
    # H_ward - General Ward
    _plot_by_age(axes[0, 3], times, results['H_ward'], age_labels,
                 'Ward Patients', 'H_ward: General Ward by Age', colors=colors, linewidth=2)
    axes[0, 3].plot(times, results['H_ward_total'], 'k--', linewidth=2, alpha=0.7, label='Total')
    _plot_capacity_line(axes[0, 3], ward_capacity, color='blue', label_prefix='Capacity')
    axes[0, 3].legend(fontsize=8)
    
    # ═══════════════════════════════════════════════════════════════
    # Row 2: Remaining compartments (H_icu, R, D, Combined)
    # ═══════════════════════════════════════════════════════════════
    
    # H_icu - ICU
    _plot_by_age(axes[1, 0], times, results['H_icu'], age_labels,
                 'ICU Patients', 'H_icu: ICU by Age', colors=colors, linewidth=2)
    axes[1, 0].plot(times, results['H_icu_total'], 'k--', linewidth=2, alpha=0.7, label='Total')
    _plot_capacity_line(axes[1, 0], icu_capacity, color='red', label_prefix='Capacity')
    axes[1, 0].legend(fontsize=8)
    
    # R - Recovered
    _plot_by_age(axes[1, 1], times, results['R'], age_labels,
                 'Recovered', 'R: Recovered by Age', colors=colors, linewidth=2)
    R_total = [sum(results['R'][a][t] for a in range(n_ages)) for t in range(len(times))]
    axes[1, 1].plot(times, R_total, 'k--', linewidth=2, alpha=0.7, label='Total')
    axes[1, 1].legend(fontsize=8)
    
    # D - Deaths (cumulative)
    _plot_by_age(axes[1, 2], times, results['D'], age_labels,
                 'Cumulative Deaths', 'D: Deaths by Age', colors=colors, linewidth=2)
    D_total = [sum(results['D'][a][t] for a in range(n_ages)) for t in range(len(times))]
    axes[1, 2].plot(times, D_total, 'k--', linewidth=2, alpha=0.7, label='Total')
    axes[1, 2].legend(fontsize=8)
    
    # Combined hospital burden (stacked area)
    axes[1, 3].stackplot(times, results['H_ward_total'], results['H_icu_total'],
                         labels=['Ward', 'ICU'], colors=['#3498db', '#e74c3c'], alpha=0.7)
    _plot_capacity_line(axes[1, 3], ward_capacity + icu_capacity, color='purple',
                        label_prefix='Total Capacity')
    _setup_axis(axes[1, 3], 'Time (days)', 'Patients', 'Combined Hospital Burden')
    
    # ═══════════════════════════════════════════════════════════════
    # Row 3: Performance metrics
    # ═══════════════════════════════════════════════════════════════
    
    # Overflow (ward + ICU)
    axes[2, 0].plot(times, results['ward_overflow'], label='Ward Overflow',
                    color='#3498db', linewidth=2)
    axes[2, 0].plot(times, results['icu_overflow'], label='ICU Overflow',
                    color='#e74c3c', linewidth=2)
    axes[2, 0].fill_between(times, results['ward_overflow'], alpha=0.3, color='#3498db')
    axes[2, 0].fill_between(times, results['icu_overflow'], alpha=0.3, color='#e74c3c')
    _setup_axis(axes[2, 0], 'Time (days)', 'Overflow (patients)', 'Capacity Overflow')
    
    # Differential mortality (if available)
    if 'D_treated' in results and 'D_untreated' in results:
        D_treated_total = [sum(results['D_treated'][a][t] for a in range(n_ages)) 
                          for t in range(len(times))]
        D_untreated_total = [sum(results['D_untreated'][a][t] for a in range(n_ages)) 
                            for t in range(len(times))]
        axes[2, 1].stackplot(times, D_treated_total, D_untreated_total,
                             labels=['Treated (baseline)', 'Untreated (preventable)'],
                             colors=['#3498db', '#e74c3c'], alpha=0.7)
        axes[2, 1].plot(times, D_total, 'k--', linewidth=2, label='Total')
        _setup_axis(axes[2, 1], 'Time (days)', 'Cumulative Deaths', 'Differential Mortality')
    else:
        # Fallback: show death rate over time
        death_rate = np.gradient(D_total, times)
        axes[2, 1].plot(times, death_rate, color='#e74c3c', linewidth=2)
        axes[2, 1].fill_between(times, death_rate, alpha=0.3, color='#e74c3c')
        _setup_axis(axes[2, 1], 'Time (days)', 'Deaths per day', 'Death Rate')
    
    # Deaths breakdown bar chart
    deaths_final, total_deaths = _compute_deaths_summary(results, n_ages)
    x = np.arange(n_ages)
    bars = axes[2, 2].bar(x, deaths_final, 0.6, color=colors[:n_ages], edgecolor='white')
    axes[2, 2].set_xticks(x)
    axes[2, 2].set_xticklabels(age_labels)
    # Add value labels on bars
    for bar, val in zip(bars, deaths_final):
        if val > 0:
            axes[2, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    _setup_axis(axes[2, 2], 'Age Group', 'Deaths', 'Final Deaths by Age',
                legend=False, grid_alpha=config.grid_alpha)
    
    # Unmet care by age (stacked bar)
    ward_unmet = [results['cum_unmet_ward'][a] for a in range(n_ages)]
    icu_unmet = [results['cum_unmet_icu'][a] for a in range(n_ages)]
    axes[2, 3].bar(x - 0.2, ward_unmet, 0.35, label='Ward', color='#3498db', alpha=0.8)
    axes[2, 3].bar(x + 0.2, icu_unmet, 0.35, label='ICU', color='#e74c3c', alpha=0.8)
    axes[2, 3].set_xticks(x)
    axes[2, 3].set_xticklabels(age_labels)
    _setup_axis(axes[2, 3], 'Age Group', 'Patient-days', 'Cumulative Unmet Care')
    
    # ═══════════════════════════════════════════════════════════════
    # Row 4: Summary statistics (spans columns)
    # ═══════════════════════════════════════════════════════════════
    
    # Compute epidemiological metrics
    S_final = sum(results['S'][a][-1] for a in range(n_ages))
    R_final = sum(results['R'][a][-1] for a in range(n_ages))
    infected_total = total_pop - S_final
    attack_rate = (infected_total / total_pop) * 100 if total_pop > 0 else 0
    ifr = (total_deaths / infected_total) * 100 if infected_total > 0 else 0
    
    peak_ward = max(results['H_ward_total'])
    peak_icu = max(results['H_icu_total'])
    peak_I = max(I_total)
    peak_X = max(X_total)
    
    # Calculate R0 proxy (peak timing)
    peak_I_day = times[I_total.index(peak_I)]
    
    # Left summary panel
    summary_left = [
        "EPIDEMIC SUMMARY",
        "━" * 32,
        "",
        f"Population: {total_pop:,.0f}",
        f"Total Infected: {infected_total:,.0f}",
        f"Attack Rate: {attack_rate:.1f}%",
        f"IFR: {ifr:.2f}%",
        "",
        f"Peak Infected: {peak_I:.0f} (day {peak_I_day:.0f})",
        f"Peak Severe: {peak_X:.0f}",
        f"Total Deaths: {total_deaths:.0f}",
        f"Total Recovered: {R_final:.0f}",
    ]
    
    # Add differential mortality if available
    if 'D_treated' in results:
        treated_final = sum(results['D_treated'][a][-1] for a in range(n_ages))
        untreated_final = sum(results['D_untreated'][a][-1] for a in range(n_ages))
        summary_left.extend([
            "",
            "Differential Mortality:",
            f"  Treated: {treated_final:.0f}",
            f"  Untreated: {untreated_final:.0f}",
            f"  Preventable: {untreated_final:.0f}",
        ])
    
    axes[3, 0].axis('off')
    _setup_summary_panel(axes[3, 0], summary_left, fontsize=10)
    
    # Middle summary panel - Hospital metrics
    summary_middle = [
        "HOSPITAL METRICS",
        "━" * 32,
        "",
        f"Ward Capacity: {ward_capacity}",
        f"ICU Capacity: {icu_capacity}",
        "",
        "Peak Occupancy:",
        f"  Ward: {peak_ward:.1f} ({peak_ward/ward_capacity*100:.0f}%)",
        f"  ICU: {peak_icu:.1f} ({peak_icu/icu_capacity*100:.0f}%)",
        "",
        "Cumulative Overflow:",
        f"  Ward: {results['cum_ward_overflow']:.1f} pt-days",
        f"  ICU: {results['cum_icu_overflow']:.1f} pt-days",
        "",
        "Total Unmet Care:",
        f"  Ward: {sum(ward_unmet):.1f} pt-days",
        f"  ICU: {sum(icu_unmet):.1f} pt-days",
    ]
    
    axes[3, 1].axis('off')
    _setup_summary_panel(axes[3, 1], summary_middle, fontsize=10)
    
    # Right summary panel - Age breakdown
    summary_right = [
        "AGE BREAKDOWN",
        "━" * 32,
        "",
        "Deaths by Age:",
    ]
    for a in range(n_ages):
        pct = (deaths_final[a] / total_deaths * 100) if total_deaths > 0 else 0
        summary_right.append(f"  {age_labels[a]}: {deaths_final[a]:.0f} ({pct:.1f}%)")
    
    summary_right.extend([
        "",
        "Attack Rate by Age:",
    ])
    for a in range(n_ages):
        age_pop = results['age_pops'][a]
        age_infected = age_pop - results['S'][a][-1]
        age_ar = (age_infected / age_pop * 100) if age_pop > 0 else 0
        summary_right.append(f"  {age_labels[a]}: {age_ar:.1f}%")
    
    summary_right.extend([
        "",
        "IFR by Age:",
    ])
    for a in range(n_ages):
        age_pop = results['age_pops'][a]
        age_infected = age_pop - results['S'][a][-1]
        age_ifr = (deaths_final[a] / age_infected * 100) if age_infected > 0 else 0
        summary_right.append(f"  {age_labels[a]}: {age_ifr:.2f}%")
    
    axes[3, 2].axis('off')
    _setup_summary_panel(axes[3, 2], summary_right, fontsize=10)
    
    # Validation panel
    validation_lines = [
        "VALIDATION",
        "━" * 32,
        "",
    ]
    
    # Check epidemiological plausibility
    warnings = []
    if attack_rate > 90:
        warnings.append("⚠ Attack rate >90%")
    if ifr > 5:
        warnings.append("⚠ IFR >5% (high)")
    if ifr < 0.1:
        warnings.append("⚠ IFR <0.1% (low)")
    if peak_ward > ward_capacity * 2:
        warnings.append("⚠ Ward overflow severe")
    if peak_icu > icu_capacity * 2:
        warnings.append("⚠ ICU overflow severe")
    
    if warnings:
        validation_lines.append("Warnings:")
        validation_lines.extend([f"  {w}" for w in warnings])
    else:
        validation_lines.append("✓ All metrics plausible")
    
    validation_lines.extend([
        "",
        "Expected ranges:",
        "  Attack rate: 30-70%",
        "  IFR: 0.1-3%",
        "  Peak <2x capacity",
    ])
    
    axes[3, 3].axis('off')
    _setup_summary_panel(axes[3, 3], validation_lines, fontsize=10)
    
    # Add title
    fig.suptitle('SIXHRD Model: Complete Compartment Overview', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    _finalize_figure(fig, config)
    
    # Console summary
    if config.verbose:
        _print_results_header("SIXHRD Full Model Results")
        print(f"Population: {total_pop:,}")
        print(f"Attack rate: {attack_rate:.1f}%")
        print(f"IFR: {ifr:.2f}%")
        print(f"Total deaths: {total_deaths:.0f}")
        print(f"Peak ward: {peak_ward:.1f} / {ward_capacity} ({peak_ward/ward_capacity*100:.0f}%)")
        print(f"Peak ICU: {peak_icu:.1f} / {icu_capacity} ({peak_icu/icu_capacity*100:.0f}%)")
        if warnings:
            print("\nValidation warnings:")
            for w in warnings:
                print(f"  {w}")
    
    return fig, axes