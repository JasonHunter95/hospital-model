# Hospital Capacity SIXHRD Epidemic Model

A comprehensive compartmental epidemic model with hospital capacity constraints, age structure, and ICU separation for analyzing infectious disease dynamics under healthcare system stress.

## Table of Contents

- [Overview](#overview)
- [Model Description](#model-description)
  - [Basic SIXHRD Model](#basic-sixhrd-model)
  - [Age-Structured Extension](#age-structured-extension)
  - [Ward/ICU Separation](#wardicu-separation)
  - [Time-Varying Extensions](#time-varying-extensions)
- [Mathematical Foundations](#mathematical-foundations)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Basic Simulation](#basic-simulation)
  - [Age-Structured Model](#age-structured-model)
  - [Ward/ICU Model](#wardicu-model)
  - [Time-Varying Scenarios](#time-varying-scenarios)
- [Configuration](#configuration)
- [Key Features](#key-features)
- [Visualization](#visualization)
- [Examples](#examples)
- [API Reference](#api-reference)

---

## Overview

This project implements an extended **SIXHRD compartmental model** designed to simulate epidemic dynamics while accounting for real-world healthcare system constraints. Unlike traditional SIR models, this implementation:

1. **Models hospital capacity constraints** using smooth Hill function gating
2. **Separates ward and ICU capacity** with distinct mortality implications
3. **Supports age-structured populations** with heterogeneous contact patterns
4. **Includes time-varying transmission** (seasonality, policy interventions, waning immunity)
5. **Tracks unmet care needs** and overflow burden for policy analysis

The model is designed for academic research and educational purposes to explore:
- Epidemic trajectory under hospital capacity limitations
- Optimal vaccine allocation across age groups
- Policy trade-offs (lockdowns vs. hospital surge capacity)
- Impact of ward/ICU allocation decisions

---

## Model Description

### Basic SIXHRD Model

The core model divides the population into six compartments:

| Compartment | Description |
|-------------|-------------|
| **S** | Susceptible - individuals who can become infected |
| **I** | Infected - mild or early-stage infections |
| **X** | Severe cases - requiring hospitalization |
| **H** | Hospitalized - admitted patients receiving care |
| **R** | Recovered - immune individuals |
| **D** | Dead - disease-related fatalities |

**Flow diagram:**
```
S → I → X → H → R
        ↓   ↓   ↑
        D   D   ↑
            ↑___↑
```

### Age-Structured Extension

The model supports **N age groups** (default: 3) with:
- **Age-specific disease parameters**: Different severity, mortality, and recovery rates
- **Contact matrix**: Heterogeneous mixing patterns between age groups
- **Age-targeted vaccination**: Different coverage levels per age group

Default age groups:
- **Young (0-19 years)**: Low severity, low mortality
- **Middle (20-64 years)**: Moderate severity and mortality
- **Elderly (65+ years)**: High severity, high mortality

### Ward/ICU Separation

The extended model splits hospitalization into two stages:

```
S → I → X → H_ward → H_icu → R or D
```

| Compartment | Description |
|-------------|-------------|
| **H_ward** | General ward beds for standard hospital care |
| **H_icu** | ICU beds for critical care with ventilators |

Key differences:
- **Separate capacity constraints**: Each with independent Hill function gating
- **Different mortality rates**: ICU mortality typically higher (sicker patients)
- **Slower ICU recovery**: Critical care takes longer
- **Escalation flow**: Ward patients may need ICU; ICU is downstream of ward

### Differential Mortality by Care Status

The model tracks deaths separately based on whether patients received hospital care:

| Category | Description |
|----------|-------------|
| **D_treated** | Deaths that occurred while receiving care (ward or ICU) |
| **D_untreated** | Deaths due to being denied admission (overflow deaths) |

**Why this matters:**
- When hospital capacity is constrained, patients in compartment X who need admission may be denied
- These denied patients experience **higher mortality** (mu_X_untreated > mu_X)
- Similarly, ward patients who need ICU but are denied have elevated mortality
- Tracking these separately reveals the **excess mortality burden** from capacity constraints

**Mortality rates:**
```
mu_X_untreated = mu_X × mu_X_untreated_multiplier  (default: 2.0×)
mu_ward_denied_icu = mu_ward × 1.5  (for patients denied ICU escalation)
```

**Key metric - Excess Mortality:**
$$\text{Excess Mortality \%} = \frac{D_{untreated}}{D_{total}} \times 100$$

This represents the percentage of deaths attributable to capacity constraints—i.e., deaths that might have been prevented with unlimited capacity.

### Time-Varying Extensions

The `time_varying_models` module supports:

1. **Seasonal transmission**: 
   ```
   β(t) = β₀ × (1 + A × cos(2π(t - t_peak)/T))
   ```

2. **Policy interventions**: Step functions reducing transmission during lockdowns

3. **Waning immunity**: Flow from R → S at rate ω, enabling reinfection dynamics

---

## Mathematical Foundations

### Force of Infection

For the age-structured model, the force of infection for age group *a* is:

$$\lambda_a = \beta_{eff} \sum_b C_{ab} \frac{I_b + \theta_X X_b + \theta_H H_b}{N_b}$$

Where:
- $\beta_{eff} = \beta \times (1 - \text{coverage}_a \times VE)$ (leaky vaccine model)
- $C_{ab}$ = contact rate from age group *a* to *b*
- $\theta_X, \theta_H$ = relative infectiousness of severe/hospitalized

### Hospital Admission Gating (Hill Function)

Admissions are gated by a smooth Hill function:

$$g(H) = \frac{1}{1 + (H/K)^n}$$

Where:
- $K$ = hospital capacity
- $n$ = Hill coefficient (steepness of constraint)

**Behavior:**
- When $H \ll K$: $g \approx 1$ (unrestricted admissions)
- When $H = K$: $g = 0.5$ (admissions halved)
- When $H \gg K$: $g \rightarrow 0$ (severe restriction)

### ODE System (Basic Model)

$$\begin{aligned}
\frac{dS}{dt} &= -\lambda S \\
\frac{dI}{dt} &= \lambda S - (\gamma_I + \mu_I + \sigma) I \\
\frac{dX}{dt} &= \sigma I - (\gamma_X + \mu_X) X - \text{admit} \\
\frac{dH}{dt} &= \text{admit} - (\gamma_H + \mu_H) H \\
\frac{dR}{dt} &= \gamma_I I + \gamma_X X + \gamma_H H \\
\frac{dD}{dt} &= \mu_I I + \mu_X X + \mu_H H
\end{aligned}$$

Where $\text{admit} = \eta \cdot X \cdot g(H)$

### Numerical Integration

The model uses the **Euler method** with configurable time step (default: 0.1 days):

$$X(t + dt) = X(t) + \frac{dX}{dt} \cdot dt$$

---

## Installation

### Requirements

- Python 3.8+
- NumPy
- Matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/JasonHunter95/hospital-model.git
cd hospital-model

# Install dependencies
pip install numpy matplotlib

# (Optional) Install Jupyter for notebook exploration
pip install jupyter
```

---

## Project Structure

```
hospital-model/
├── config.py                  # Centralized configuration parameters
├── hospital_models.py         # Core simulation functions
├── simulation_helpers.py      # Helper functions for sweeps and comparisons
├── plotting_utils.py          # Visualization functions
├── time_varying_models.py     # Time-varying transmission extensions
├── hospital.ipynb             # Interactive notebook with examples
└── README.md                  # This file
```

### File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | All default parameters, age-specific disease params, contact matrices, vaccination strategies |
| `hospital_models.py` | Core simulation functions: `simulate_hospital_model()`, `simulate_age_structured_model()`, `simulate_age_structured_model_icu()` |
| `simulation_helpers.py` | Reusable analysis functions: `compare_vaccination_strategies()`, `optimize_vaccine_allocation()`, `sweep_icu_capacity()` |
| `plotting_utils.py` | Comprehensive visualization: `plot_hospital_simulation_stats()`, `plot_age_structured_results()`, `plot_age_structured_icu_results()` |
| `time_varying_models.py` | Time-varying extensions: `simulate_age_structured_time_varying()` |

---

## Usage

### Basic Simulation

```python
from hospital_models import simulate_hospital_model
from plotting_utils import plot_hospital_simulation_stats

# Run basic SIXHRD simulation
times, S, I, X, H, R, D, overflow, cum_overflow, cum_unmet, unmet = \
    simulate_hospital_model(
        beta=0.3,           # Transmission rate
        sigma=0.2,          # Progression to severe
        eta=0.3,            # Hospitalization need rate
        gamma_I=0.1,        # Recovery from I
        mu_I=0.01,          # Mortality in I
        gamma_X=0.15,       # Recovery from X
        mu_X=0.05,          # Mortality in X
        gamma_H=0.2,        # Recovery from H
        mu_H=0.02,          # Mortality in H
        theta_X=0.5,        # Relative infectiousness of X
        theta_H=0.3,        # Relative infectiousness of H
        hosp_capacity=100,  # Hospital bed capacity
        hill_coef=4,        # Hill coefficient
        coverage=0.1,       # Vaccine coverage
        VE=0.7,             # Vaccine efficacy
        N=10000,            # Population size
        Tmax=200            # Simulation duration (days)
    )

# Visualize results
plot_hospital_simulation_stats(
    times, S, I, X, H, R, D, overflow, cum_overflow, cum_unmet,
    hosp_capacity=100, N=10000
)
```

### Age-Structured Model

```python
from hospital_models import simulate_age_structured_model
from plotting_utils import plot_age_structured_results
import config

# Run age-structured simulation with config defaults
results = simulate_age_structured_model(
    beta=0.3,
    age_params=config.AGE_PARAMS_DEFAULT,
    contact_matrix=config.CONTACT_MATRIX_DEFAULT,
    hosp_capacity=100,
    hill_coef=4,
    coverage=[0.1, 0.2, 0.7],  # Age-specific vaccination
    VE=0.7,
    age_pops=config.AGE_POPS_DEFAULT
)

# Visualize
plot_age_structured_results(results, hosp_capacity=100,
                           age_labels=config.AGE_LABELS_SHORT)
```

### Ward/ICU Model

```python
from hospital_models import simulate_age_structured_model_icu
from plotting_utils import plot_age_structured_icu_results
import config

# Run ward/ICU separated model
results = simulate_age_structured_model_icu(
    beta=0.4,
    age_params=config.AGE_PARAMS_DEFAULT,
    contact_matrix=config.CONTACT_MATRIX_DEFAULT,
    ward_capacity=80,           # 80 general ward beds
    icu_capacity=20,            # 20 ICU beds
    hill_coef_ward=4,
    hill_coef_icu=4,
    coverage=[0.1, 0.2, 0.7],   # Elderly priority
    VE=0.7,
    age_pops=config.AGE_POPS_DEFAULT
)

# Visualize with comprehensive 3x3 grid
plot_age_structured_icu_results(results)
```

### Differential Mortality Analysis

```python
from hospital_models import simulate_age_structured_model_icu
from plotting_utils import plot_mortality_breakdown, plot_mortality_comparison
import config

# Run simulation with differential mortality tracking (enabled by default)
results = simulate_age_structured_model_icu(
    beta=0.5,
    age_params=config.AGE_PARAMS_DEFAULT,
    contact_matrix=config.CONTACT_MATRIX_DEFAULT,
    ward_capacity=60,           # Constrained capacity to see overflow
    icu_capacity=15,
    hill_coef_ward=4,
    hill_coef_icu=4,
    coverage=[0.1, 0.2, 0.5],
    VE=0.7,
    age_pops=config.AGE_POPS_DEFAULT,
    track_differential_mortality=True  # Default is True
)

# Visualize mortality breakdown by care status
stats = plot_mortality_breakdown(results)
print(f"Excess mortality from capacity: {stats['excess_mortality_pct']:.1f}%")

# Compare scenarios with different capacities
results_baseline = simulate_age_structured_model_icu(..., ward_capacity=60, icu_capacity=15)
results_expanded = simulate_age_structured_model_icu(..., ward_capacity=100, icu_capacity=25)

plot_mortality_comparison(
    [results_baseline, results_expanded],
    labels=['Baseline Capacity', 'Expanded Capacity']
)
```

### Time-Varying Scenarios

```python
from time_varying_models import simulate_age_structured_time_varying
from plotting_utils import plot_time_varying_results
import config

# Seasonal transmission with lockdown
results = simulate_age_structured_time_varying(
    beta_base=0.35,
    age_params=config.AGE_PARAMS_DEFAULT,
    contact_matrix=config.CONTACT_MATRIX_DEFAULT,
    hosp_capacity=100,
    hill_coef=4,
    coverage=[0.1, 0.3, 0.6],
    VE=0.7,
    age_pops=config.AGE_POPS_DEFAULT,
    seasonal_params={'amplitude': 0.3, 'period': 365, 'peak_day': 0},
    waning_params={'omega': 0.003},  # ~333 day immunity
    interventions=[
        {'start_day': 60, 'end_day': 100, 'transmission_reduction': 0.5}
    ],
    Tmax=400
)

plot_time_varying_results(results, hosp_capacity=100)
```

---

## Configuration

### Default Parameters (`config.py`)

#### Simulation Parameters
```python
DEFAULT_SIM_PARAMS = {
    'Tmax': 200,          # Simulation duration (days)
    'time_step': 0.1,     # Euler step size
    'hill_coef': 4,       # Hill coefficient
    'theta_X': 0.5,       # Relative infectiousness of X
    'theta_H': 0.3,       # Relative infectiousness of H
    'VE': 0.7             # Vaccine efficacy
}
```

#### Capacity Parameters
```python
DEFAULT_CAPACITY_PARAMS = {
    'ward_capacity': 80,
    'icu_capacity': 20,
    'total_capacity': 100
}
```

#### Differential Mortality Parameters
```python
DIFFERENTIAL_MORTALITY_PARAMS = {
    'mu_X_untreated_multiplier': 2.0,  # Mortality multiplier when care denied
    'mu_X_untreated_young': None,      # Age-specific overrides (None = use multiplier)
    'mu_X_untreated_middle': None,
    'mu_X_untreated_elderly': None,
    'track_mortality_source': True     # Enable D_treated/D_untreated tracking
}
```

#### Age-Specific Disease Parameters

| Parameter | Young (0-19) | Middle (20-64) | Elderly (65+) |
|-----------|--------------|----------------|---------------|
| `sigma` (→ severe) | 0.1 | 0.2 | 0.3 |
| `eta` (→ ward) | 0.2 | 0.3 | 0.5 |
| `eta_icu` (→ ICU) | 0.05 | 0.15 | 0.3 |
| `mu_ward` | 0.003 | 0.01 | 0.04 |
| `mu_icu` | 0.02 | 0.06 | 0.20 |

#### Contact Matrices

Three predefined contact matrices:
- `CONTACT_MATRIX_DEFAULT`: Assortative mixing (age groups prefer same-age contacts)
- `CONTACT_MATRIX_HOMOGENEOUS`: Equal mixing across all ages
- `CONTACT_MATRIX_ASSORTATIVE`: Strong within-age-group preference

#### Vaccination Strategies

```python
VACCINATION_STRATEGIES = {
    'No vaccination': [0.0, 0.0, 0.0],
    'Uniform 30%': [0.3, 0.3, 0.3],
    'Elderly priority': [0.1, 0.2, 0.7],
    'Young priority': [0.7, 0.2, 0.1],
    'Middle priority': [0.1, 0.7, 0.2]
}
```

---

## Key Features

### Hospital Capacity Constraints

- **Soft capacity constraint**: Hill function provides smooth transition rather than hard cutoff
- **Overflow tracking**: Measures patients exceeding capacity at each time step
- **Cumulative overflow**: Patient-days of overflow burden (integral over time)
- **Unmet care tracking**: Care needed but not delivered due to capacity

### Age Structure

- **Heterogeneous disease severity**: Young have lower progression and mortality
- **Contact matrices**: Model realistic social mixing patterns
- **Age-targeted interventions**: Different vaccine coverage by age group

### Ward/ICU Separation

- **Two-stage hospitalization**: Ward → ICU progression
- **Independent capacity gating**: Each level has own Hill function
- **Differential mortality**: ICU patients typically sicker, higher baseline mortality
- **Escalation tracking**: Monitors ward patients needing ICU transfer

### Policy Analysis Tools

- **Strategy comparison**: Compare multiple vaccination approaches
- **Optimal allocation**: Grid search for best vaccine distribution
- **ICU capacity sweep**: Find optimal ward/ICU bed allocation
- **Scenario comparison**: Test different capacity configurations

---

## Visualization

The `plotting_utils` module provides comprehensive visualizations:

| Function | Description |
|----------|-------------|
| `plot_hospital_simulation_stats()` | 2×2 grid for basic model |
| `plot_hospital_icu_stats()` | 2×3 grid for ward/ICU model |
| `plot_age_structured_results()` | 2×3 grid for age-structured model |
| `plot_age_structured_icu_results()` | 3×3 comprehensive grid for age+ICU |
| `plot_strategy_comparison()` | Bar charts comparing vaccination strategies |
| `plot_optimal_allocation()` | Heatmaps for vaccine allocation optimization |
| `plot_time_varying_results()` | Time-varying transmission visualization |
| `plot_icu_capacity_sweep()` | ICU allocation optimization curves |
| `plot_mortality_breakdown()` | Treated vs untreated deaths analysis |
| `plot_mortality_comparison()` | Compare mortality across scenarios |

---

## Examples

### Example 1: Comparing Vaccination Strategies

```python
from simulation_helpers import compare_vaccination_strategies
from plotting_utils import plot_strategy_comparison
import config

results = compare_vaccination_strategies(
    beta=0.3,
    age_params=config.AGE_PARAMS_DEFAULT,
    contact_matrix=config.CONTACT_MATRIX_DEFAULT,
    hosp_capacity=100,
    age_pops=config.AGE_POPS_DEFAULT,
    strategies=config.VACCINATION_STRATEGIES
)

plot_strategy_comparison(results, hosp_capacity=100)
```

### Example 2: Optimal Vaccine Allocation

```python
from simulation_helpers import optimize_vaccine_allocation
from plotting_utils import plot_optimal_allocation
import config

deaths_grid, overflow_grid, young_range, middle_range = optimize_vaccine_allocation(
    beta=0.3,
    age_params=config.AGE_PARAMS_DEFAULT,
    contact_matrix=config.CONTACT_MATRIX_DEFAULT,
    hosp_capacity=100,
    age_pops=config.AGE_POPS_DEFAULT,
    total_coverage_target=0.3,  # 30% of population
    n_grid=20
)

opt_young, opt_middle, opt_elderly = plot_optimal_allocation(
    deaths_grid, overflow_grid, young_range, middle_range,
    age_pops=config.AGE_POPS_DEFAULT,
    total_doses=0.3 * sum(config.AGE_POPS_DEFAULT)
)
```

### Example 3: Finding Optimal Ward/ICU Split

```python
from simulation_helpers import sweep_icu_capacity
from plotting_utils import plot_icu_capacity_sweep
import config

sweep_results = sweep_icu_capacity(
    beta=0.4,
    age_params=config.AGE_PARAMS_DEFAULT,
    contact_matrix=config.CONTACT_MATRIX_DEFAULT,
    age_pops=config.AGE_POPS_DEFAULT,
    total_beds=100,
    coverage=[0.1, 0.2, 0.7],
    n_points=20
)

plot_icu_capacity_sweep(sweep_results)
# Output: Optimal ICU fraction and minimum achievable deaths
```

---

## API Reference

### Core Simulation Functions

#### `simulate_hospital_model()`
Basic SIXHRD model with single hospital compartment.

#### `simulate_age_structured_model()`
Age-structured model with shared hospital capacity.

#### `simulate_age_structured_model_icu()`
Full age-structured model with ward/ICU separation.

#### `simulate_age_structured_time_varying()`
Age-structured model with seasonal, policy, and waning dynamics.

### Helper Functions

#### `compare_vaccination_strategies()`
Run multiple vaccination scenarios and compile results.

#### `compare_vaccination_strategies_icu()`
Compare strategies using ward/ICU model.

#### `optimize_vaccine_allocation()`
Grid search for optimal age-specific vaccine distribution.

#### `compare_capacity_scenarios()`
Compare different ward/ICU capacity allocations.

#### `sweep_icu_capacity()`
Sweep ICU fraction to find optimal allocation.

### Utility Functions

#### `hill_gate(occupancy, capacity, hill_coef)`
Calculate Hill function gating factor.

#### `seasonal_forcing(t, beta_base, amplitude, period, peak_day)`
Calculate time-varying transmission with seasonality.

#### `policy_multiplier(t, interventions)`
Calculate transmission multiplier from active interventions.

---

## Output Interpretation

### Key Metrics

| Metric | Description | Units |
|--------|-------------|-------|
| **Peak H** | Maximum hospital occupancy | Patients |
| **Peak Ward/ICU** | Maximum ward and ICU occupancy | Patients |
| **Total Deaths** | Final cumulative deaths | Persons |
| **D_treated** | Deaths with hospital care received | Persons |
| **D_untreated** | Deaths without care (overflow) | Persons |
| **Excess Mortality %** | Proportion of deaths from capacity constraints | Percentage |
| **Cumulative Overflow** | Total patient-days exceeding capacity | Patient-days |
| **Cumulative Unmet Care** | Care needed but not provided | Patient-days |
| **Attack Rate** | Fraction of population infected | Percentage |

### Understanding Overflow

- **Overflow = max(0, H - K)**: Patients exceeding capacity at each time point
- **Cumulative overflow = ∫ overflow dt**: Total burden over simulation
- High cumulative overflow indicates sustained capacity crisis
- Ward and ICU overflow have different mortality implications

---

## Limitations

1. **Euler method**: Simple but can be unstable for very large time steps
2. **Deterministic model**: No stochastic variation (suitable for large populations)
3. **No spatial structure**: Assumes well-mixed population within age groups
4. **Simplified vaccine model**: Leaky model only; no all-or-nothing protection
5. **No healthcare worker dynamics**: Ignores staff shortages during surges

---

## Future Extensions

- Stochastic simulation using Gillespie algorithm
- Spatial/metapopulation structure
- Multiple pathogen strains
- Healthcare worker compartments
- Economic cost modeling
- Hospitalization duration distributions
---
