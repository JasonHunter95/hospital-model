# Hospital Capacity SEIXHRD Epidemic Model

A comprehensive compartmental epidemic model with hospital capacity constraints, age structure, an exposed (latent) compartment, and ICU separation for analyzing infectious disease dynamics under healthcare system stress.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Model Description](#model-description)
  - [Basic SEIXHRD Model](#basic-seixhrd-model)
  - [Age-Structured Extension](#age-structured-extension)
  - [Ward/ICU Separation](#wardicu-separation)
  - [Master Model](#master-model)
  - [Time-Varying Extensions](#time-varying-extensions)
- [Mathematical Foundations](#mathematical-foundations)
  - [Force of Infection](#force-of-infection)
  - [Hill Function Gating](#hill-function-gating)
  - [Complete ODE System](#complete-ode-system)
  - [Differential Mortality](#differential-mortality-equations)
  - [Time-Varying Transmission](#time-varying-transmission-equations)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Basic Simulation](#basic-simulation)
  - [Age-Structured Model](#age-structured-model)
  - [Ward/ICU Model](#wardicu-model-usage)
  - [Master Model](#master-model-usage)
  - [Time-Varying Scenarios](#time-varying-scenarios)
- [Configuration](#configuration)
- [Key Features](#key-features)
- [Visualization](#visualization)
- [Examples](#examples)
- [API Reference](#api-reference)

---

## Overview

This project implements an extended **SEIXHRD compartmental model** designed to simulate epidemic dynamics while accounting for real-world healthcare system constraints. Unlike traditional SIR models, this implementation:

1. **Includes an Exposed (E) compartment** for realistic latent period dynamics
2. **Models hospital capacity constraints** using smooth Hill function gating
3. **Separates ward and ICU capacity** with distinct mortality implications
4. **Supports age-structured populations** with heterogeneous contact patterns
5. **Includes time-varying transmission** (seasonality, policy interventions, waning immunity)
6. **Tracks unmet care needs** and overflow burden for policy analysis

The model is designed for academic research and educational purposes to explore:

- Epidemic trajectory under hospital capacity limitations
- Optimal vaccine allocation across age groups
- Policy trade-offs (lockdowns vs. hospital surge capacity)
- Impact of ward/ICU allocation decisions

---

## Model Description

### Basic SEIXHRD Model

The core model divides the population into seven compartments:

| Compartment | Description |
|-------------|-------------|
| **S** | Susceptible - individuals who can become infected |
| **E** | Exposed - infected but in latent period (not yet infectious) |
| **I** | Infected - mild or early-stage infections (infectious) |
| **X** | Severe cases - requiring hospitalization |
| **H** | Hospitalized - admitted patients receiving care |
| **R** | Recovered - immune individuals |
| **D** | Dead - disease-related fatalities |

**Flow diagram:**

```text
S → E → I → X → H → R
            ↓   ↓   ↑
            D   D   ↑
                ↑___↑
```

The **Exposed (E) compartment** represents the latent period where individuals are infected but not yet infectious. This is controlled by the parameter $\alpha$ (E→I rate), where the mean latent period is $1/\alpha$ days.

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

```text
S → E → I → X → H_ward → H_icu → R or D
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

### Master Model

The **Master Model** (`master_hospital_model.py`) unifies all features into a single comprehensive simulation:

```python
simulate_master_hospital_model(
    beta_base,              # Baseline transmission (modified by seasonality/policy)
    age_params,             # Age-specific disease parameters
    contact_matrix,         # Contact rates between age groups
    ward_capacity,          # General ward capacity
    icu_capacity,           # ICU capacity
    coverage,               # Age-specific vaccination coverage
    VE,                     # Vaccine efficacy
    age_pops,               # Population by age group
    seasonal_params,        # Seasonal transmission parameters
    waning_params,          # Immunity waning rates
    interventions,          # Policy interventions (lockdowns)
    track_differential_mortality,  # Track treated vs untreated deaths
    track_compartment_flows        # Track daily transition flows
)
```

**Features combined in the Master Model:**

- ✅ Age-structured compartments (S, E, I, X, H_ward, H_icu, R, D)
- ✅ Exposed (E) compartment with age-specific latent periods
- ✅ Separate ward and ICU with independent Hill gating
- ✅ Differential mortality tracking (D_treated vs D_untreated)
- ✅ Seasonal forcing of transmission
- ✅ Policy interventions with configurable timing
- ✅ Age-specific waning immunity
- ✅ Vaccination with age-targeted coverage

### Differential Mortality by Care Status

The model tracks deaths separately based on whether patients received hospital care:

| Category | Description |
|----------|-------------|
| **D_treated** | Deaths that occurred while receiving care (ward or ICU) |
| **D_untreated** | Deaths due to being denied admission (overflow deaths) |

**Why this matters:**

- When hospital capacity is constrained, patients in compartment X who need admission may be denied
- These denied patients experience **higher mortality** (μ_X_untreated > μ_X)
- Similarly, ward patients who need ICU but are denied have elevated mortality
- Tracking these separately reveals the **excess mortality burden** from capacity constraints

**Key metric - Preventable Mortality:**

$$\text{Preventable Mortality \%} = \frac{D_{\text{untreated}}}{D_{\text{total}}} \times 100$$

This represents the percentage of deaths attributable to capacity constraints—i.e., deaths that might have been prevented with unlimited capacity.

### Time-Varying Extensions

The `time_varying_models` module supports:

1. **Seasonal transmission**: Periodic variation in transmission rate
2. **Policy interventions**: Step functions reducing transmission during lockdowns
3. **Waning immunity**: Flow from R → S at rate ω, enabling reinfection dynamics

---

## Mathematical Foundations

### Force of Infection

For the age-structured model, the force of infection for age group $a$ is:

$$\lambda_a = \beta_{\text{eff}} \sum_{b} C_{ab} \frac{I_b + \theta_X X_b + \theta_H (H_{\text{ward},b} + H_{\text{icu},b})}{N_b}$$

Where:

- $\beta_{\text{eff}} = \beta \times (1 - \text{coverage}_a \times VE)$ — leaky vaccine model
- $C_{ab}$ — contact rate from age group $a$ to $b$
- $\theta_X, \theta_H$ — relative infectiousness of severe/hospitalized compartments
- $N_b$ — living population in age group $b$

### Hill Function Gating

Hospital admissions are gated by a smooth Hill function that models capacity constraints:

$$g(H) = \frac{1}{1 + \left(\frac{H}{K}\right)^n}$$

Where:

- $K$ — hospital capacity (ward or ICU)
- $n$ — Hill coefficient controlling steepness of constraint
- $H$ — current occupancy

**Behavioral properties:**

| Condition | Gating Factor | Interpretation |
|-----------|---------------|----------------|
| $H \ll K$ | $g \approx 1$ | Unrestricted admissions |
| $H = K$ | $g = 0.5$ | Admissions halved |
| $H \gg K$ | $g \to 0$ | Severe admission restriction |

The Hill coefficient $n$ controls how sharply admissions decline as capacity is approached. Higher values create a sharper transition.

### Complete ODE System

#### Basic SEIXHRD Hospital Model

$$\frac{dS}{dt} = -\lambda S$$

$$\frac{dE}{dt} = \lambda S - \alpha E$$

$$\frac{dI}{dt} = \alpha E - (\gamma_I + \mu_I + \sigma) I$$

$$\frac{dX}{dt} = \sigma I - (\gamma_X + \mu_X) X - \eta \cdot X \cdot g(H)$$

$$\frac{dH}{dt} = \eta \cdot X \cdot g(H) - (\gamma_H + \mu_H) H$$

$$\frac{dR}{dt} = \gamma_I I + \gamma_X X + \gamma_H H$$

$$\frac{dD}{dt} = \mu_I I + \mu_X X + \mu_H H$$

Where $\alpha$ is the rate of progression from Exposed to Infectious (1/latent period).

#### Master Model with Ward/ICU Split

For each age group $a$:

**Susceptible:**
$$\frac{dS_a}{dt} = -\lambda_a S_a + \omega_a R_a$$

**Exposed (latent period):**
$$\frac{dE_a}{dt} = \lambda_a S_a - \alpha_a E_a$$

**Infected (mild):**
$$\frac{dI_a}{dt} = \alpha_a E_a - (\gamma_{I,a} + \mu_{I,a} + \sigma_a) I_a$$

**Severe (needs hospitalization):**
$$\frac{dX_a}{dt} = \sigma_a I_a - (\gamma_{X,a} + \mu_{X,\text{eff}}) X_a - \text{admit}_{\text{ward},a}$$

**General Ward:**
$$\frac{dH_{\text{ward},a}}{dt} = \text{admit}_{\text{ward},a} - (\gamma_{\text{ward},a} + \mu_{\text{ward},\text{eff}}) H_{\text{ward},a} - \text{admit}_{\text{icu},a}$$

**ICU:**
$$\frac{dH_{\text{icu},a}}{dt} = \text{admit}_{\text{icu},a} - (\gamma_{\text{icu},a} + \mu_{\text{icu},a}) H_{\text{icu},a}$$

**Recovered:**
$$\frac{dR_a}{dt} = \gamma_{I,a} I_a + \gamma_{X,a} X_a + \gamma_{\text{ward},a} H_{\text{ward},a} + \gamma_{\text{icu},a} H_{\text{icu},a} - \omega_a R_a$$

**Deaths:**
$$\frac{dD_a}{dt} = \mu_{I,a} I_a + \mu_{X,\text{eff}} X_a + \mu_{\text{ward},\text{eff}} H_{\text{ward},a} + \mu_{\text{icu},a} H_{\text{icu},a}$$

Where the admission flows are:

$$\text{admit}_{\text{ward},a} = \eta_a \cdot X_a \cdot g_{\text{ward}}(H_{\text{ward,total}})$$

$$\text{admit}_{\text{icu},a} = \eta_{\text{icu},a} \cdot H_{\text{ward},a} \cdot g_{\text{icu}}(H_{\text{icu,total}})$$

### Differential Mortality Equations

The effective mortality rates account for capacity-constrained care:

**X compartment (severe cases):**
$$\mu_{X,\text{eff}} = \mu_X \cdot g_{\text{ward}} + \mu_{X,\text{untreated}} \cdot (1 - g_{\text{ward}})$$

Where:
$$\mu_{X,\text{untreated}} = \mu_X \times m_{\text{X,untreated}}$$

The multiplier $m_{\text{X,untreated}}$ is age-specific:

| Age Group | Multiplier | Interpretation |
|-----------|------------|----------------|
| Young | 1.5× | Better compensatory reserve |
| Middle | 2.0× | Baseline |
| Elderly | 3.0× | Most vulnerable without care |

**Ward patients denied ICU:**
$$\mu_{\text{ward,eff}} = \mu_{\text{ward}} + (\mu_{\text{ward,denied}} - \mu_{\text{ward}}) \cdot \eta_{\text{icu}} \cdot (1 - g_{\text{icu}})$$

**Tracking treated vs untreated deaths:**

$$\frac{dD_{\text{treated}}}{dt} = \mu_I I + \mu_X \cdot g_{\text{ward}} \cdot X + \mu_{\text{ward}} H_{\text{ward}} + \mu_{\text{icu}} H_{\text{icu}}$$

$$\frac{dD_{\text{untreated}}}{dt} = \mu_{X,\text{untreated}} \cdot (1 - g_{\text{ward}}) \cdot X + (\mu_{\text{ward,denied}} - \mu_{\text{ward}}) \cdot \eta_{\text{icu}} \cdot (1 - g_{\text{icu}}) \cdot H_{\text{ward}}$$

### Time-Varying Transmission Equations

**Seasonal forcing:**
$$\beta(t) = \beta_0 \left(1 + A \cos\left(\frac{2\pi(t - t_{\text{peak}})}{T}\right)\right)$$

Where:

- $\beta_0$ — baseline transmission rate
- $A$ — seasonal amplitude (0 to 1)
- $T$ — period (typically 365 days)
- $t_{\text{peak}}$ — day of peak transmission

**Policy interventions:**
$$\beta_{\text{eff}}(t) = \beta(t) \times (1 - r_{\text{intervention}})$$

Where $r_{\text{intervention}}$ is the transmission reduction during active interventions (e.g., 0.5 for 50% reduction).

**Waning immunity:**

The recovered compartment loses immunity at rate $\omega$:
$$\frac{dR_a}{dt} = \text{(recoveries)} - \omega_a R_a$$
$$\frac{dS_a}{dt} = -\lambda_a S_a + \omega_a R_a$$

Age-specific waning rates allow modeling of differential immune durability across age groups.

### Numerical Integration

The model uses the **Euler method** with configurable time step (default: $\Delta t = 0.1$ days):

$$X(t + \Delta t) = X(t) + \frac{dX}{dt} \cdot \Delta t$$

All compartments are constrained to be non-negative after each update.

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

```text
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
from hospital_models import simulate_basic_hospital_model
from plotting_utils import plot_hospital_simulation_stats

# Run basic SIXHRD simulation
times, S, I, X, H, R, D, overflow, cum_overflow, cum_unmet, unmet = \
    simulate_basic_hospital_model(
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
from hospital_models import simulate_age_structured_hospital_model
from plotting_utils import plot_age_structured_results
import config

# Run age-structured simulation with config defaults
results = simulate_age_structured_hospital_model(
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

### Ward/ICU Model Usage

```python
from hospital_models import simulate_age_structured_hospital_model_with_icu_ward_split
from plotting_utils import plot_age_structured_icu_results
import config

# Run ward/ICU separated model
results = simulate_age_structured_hospital_model_with_icu_ward_split(
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

### Master Model Usage

The Master Model combines all features for comprehensive pandemic simulations:

```python
from master_experiments.master_hospital_model import simulate_master_hospital_model
from plotting_utils import plot_age_structured_icu_results
import config

# Full-featured simulation with all extensions
results = simulate_master_hospital_model(
    beta_base=0.25,
    age_params=config.AGE_PARAMS_DEFAULT,
    contact_matrix=config.CONTACT_MATRIX_DEFAULT,
    
    # Hospital capacity
    ward_capacity=1600,
    icu_capacity=400,
    
    # Vaccination
    coverage=[0.3, 0.5, 0.8],  # Age-targeted
    VE=0.7,
    
    # Population
    age_pops=config.AGE_POPS_REGIONAL_DEFAULT,  # 200,000 total
    
    # Seasonal transmission
    seasonal_params={
        'amplitude': 0.25,
        'period': 365,
        'peak_day': 0  # Winter peak
    },
    
    # Waning immunity (age-specific)
    waning_params={
        'omega_young': 0.002,
        'omega_middle': 0.003,
        'omega_elderly': 0.005
    },
    
    # Policy interventions
    interventions=[
        {'start_day': 30, 'end_day': 75, 'transmission_reduction': 0.5},
        {'start_day': 150, 'end_day': 200, 'transmission_reduction': 0.3}
    ],
    
    Tmax=730,  # 2 years
    track_differential_mortality=True,
    track_compartment_flows=True
)

# Access comprehensive results
print(f"Total deaths: {sum(results['D'][a][-1] for a in range(3)):.0f}")
print(f"Preventable deaths: {results['D_untreated_total'][-1]:.0f}")
print(f"Peak ICU: {max(results['H_icu_total']):.0f}")
```

**Key output fields from Master Model:**

| Field | Description |
|-------|-------------|
| `S`, `E`, `I`, `X`, `R`, `D` | Compartments by age (list of arrays) |
| `H_ward`, `H_icu` | Hospital compartments by age |
| `H_ward_total`, `H_icu_total` | Aggregated hospital occupancy |
| `D_treated`, `D_untreated` | Deaths by care status |
| `beta_t`, `seasonal_factor`, `policy_mult` | Time-varying parameters |
| `g_ward`, `g_icu` | Admission gating factors over time |
| `cum_ward_overflow`, `cum_icu_overflow` | Cumulative overflow burden |

### Time-Varying Scenarios

```python
from time_varying_models import simulate_age_structured_hospital_model_with_time_variance
from plotting_utils import plot_time_varying_results
import config

# Seasonal transmission with lockdown
results = simulate_age_structured_hospital_model_with_time_variance(
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
    'mu_X_untreated_multiplier': 2.0,            # Multiplier when hospital admission is denied
    'mu_ward_denied_icu_multiplier': 1.5,        # Multiplier when ICU escalation is denied
    'mu_X_untreated_multiplier_young': 1.5,      # Age-specific overrides
    'mu_X_untreated_multiplier_middle': 2.0,
    'mu_X_untreated_multiplier_elderly': 3.0,
    'mu_ward_denied_icu_multiplier_young': 1.3,
    'mu_ward_denied_icu_multiplier_middle': 1.5,
    'mu_ward_denied_icu_multiplier_elderly': 2.0
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

### Ward versus ICU Separation

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
| `plot_age_structured_results()` | 2×3 grid for age-structured model |
| `plot_age_structured_icu_results()` | 3×3 comprehensive grid for age+ICU |
| `plot_strategy_comparison()` | Bar charts comparing vaccination strategies |
| `plot_optimal_allocation()` | Heatmaps for vaccine allocation optimization |
| `plot_time_varying_results()` | Time-varying transmission visualization |

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

print(f"Optimal ICU fraction: {sweep_results['optimal_icu_fraction']:.1%}")
print(f"Minimum deaths: {sweep_results['optimal_deaths']:.0f}")
# sweep_results contains arrays you can plot externally (e.g., matplotlib contour/line plots)
```

---

## API Reference

### Core Simulation Functions

#### `simulate_basic_hospital_model()`

Basic SIXHRD model with single hospital compartment.

#### `simulate_age_structured_hospital_model()`

Age-structured model with shared hospital capacity.

#### `simulate_age_structured_hospital_model_with_icu_ward_split()`

Full age-structured model with ward/ICU separation.

#### `simulate_age_structured_hospital_model_with_time_variance()`

Age-structured model with seasonal, policy, and waning dynamics.

#### `simulate_master_hospital_model()`

Unified model combining all features: ward/ICU split, seasonality, interventions, waning immunity, and differential mortality tracking.

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

## Citation

If you use this model in your research, please cite:

```bibtex
@software{hospital_sixhrd_model,
  author = {Hunter, Jason},
  title = {Hospital Capacity SIXHRD Epidemic Model},
  year = {2025},
  url = {https://github.com/JasonHunter95/hospital-model}
}
```

---

## License

MIT License - see LICENSE file for details.
