# Hospital Capacity SEIXHRD Epidemic Model

An attempt at a comprehensive, age-structured compartmental epidemic model designed for analyzing infectious disease dynamics under healthcare system stress. This model features a split-stage severe disease progression (Queued vs. Admitted), distinct Ward and ICU capacity constraints, and a Three-Factor vaccination model, support for seasonality and policy interventions (lockdowns), and support for birth and death processes. Most of these features are optional, so that the features can be toggled and tested in isolation, or combined.

## Scientific Overview

The **SEIXHRD model** extends the traditional SEIR framework to explicitly model the interaction between epidemic dynamics and healthcare capacity. It is designed to answer critical policy questions regarding hospital surge capacity, vaccine allocation, and non-pharmaceutical interventions.

Key differentiators from standard models:
*   **Explicit Capacity Constraints**: Models the "tipping point" where hospital capacity is exceeded, leading to increased mortality.
*   **Split-X Architecture**: Separates severe cases into `X_queued` (waiting for care) and `X_admitted` (receiving care), allowing for rigorous tracking of unmet care needs.
*   **Differential Mortality**: Mechanistically calculates excess deaths due to lack of ward or ICU beds.
*   **Three-Factor Vaccination**: Models vaccine efficacy against infection ($VE_{I}$), severe disease ($VE_{S}$), and death ($VE_{D}$) separately.

## Key Features

*   **Compartmental Structure**: `S` (Susceptible) $\to$ `E` (Exposed) $\to$ `I` (Infected) $\to$ `X` (Severe) $\to$ `H` (Hospitalized) $\to$ `R` (Recovered) or `D` (Dead).
*   **Age Structure**: Supports $N$ age groups with configurable contact matrices and age-specific disease parameters.
*   **Hospital Dynamics**:
    *   **Ward vs. ICU**: Separate capacity constraints and gating functions.
    *   **Hill Function Gating**: Smooth capacity constraints ($g(H) = \frac{1}{1 + (H/K)^n}$) rather than hard cutoffs.
*   **Time-Varying Parameters**: Supports seasonality, policy interventions (lockdowns), and waning immunity.
*   **Demographics**: Optional open population dynamics with births and background mortality for long-term endemic simulations.

## Installation

### Prerequisites
*   Python 3.8 or higher
*   `pip` package manager

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run a simulation using model with a pre-defined scenario:

```python
from simulate_model import simulate_model
from scenario_helpers import get_scenario_params

# Load the 'covid_delta' scenario
params = get_scenario_params('covid_delta')

# Run the simulation
results = simulate_model(**params)

# Access results
print(f"Total Deaths: {results['D_total'][-1]:.0f}")
print(f"Peak ICU Occupancy: {max(results['H_icu_total']):.0f}")
print(f"Preventable Deaths (Untreated): {results['D_untreated_total'][-1]:.0f}")
```

## Model Architecture

### Compartmental Flow
The model tracks individuals through the following states for each age group:

1.  **Susceptible ($S$)**: Vulnerable to infection.
2.  **Exposed ($E$)**: Infected but not yet infectious (latent period).
3.  **Infected ($I$)**: Infectious, mild symptoms.
4.  **Severe ($X$)**: Requires hospitalization.
    *   **$X_{queued}$**: Waiting for admission. Subject to *untreated* mortality.
    *   **$X_{admitted}$**: Admitted to stabilization. Subject to *treated* mortality.
5.  **Hospitalized ($H$)**:
    *   **$H_{ward}$**: General ward care.
    *   **$H_{icu}$**: Intensive care.
6.  **Recovered ($R$)**: Immune (temporarily or permanently).
7.  **Dead ($D$)**: Disease-related fatalities.

### Vaccination Model
The **Three-Factor Vaccine Model** runs a parallel set of compartments ($S_{vax}$, $E_{vax}$, $\dots$) for vaccinated individuals.

*   **$VE_{infection}$**: Reduces probability of $S_{vax} \to E_{vax}$.
*   **$VE_{severe}$**: Reduces probability of $I_{vax} \to X_{vax}$.
*   **$VE_{death}$**: Reduces mortality rates for all vaccinated compartments.

### Mathematical Specification

The model is defined by a system of ordinary differential equations (ODEs) for each age group $i$. For brevity, the age index $i$ is implied.

#### 1. Force of Infection
The force of infection $\lambda_i(t)$ represents the risk of a susceptible individual becoming infected at time $t$:

$$
\lambda_i(t) = \beta(t) \sum_j \frac{C_{ji} \cdot I_{eff, j}}{N_j}
$$

Where $I_{eff}$ is the effective infectious population, weighted by relative infectiousness parameters ($\theta$):

$$
I_{eff} = (I + \theta_X X_{total}) + \theta_{vax}(I_{vax} + \theta_X X_{vax_{total}}) + \theta_H H_{total}
$$

#### 2. Core Dynamics (Unvaccinated)
The flow of individuals through the unvaccinated compartments is governed by:

$$
\begin{aligned}
\frac{dS}{dt} &= \nu N_{total} - \lambda S + \omega R - \phi S - \mu_{bg} S \\
\frac{dE}{dt} &= \lambda S - \alpha E - \mu_{bg} E \\
\frac{dI}{dt} &= \alpha E - (\gamma_I + \mu_I + \sigma) I - \mu_{bg} I \\
\frac{dX_{queued}}{dt} &= \sigma I - (\gamma_X + \mu_{X_{untreated}}) X_{queued} - \eta g_{ward} X_{queued} - \mu_{bg} X_{queued} \\
\frac{dX_{admitted}}{dt} &= \eta g_{ward} X_{queued} - (\gamma_X + \mu_X) X_{admitted} - \gamma_{admit} X_{admitted} - \mu_{bg} X_{admitted} \\
\frac{dH_{ward}}{dt} &= \gamma_{admit} X_{admitted} - (\gamma_{ward} + \mu_{ward_{eff}}) H_{ward} - \eta_{icu} g_{icu} H_{ward} - \mu_{bg} H_{ward} \\
\frac{dH_{icu}}{dt} &= \eta_{icu} g_{icu} H_{ward} - (\gamma_{icu} + \mu_{icu}) H_{icu} - \mu_{bg} H_{icu} \\
\frac{dR}{dt} &= \gamma_I I + \gamma_X (X_{queued} + X_{admitted}) + \gamma_{ward} H_{ward} + \gamma_{icu} H_{icu} - \omega R - \mu_{bg} R \\
\frac{dD}{dt} &= \mu_I I + \mu_{X_{untreated}} X_{queued} + \mu_X X_{admitted} + \mu_{ward_{eff}} H_{ward} + \mu_{icu} H_{icu}
\end{aligned}
$$

**Key Parameters:**
*   $\nu$: Birth rate (enters $S$)
*   $\phi$: Vaccination rate ($S \to S_{vax}$)
*   $\sigma$: Progression to severe disease ($I \to X$)
*   $\eta$: Hospital admission rate ($X \to H$)
*   $\gamma_{admit}$: Transfer from stabilization to ward
*   $g_{ward}$, $g_{icu}$: Hill gating functions (0 to 1) based on capacity

#### 3. Differential Mortality
The model explicitly accounts for excess mortality due to denied care. The effective mortality rate in the Ward ($\mu_{ward_{eff}}$) increases when ICU capacity is full:

$$
\mu_{ward_{eff}} = \mu_{ward} + (\mu_{ward_{denied}} - \mu_{ward}) \cdot \eta_{icu} \cdot (1 - g_{icu})
$$

*   **Treated Deaths**: Occur in $X_{admitted}$, $H_{ward}$ (baseline), and $H_{icu}$.
*   **Untreated Deaths**: Occur in $X_{queued}$ (waiting for bed) and $H_{ward}$ (denied ICU).

#### 4. Vaccinated Dynamics
A parallel system exists for vaccinated individuals ($S_{vax}, E_{vax}, \dots$). Transitions are identical but parameters are modified by Vaccine Efficacy ($VE$):
*   **Infection**: $\lambda_{vax} = (1 - VE_{infection}) \lambda$
*   **Severe Disease**: $\sigma_{vax} = (1 - VE_{severe}) \sigma$
*   **Mortality**: $\mu_{vax} = (1 - VE_{death}) \mu$

## Configuration & Scenarios

The project uses a hierarchical configuration system located in `config.py` and `scenarios.py`.

*   **Scenarios**: Pre-packaged bundles of parameters (e.g., `covid_delta`, `seasonal_flu`).
*   **Overrides**: You can override any specific parameter when calling `simulate_model`.

### Example: Custom Scenario
```python
results = simulate_model(
    beta_base=0.35,
    ward_capacity=1000,
    icu_capacity=200,
    vaccination_rate=0.01,
    VE_infection=0.8
)
```

## Project Structure

```text
hospital-model/
├── simulate_model.py   # Main simulation entry point
├── derivatives.py             # ODE system and physics
├── scenario_helpers.py        # Scenario helpers
├── scenarios.py               # Scenario definitions
├── capacity_helpers.py        # Capacity helpers (gating functions)
├── demographic_helpers.py     # Demographic helpers (births, natural deaths)
├── model_types.py             # Model type definitions for enforcing type safety
├── result_processor.py        # Post-processing of simulation results
├── time_varying_helpers.py    # Time-varying parameter helpers (seasonality, NPIs)
├── tests/                     # Comprehensive test suite
└── notebooks/                 # Jupyter notebooks for experiments
```

## Testing & Verification

Scientific accuracy is verified through a comprehensive test suite, including symbolic verification of conservation laws.

```bash
# Run all tests
pytest

# Run symbolic verification tests
pytest tests/test_symbolic_verification.py
```
