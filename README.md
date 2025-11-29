# Hospital Capacity SEIXHRD Epidemic Model

A comprehensive compartmental epidemic model with hospital capacity constraints, age structure, an exposed (latent) compartment, and ICU separation for analyzing infectious disease dynamics under healthcare system stress.

## Table of Contents

- [Overview](#overview)
- [Model Description](#model-description)
  - [SEIXHRD Model](#seixhrd-model)
  - [Age-Structured Extension](#age-structured-extension)
  - [Ward/ICU Separation](#wardicu-separation)
  - [Master Model](#master-model)
  - [Time-Varying Extensions](#time-varying-extensions)
  - [Three-Factor Vaccine Model](#three-factor-vaccine-model)
- [Mathematical Foundations](#mathematical-foundations)
  - [Force of Infection](#force-of-infection)
  - [Hill Function Gating](#hill-function-gating)
  - [Complete ODE System](#complete-ode-system)
  - [Differential Mortality](#differential-mortality-equations)
  - [Time-Varying Transmission](#time-varying-transmission-equations)
  - [Numerical Integration](#numerical-integration)
  - [Vaccination Dynamics](#vaccination-dynamics-equations)
- [Installation](#installation)
- [Testing](#testing)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Basic Simulation](#basic-simulation)
  - [Age-Structured Model](#age-structured-model)
  - [Ward/ICU Model](#wardicu-model-usage)
  - [Master Model](#master-model-usage)
  - [Vaccination Scenarios](#vaccination-usage)
  - [Time-Varying Scenarios](#time-varying-scenarios)
- [Configuration](#configuration)
- [Key Features](#key-features)
- [Examples](#examples)
- [API Reference](#api-reference)

---

## Overview

This project implements an **SEIXHRD compartmental model** designed to simulate epidemic dynamics while accounting for real-world healthcare system constraints. Unlike traditional SIR models, this implementation:

1. **Includes an Exposed (E) compartment** for realistic latent period dynamics
2. **Include a SEVERE (X) compartment** for cases needing hospitalization
3. **Includes a HOSPITALIZED (H) compartment** for admitted patients
4. **Separates ward and ICU capacity** with distinct mortality implications
5. **The D (dead) compartment** tracks disease-related fatalities
6. **Models hospital capacity constraints** using smooth Hill function gating
7. **Supports age-structured populations** with varying contact patterns
8. **Includes time-varying transmission** (seasonality, policy interventions, waning immunity)
9. **Implements a Three-Factor Vaccine Model** with breakthrough infections and differential efficacy
10. **Differential mortality tracking** based on care status (treated vs untreated)
11. **Tracks unmet care needs** and overflow burden for policy analysis
12. **Utilizes adaptive ODE solvers** for numerical stability and accuracy
13. **Extensive Configurability** for disease parameters, capacities, vaccination, and interventions, etc...

The model is designed for educational purposes to explore:

- Epidemic trajectory under hospital capacity limitations
- Optimal vaccine allocation across age groups
- Policy trade-offs (lockdowns vs. hospital surge capacity)
- Impact of ward/ICU allocation decisions
- Differential mortality due to care denial
- Effects of seasonality and waning immunity on outbreaks
- Vaccine breakthrough dynamics with varying efficacy profiles
- And more...

---

## Model Description

### SEIXHRD Model

The core model divides the population into **nine compartments**, with a critical distinction in the Severe (X) stage to model queuing dynamics and differential mortality:

| Compartment | Description |
|-------------|-------------|
| **S** | Susceptible - individuals who can become infected |
| **E** | Exposed - infected but in latent period (not yet infectious) |
| **I** | Infected - mild or early-stage infections (infectious) |
| **X_queued** | Severe cases **waiting** for ward admission (subject to untreated mortality μ_X_untreated) |
| **X_admitted** | Severe cases **admitted** to ward stabilization (subject to treated mortality μ_X) |
| **H_ward** | General Ward - admitted patients receiving standard care |
| **H_icu** | ICU - critical care patients |
| **R** | Recovered - immune individuals |
| **D** | Dead - disease-related fatalities |

**Flow diagram:**

```text
S → E → I → X_queued ──(gated by g_ward)──> X_admitted → H_ward ──(gated by g_icu)──> H_icu
            ↓                                  ↓            ↓                          ↓
        (μ_X_untreated)                     (μ_X)      (μ_ward)                    (μ_icu)
            ↓                                  ↓            ↓                          ↓
        D_untreated ←──────────────────────────┴────────────┴──────────────────────> D_treated
                                                            ↑
                                              (also μ_ward_denied when ICU gated)
```

**Key insight:** The split X architecture eliminates the need for "blended effective mortality rates." Patients in `X_queued` always experience the untreated rate, while those in `X_admitted` always experience the treated rate. The Hill gating function controls the **flow** between compartments, not the mortality rate within a compartment.

The **Exposed (E) compartment** represents the latent period where individuals are infected but not yet infectious. This is controlled by the parameter $\alpha$ (E→I rate), where the mean latent period is $1/\alpha$ days.

### Age-Structured Extension

The model supports **N age groups** (default: 3) with:

- **Age-specific disease parameters**: Different severity, mortality, and recovery rates
- **Contact matrix**: Heterogeneous mixing patterns between age groups
- **Age-targeted vaccination**: Different coverage levels per age group
- **Age-specific waning immunity**: Different rates of immunity loss
- **Age-specific force of infection**: Calculated using the contact matrix and age-specific infectious populations

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
| **X_queued** | Severe cases waiting for ward bed (untreated mortality applies) |
| **X_admitted** | Severe cases that secured a ward spot, stabilizing before transfer |
| **H_ward** | General ward beds for standard hospital care |
| **H_icu** | ICU beds for critical care with ventilators |

Key features:

- **Two-stage ward admission**: X_queued → X_admitted (gated) → H_ward (ungated transfer)
- **Separate capacity constraints**: Ward and ICU each with independent Hill function gating
- **X_admitted counts toward ward capacity**: `H_ward_total = H_ward + H_ward_vax + X_admitted + X_admitted_vax`
- **Different mortality rates**: ICU mortality typically higher (sicker patients)
- **Differential mortality tracking**: Untreated deaths (X_queued, ICU-denied) vs treated deaths
- **Escalation flow**: Ward patients may need ICU; ICU admission is gated separately

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

### Three-Factor Vaccine Model

The model implements a **Three-Factor Vaccine Model** with vaccinated compartments that mirror the unvaccinated pathway:

#### Vaccinated Compartments

| Compartment | Description |
|-------------|-------------|
| **S_vax** | Vaccinated susceptible - can still be infected (breakthrough) |
| **E_vax** | Vaccinated exposed - breakthrough infection in latent period |
| **I_vax** | Vaccinated infected - breakthrough with mild symptoms |
| **X_queued_vax** | Vaccinated severe (queued) - breakthrough waiting for ward bed |
| **X_admitted_vax** | Vaccinated severe (admitted) - breakthrough secured ward spot |
| **H_ward_vax** | Vaccinated in ward - breakthrough requiring ward care |
| **H_icu_vax** | Vaccinated in ICU - breakthrough requiring critical care |
| **R_vax** | Vaccinated recovered - recovered from breakthrough infection |
| **D_vax** | Vaccinated deaths - deaths despite vaccination |

**Flow diagram with vaccination:**

```text
Unvaccinated:  S  → E  → I  → X_queued ──(gated)──> X_admitted → H_ward → H_icu → R
               ↓                                                                   ↓
               ↓ vaccination                                                      D
               ↓
Vaccinated:   S_vax → E_vax → I_vax → X_queued_vax ──(gated)──> X_admitted_vax → H_ward_vax → H_icu_vax → R_vax
                      (reduced λ)  (reduced σ)                    (reduced μ)                              ↓
                                                                                                       D_vax (reduced μ)
```

#### Three-Factor Efficacy Mechanism

The vaccine provides protection through three distinct efficacy parameters:

| Parameter | Symbol | Description | Effect |
|-----------|--------|-------------|--------|
| **VE_infection** | $VE_I$ | Vaccine efficacy against infection | Reduces force of infection on S_vax |
| **VE_severe** | $VE_S$ | Vaccine efficacy against severe disease | Reduces I_vax → X_vax progression |
| **VE_death** | $VE_D$ | Vaccine efficacy against death | Reduces all mortality rates in vaccinated |

**Example values (mRNA vaccine vs Delta variant):**

```python
VE_infection = 0.80  # 80% reduction in infection risk
VE_severe = 0.90     # 90% reduction in severe disease progression
VE_death = 0.95      # 95% reduction in mortality
```

#### Vaccine Profiles

Pre-configured vaccine profiles are available:

| Profile | VE_infection | VE_severe | VE_death | Description |
|---------|--------------|-----------|----------|-------------|
| `mrna_original` | 0.80 | 0.90 | 0.95 | Pfizer/Moderna vs original strain |
| `mrna_omicron` | 0.30 | 0.70 | 0.85 | mRNA vs Omicron (immune escape) |
| `adenovirus` | 0.65 | 0.80 | 0.90 | AstraZeneca/J&J type |
| `inactivated` | 0.50 | 0.70 | 0.85 | Sinovac/Sinopharm type |
| `influenza_typical` | 0.40 | 0.60 | 0.75 | Seasonal flu vaccine |
| `ideal` | 0.95 | 0.98 | 0.99 | Ideal vaccine benchmark |
| `minimal` | 0.20 | 0.40 | 0.50 | Minimal protection scenario |

```python
from config import get_vaccine_profile, list_vaccine_profiles

# List available profiles
print(list_vaccine_profiles())

# Get profile parameters
profile = get_vaccine_profile('mrna_original')
# {'VE_infection': 0.80, 'VE_severe': 0.90, 'VE_death': 0.95, ...}
```

#### Vaccine Waning

Vaccine-induced immunity can wane over time:

```python
vaccine_waning_params = {
    'omega_vax': 0.005,  # Waning rate (~200 day immunity duration)
    'wane_to_S': True    # Wane to S (unvaccinated) or S_vax (remains vaccinated)
}
```

When `wane_to_S=True`, vaccinated recovered individuals (R_vax) return to unvaccinated susceptible (S), losing vaccine protection. When `wane_to_S=False`, they return to S_vax, remaining in the vaccinated population.

---

## Mathematical Foundations

### Force of Infection

The force of infection $\lambda_a$ acting on age group $a$ is calculated via the contact matrix $C_{ab}$ using **vectorized matrix operations**. The infectious population includes $I$, $X$ (both queued and admitted), and $H$ (ward and ICU), for both vaccinated and unvaccinated groups.

$$\lambda_a(t) = \beta(t) \cdot \frac{\sum_b C_{ba} \cdot I_{\text{eff},b}}{N_a}$$

Where the **effective infectious contributions** from each age group $b$ are:

$$I_{\text{eff},b} = I_b + \theta_X (X_{\text{queued},b} + X_{\text{admitted},b}) + \theta_{\text{vax}} \left[ I_{\text{vax},b} + \theta_X (X_{\text{queued,vax},b} + X_{\text{admitted,vax},b}) \right] + \theta_H H_{\text{total},b}$$

With hospital contributions:

$$H_{\text{total},b} = H_{\text{ward},b} + H_{\text{icu},b} + H_{\text{ward,vax},b} + H_{\text{icu,vax},b}$$

**Key parameters:**

| Parameter | Symbol | Description | Default |
|-----------|--------|-------------|---------|
| Contact matrix | $C_{ab}$ | Contacts per day from age $a$ to $b$ | Age-assortative |
| Severe infectiousness | $\theta_X$ | Relative infectiousness of X compartments | 0.5 |
| Hospital infectiousness | $\theta_H$ | Relative infectiousness of H compartments | 0.3 |
| Vaccinated infectiousness | $\theta_{\text{vax}}$ | Infectiousness reduction for breakthrough cases | 0.5 |

**For vaccinated susceptibles**, the force of infection is reduced by vaccine efficacy against infection:

$$\lambda_{\text{vax},a}(t) = (1 - VE_{\text{infection}}) \cdot \lambda_a(t)$$

> **Note on matrix orientation:** The code uses `C.T @ I_eff` (transpose), meaning $C_{ba}$ represents contacts *received by* age group $a$ *from* age group $b$. This is the standard epidemiological convention where rows represent infectees and columns represent infectors after transposition.

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

The model uses a system of Ordinary Differential Equations (ODEs) for each age group $a$. The key innovation is the **separation of severe cases into queued and admitted states** to rigorously model capacity constraints and differential mortality.

#### Basic SEIXHRD Hospital Model (Simplified)

For a single population without age structure:

$$\frac{dS}{dt} = -\lambda S$$

$$\frac{dE}{dt} = \lambda S - \alpha E$$

$$\frac{dI}{dt} = \alpha E - (\gamma_I + \mu_I + \sigma) I$$

$$\frac{dX_{\text{queued}}}{dt} = \sigma I - (\gamma_X + \mu_{X,\text{untreated}}) X_{\text{queued}} - \eta \cdot X_{\text{queued}} \cdot g_{\text{ward}}$$

$$\frac{dX_{\text{admitted}}}{dt} = \eta \cdot X_{\text{queued}} \cdot g_{\text{ward}} - (\gamma_X + \mu_X) X_{\text{admitted}} - \gamma_{X,\text{admit}} \cdot X_{\text{admitted}}$$

$$\frac{dH_{\text{ward}}}{dt} = \gamma_{X,\text{admit}} \cdot X_{\text{admitted}} - (\gamma_{\text{ward}} + \mu_{\text{ward}}) H_{\text{ward}} - \eta_{\text{icu}} \cdot H_{\text{ward}} \cdot g_{\text{icu}}$$

$$\frac{dR}{dt} = \gamma_I I + \gamma_X (X_{\text{queued}} + X_{\text{admitted}}) + \gamma_{\text{ward}} H_{\text{ward}} + \gamma_{\text{icu}} H_{\text{icu}}$$

$$\frac{dD}{dt} = \mu_I I + \mu_{X,\text{untreated}} X_{\text{queued}} + \mu_X X_{\text{admitted}} + \mu_{\text{ward}} H_{\text{ward}} + \mu_{\text{icu}} H_{\text{icu}}$$

Where $\alpha$ is the rate of progression from Exposed to Infectious (1/latent period).

#### Master Model with Ward/ICU Split (Full Age-Structured)

For each age group $a$:

**Susceptible:**
$$\frac{dS_a}{dt} = -\lambda_a S_a + \omega_a R_a - \nu_a S_a$$

where $\nu_a$ is the vaccination rate.

**Exposed (latent period):**
$$\frac{dE_a}{dt} = \lambda_a S_a - \alpha_a E_a$$

**Infected (mild):**
$$\frac{dI_a}{dt} = \alpha_a E_a - (\gamma_{I,a} + \mu_{I,a} + \sigma_a) I_a$$

**Severe Queued (waiting for ward bed):**

Patients enter `X_queued` when they become severe. They compete for ward admission via the gating function $g_{\text{ward}}$. While waiting, they experience **untreated mortality** $\mu_{X,\text{untreated}}$.

$$\frac{dX_{\text{queued},a}}{dt} = \sigma_a I_a - (\gamma_{X,a} + \mu_{X,\text{untreated},a}) X_{\text{queued},a} - \text{admit}_{\text{X},a}$$

where:
$$\text{admit}_{\text{X},a} = \eta_a \cdot X_{\text{queued},a} \cdot g_{\text{ward}}(H_{\text{ward,total}})$$

**Severe Admitted (secured ward spot, stabilizing):**

Patients who cross the capacity gate enter `X_admitted`. They receive care (treated mortality $\mu_X$) and transfer to the general ward at rate $\gamma_{X,\text{admit}}$.

$$\frac{dX_{\text{admitted},a}}{dt} = \text{admit}_{\text{X},a} - (\gamma_{X,a} + \mu_{X,a}) X_{\text{admitted},a} - \text{transfer}_{\text{ward},a}$$

where:
$$\text{transfer}_{\text{ward},a} = \gamma_{X,\text{admit}} \cdot X_{\text{admitted},a}$$

> **Note:** Transfer from `X_admitted` to `H_ward` is **not gated**—once a patient secures a spot in `X_admitted`, their progression to the ward is guaranteed. The parameter $\gamma_{X,\text{admit}} \approx 0.5$ day$^{-1}$ represents a ~2 day administrative/stabilization period.

**General Ward:**

Ward patients may recover, die, or require ICU escalation. ICU admission is gated by $g_{\text{icu}}$.

$$\frac{dH_{\text{ward},a}}{dt} = \text{transfer}_{\text{ward},a} - (\gamma_{\text{ward},a} + \mu_{\text{ward,eff},a}) H_{\text{ward},a} - \text{admit}_{\text{icu},a}$$

where:
$$\text{admit}_{\text{icu},a} = \eta_{\text{icu},a} \cdot H_{\text{ward},a} \cdot g_{\text{icu}}(H_{\text{icu,total}})$$

**ICU:**
$$\frac{dH_{\text{icu},a}}{dt} = \text{admit}_{\text{icu},a} - (\gamma_{\text{icu},a} + \mu_{\text{icu},a}) H_{\text{icu},a}$$

**Recovered:**
$$\frac{dR_a}{dt} = \gamma_{I,a} I_a + \gamma_{X,a} (X_{\text{queued},a} + X_{\text{admitted},a}) + \gamma_{\text{ward},a} H_{\text{ward},a} + \gamma_{\text{icu},a} H_{\text{icu},a} - \omega_a R_a$$

**Deaths:**
$$\frac{dD_a}{dt} = \mu_{I,a} I_a + \mu_{X,\text{untreated},a} X_{\text{queued},a} + \mu_{X,a} X_{\text{admitted},a} + \mu_{\text{ward,eff},a} H_{\text{ward},a} + \mu_{\text{icu},a} H_{\text{icu},a}$$

#### Ward Capacity Calculation

The ward gating function uses a combined occupancy that includes patients in `X_admitted` (who have secured a ward spot but haven't transferred yet):

$$H_{\text{ward,total}} = \sum_a \left( H_{\text{ward},a} + H_{\text{ward,vax},a} + X_{\text{admitted},a} + X_{\text{admitted,vax},a} \right)$$

This "closes the ghost ward gap"—ensuring that patients stabilizing in `X_admitted` are counted toward capacity.

### Differential Mortality Equations

The split-X architecture **eliminates the need for blended effective mortality rates**. Instead, patients in different compartments simply experience different mortality rates:

| Compartment | Mortality Rate | Care Status |
|-------------|----------------|-------------|
| `X_queued` | $\mu_{X,\text{untreated}}$ | Waiting for bed (no care) |
| `X_admitted` | $\mu_X$ | Secured bed (receiving care) |
| `H_ward` (baseline) | $\mu_{\text{ward}}$ | Ward care |
| `H_ward` (ICU denied) | $\mu_{\text{ward,denied}}$ | Needed ICU but denied |
| `H_icu` | $\mu_{\text{icu}}$ | ICU care |

**Untreated mortality multipliers** (age-specific):

| Age Group | $\mu_{X,\text{untreated}}$ Multiplier | $\mu_{\text{ward,denied}}$ Multiplier |
|-----------|--------------------------------------|---------------------------------------|
| Young | 1.5× | 1.3× |
| Middle | 2.0× | 1.5× |
| Elderly | 3.0× | 2.0× |

#### Ward Effective Mortality

For ward patients, effective mortality increases when ICU is constrained. The fraction of ICU-needing patients who are denied care is $(1 - g_{\text{icu}})$:

$$\mu_{\text{ward,eff}} = \mu_{\text{ward}} + (\mu_{\text{ward,denied}} - \mu_{\text{ward}}) \cdot \eta_{\text{icu}} \cdot (1 - g_{\text{icu}})$$

where $\eta_{\text{icu}}$ is the rate at which ward patients need ICU escalation.

#### Death Accumulator Equations

The model tracks deaths by care status using accumulator variables integrated alongside the main state:

**Treated deaths** (received appropriate care):
$$\frac{dD_{\text{treated}}}{dt} = \sum_a \left[ \mu_{I,a} I_a + \mu_{X,a} X_{\text{admitted},a} + \mu_{\text{ward},a} H_{\text{ward},a} + \mu_{\text{icu},a} H_{\text{icu},a} \right]$$

**Untreated deaths** (denied care due to capacity):
$$\frac{dD_{\text{untreated}}}{dt} = \sum_a \left[ \mu_{X,\text{untreated},a} X_{\text{queued},a} + (\mu_{\text{ward,denied},a} - \mu_{\text{ward},a}) \cdot \eta_{\text{icu},a} \cdot (1 - g_{\text{icu}}) \cdot H_{\text{ward},a} \right]$$

> **Key insight:** Deaths in `X_queued` are *always* counted as untreated (these patients never secured a bed). Deaths in `X_admitted`, `H_ward`, and `H_icu` are counted as treated. The excess mortality from ICU denial (the $(\mu_{\text{ward,denied}} - \mu_{\text{ward}})$ term) is counted as untreated.

**Preventable Mortality Metric:**
$$\text{Preventable Mortality \%} = \frac{D_{\text{untreated}}}{D_{\text{total}}} \times 100$$

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

The master model uses **adaptive ODE solvers** from `scipy.integrate` for improved numerical stability and accuracy:

#### Default Solver: LSODA (via `odeint`)

The default solver is `scipy.integrate.odeint`, which uses the **LSODA** algorithm that automatically switches between stiff (BDF) and non-stiff (Adams) methods:

```python
results = simulate_master_hospital_model(
    beta_base=0.3,
    age_pops=[3000, 5000, 2000],
    solver='odeint',           # Default
    rtol=1e-6,                 # Relative tolerance
    atol=1e-8,                 # Absolute tolerance
    ...
)
```

#### Alternative Solver: `solve_ivp`

For more control over the integration method, use `scipy.integrate.solve_ivp`:

```python
results = simulate_master_hospital_model(
    beta_base=0.3,
    age_pops=[3000, 5000, 2000],
    solver='solve_ivp',
    solver_method='BDF',       # Good for stiff systems
    rtol=1e-6,
    atol=1e-8,
    ...
)
```

Available `solver_method` options:

- `'LSODA'` (default) - Automatic stiff/non-stiff switching
- `'BDF'` - Backward Differentiation Formula for stiff problems
- `'Radau'` - Implicit Runge-Kutta for stiff problems
- `'RK45'` - Explicit Runge-Kutta for non-stiff problems
- `'RK23'` - Lower-order explicit Runge-Kutta
- `'DOP853'` - High-order explicit Runge-Kutta

#### Handling Discontinuities with `tcrit`

When policy interventions are specified, the solver uses **critical time points** (`tcrit` for `odeint`, `t_eval` dense output for `solve_ivp`) to accurately capture discontinuities in the transmission rate:

```python
interventions = [
    {'start_day': 30, 'end_day': 75, 'transmission_reduction': 0.5},
    {'start_day': 150, 'end_day': 200, 'transmission_reduction': 0.3}
]

# The solver will step exactly at days 30, 75, 150, 200
results = simulate_master_hospital_model(
    interventions=interventions,
    ...
)
```

#### Post-Integration Clipping

To handle potential numerical artifacts, the solver applies post-integration clipping with a configurable warning threshold:

```python
results = simulate_master_hospital_model(
    clip_warning_threshold=-1e-6,  # Warn if values < this
    ...
)
```

If any compartment value falls below the threshold, a warning is issued. All compartment values are then clipped to `[0, ∞)` to ensure non-negativity.

#### State Vector Packing

The **18 compartments** × N age groups are packed into a single 1D state vector, plus **5 tracked accumulators** for differential mortality:

```python
# Main compartments (18 per age group):
STATE_ORDER = [
    'S', 'E', 'I', 'X_queued', 'X_admitted', 'H_ward', 'H_icu', 'R', 'D',
    'S_vax', 'E_vax', 'I_vax', 'X_queued_vax', 'X_admitted_vax', 'H_ward_vax', 'H_icu_vax', 'R_vax', 'D_vax'
]

# Tracked accumulators (5 total, not per-age):
TRACKED_ORDER = ['D_treated', 'D_untreated', 'D_vax_treated', 'D_vax_untreated', 'cum_breakthrough']

# For 3 age groups: 18 * 3 + 5 = 59 state variables
# Layout: [S_0, S_1, S_2, E_0, E_1, E_2, ..., D_vax_0, D_vax_1, D_vax_2, D_treated, D_untreated, ...]
```

This enables efficient vectorized computation of the force of infection:

$$\lambda = \beta_{\text{eff}}(t) \cdot C \cdot \frac{\mathbf{I}_{\text{infectious}}}{\mathbf{N}}$$

Where:

- $C$ is the contact matrix
- $\mathbf{I}_{\text{infectious}}$ includes contributions from I, X, H compartments (both vaccinated and unvaccinated)
- The matrix multiplication is fully vectorized using NumPy

### Vaccination Dynamics Equations

The Three-Factor Vaccine Model adds 8 vaccinated compartments parallel to the unvaccinated pathway.

#### Vaccination Flow

Susceptible individuals are vaccinated at rate $v$ (vaccination_rate):

$$\frac{dS_a}{dt} = -\lambda_a S_a - v_a S_a + \omega_a R_a$$

$$\frac{dS_{\text{vax},a}}{dt} = v_a S_a - \lambda_{\text{vax},a} S_{\text{vax},a} + \omega_{\text{vax},a} R_{\text{vax},a} \cdot (1 - w_{S})$$

Where:

- $v_a$ — age-specific vaccination rate
- $w_S$ — wane_to_S flag (1 = wane to S, 0 = wane to S_vax)

#### Reduced Force of Infection for Vaccinated

VE_infection reduces the force of infection for vaccinated susceptibles:

$$\lambda_{\text{vax},a} = (1 - VE_I) \cdot \lambda_a$$

Where $\lambda_a$ is the standard force of infection including both unvaccinated and vaccinated infectious individuals:

$$\lambda_a = \beta_{\text{eff}} \sum_{b} C_{ab} \frac{I_b + I_{\text{vax},b} \cdot \theta_{\text{vax}} + \theta_X (X_b + X_{\text{vax},b}) + \theta_H (H_b + H_{\text{vax},b})}{N_b}$$

The parameter $\theta_{\text{vax}}$ represents the relative infectiousness of breakthrough infections (default: 0.5).

#### Reduced Progression to Severe Disease

VE_severe reduces the rate at which vaccinated infected individuals progress to severe disease:

$$\sigma_{\text{vax},a} = (1 - VE_S) \cdot \sigma_a$$

**Compensatory Recovery Rate:** To preserve the total exit rate from compartment I (ensuring vaccinated individuals don't remain infectious longer), the recovery rate is increased:

$$\gamma_{I,\text{vax},a} = \gamma_{I,a} + (\sigma_a - \sigma_{\text{vax},a})$$

This ensures that the reduced flow to severe disease is compensated by increased recovery, maintaining epidemiological consistency.

#### Reduced Mortality

VE_death reduces all mortality rates in vaccinated compartments:

$$\mu_{I,\text{vax}} = (1 - VE_D) \cdot \mu_I$$
$$\mu_{X,\text{vax}} = (1 - VE_D) \cdot \mu_X$$
$$\mu_{\text{ward,vax}} = (1 - VE_D) \cdot \mu_{\text{ward}}$$
$$\mu_{\text{icu,vax}} = (1 - VE_D) \cdot \mu_{\text{icu}}$$

#### Vaccinated Compartment ODEs

The vaccinated pathway mirrors the unvaccinated pathway, including the split severe compartments.

**Vaccinated Exposed:**
$$\frac{dE_{\text{vax},a}}{dt} = \lambda_{\text{vax},a} S_{\text{vax},a} - \alpha_a E_{\text{vax},a}$$

**Vaccinated Infected (mild):**
$$\frac{dI_{\text{vax},a}}{dt} = \alpha_a E_{\text{vax},a} - (\gamma_{I,\text{vax},a} + \mu_{I,\text{vax}} + \sigma_{\text{vax},a}) I_{\text{vax},a}$$

Note: $\gamma_{I,\text{vax}}$ is accelerated (see compensatory recovery rate above).

**Vaccinated Severe Queued:**
$$\frac{dX_{\text{queued,vax},a}}{dt} = \sigma_{\text{vax},a} I_{\text{vax},a} - (\gamma_{X,a} + \mu_{X,\text{untreated,vax}}) X_{\text{queued,vax},a} - \text{admit}_{\text{X,vax},a}$$

**Vaccinated Severe Admitted:**
$$\frac{dX_{\text{admitted,vax},a}}{dt} = \text{admit}_{\text{X,vax},a} - (\gamma_{X,a} + \mu_{X,\text{vax}}) X_{\text{admitted,vax},a} - \text{transfer}_{\text{ward,vax},a}$$

**Vaccinated Ward:**
$$\frac{dH_{\text{ward,vax},a}}{dt} = \text{transfer}_{\text{ward,vax},a} - (\gamma_{\text{ward},a} + \mu_{\text{ward,vax,eff}}) H_{\text{ward,vax},a} - \text{admit}_{\text{icu,vax},a}$$

**Vaccinated ICU:**
$$\frac{dH_{\text{icu,vax},a}}{dt} = \text{admit}_{\text{icu,vax},a} - (\gamma_{\text{icu},a} + \mu_{\text{icu,vax}}) H_{\text{icu,vax},a}$$

**Vaccinated Recovered:**
$$\frac{dR_{\text{vax},a}}{dt} = \gamma_{I,\text{vax},a} I_{\text{vax},a} + \gamma_{X,a} (X_{\text{queued,vax},a} + X_{\text{admitted,vax},a}) + \gamma_{\text{ward},a} H_{\text{ward,vax},a} + \gamma_{\text{icu},a} H_{\text{icu,vax},a} - \omega_a R_{\text{vax},a}$$

**Vaccinated Deaths:**
$$\frac{dD_{\text{vax},a}}{dt} = \mu_{I,\text{vax}} I_{\text{vax},a} + \mu_{X,\text{untreated,vax}} X_{\text{queued,vax},a} + \mu_{X,\text{vax}} X_{\text{admitted,vax},a} + \mu_{\text{ward,vax,eff}} H_{\text{ward,vax},a} + \mu_{\text{icu,vax}} H_{\text{icu,vax},a}$$

#### Hybrid Immunity Waning

Individuals in R_vax have "hybrid immunity" (vaccination + natural infection). The model wanes R_vax at the **natural immunity rate** $\omega_a$ (not a separate $\omega_{\text{vax}}$), reflecting that hybrid immunity duration is driven by infection-induced immunity:

$$\frac{dR_{\text{vax},a}}{dt} = \text{(recoveries)} - \omega_a R_{\text{vax},a}$$

The destination depends on the `waning_destination` parameter:

- If `waning_destination = 'S'`: R_vax → S (lose all protection)
- If `waning_destination = 'S_vax'`: R_vax → S_vax (retain vaccine-induced protection)

$$\frac{dS_a}{dt} += \omega_a R_{\text{vax},a} \cdot \mathbb{1}[\text{dest}=S]$$
$$\frac{dS_{\text{vax},a}}{dt} += \omega_a R_{\text{vax},a} \cdot \mathbb{1}[\text{dest}=S_{\text{vax}}]$$

#### Population Conservation

With vaccination and split-X compartments, the total population across all **18 compartments** is conserved:

$$N = \sum_a \big( S_a + E_a + I_a + X_{\text{queued},a} + X_{\text{admitted},a} + H_{\text{ward},a} + H_{\text{icu},a} + R_a + D_a$$
$$\quad + S_{\text{vax},a} + E_{\text{vax},a} + I_{\text{vax},a} + X_{\text{queued,vax},a} + X_{\text{admitted,vax},a} + H_{\text{ward,vax},a} + H_{\text{icu,vax},a} + R_{\text{vax},a} + D_{\text{vax},a} \big)$$

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

## Testing

The project includes a comprehensive test suite with **229 tests** covering all simulation functions, helper utilities, vaccination compartments, and edge cases.

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with short traceback (recommended)
python -m pytest tests/ -v --tb=short

# Run specific test file
python -m pytest tests/test_helper_functions.py -v
python -m pytest tests/test_master_model_integration.py -v
python -m pytest tests/test_time_varying_behavior.py -v
python -m pytest tests/test_edge_cases.py -v
python -m pytest tests/test_differential_mortality.py -v

# Run specific test class
python -m pytest tests/test_helper_functions.py::TestHillGate -v

# Run specific test
python -m pytest tests/test_helper_functions.py::TestHillGate::test_half_occupancy -v

# Run tests matching a pattern
python -m pytest tests/ -v -k "mortality"
python -m pytest tests/ -v -k "conservation"
```

### Test Categories

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_helper_functions.py` | 48 | `hill_gate`, `_validate_age_structured_inputs`, `_coerce_initial_vector`, `seasonal_forcing`, `policy_multiplier` |
| `test_master_model_integration.py` | 45 | Smoke tests, output structure, population conservation, death monotonicity, compartment flows |
| `test_time_varying_behavior.py` | 25 | Seasonal forcing, policy interventions, waning immunity, combined effects |
| `test_edge_cases.py` | 40 | No infection, full vaccination, zero/high capacity, single age group, extreme parameters |
| `test_differential_mortality.py` | 31 | D_treated vs D_untreated tracking, capacity-dependent mortality, gating correlation |
| `test_vaccination_compartments.py` | 40 | Three-Factor Vaccine Model, breakthrough infections, VE efficacy, vaccine waning, population conservation |

### Test Organization

Tests are organized using pytest classes for logical grouping:

```python
# Example: tests/test_helper_functions.py
class TestHillGate:           # 13 tests for hill_gate()
class TestValidateInputs:     # 8 tests for _validate_age_structured_inputs()
class TestCoerceVector:       # 8 tests for _coerce_initial_vector()
class TestSeasonalForcing:    # 12 tests for seasonal_forcing()
class TestPolicyMultiplier:   # 14 tests for policy_multiplier()
```

### Pytest Options

```bash
# Show test durations (find slow tests)
python -m pytest tests/ --durations=10

# Stop on first failure
python -m pytest tests/ -x

# Run last failed tests only
python -m pytest tests/ --lf

# Show local variables in tracebacks
python -m pytest tests/ --tb=long --showlocals

# Quiet mode (just pass/fail summary)
python -m pytest tests/ -q

# Very verbose (show each assertion)
python -m pytest tests/ -vv
```

### Fixtures

Shared test fixtures are defined in `tests/conftest.py`:

| Fixture | Description |
|---------|-------------|
| `minimal_inputs` | Standard 3-age-group simulation inputs |
| `minimal_inputs_single_age` | Single age group inputs |
| `high_capacity_inputs` | Abundant hospital capacity (no overflow) |
| `low_capacity_inputs` | Severely constrained capacity |
| `zero_capacity_inputs` | Zero hospital beds |
| `seasonal_inputs` | Inputs with seasonal forcing enabled |
| `intervention_inputs` | Inputs with policy interventions |
| `waning_inputs` | Inputs with waning immunity |
| `full_vaccination_inputs` | 100% vaccine coverage |
| `no_infection_inputs` | Zero initial infections |
| `config_defaults` | Default parameters from config.py |

---

## Development Setup

For contributors and developers:

### Virtual Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib pytest

# (Optional) Install Jupyter for notebooks
pip install jupyter
```

### Running the Test Suite

```bash
# Verify installation
python -c "import numpy; import matplotlib; import pytest; print('All dependencies installed')"

# Run full test suite
python -m pytest tests/ -v --tb=short

# Expected output: 229 passed
```

### Code Quality Checks

Before submitting changes, ensure:

1. **All tests pass**: `python -m pytest tests/ -v`
2. **No regressions**: Run tests related to modified code
3. **New features have tests**: Add tests for any new functionality

---

## Project Structure

```text
hospital-model/
├── config.py                  # Scenario bundles, presets, and helper functions
├── master_hospital_model.py   # Unified master model with all features
├── hospital_models.py         # Core simulation functions
├── simulation_helpers.py      # Helper functions for sweeps and comparisons
├── plotting_utils.py          # Visualization functions
├── time_varying_helpers.py    # Time-varying transmission utilities
├── hospital.ipynb             # Interactive notebook with examples
├── master_model_experiments.ipynb  # Master model experiments
├── tests/                     # Comprehensive test suite (189 tests)
│   ├── __init__.py            # Test package marker
│   ├── conftest.py            # Shared pytest fixtures
│   ├── test_helper_functions.py       # Unit tests for helper functions
│   ├── test_master_model_integration.py  # Integration tests for main simulation
│   ├── test_time_varying_behavior.py  # Time-varying dynamics tests
│   ├── test_edge_cases.py     # Boundary condition tests
│   └── test_differential_mortality.py  # D_treated/D_untreated tests
└── README.md                  # This file
```

### File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | Complete scenario bundles, transmission/healthcare/intervention presets, vaccination strategies, and helper functions (`get_scenario_params()`, `list_scenarios()`, etc.) |
| `master_hospital_model.py` | Unified simulation: `simulate_master_hospital_model()` combining age structure, ward/ICU, seasonality, interventions, waning immunity, and differential mortality |
| `hospital_models.py` | Core simulation functions: `simulate_hospital_model()`, `simulate_age_structured_model()`, `simulate_age_structured_model_icu()` |
| `simulation_helpers.py` | Reusable analysis functions: `compare_vaccination_strategies()`, `optimize_vaccine_allocation()`, `sweep_icu_capacity()` |
| `plotting_utils.py` | Comprehensive visualization: `plot_hospital_simulation_stats()`, `plot_age_structured_results()`, `plot_age_structured_icu_results()` |
| `time_varying_helpers.py` | Time-varying transmission utilities |
| `tests/conftest.py` | Shared pytest fixtures for all test files |
| `tests/test_*.py` | Test modules covering helpers, integration, edge cases, and mortality tracking |

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
| `S_vax`, `E_vax`, `I_vax`, `X_vax`, `R_vax`, `D_vax` | Vaccinated compartments by age |
| `H_ward_vax`, `H_icu_vax` | Vaccinated hospital compartments by age |
| `H_ward_total`, `H_icu_total` | Aggregated hospital occupancy |
| `D_treated`, `D_untreated` | Deaths by care status |
| `D_vax`, `D_vax_total` | Vaccinated deaths |
| `beta_t`, `seasonal_factor`, `policy_mult` | Time-varying parameters |
| `g_ward`, `g_icu` | Admission gating factors over time |
| `cum_ward_overflow`, `cum_icu_overflow` | Cumulative overflow burden |
| `breakthrough_infections` | Cumulative breakthrough infections |

### Vaccination Usage

The Three-Factor Vaccine Model supports comprehensive vaccination dynamics:

#### Basic Vaccination

```python
from master_hospital_model import simulate_master_hospital_model
import config

# Simple vaccination scenario
results = simulate_master_hospital_model(
    beta_base=0.3,
    age_params=config.AGE_PARAMS_DEFAULT,
    contact_matrix=config.CONTACT_MATRIX_DEFAULT,
    age_pops=[3000, 5000, 2000],
    
    # Three-Factor Vaccine Efficacy
    vaccination_rate=0.02,     # 2% of susceptibles vaccinated per day
    VE_infection=0.7,          # 70% reduction in infection risk
    VE_severe=0.85,            # 85% reduction in severe disease
    VE_death=0.95,             # 95% reduction in mortality
    
    Tmax=200
)

# Access vaccination results
print(f"Total vaccinated susceptible: {sum(results['S_vax'][a][-1] for a in range(3)):.0f}")
print(f"Breakthrough infections (cumulative): {results['breakthrough_infections'][-1]:.0f}")
print(f"Vaccinated deaths: {sum(results['D_vax'][a][-1] for a in range(3)):.0f}")
```

#### Using Vaccine Profiles

```python
from config import get_vaccine_profile, list_vaccine_profiles, describe_vaccine_profile

# List available profiles
print(list_vaccine_profiles())
# ['mrna_original', 'mrna_omicron', 'adenovirus', 'inactivated', 'influenza_typical', 'ideal', 'minimal']

# Describe a profile
print(describe_vaccine_profile('mrna_original'))

# Use profile in simulation
profile = get_vaccine_profile('mrna_original')
results = simulate_master_hospital_model(
    beta_base=0.3,
    age_params=config.AGE_PARAMS_DEFAULT,
    contact_matrix=config.CONTACT_MATRIX_DEFAULT,
    age_pops=[3000, 5000, 2000],
    vaccination_rate=0.01,
    VE_infection=profile['VE_infection'],
    VE_severe=profile['VE_severe'],
    VE_death=profile['VE_death'],
    Tmax=200
)
```

#### Age-Specific Vaccination Rates

```python
# Prioritize elderly vaccination
results = simulate_master_hospital_model(
    beta_base=0.3,
    age_params=config.AGE_PARAMS_DEFAULT,
    contact_matrix=config.CONTACT_MATRIX_DEFAULT,
    age_pops=[3000, 5000, 2000],
    
    vaccination_rate=[0.005, 0.01, 0.03],  # Elderly 3x faster
    VE_infection=0.7,
    VE_severe=0.85,
    VE_death=0.95,
    
    Tmax=200
)
```

#### Immunity Waning Example

```python
# Vaccine immunity wanes over time
results = simulate_master_hospital_model(
    beta_base=0.3,
    age_params=config.AGE_PARAMS_DEFAULT,
    contact_matrix=config.CONTACT_MATRIX_DEFAULT,
    age_pops=[3000, 5000, 2000],
    
    vaccination_rate=0.02,
    VE_infection=0.7,
    VE_severe=0.85,
    VE_death=0.95,
    
    # Vaccine waning configuration
    vaccine_waning_params={
        'omega_vax': 0.005,    # ~200 day immunity duration
        'wane_to_S': True,     # Waned individuals become fully susceptible
    },
    
    Tmax=365
)
```

#### Initial Vaccinated Population

```python
# Start with some population already vaccinated
results = simulate_master_hospital_model(
    beta_base=0.3,
    age_params=config.AGE_PARAMS_DEFAULT,
    contact_matrix=config.CONTACT_MATRIX_DEFAULT,
    age_pops=[3000, 5000, 2000],
    
    initial_conditions={
        'S_vax_by_age': [500, 1500, 1200],  # Already vaccinated
        'I_by_age': [10, 10, 10],           # Initial infections
    },
    
    vaccination_rate=0.01,
    VE_infection=0.7,
    VE_severe=0.85,
    VE_death=0.95,
    
    Tmax=200
)
```

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

The `config.py` module provides a hierarchical configuration system with ready-to-use scenario bundles, modular presets, and helper functions for the master hospital model.

### Quick Start with Scenarios

The easiest way to run simulations is using pre-built scenario bundles:

```python
from config import get_scenario_params, list_scenarios, describe_scenario
from master_hospital_model import simulate_master_hospital_model

# List available scenarios
print(list_scenarios())
# ['baseline', 'covid_early_2020', 'covid_delta', 'covid_omicron', 'seasonal_flu',
#  'endemic', 'stress_test', 'optimal_response', 'resource_limited', ...]

# Get detailed description
print(describe_scenario('covid_delta'))

# Run simulation with scenario defaults
params = get_scenario_params('covid_delta')
results = simulate_master_hospital_model(**params)
```

### Available Scenario Bundles

| Scenario | Description | Key Features |
|----------|-------------|---------------|
| `baseline` | Default moderate outbreak | Urban setting, no interventions |
| `covid_early_2020` | Early pandemic wave | Delayed lockdown, no vaccines |
| `covid_delta` | Delta variant wave | High transmission, partial vaccination |
| `covid_omicron` | Omicron wave | Very high transmission, immune escape |
| `seasonal_flu` | Typical flu season | Strong seasonality, elderly priority vaccines |
| `endemic` | Long-term dynamics | 2-year simulation with waning immunity |
| `stress_test` | Hospital capacity crisis | High transmission, limited rural capacity |
| `optimal_response` | Best-case intervention | Early lockdown + high vaccination |
| `resource_limited` | Low-resource setting | Constrained healthcare system |
| `school_outbreak` | School-seeded outbreak | Young seed, school closure |
| `care_home_outbreak` | Care home outbreak | Elderly seed, shielding |
| `surge_capacity` | Surge capacity test | Field hospital activation |
| `cyclical_policy` | On-off lockdowns | Intermittent intervention strategy |

### Configuration Hierarchy

`config.py` is organized into modular presets that can be mixed and matched:

#### 1. Transmission Rate Presets (`TRANSMISSION_PRESETS`)

```python
TRANSMISSION_PRESETS = {
    'very_mild': {'beta_base': 0.12, 'approx_R0': 1.2},  # Seasonal cold
    'mild':      {'beta_base': 0.18, 'approx_R0': 1.5},  # Seasonal flu
    'moderate':  {'beta_base': 0.28, 'approx_R0': 2.5},  # Early COVID
    'high':      {'beta_base': 0.38, 'approx_R0': 3.5},  # Delta variant
    'severe':    {'beta_base': 0.45, 'approx_R0': 4.5},  # Omicron
    'extreme':   {'beta_base': 0.55, 'approx_R0': 6.0},  # Measles-like
}
```

#### 2. Healthcare System Configurations

Bundled capacity + population presets:

| System | Ward | ICU | Population | Description |
|--------|------|-----|------------|-------------|
| `HEALTHCARE_SYSTEM_SMALL` | 40 | 8 | 40,000 | Rural community hospital |
| `HEALTHCARE_SYSTEM_RURAL` | 80 | 20 | 100,000 | Regional rural network |
| `HEALTHCARE_SYSTEM_SUBURBAN` | 160 | 40 | 200,000 | Mid-sized suburban |
| `HEALTHCARE_SYSTEM_URBAN` | 400 | 100 | 400,000 | Urban medical center |
| `HEALTHCARE_SYSTEM_METROPOLITAN` | 1,600 | 400 | 2,000,000 | Large metro area |
| `HEALTHCARE_SYSTEM_WELL_RESOURCED` | 2,400 | 600 | 2,000,000 | High-income region |
| `HEALTHCARE_SYSTEM_RESOURCE_LIMITED` | 200 | 25 | 2,000,000 | Low-resource setting |
| `HEALTHCARE_SYSTEM_SURGE_MILD` | 500 | 125 | 400,000 | +25% surge capacity |
| `HEALTHCARE_SYSTEM_SURGE_MAJOR` | 600 | 150 | 400,000 | +50% field hospitals |

Access via helper function:

```python
from config import get_healthcare_systems
systems = get_healthcare_systems()  # Returns dict of all systems
```

#### 3. Intervention Templates

```python
INTERVENTION_EARLY_STRONG      # Day 14-60, 60% reduction
INTERVENTION_DELAYED_STRONG    # Day 45-105, 60% reduction
INTERVENTION_TIERED_ESCALATING # 3 phases: 20% → 40% → 60%
INTERVENTION_TIERED_DEESCALATING  # 3 phases: 60% → 40% → 20%
INTERVENTION_CYCLICAL          # On-off cycling (21 days on/off)
INTERVENTION_SUSTAINED_MODERATE # Day 30-150, 30% reduction
INTERVENTION_SHORT_SHARP       # Day 25-39, 70% reduction
INTERVENTION_MULTI_WAVE        # 3 separate intervention periods
```

#### 4. Waning Immunity Presets

```python
WANING_NONE       # No waning (permanent immunity)
WANING_SLOW       # omega=0.001, ~3 year immunity
WANING_MODERATE   # omega=0.003, ~1 year immunity
WANING_FAST       # omega=0.005, ~6 month immunity
WANING_VERY_FAST  # omega=0.01, ~3 month immunity
WANING_AGE_DIFFERENTIAL  # Age-specific: elderly wane faster
```

#### 5. Vaccination Strategies

```python
from config import get_vaccination_strategies

strategies = get_vaccination_strategies()  # Returns normalized {name: [coverage_list]}
# Available: 'none', 'uniform_low/moderate/high', 'elderly_priority', 
# 'elderly_only', 'working_age_priority', 'young_priority', 
# 'balanced_risk', 'herd_immunity_target'
```

#### 6. Contact Matrices

```python
CONTACT_MATRIX_DEFAULT       # Assortative mixing by age
CONTACT_MATRIX_HOMOGENEOUS   # Equal mixing across ages
CONTACT_MATRIX_ASSORTATIVE   # Strong within-age preference
CONTACT_MATRIX_SCHOOL_CLOSURE    # Reduced young contacts
CONTACT_MATRIX_WORK_FROM_HOME    # Reduced middle-age contacts
CONTACT_MATRIX_ELDERLY_SHIELDING # Reduced elderly contacts
```

### Creating Custom Scenarios

```python
from config import create_custom_scenario, HEALTHCARE_SYSTEM_URBAN, INTERVENTION_EARLY_STRONG

my_scenario = create_custom_scenario(
    name='My Custom Outbreak',
    beta_base=0.35,
    healthcare_system=HEALTHCARE_SYSTEM_URBAN,
    vaccination_coverage=[0.3, 0.5, 0.9],
    interventions=INTERVENTION_EARLY_STRONG,
    VE=0.8,
    Tmax=300,
    description='Custom scenario with high elderly vaccination'
)

# Convert to simulation parameters
from config import get_scenario_params
# Add to registry temporarily or use directly
```

### Backward Compatibility

> **Note:** All legacy exports continue to work for existing code. The following are preserved:

```python
# Legacy exports (still work)
from config import (
    AGE_POPS_DEFAULT,           # [3000, 5000, 2000]
    AGE_PARAMS_DEFAULT,         # Active parameter set
    CONTACT_MATRIX_DEFAULT,     # Default contact matrix
    DEFAULT_SIM_PARAMS,         # Simulation parameters
    DEFAULT_CAPACITY_PARAMS,    # Capacity parameters
    LOCKDOWN_SCENARIO,          # Legacy intervention
    MULTIPLE_WAVES_SCENARIO,    # Legacy intervention
    SEASONAL_PARAMS,            # Legacy seasonal params
    WANING_PARAMS,              # Legacy waning params
    VACCINATION_STRATEGIES,     # Includes both old and new formats
    YOUNG_PARAMS, MIDDLE_PARAMS, ELDERLY_PARAMS,  # Teaching params
)
```

### Default Parameters Reference

#### Simulation Parameters

```python
DEFAULT_SIM_PARAMS = {
    'Tmax': 200,          # Simulation duration (days)
    'time_step': 0.1,     # Euler step size
    'hill_coef': 4,       # Hill coefficient
    'theta_X': 0.5,       # Relative infectiousness of X compartments
    'theta_H': 0.3,       # Relative infectiousness of H compartments
    'theta_vax': 0.5,     # Relative infectiousness of vaccinated individuals
    'VE': 0.7             # Legacy vaccine efficacy (use Three-Factor model instead)
}
```

#### Split-X Architecture Parameters

| Parameter | Symbol | Description | Default |
|-----------|--------|-------------|---------|
| `gamma_X_admit` | $\gamma_{X,\text{admit}}$ | Rate of transfer from X_admitted to H_ward | 0.5 day⁻¹ (~2 day wait) |
| `mu_X` | $\mu_X$ | Treated mortality in X_admitted | Age-specific |
| `mu_X_untreated` | $\mu_{X,\text{untreated}}$ | Untreated mortality in X_queued | `mu_X × multiplier` |
| `mu_ward_denied_icu` | $\mu_{\text{ward,denied}}$ | Ward mortality when ICU denied | Age-specific |

#### Age-Specific Disease Parameters (Empirical/COVID-calibrated)

| Parameter | Young (0-19) | Middle (20-64) | Elderly (65+) |
|-----------|--------------|----------------|---------------|
| `alpha` (E→I) | 0.2 | 0.2 | 0.18 |
| `sigma` (→ severe) | 0.02 | 0.08 | 0.15 |
| `eta` (→ ward admission) | 0.05 | 0.15 | 0.35 |
| `eta_icu` (→ ICU) | 0.02 | 0.10 | 0.25 |
| `gamma_X` (recovery from X) | 0.2 | 0.15 | 0.10 |
| `gamma_X_admit` (X_admitted → H_ward) | 0.5 | 0.5 | 0.5 |
| `mu_X` (treated severe mortality) | 0.002 | 0.008 | 0.025 |
| `mu_X_untreated` (untreated severe) | 0.006 | 0.024 | 0.10 |
| `mu_ward` | 0.001 | 0.005 | 0.015 |
| `mu_ward_denied_icu` | 0.02 | 0.06 | 0.12 |
| `mu_icu` | 0.005 | 0.02 | 0.04 |

#### Differential Mortality Parameters

```python
DIFFERENTIAL_MORTALITY_PARAMS = {
    'mu_X_untreated_multiplier': 2.0,            # Baseline multiplier when hospital denied
    'mu_ward_denied_icu_multiplier': 1.5,        # Baseline multiplier when ICU denied
    # Age-specific overrides (elderly are most vulnerable when denied care):
    'mu_X_untreated_multiplier_young': 1.5,
    'mu_X_untreated_multiplier_middle': 2.0,
    'mu_X_untreated_multiplier_elderly': 3.0,
    'mu_ward_denied_icu_multiplier_young': 1.3,
    'mu_ward_denied_icu_multiplier_middle': 1.5,
    'mu_ward_denied_icu_multiplier_elderly': 2.0,
}
```

#### Three-Factor Vaccine Efficacy Parameters

| Parameter | Symbol | Description | Default (mRNA) |
|-----------|--------|-------------|----------------|
| `VE_infection` | $VE_I$ | Efficacy against infection | 0.80 |
| `VE_severe` | $VE_S$ | Efficacy against severe disease | 0.90 |
| `VE_death` | $VE_D$ | Efficacy against death | 0.95 |
| `theta_vax` | $\theta_{\text{vax}}$ | Relative infectiousness of breakthrough | 0.30 |
| `omega_vax` | $\omega_{\text{vax}}$ | Vaccine immunity waning rate (S_vax) | 0.002 day⁻¹ |

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

### Configuration Helper Functions (`config.py`)

#### `get_scenario_params(scenario_name)`

Extract parameters from a scenario bundle for `simulate_master_hospital_model()`.

```python
params = get_scenario_params('covid_delta')
results = simulate_master_hospital_model(**params)
```

#### `list_scenarios()`

Return list of available scenario names.

#### `describe_scenario(scenario_name)`

Return detailed multi-line description of a scenario.

#### `get_healthcare_systems()`

Return dictionary of all healthcare system configurations.

#### `get_vaccination_strategies()`

Return vaccination strategies as normalized `{name: [coverage_list]}` format, compatible with `compare_vaccination_strategies()` and other helper functions.

#### `validate_age_params(age_params)`

Validate that age parameter dictionaries contain all required keys. Returns `True` if valid, raises `ValueError` with details if not.

#### `create_custom_scenario(name, beta_base, healthcare_system, vaccination_coverage, **kwargs)`

Create a custom scenario configuration dictionary.

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

## Future Extensions

- Stochastic simulation using Gillespie algorithm
- Spatial/metapopulation structure
- Multiple pathogen strains
- Healthcare worker compartments
- Economic cost modeling
- Hospitalization duration distributions

---
