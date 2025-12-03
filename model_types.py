from typing import TypedDict, List, Optional, Union, Literal
import numpy as np
import numpy.typing as npt

# Type alias for contact matrix (numpy array)
ContactMatrix = npt.NDArray[np.float64]

class AgeParams(TypedDict):
    """Disease parameters for a single age group."""
    alpha: float
    sigma: float
    eta: float
    eta_icu: float
    gamma_I: float
    gamma_X: float
    gamma_X_admit: float
    gamma_ward: float
    gamma_icu: float
    mu_I: float
    mu_X: float
    mu_X_untreated: float
    mu_ward: float
    mu_ward_denied_icu: float
    mu_icu: float
    # Optional overrides
    gamma_H: Optional[float]
    mu_H: Optional[float]

class SimParams(TypedDict):
    """Core simulation parameters."""
    Tmax: float
    time_step: float
    theta_X: float
    theta_H: float
    VE: float
    theta_vax: float

class CapacityParams(TypedDict):
    """Healthcare capacity parameters."""
    ward_capacity: float
    icu_capacity: float
    total_capacity: float
    hill_coef_ward: float
    hill_coef_icu: float

class VaccineEfficacyParams(TypedDict):
    """Three-factor vaccine efficacy parameters."""
    VE_infection: float
    VE_severe: float
    VE_death: float
    theta_vax: float

class VaccineWaningParams(TypedDict):
    """Vaccine waning parameters."""
    omega_vax: float
    omega_vax_by_age: Optional[List[float]]
    waning_destination: Literal['S', 'S_vax']

class SeasonalParams(TypedDict):
    """Seasonality settings."""
    amplitude: float
    period: float
    peak_day: float
    description: str

class Intervention(TypedDict):
    """Policy intervention structure."""
    start_day: float
    end_day: float
    transmission_reduction: float

class DemographicParams(TypedDict):
    """Demographic parameters (births and deaths)."""
    birth_rate: float
    birth_age_distribution: List[float]
    mu_background: List[float]
    neonatal_vaccination_rate: float
    description: Optional[str]

class DifferentialMortalityParams(TypedDict):
    """Mortality multipliers for denied care."""
    mu_X_untreated_multiplier: float
    mu_ward_denied_icu_multiplier: float
    mu_X_untreated_multiplier_young: Optional[float]
    mu_X_untreated_multiplier_middle: Optional[float]
    mu_X_untreated_multiplier_elderly: Optional[float]
    mu_ward_denied_icu_multiplier_young: Optional[float]
    mu_ward_denied_icu_multiplier_middle: Optional[float]
    mu_ward_denied_icu_multiplier_elderly: Optional[float]

class ODEParams(TypedDict):
    """Aggregated parameters passed to the ODE solver."""
    n_ages: int
    contact_matrix: ContactMatrix
    
    # Vaccine Efficacy
    VE_infection: float
    VE_severe: float
    VE_death: float
    
    # Disease Parameters
    age_params: List[AgeParams]
    theta_X: float
    theta_H: float
    theta_vax: float
    
    # Waning and Vaccination
    omega: float
    vaccination_rate: float
    vax_waning_destination: Literal['S', 'S_vax']
    
    # Differential Mortality
    dm_params: DifferentialMortalityParams
    
    # Transmission and Seasonality
    beta_base: float
    seasonal_params: SeasonalParams
    intervention_params: List[Intervention]
    
    # Capacity
    ward_capacity: float
    icu_capacity: float
    hill_coef_ward: float
    hill_coef_icu: float
    
    # Demographics
    demographic_params: DemographicParams
