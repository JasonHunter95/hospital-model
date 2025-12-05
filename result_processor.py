"""
Result processing module for hospital model.

This module handles the unpacking and post-processing of raw ODE solver output
into the comprehensive results dictionary returned by simulate_master_hospital_model.
It contains functionality that was originally in the main simulation function,
but was refactored here because the model was becoming too complex for a single function.
"""

import numpy as np
from typing import Dict, List, Any
from utils import unpack_state
from capacity import hill_gate
from time_varying_helpers import seasonal_forcing, policy_multiplier
from model_types import ODEParams


class ResultProcessor:
    """
    Processes raw ODE solver output into comprehensive simulation results.

    This class handles:
    - Unpacking state vectors into compartment histories
    - Computing aggregates across age groups
    - Computing derived metrics (overflow, gating factors, etc.)
    - Computing compartment flows (optional)
    - Assembling the final results dictionary
    """
    
    def __init__(
        self,
        solution: np.ndarray,
        times: List[float],
        ode_params: ODEParams,
        coverage: List[float],
        vaccination_rate: np.ndarray,
        vaccine_waning_params: Dict,
        seasonal_params: Dict,
        waning_params: Dict,
        interventions: List[Dict],
        demographic_params: Dict,
        track_differential_mortality: bool,
        track_compartment_flows: bool,
        solver: str, # Name of ODE solver used (e.g., 'RK45', 'BDF', etc.)
        solver_method: str, # Solver method (there are several we can use)
        rtol: float, # Relative tolerance
        atol: float, # Absolute tolerance
        Tmax: float,
        time_step: float,
    ):
        """
        Initialize the ResultProcessor.
        
        Parameters
        ----------
        solution : np.ndarray
            Raw ODE solver output (n_times x n_states)
        times : list of float
            Time points corresponding to solution rows
        ode_params : ODEParams
            ODE parameters used in simulation
        coverage : list of float
            Vaccine coverage per age group
        vaccination_rate : np.ndarray
            Vaccination rate per age group
        vaccine_waning_params : dict
            Vaccine waning parameters
        seasonal_params : dict
            Seasonal forcing parameters
        waning_params : dict
            Natural immunity waning parameters
        interventions : list of dict
            Policy interventions
        demographic_params : dict
            Demographic parameters
        track_differential_mortality : bool
            Whether to track treated vs untreated deaths
        track_compartment_flows : bool
            Whether to track daily flows between compartments
        solver : str
            Name of ODE solver used
        solver_method : str
            ODE solver method
        rtol : float
            Relative tolerance
        atol : float
            Absolute tolerance
        Tmax : float
            Simulation end time
        time_step : float
            Integration time step
        """
        self.solution = solution
        self.times = times
        self.n_times = len(times)
        self.n_ages = ode_params['n_ages']
        self.age_params = ode_params['age_params']
        self.age_pops = ode_params['age_pops']
        self.beta_base = ode_params['beta_base']
        self.contact_matrix = ode_params['contact_matrix']
        self.K_ward = ode_params['K_ward']
        self.K_icu = ode_params['K_icu']
        self.n_ward = ode_params['n_ward']
        self.n_icu = ode_params['n_icu']
        self.VE_infection = ode_params['VE_infection']
        self.VE_severe = ode_params['VE_severe']
        self.VE_death = ode_params['VE_death']
        self.theta_X = ode_params['theta_X']
        self.theta_H = ode_params['theta_H']
        self.theta_vax = ode_params['theta_vax']
        
        self.coverage = coverage
        self.vaccination_rate = vaccination_rate
        self.vaccine_waning_params = vaccine_waning_params
        self.seasonal_params = seasonal_params
        self.waning_params = waning_params
        self.interventions = interventions
        self.demographic_params = demographic_params
        self.track_differential_mortality = track_differential_mortality
        self.track_compartment_flows = track_compartment_flows
        self.solver = solver
        self.solver_method = solver_method
        self.rtol = rtol
        self.atol = atol
        self.Tmax = Tmax
        self.time_step = time_step
        
        # Storage for computed results
        self.compartment_histories = {}
        self.aggregate_histories = {}
        self.derived_metrics = {}
        self.flow_histories = {}
    
    def process(self) -> Dict[str, Any]:
        """
        Process the raw ODE solution into comprehensive results.
        
        Returns
        -------
        dict
            Complete results dictionary with all compartments, aggregates,
            metrics, and metadata.
        """
        self.unpack_compartments()
        self.compute_aggregates_and_metrics()
        if self.track_compartment_flows:
            self.compute_flows()
        return self.build_results_dict()
    
    def unpack_compartments(self):
        """Extract raw compartment histories from the state vector."""
        # Initialize per-age compartment histories (unvaccinated)
        S_history = [[] for _ in range(self.n_ages)]
        E_history = [[] for _ in range(self.n_ages)]
        I_history = [[] for _ in range(self.n_ages)]
        X_queued_history = [[] for _ in range(self.n_ages)]
        X_admitted_history = [[] for _ in range(self.n_ages)]
        X_history = [[] for _ in range(self.n_ages)]  # Combined
        H_ward_history = [[] for _ in range(self.n_ages)]
        H_icu_history = [[] for _ in range(self.n_ages)]
        R_history = [[] for _ in range(self.n_ages)]
        D_history = [[] for _ in range(self.n_ages)]
        D_treated_history = [[] for _ in range(self.n_ages)]
        D_untreated_history = [[] for _ in range(self.n_ages)]
        
        # Initialize per-age compartment histories (vaccinated)
        S_vax_history = [[] for _ in range(self.n_ages)]
        E_vax_history = [[] for _ in range(self.n_ages)]
        I_vax_history = [[] for _ in range(self.n_ages)]
        X_queued_vax_history = [[] for _ in range(self.n_ages)]
        X_admitted_vax_history = [[] for _ in range(self.n_ages)]
        X_vax_history = [[] for _ in range(self.n_ages)]  # Combined
        H_ward_vax_history = [[] for _ in range(self.n_ages)]
        H_icu_vax_history = [[] for _ in range(self.n_ages)]
        R_vax_history = [[] for _ in range(self.n_ages)]
        D_vax_history = [[] for _ in range(self.n_ages)]
        D_vax_treated_history = [[] for _ in range(self.n_ages)]
        D_vax_untreated_history = [[] for _ in range(self.n_ages)]
        
        # Demographic tracking histories
        cum_births_history = [[] for _ in range(self.n_ages)]
        cum_background_deaths_history = [[] for _ in range(self.n_ages)]
        
        # Extract per-age histories from solution
        for t_idx in range(self.n_times):
            state = unpack_state(self.solution[t_idx], self.n_ages)
            for a in range(self.n_ages):
                S_history[a].append(state['S'][a])
                E_history[a].append(state['E'][a])
                I_history[a].append(state['I'][a])
                X_queued_history[a].append(state['X_queued'][a])
                X_admitted_history[a].append(state['X_admitted'][a])
                X_history[a].append(state['X_queued'][a] + state['X_admitted'][a])
                H_ward_history[a].append(state['H_ward'][a])
                H_icu_history[a].append(state['H_icu'][a])
                R_history[a].append(state['R'][a])
                D_history[a].append(state['D'][a])
                D_treated_history[a].append(state['D_treated'][a])
                D_untreated_history[a].append(state['D_untreated'][a])
                
                S_vax_history[a].append(state['S_vax'][a])
                E_vax_history[a].append(state['E_vax'][a])
                I_vax_history[a].append(state['I_vax'][a])
                X_queued_vax_history[a].append(state['X_queued_vax'][a])
                X_admitted_vax_history[a].append(state['X_admitted_vax'][a])
                X_vax_history[a].append(state['X_queued_vax'][a] + state['X_admitted_vax'][a])
                H_ward_vax_history[a].append(state['H_ward_vax'][a])
                H_icu_vax_history[a].append(state['H_icu_vax'][a])
                R_vax_history[a].append(state['R_vax'][a])
                D_vax_history[a].append(state['D_vax'][a])
                D_vax_treated_history[a].append(state['D_vax_treated'][a])
                D_vax_untreated_history[a].append(state['D_vax_untreated'][a])
                cum_births_history[a].append(state['cum_births'][a])
                cum_background_deaths_history[a].append(state['cum_background_deaths'][a])
        
        # Store in compartment_histories
        self.compartment_histories = {
            'S': S_history,
            'E': E_history,
            'I': I_history,
            'X_queued': X_queued_history,
            'X_admitted': X_admitted_history,
            'X': X_history,
            'H_ward': H_ward_history,
            'H_icu': H_icu_history,
            'R': R_history,
            'D': D_history,
            'D_treated': D_treated_history,
            'D_untreated': D_untreated_history,
            'S_vax': S_vax_history,
            'E_vax': E_vax_history,
            'I_vax': I_vax_history,
            'X_queued_vax': X_queued_vax_history,
            'X_admitted_vax': X_admitted_vax_history,
            'X_vax': X_vax_history,
            'H_ward_vax': H_ward_vax_history,
            'H_icu_vax': H_icu_vax_history,
            'R_vax': R_vax_history,
            'D_vax': D_vax_history,
            'D_vax_treated': D_vax_treated_history,
            'D_vax_untreated': D_vax_untreated_history,
            'cum_births': cum_births_history,
            'cum_background_deaths': cum_background_deaths_history,
        }
    
    def compute_aggregates_and_metrics(self):
        """Compute aggregated totals and derived metrics."""
        # Initialize aggregate histories
        H_ward_total_history = []
        H_icu_total_history = []
        H_total_history = []
        E_total_history = []
        I_total_history = []
        X_total_history = []
        D_total_history = []
        D_treated_total_history = []
        D_untreated_total_history = []
        
        # Vaccinated aggregates
        H_ward_vax_total_history = []
        H_icu_vax_total_history = []
        H_vax_total_history = []
        E_vax_total_history = []
        I_vax_total_history = []
        X_vax_total_history = []
        D_vax_total_history = []
        vaccinated_total_history = []
        breakthrough_infections_history = []
        
        # Capacity metrics
        ward_overflow_history = []
        icu_overflow_history = []
        g_ward_history = []
        g_icu_history = []
        
        # Demographic aggregates
        cum_births_total_history = []
        cum_background_deaths_total_history = []
        live_population_history = []
        
        # Time-varying parameter tracking
        beta_t_history = []
        seasonal_factor_history = []
        policy_mult_history = []
        
        # Cumulative overflow (computed via trapezoidal integration)
        cum_ward_overflow = 0.0
        cum_icu_overflow = 0.0
        cum_unmet_ward = [0.0] * self.n_ages
        cum_unmet_icu = [0.0] * self.n_ages
        
        # Iterate through solution to compute auxiliary metrics
        for t_idx in range(self.n_times):
            t = self.times[t_idx]
            state = unpack_state(self.solution[t_idx], self.n_ages)
            
            # Extract compartments
            S_t = state['S']
            E_t = state['E']
            I_t = state['I']
            X_t = state['X_queued'] + state['X_admitted']
            H_ward_t = state['H_ward']
            H_icu_t = state['H_icu']
            R_t = state['R']
            D_t = state['D']
            S_vax_t = state['S_vax']
            E_vax_t = state['E_vax']
            I_vax_t = state['I_vax']
            X_vax_t = state['X_queued_vax'] + state['X_admitted_vax']
            H_ward_vax_t = state['H_ward_vax']
            H_icu_vax_t = state['H_icu_vax']
            R_vax_t = state['R_vax']
            D_vax_t = state['D_vax']
            D_treated_t = state['D_treated']
            D_untreated_t = state['D_untreated']
            D_vax_treated_t = state['D_vax_treated']
            D_vax_untreated_t = state['D_vax_untreated']
            cum_breakthrough_t = state['cum_breakthrough']
            cum_births_t = state['cum_births']
            cum_background_deaths_t = state['cum_background_deaths']
            
            # Compute aggregates
            H_ward_total = np.sum(H_ward_t) + np.sum(H_ward_vax_t)
            H_icu_total = np.sum(H_icu_t) + np.sum(H_icu_vax_t)
            H_total = H_ward_total + H_icu_total
            E_total = np.sum(E_t) + np.sum(E_vax_t)
            I_total = np.sum(I_t) + np.sum(I_vax_t)
            X_total = np.sum(X_t) + np.sum(X_vax_t)
            D_total = np.sum(D_t) + np.sum(D_vax_t)
            
            # Demographic aggregates
            cum_births_total = np.sum(cum_births_t)
            cum_background_deaths_total = np.sum(cum_background_deaths_t)
            # Live population = all compartments except D (dead)
            live_pop = (np.sum(S_t) + np.sum(E_t) + np.sum(I_t) + np.sum(X_t) + 
                        np.sum(H_ward_t) + np.sum(H_icu_t) + np.sum(R_t) +
                        np.sum(S_vax_t) + np.sum(E_vax_t) + np.sum(I_vax_t) + np.sum(X_vax_t) + 
                        np.sum(H_ward_vax_t) + np.sum(H_icu_vax_t) + np.sum(R_vax_t))
            
            H_ward_vax_total = np.sum(H_ward_vax_t)
            H_icu_vax_total = np.sum(H_icu_vax_t)
            H_vax_total = H_ward_vax_total + H_icu_vax_total
            vaccinated_total = (np.sum(S_vax_t) + np.sum(E_vax_t) + np.sum(I_vax_t) + np.sum(X_vax_t) + 
                               H_vax_total + np.sum(R_vax_t) + np.sum(D_vax_t))
            
            H_ward_total_history.append(H_ward_total)
            H_icu_total_history.append(H_icu_total)
            H_total_history.append(H_total)
            E_total_history.append(E_total)
            I_total_history.append(I_total)
            X_total_history.append(X_total)
            D_total_history.append(D_total)
            D_treated_total_history.append(np.sum(D_treated_t) + np.sum(D_vax_treated_t))
            D_untreated_total_history.append(np.sum(D_untreated_t) + np.sum(D_vax_untreated_t))
            
            H_ward_vax_total_history.append(H_ward_vax_total)
            H_icu_vax_total_history.append(H_icu_vax_total)
            H_vax_total_history.append(H_vax_total)
            E_vax_total_history.append(np.sum(E_vax_t))
            I_vax_total_history.append(np.sum(I_vax_t))
            X_vax_total_history.append(np.sum(X_vax_t))
            D_vax_total_history.append(np.sum(D_vax_t))
            vaccinated_total_history.append(vaccinated_total)
            breakthrough_infections_history.append(np.sum(cum_breakthrough_t))
            
            # Demographic aggregates
            cum_births_total_history.append(cum_births_total)
            cum_background_deaths_total_history.append(cum_background_deaths_total)
            live_population_history.append(live_pop)
            
            # Time-varying parameters
            seasonal_factor = seasonal_forcing(
                t, 1.0,
                amplitude=self.seasonal_params.get('amplitude', 0.0),
                period=self.seasonal_params.get('period', 365),
                peak_day=self.seasonal_params.get('peak_day', 0)
            )
            policy_mult = policy_multiplier(t, self.interventions)
            beta_t = self.beta_base * seasonal_factor * policy_mult
            
            beta_t_history.append(beta_t)
            seasonal_factor_history.append(seasonal_factor)
            policy_mult_history.append(policy_mult)
            
            # Capacity gating
            g_ward = hill_gate(H_ward_total, self.K_ward, self.n_ward)
            g_icu = hill_gate(H_icu_total, self.K_icu, self.n_icu)
            g_ward_history.append(g_ward)
            g_icu_history.append(g_icu)
            
            # Overflow
            ward_overflow = max(0, H_ward_total - self.K_ward)
            icu_overflow = max(0, H_icu_total - self.K_icu)
            ward_overflow_history.append(ward_overflow)
            icu_overflow_history.append(icu_overflow)
            
            # Cumulative overflow (trapezoidal integration)
            if t_idx > 0:
                dt_local = self.times[t_idx] - self.times[t_idx - 1]
                cum_ward_overflow += 0.5 * (ward_overflow_history[-1] + ward_overflow_history[-2]) * dt_local
                cum_icu_overflow += 0.5 * (icu_overflow_history[-1] + icu_overflow_history[-2]) * dt_local
                
                # Compute unmet care per age group
                for a in range(self.n_ages):
                    eta_a = self.age_params[a]['eta']
                    eta_icu_a = self.age_params[a].get('eta_icu', 0.1)
                    desired_ward = eta_a * (X_t[a] + X_vax_t[a])
                    actual_ward = desired_ward * g_ward
                    unmet_ward_a = max(0, desired_ward - actual_ward)
                    
                    desired_icu = eta_icu_a * (H_ward_t[a] + H_ward_vax_t[a])
                    actual_icu = desired_icu * g_icu
                    unmet_icu_a = max(0, desired_icu - actual_icu)
                    
                    cum_unmet_ward[a] += unmet_ward_a * dt_local
                    cum_unmet_icu[a] += unmet_icu_a * dt_local
        
        # Store aggregates and derived metrics
        self.aggregate_histories = {
            'H_ward_total': H_ward_total_history,
            'H_icu_total': H_icu_total_history,
            'H_total': H_total_history,
            'E_total': E_total_history,
            'I_total': I_total_history,
            'X_total': X_total_history,
            'D_total': D_total_history,
            'D_treated_total': D_treated_total_history,
            'D_untreated_total': D_untreated_total_history,
            'H_ward_vax_total': H_ward_vax_total_history,
            'H_icu_vax_total': H_icu_vax_total_history,
            'H_vax_total': H_vax_total_history,
            'E_vax_total': E_vax_total_history,
            'I_vax_total': I_vax_total_history,
            'X_vax_total': X_vax_total_history,
            'D_vax_total': D_vax_total_history,
            'vaccinated_total': vaccinated_total_history,
            'breakthrough_infections': breakthrough_infections_history,
            'cum_births_total': cum_births_total_history,
            'cum_background_deaths_total': cum_background_deaths_total_history,
            'live_population': live_population_history,
        }
        
        self.derived_metrics = {
            'ward_overflow': ward_overflow_history,
            'icu_overflow': icu_overflow_history,
            'cum_ward_overflow': cum_ward_overflow,
            'cum_icu_overflow': cum_icu_overflow,
            'cum_overflow': cum_ward_overflow + cum_icu_overflow,
            'cum_unmet_ward': cum_unmet_ward,
            'cum_unmet_icu': cum_unmet_icu,
            'cum_unmet': [cum_unmet_ward[a] + cum_unmet_icu[a] for a in range(self.n_ages)],
            'g_ward': g_ward_history,
            'g_icu': g_icu_history,
            'beta_t': beta_t_history,
            'seasonal_factor': seasonal_factor_history,
            'policy_mult': policy_mult_history,
        }
    
    def compute_flows(self):
        """Compute daily flows between compartments."""
        new_infections_history = []
        ward_admissions_history = []
        icu_admissions_history = []
        new_vaccinations_history = []
        breakthrough_infections_daily_history = []
        
        # Flow tracking from state differences
        for t_idx in range(1, self.n_times):
            state_prev = unpack_state(self.solution[t_idx - 1], self.n_ages)
            state = unpack_state(self.solution[t_idx], self.n_ages)
            dt_local = self.times[t_idx] - self.times[t_idx - 1]
            g_ward_prev = self.derived_metrics['g_ward'][t_idx - 1]
            g_icu_prev = self.derived_metrics['g_icu'][t_idx - 1]
            
            # Approximate flows from compartment changes
            # new_infections ~ alpha * E (rate of E -> I)
            alpha_arr = np.array([self.age_params[a].get('alpha', 0.2) for a in range(self.n_ages)])
            new_inf = alpha_arr * state_prev['E']
            new_infections_history.append(list(new_inf))
            
            # Ward admissions come from X_admitted (already past the admission gate)
            # ward_admissions ~ eta * X_admitted * g_ward
            eta_arr = np.array([self.age_params[a]['eta'] for a in range(self.n_ages)])
            X_prev = state_prev['X_queued'] + state_prev['X_admitted']
            X_vax_prev = state_prev['X_queued_vax'] + state_prev['X_admitted_vax']
            ward_adm = eta_arr * (X_prev + X_vax_prev) * g_ward_prev
            ward_admissions_history.append(list(ward_adm))
            
            # ICU admissions ~ eta_icu * H_ward * g_icu
            eta_icu_arr = np.array([self.age_params[a].get('eta_icu', 0.1) for a in range(self.n_ages)])
            icu_adm = eta_icu_arr * (state_prev['H_ward'] + state_prev['H_ward_vax']) * g_icu_prev
            icu_admissions_history.append(list(icu_adm))
            
            # New vaccinations ~ vaccination_rate * S
            new_vax = self.vaccination_rate * state_prev['S']
            new_vaccinations_history.append(list(new_vax))
            
            # Breakthrough infections (rate of change)
            cum_breakthrough_t = state['cum_breakthrough']
            breakthrough_rate = (cum_breakthrough_t - state_prev['cum_breakthrough']) / dt_local
            breakthrough_infections_daily_history.append(list(breakthrough_rate))
        
        self.flow_histories = {
            'new_infections': new_infections_history,
            'ward_admissions': ward_admissions_history,
            'icu_admissions': icu_admissions_history,
            'new_vaccinations': new_vaccinations_history,
            'breakthrough_infections_daily': breakthrough_infections_daily_history,
        }
    
    def build_results_dict(self) -> Dict[str, Any]:
        """Build the final results dictionary."""
        results = {
            # Time
            'times': self.times,
            
            # Per-age compartments (unvaccinated)
            'S': self.compartment_histories['S'],
            'E': self.compartment_histories['E'],
            'I': self.compartment_histories['I'],
            'X': self.compartment_histories['X'],
            'X_queued': self.compartment_histories['X_queued'],
            'X_admitted': self.compartment_histories['X_admitted'],
            'H_ward': self.compartment_histories['H_ward'],
            'H_icu': self.compartment_histories['H_icu'],
            'H': [[(self.compartment_histories['H_ward'][a][t] + self.compartment_histories['H_icu'][a][t]) 
                   for t in range(len(self.times))] for a in range(self.n_ages)],
            'R': self.compartment_histories['R'],
            'D': self.compartment_histories['D'],
            
            # Per-age compartments (vaccinated)
            'S_vax': self.compartment_histories['S_vax'],
            'E_vax': self.compartment_histories['E_vax'],
            'I_vax': self.compartment_histories['I_vax'],
            'X_vax': self.compartment_histories['X_vax'],
            'X_queued_vax': self.compartment_histories['X_queued_vax'],
            'X_admitted_vax': self.compartment_histories['X_admitted_vax'],
            'H_ward_vax': self.compartment_histories['H_ward_vax'],
            'H_icu_vax': self.compartment_histories['H_icu_vax'],
            'H_vax': [[(self.compartment_histories['H_ward_vax'][a][t] + self.compartment_histories['H_icu_vax'][a][t]) 
                       for t in range(len(self.times))] for a in range(self.n_ages)],
            'R_vax': self.compartment_histories['R_vax'],
            'D_vax': self.compartment_histories['D_vax'],
            
            # Aggregated totals
            **self.aggregate_histories,
            
            # Demographic outputs
            'cum_births': self.compartment_histories['cum_births'],
            'cum_background_deaths': self.compartment_histories['cum_background_deaths'],
            
            # Derived metrics
            **self.derived_metrics,
            
            # Metadata
            'ward_capacity': self.K_ward,
            'icu_capacity': self.K_icu,
            'age_pops': self.age_pops,
            
            # Parameters for reproducibility
            'parameters': {
                'beta_base': self.beta_base,
                'coverage': self.coverage,
                'VE_infection': self.VE_infection,
                'VE_severe': self.VE_severe,
                'VE_death': self.VE_death,
                'vaccination_rate': list(self.vaccination_rate) if isinstance(self.vaccination_rate, np.ndarray) else self.vaccination_rate,
                'theta_vax': self.theta_vax,
                'vaccine_waning_params': self.vaccine_waning_params,
                'theta_X': self.theta_X,
                'theta_H': self.theta_H,
                'seasonal_params': self.seasonal_params,
                'waning_params': self.waning_params,
                'interventions': self.interventions,
                'demographic_params': self.demographic_params,
                'Tmax': self.Tmax,
                'time_step': self.time_step,
                'track_differential_mortality': self.track_differential_mortality,
                'track_compartment_flows': self.track_compartment_flows,
                'age_params': self.age_params,
                'contact_matrix': self.contact_matrix.tolist() if isinstance(self.contact_matrix, np.ndarray) else self.contact_matrix,
                'solver': self.solver,
                'solver_method': self.solver_method,
                'rtol': self.rtol,
                'atol': self.atol,
            }
        }
        
        # Differential mortality results
        if self.track_differential_mortality:
            results.update({
                'D_treated': self.compartment_histories['D_treated'],
                'D_untreated': self.compartment_histories['D_untreated'],
                'D_treated_total': self.aggregate_histories['D_treated_total'],
                'D_untreated_total': self.aggregate_histories['D_untreated_total'],
                'D_vax_treated': self.compartment_histories['D_vax_treated'],
                'D_vax_untreated': self.compartment_histories['D_vax_untreated'],
            })
        
        # Compartment flows
        if self.track_compartment_flows:
            results.update(self.flow_histories)
        
        return results
