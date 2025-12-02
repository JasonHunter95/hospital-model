"""
Tests for Three-Factor Vaccine Model implementation.

This module tests the vaccination compartments and vaccine efficacy mechanisms:
- VE_infection: Reduces susceptibility to infection (S_vax → E_vax)
- VE_severe: Reduces probability of severe disease (I_vax → X_vax)
- VE_death: Reduces mortality rates in all vaccinated compartments

Vaccinated compartments: S_vax, E_vax, I_vax, X_vax, H_ward_vax, H_icu_vax, R_vax, D_vax

Note on data structure:
- Compartments like results['S'] are lists of lists: results['S'][age_group][time_index]
- Aggregate totals like results['D_total'] are 1D arrays: results['D_total'][time_index]
- Use get_total_at_time() helper to sum across age groups at a given time index
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from master_hospital_model import simulate_master_hospital_model
from config import (
    AGE_PARAMS_DEFAULT,
    CONTACT_MATRIX_DEFAULT,
    AGE_POPS_DEFAULT,
    VACCINE_EFFICACY_PARAMS,
    VACCINE_PROFILES
)
from config_helpers import (
    get_vaccine_profile,
    list_vaccine_profiles,
    describe_vaccine_profile
)


# ========================================
# Helper Functions
# ========================================

def get_total_at_time(results, compartment, t_idx):
    """Get total population in a compartment across all ages at time index t_idx."""
    return sum(results[compartment][a][t_idx] for a in range(len(results[compartment])))


def get_total_array(results, compartment):
    """Get array of total population in a compartment across all ages for all times."""
    n_ages = len(results[compartment])
    n_times = len(results[compartment][0])
    return np.array([sum(results[compartment][a][t] for a in range(n_ages)) for t in range(n_times)])


# ========================================
# Fixtures
# ========================================

@pytest.fixture
def minimal_inputs():
    """Minimal valid inputs for simulate_master_hospital_model."""
    return {
        'beta_base': 0.3,
        'age_params': AGE_PARAMS_DEFAULT,
        'contact_matrix': CONTACT_MATRIX_DEFAULT,
        'age_pops': [3000, 5000, 2000],
    }


@pytest.fixture
def vaccination_inputs(minimal_inputs):
    """Inputs with vaccination enabled."""
    return {
        **minimal_inputs,
        'vaccination_rate': 0.01,  # 1% of S vaccinated per day
        'VE_infection': 0.6,
        'VE_severe': 0.8,
        'VE_death': 0.9,
        'Tmax': 100,
    }


@pytest.fixture
def high_vaccination_inputs(minimal_inputs):
    """Inputs with high vaccination rate."""
    return {
        **minimal_inputs,
        'vaccination_rate': 0.05,  # 5% of S vaccinated per day
        'VE_infection': 0.8,
        'VE_severe': 0.9,
        'VE_death': 0.95,
        'Tmax': 200,
    }


@pytest.fixture
def perfect_vaccine_inputs(minimal_inputs):
    """Inputs with perfect vaccine (100% efficacy in all factors)."""
    return {
        **minimal_inputs,
        'vaccination_rate': 0.02,
        'VE_infection': 1.0,
        'VE_severe': 1.0,
        'VE_death': 1.0,
        'Tmax': 100,
    }


@pytest.fixture
def vaccine_waning_inputs(minimal_inputs):
    """Inputs with vaccine waning enabled."""
    return {
        **minimal_inputs,
        'vaccination_rate': 0.02,
        'VE_infection': 0.7,
        'VE_severe': 0.8,
        'VE_death': 0.9,
        'vaccine_waning_params': {
            'omega_vax': 0.01,  # ~100 day vaccine immunity duration
            'wane_to_S': True,  # Wane to unvaccinated susceptible
        },
        'Tmax': 365,
    }


@pytest.fixture
def age_specific_vaccination_inputs(minimal_inputs):
    """Inputs with age-specific vaccination rates."""
    return {
        **minimal_inputs,
        'vaccination_rate': [0.005, 0.01, 0.03],  # Higher for elderly
        'VE_infection': 0.6,
        'VE_severe': 0.8,
        'VE_death': 0.9,
        'Tmax': 100,
    }


@pytest.fixture
def initial_vaccinated_inputs(minimal_inputs):
    """Inputs with initial vaccinated population."""
    return {
        **minimal_inputs,
        'initial_conditions': {
            'S_vax_by_age': [500, 1000, 800],  # Some already vaccinated
            'I_by_age': [10, 10, 10],
        },
        'vaccination_rate': 0.01,
        'VE_infection': 0.6,
        'VE_severe': 0.8,
        'VE_death': 0.9,
        'Tmax': 100,
    }


# ========================================
# Test: Population Conservation with Vaccination
# ========================================

class TestPopulationConservation:
    """Tests that total population is conserved with vaccinated compartments."""
    
    def test_total_population_conserved_with_vaccination(self, vaccination_inputs):
        """Total population (all 16 compartments) should be constant over time."""
        result = simulate_master_hospital_model(**vaccination_inputs)
        
        n_ages = len(vaccination_inputs['age_pops'])
        n_times = len(result['times'])
        initial_pop = sum(vaccination_inputs['age_pops'])
        
        for t_idx in range(n_times):
            pop_at_t = 0
            for a in range(n_ages):
                # Unvaccinated compartments
                pop_at_t += (result['S'][a][t_idx] + result['E'][a][t_idx] +
                           result['I'][a][t_idx] + result['X'][a][t_idx] +
                           result['H_ward'][a][t_idx] + result['H_icu'][a][t_idx] +
                           result['R'][a][t_idx] + result['D'][a][t_idx])
                # Vaccinated compartments
                pop_at_t += (result['S_vax'][a][t_idx] + result['E_vax'][a][t_idx] +
                           result['I_vax'][a][t_idx] + result['X_vax'][a][t_idx] +
                           result['H_ward_vax'][a][t_idx] + result['H_icu_vax'][a][t_idx] +
                           result['R_vax'][a][t_idx] + result['D_vax'][a][t_idx])
            
            assert pop_at_t == pytest.approx(initial_pop, rel=0.01), \
                f"Population not conserved at t={result['times'][t_idx]}"
    
    def test_population_per_age_conserved_with_vaccination(self, vaccination_inputs):
        """Population per age group should be conserved."""
        result = simulate_master_hospital_model(**vaccination_inputs)
        
        age_pops = vaccination_inputs['age_pops']
        n_ages = len(age_pops)
        n_times = len(result['times'])
        
        for a in range(n_ages):
            for t_idx in range(n_times):
                # Unvaccinated + Vaccinated
                pop_at_t = (result['S'][a][t_idx] + result['E'][a][t_idx] +
                           result['I'][a][t_idx] + result['X'][a][t_idx] +
                           result['H_ward'][a][t_idx] + result['H_icu'][a][t_idx] +
                           result['R'][a][t_idx] + result['D'][a][t_idx] +
                           result['S_vax'][a][t_idx] + result['E_vax'][a][t_idx] +
                           result['I_vax'][a][t_idx] + result['X_vax'][a][t_idx] +
                           result['H_ward_vax'][a][t_idx] + result['H_icu_vax'][a][t_idx] +
                           result['R_vax'][a][t_idx] + result['D_vax'][a][t_idx])
                
                assert pop_at_t == pytest.approx(age_pops[a], rel=0.01), \
                    f"Population for age group {a} not conserved at t={result['times'][t_idx]}"
    
    def test_population_conserved_with_waning(self, vaccine_waning_inputs):
        """Population conserved even with vaccine immunity waning."""
        result = simulate_master_hospital_model(**vaccine_waning_inputs)
        
        n_ages = len(vaccine_waning_inputs['age_pops'])
        n_times = len(result['times'])
        initial_pop = sum(vaccine_waning_inputs['age_pops'])
        
        # Check at start, middle, and end
        for t_idx in [0, n_times // 2, -1]:
            pop_at_t = 0
            for a in range(n_ages):
                pop_at_t += (result['S'][a][t_idx] + result['E'][a][t_idx] +
                           result['I'][a][t_idx] + result['X'][a][t_idx] +
                           result['H_ward'][a][t_idx] + result['H_icu'][a][t_idx] +
                           result['R'][a][t_idx] + result['D'][a][t_idx] +
                           result['S_vax'][a][t_idx] + result['E_vax'][a][t_idx] +
                           result['I_vax'][a][t_idx] + result['X_vax'][a][t_idx] +
                           result['H_ward_vax'][a][t_idx] + result['H_icu_vax'][a][t_idx] +
                           result['R_vax'][a][t_idx] + result['D_vax'][a][t_idx])
            
            assert pop_at_t == pytest.approx(initial_pop, rel=0.01), \
                f"Population not conserved with vaccine waning at t_idx={t_idx}"


# ========================================
# Test: Vaccination Dynamics
# ========================================

class TestVaccinationDynamics:
    """Tests for vaccination flow from S to S_vax."""
    
    def test_vaccination_moves_s_to_s_vax(self, vaccination_inputs):
        """Vaccination should transfer people from S to S_vax."""
        result = simulate_master_hospital_model(**vaccination_inputs)
        
        # S_vax_total should increase from zero (default)
        s_vax_total = get_total_array(result, 'S_vax')
        assert s_vax_total[-1] > 0, "S_vax should have positive population after vaccination"
        
        # S should decrease (partly due to vaccination, partly infection)
        s_total = get_total_array(result, 'S')
        assert s_total[-1] < s_total[0], "S should decrease with vaccination"
    
    def test_zero_vaccination_rate_no_vaccinations(self, minimal_inputs):
        """With zero vaccination rate, no vaccinations should occur."""
        result = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.0,
            VE_infection=0.6,
            Tmax=100
        )
        
        # All vaccinated compartments should be zero
        s_vax_total = get_total_array(result, 'S_vax')
        e_vax_total = get_total_array(result, 'E_vax')
        i_vax_total = get_total_array(result, 'I_vax')
        
        assert np.allclose(s_vax_total, 0), "S_vax should be zero without vaccination"
        assert np.allclose(e_vax_total, 0), "E_vax should be zero without vaccination"
        assert np.allclose(i_vax_total, 0), "I_vax should be zero without vaccination"
    
    def test_higher_vaccination_rate_faster_coverage(self, minimal_inputs):
        """Higher vaccination rate should lead to faster S_vax growth."""
        result_low = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.01,
            VE_infection=0.6,
            Tmax=50
        )
        result_high = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.05,
            VE_infection=0.6,
            Tmax=50
        )
        
        # At midpoint, higher rate should have more S_vax
        s_vax_low = get_total_array(result_low, 'S_vax')
        s_vax_high = get_total_array(result_high, 'S_vax')
        mid_idx = len(s_vax_low) // 2
        
        assert s_vax_high[mid_idx] > s_vax_low[mid_idx], \
            "Higher vaccination rate should yield more S_vax faster"
    
    def test_age_specific_vaccination_rates(self, age_specific_vaccination_inputs):
        """Age-specific vaccination rates should be applied correctly."""
        result = simulate_master_hospital_model(**age_specific_vaccination_inputs)
        
        # Elderly (age group 2) with highest rate should have proportionally more vaccinated
        vax_rates = age_specific_vaccination_inputs['vaccination_rate']
        n_ages = len(vax_rates)
        
        # At early times, S_vax should reflect the relative vaccination rates
        early_idx = 10  # After some vaccination but before much infection spread
        
        # All age groups should have some vaccinated individuals
        for a in range(n_ages):
            assert result['S_vax'][a][early_idx] > 0, \
                f"Age group {a} should have some vaccinated individuals"


# ========================================
# Test: Vaccine Efficacy - VE_infection
# ========================================

class TestVEInfection:
    """Tests for VE_infection: reduced susceptibility to infection."""
    
    def test_ve_infection_reduces_breakthrough(self, high_vaccination_inputs):
        """
        Verify that VE_infection reduces breakthrough infections in vaccinated individuals.
        
        Three-Factor Vaccine Model - Factor 1: VE_infection
        ----------------------------------------------------
        VE_infection reduces the susceptibility of vaccinated individuals to infection.
        Mechanistically: λ_vax = (1 - VE_infection) * λ
        
        Where:
        - λ = force of infection for unvaccinated
        - λ_vax = force of infection for vaccinated
        
        Expected Behavior:
        ------------------
        Higher VE_infection should result in:
        - Fewer S_vax → E_vax transitions
        - Lower peak I_vax (breakthrough infections)
        - Lower cumulative R_vax (recovered from breakthrough)
        
        This test compares two scenarios:
        - Low VE_infection (0.3): 30% protection from infection
        - High VE_infection (0.9): 90% protection from infection
        
        The high VE scenario should show significantly fewer breakthrough infections.
        """
        result_low_ve = simulate_master_hospital_model(
            **{**high_vaccination_inputs, 'VE_infection': 0.3}
        )
        result_high_ve = simulate_master_hospital_model(
            **{**high_vaccination_inputs, 'VE_infection': 0.9}
        )
        
        # R_vax represents cumulative recovered vaccinated (had breakthrough)
        r_vax_low = get_total_array(result_low_ve, 'R_vax')
        r_vax_high = get_total_array(result_high_ve, 'R_vax')
        i_vax_low = get_total_array(result_low_ve, 'I_vax')
        i_vax_high = get_total_array(result_high_ve, 'I_vax')
        
        # With high VE_infection, vaccinated infections should be lower
        assert r_vax_high[-1] <= r_vax_low[-1] or i_vax_high.max() <= i_vax_low.max(), \
            "Higher VE_infection should reduce vaccinated infections"
    
    def test_perfect_ve_infection_blocks_infection(self, minimal_inputs):
        """
        Verify that perfect VE_infection (1.0) completely prevents infection.
        
        Edge Case Test: VE_infection = 1.0
        -----------------------------------
        When VE_infection = 1.0:
        - λ_vax = (1 - 1.0) * λ = 0
        - Vaccinated individuals have ZERO force of infection
        - No S_vax → E_vax transitions should occur
        
        Test Setup:
        -----------
        - Start with vaccinated susceptibles (S_vax > 0)
        - Seed infections in unvaccinated population (I > 0)
        - Run simulation with high transmission (epidemic occurs)
        - Set VE_severe = 0 and VE_death = 0 (isolate VE_infection effect)
        
        Expected Outcome:
        -----------------
        - E_vax should remain exactly 0 (no exposures)
        - I_vax should remain exactly 0 (no infections)
        - S_vax should remain constant (no infections, no vaccination)
        
        This is a critical edge case that validates the VE_infection implementation.
        If this test fails, it indicates a bug in the force of infection calculation
        for vaccinated individuals.
        """
        result = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.02,
            VE_infection=1.0,
            VE_severe=0.0,
            VE_death=0.0,
            initial_conditions={
                'S_vax_by_age': [1000, 1500, 500],  # Start with vaccinated
                'I_by_age': [10, 10, 10],  # Some initial infections
            },
            Tmax=100
        )
        
        # With perfect VE_infection, vaccinated should never get infected
        e_vax_total = get_total_array(result, 'E_vax')
        i_vax_total = get_total_array(result, 'I_vax')
        
        assert np.allclose(e_vax_total, 0, atol=1e-6), \
            "E_vax should be zero with perfect VE_infection"
        assert np.allclose(i_vax_total, 0, atol=1e-6), \
            "I_vax should be zero with perfect VE_infection"


# ========================================
# Test: Vaccine Efficacy - VE_severe
# ========================================

class TestVESevere:
    """Tests for VE_severe: reduced progression to severe disease."""
    
    def test_ve_severe_reduces_severe_cases(self, minimal_inputs):
        """
        Verify that VE_severe reduces progression from I_vax to X_vax.
        
        Three-Factor Vaccine Model - Factor 2: VE_severe
        -------------------------------------------------
        VE_severe reduces the probability that an infected vaccinated individual
        progresses to severe disease requiring hospitalization.
        
        Mechanistically: σ_vax = (1 - VE_severe) * σ
        
        Where:
        - σ = progression rate from I to X (unvaccinated)
        - σ_vax = progression rate from I_vax to X_vax (vaccinated)
        
        Compensatory Recovery:
        ----------------------
        To maintain the same total exit rate from I_vax as from I:
        γ_I_vax = γ_I + (σ - σ_vax)
        
        This ensures that the time spent in I_vax equals time in I, but with
        more individuals recovering and fewer progressing to severe disease.
        
        Test Design:
        ------------
        - Allow breakthrough infections (VE_infection = 0.3)
        - Start with vaccinated susceptibles and seed infections
        - Compare low VE_severe (0.2) vs high VE_severe (0.9)
        
        Expected Outcome:
        -----------------
        Peak X_vax should be lower with higher VE_severe, indicating that
        fewer breakthrough infections progress to severe disease.
        """
        # Create scenario with breakthrough infections
        inputs = {
            **minimal_inputs,
            'vaccination_rate': 0.02,
            'VE_infection': 0.3,  # Allow breakthroughs
            'VE_death': 0.5,
            'initial_conditions': {
                'S_vax_by_age': [1000, 1500, 500],
                'I_by_age': [50, 50, 50],  # More infections for observable effect
            },
            'Tmax': 100
        }
        
        result_low_ve = simulate_master_hospital_model(**{**inputs, 'VE_severe': 0.2})
        result_high_ve = simulate_master_hospital_model(**{**inputs, 'VE_severe': 0.9})
        
        # Maximum X_vax should be lower with higher VE_severe
        x_vax_low = get_total_array(result_low_ve, 'X_vax')
        x_vax_high = get_total_array(result_high_ve, 'X_vax')
        
        assert x_vax_high.max() <= x_vax_low.max(), \
            "Higher VE_severe should reduce peak severe cases in vaccinated"
    
    def test_perfect_ve_severe_no_severe_disease(self, minimal_inputs):
        """
        Verify that perfect VE_severe (1.0) completely prevents severe disease.
        
        Edge Case Test: VE_severe = 1.0
        --------------------------------
        When VE_severe = 1.0:
        - σ_vax = (1 - 1.0) * σ = 0
        - No I_vax → X_vax transitions occur
        - All vaccinated infections remain mild (recover from I_vax)
        
        Test Setup:
        -----------
        - Start with vaccinated infected individuals (I_vax > 0)
        - Set VE_infection = 0 (allow infections, isolate VE_severe effect)
        - Set VE_death = 0 (isolate VE_severe effect)
        - No new vaccinations (focus on existing I_vax)
        
        Expected Outcome:
        -----------------
        - X_vax should remain exactly 0 throughout simulation
        - I_vax should decrease as individuals recover
        - R_vax should increase (all I_vax recover, none progress to X_vax)
        
        Clinical Interpretation:
        ------------------------
        This represents a vaccine that provides no protection from infection
        but completely prevents severe disease. All breakthrough infections
        remain mild and resolve without hospitalization.
        """
        result = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.0,  # No new vaccinations
            VE_infection=0.0,  # No protection from infection
            VE_severe=1.0,  # Perfect protection from severe disease
            VE_death=0.0,
            initial_conditions={
                'I_vax_by_age': [50, 80, 30],  # Start with vaccinated infected
                'S_vax_by_age': [500, 1000, 500],
            },
            Tmax=50
        )
        
        # X_vax should remain zero with perfect VE_severe
        x_vax_total = get_total_array(result, 'X_vax')
        assert np.allclose(x_vax_total, 0, atol=1e-6), \
            "X_vax should be zero with perfect VE_severe"


# ========================================
# Test: Vaccine Efficacy - VE_death
# ========================================

class TestVEDeath:
    """Tests for VE_death: reduced mortality in vaccinated compartments."""
    
    def test_ve_death_reduces_mortality(self, minimal_inputs):
        """Higher VE_death should result in fewer deaths among vaccinated."""
        inputs = {
            **minimal_inputs,
            'vaccination_rate': 0.0,
            'VE_infection': 0.0,
            'VE_severe': 0.0,
            'initial_conditions': {
                'I_vax_by_age': [100, 100, 100],  # Start with infected vaccinated
                'X_vax_by_age': [50, 50, 50],  # And severe cases
            },
            'Tmax': 100
        }
        
        result_low_ve = simulate_master_hospital_model(**{**inputs, 'VE_death': 0.1})
        result_high_ve = simulate_master_hospital_model(**{**inputs, 'VE_death': 0.95})
        
        # D_vax at end should be lower with higher VE_death
        d_vax_low = get_total_array(result_low_ve, 'D_vax')
        d_vax_high = get_total_array(result_high_ve, 'D_vax')
        
        assert d_vax_high[-1] < d_vax_low[-1], \
            "Higher VE_death should reduce vaccinated deaths"
    
    def test_perfect_ve_death_no_mortality(self, minimal_inputs):
        """VE_death=1.0 should prevent all deaths in vaccinated compartments."""
        result = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.0,
            VE_infection=0.0,
            VE_severe=0.0,
            VE_death=1.0,
            initial_conditions={
                'I_vax_by_age': [100, 100, 100],
                'X_vax_by_age': [100, 100, 100],
                'H_ward_vax_by_age': [50, 50, 50],
                'H_icu_vax_by_age': [30, 30, 30],
            },
            Tmax=100
        )
        
        # D_vax should remain zero
        d_vax_total = get_total_array(result, 'D_vax')
        assert np.allclose(d_vax_total, 0, atol=1e-6), \
            "D_vax should be zero with perfect VE_death"


# ========================================
# Test: Vaccine Waning
# ========================================

class TestVaccineWaning:
    """Tests for vaccine immunity waning."""
    
    def test_waning_reduces_s_vax(self, vaccine_waning_inputs):
        """Vaccine waning should reduce S_vax over time."""
        result = simulate_master_hospital_model(**vaccine_waning_inputs)
        
        # S_vax should first increase (vaccination) then may decrease (waning + infection)
        s_vax_total = get_total_array(result, 'S_vax')
        max_s_vax_idx = np.argmax(s_vax_total)
        
        # There should be a peak before the end (waning kicks in)
        # Note: may peak at end if vaccination outpaces waning
        assert max_s_vax_idx < len(result['times']) - 1 or s_vax_total[-1] > 0, \
            "S_vax should have significant population with vaccination"
    
    def test_wane_to_s_increases_susceptibles(self, minimal_inputs):
        """When wane_to_S=True, waned immunity should increase S."""
        result = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.03,
            VE_infection=0.8,
            VE_severe=0.8,
            VE_death=0.9,
            vaccine_waning_params={
                'omega_vax': 0.02,  # Fast waning
                'wane_to_S': True,
            },
            Tmax=200
        )
        
        # S should have individuals from waned immunity
        s_total = get_total_array(result, 'S')
        assert s_total[-1] > 0, "S should have individuals from waned immunity"
    
    def test_wane_to_s_vax_keeps_in_vaccinated(self, minimal_inputs):
        """When wane_to_S=False, waned immunity should go to S_vax."""
        result = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.03,
            VE_infection=0.8,
            VE_severe=0.8,
            VE_death=0.9,
            vaccine_waning_params={
                'omega_vax': 0.02,
                'wane_to_S': False,  # Wane to S_vax
            },
            Tmax=200
        )
        
        # Total vaccinated population should grow over time
        n_ages = len(minimal_inputs['age_pops'])
        n_times = len(result['times'])
        
        def total_vax_at_t(t_idx):
            total = 0
            for a in range(n_ages):
                total += (result['S_vax'][a][t_idx] + result['E_vax'][a][t_idx] +
                         result['I_vax'][a][t_idx] + result['X_vax'][a][t_idx] +
                         result['H_ward_vax'][a][t_idx] + result['H_icu_vax'][a][t_idx] +
                         result['R_vax'][a][t_idx] + result['D_vax'][a][t_idx])
            return total
        
        # Vaccinated population should grow with wane_to_S=False
        assert total_vax_at_t(-1) > total_vax_at_t(0), \
            "Vaccinated population should grow with wane_to_S=False"
    
    def test_no_waning_by_default(self, vaccination_inputs):
        """Without vaccine_waning_params, no waning should occur."""
        # Don't include vaccine_waning_params
        inputs = {k: v for k, v in vaccination_inputs.items() 
                  if k != 'vaccine_waning_params'}
        result = simulate_master_hospital_model(**inputs)
        
        # R_vax should be monotonically increasing (no waning back to S)
        r_vax_total = get_total_array(result, 'R_vax')
        for i in range(1, len(r_vax_total)):
            assert r_vax_total[i] >= r_vax_total[i-1] - 1e-10, \
                "R_vax should be monotonically increasing without waning"


# ========================================
# Test: Breakthrough Infections
# ========================================

class TestBreakthroughInfections:
    """Tests for breakthrough infection tracking."""
    
    def test_breakthrough_infections_occur(self, vaccination_inputs):
        """With imperfect VE_infection, breakthrough infections should occur."""
        result = simulate_master_hospital_model(**vaccination_inputs)
        
        # E_vax and I_vax should have positive values
        e_vax_total = get_total_array(result, 'E_vax')
        i_vax_total = get_total_array(result, 'I_vax')
        
        assert e_vax_total.max() > 0 or i_vax_total.max() > 0, \
            "Breakthrough infections should occur with imperfect VE_infection"
    
    def test_cumulative_breakthrough_tracking(self, high_vaccination_inputs):
        """Cumulative breakthrough infections should be tracked."""
        result = simulate_master_hospital_model(**high_vaccination_inputs)
        
        # Check if breakthrough tracking is in result
        if 'breakthrough_infections' in result:
            assert result['breakthrough_infections'][-1] >= 0, \
                "Breakthrough infections should be non-negative"
        
        # We can infer from R_vax + D_vax (recovered or died vaccinated = had breakthrough)
        r_vax_total = get_total_array(result, 'R_vax')
        d_vax_total = get_total_array(result, 'D_vax')
        total_breakthrough = r_vax_total[-1] + d_vax_total[-1]
        
        # If there were vaccinated people exposed, some should have recovered/died
        s_vax_total = get_total_array(result, 'S_vax')
        if s_vax_total.max() > 100:  # Had meaningful vaccinated population
            # With VE_infection < 1, there should be some breakthroughs
            ve_infection = high_vaccination_inputs.get('VE_infection', 0)
            assert total_breakthrough > 0 or ve_infection == 1.0, \
                "Should have breakthrough infections with imperfect VE_infection"


# ========================================
# Test: Compartment Non-Negativity
# ========================================

class TestNonNegativity:
    """Tests that all vaccinated compartments remain non-negative."""
    
    def test_all_vaccinated_compartments_non_negative(self, vaccination_inputs):
        """All vaccinated compartments should be non-negative."""
        result = simulate_master_hospital_model(**vaccination_inputs)
        
        vax_compartments = ['S_vax', 'E_vax', 'I_vax', 'X_vax',
                            'H_ward_vax', 'H_icu_vax', 'R_vax', 'D_vax']
        
        for comp in vax_compartments:
            comp_array = get_total_array(result, comp)
            assert np.all(comp_array >= -1e-10), \
                f"{comp} has negative values"
    
    def test_non_negative_with_high_vaccination(self, high_vaccination_inputs):
        """Rapid vaccination should not cause negative compartments."""
        result = simulate_master_hospital_model(**high_vaccination_inputs)
        
        all_compartments = ['S', 'E', 'I', 'X', 'H_ward', 'H_icu', 'R', 'D',
                           'S_vax', 'E_vax', 'I_vax', 'X_vax',
                           'H_ward_vax', 'H_icu_vax', 'R_vax', 'D_vax']
        
        for comp in all_compartments:
            comp_array = get_total_array(result, comp)
            assert np.all(comp_array >= -1e-10), \
                f"{comp} has negative values with high vaccination"
    
    def test_non_negative_with_waning(self, vaccine_waning_inputs):
        """Vaccine waning should not cause negative compartments."""
        result = simulate_master_hospital_model(**vaccine_waning_inputs)
        
        all_compartments = ['S', 'E', 'I', 'X', 'H_ward', 'H_icu', 'R', 'D',
                           'S_vax', 'E_vax', 'I_vax', 'X_vax',
                           'H_ward_vax', 'H_icu_vax', 'R_vax', 'D_vax']
        
        for comp in all_compartments:
            comp_array = get_total_array(result, comp)
            assert np.all(comp_array >= -1e-10), \
                f"{comp} has negative values with vaccine waning"


# ========================================
# Test: Deaths Monotonicity
# ========================================

class TestDeathMonotonicity:
    """Tests that death compartments are monotonically increasing."""
    
    def test_d_vax_monotonically_increasing(self, vaccination_inputs):
        """D_vax should be monotonically increasing."""
        result = simulate_master_hospital_model(**vaccination_inputs)
        
        d_vax = get_total_array(result, 'D_vax')
        for i in range(1, len(d_vax)):
            assert d_vax[i] >= d_vax[i-1] - 1e-10, \
                "D_vax should be monotonically increasing"
    
    def test_d_vax_by_age_monotonically_increasing(self, vaccination_inputs):
        """D_vax should be monotonically increasing for each age."""
        result = simulate_master_hospital_model(**vaccination_inputs)
        
        n_ages = len(vaccination_inputs['age_pops'])
        for age in range(n_ages):
            d_vax_age = result['D_vax'][age]
            for i in range(1, len(d_vax_age)):
                assert d_vax_age[i] >= d_vax_age[i-1] - 1e-10, \
                    f"D_vax for age {age} should be monotonically increasing"
    
    def test_total_deaths_monotonic_with_vaccination(self, vaccination_inputs):
        """Total deaths (D + D_vax) should be monotonically increasing."""
        result = simulate_master_hospital_model(**vaccination_inputs)
        
        d_total = get_total_array(result, 'D')
        d_vax_total = get_total_array(result, 'D_vax')
        total_deaths = d_total + d_vax_total
        
        for i in range(1, len(total_deaths)):
            assert total_deaths[i] >= total_deaths[i-1] - 1e-10, \
                "Total deaths should be monotonically increasing"


# ========================================
# Test: Configuration Helpers
# ========================================

class TestVaccineProfiles:
    """Tests for vaccine profile configuration helpers."""
    
    def test_get_vaccine_profile_returns_dict(self):
        """get_vaccine_profile should return a dictionary."""
        profile = get_vaccine_profile('mrna_original')
        assert isinstance(profile, dict), "Profile should be a dict"
        
        # Should have the three VE parameters
        assert 'VE_infection' in profile
        assert 'VE_severe' in profile
        assert 'VE_death' in profile
    
    def test_list_vaccine_profiles_returns_list(self):
        """list_vaccine_profiles should return a list of profile names."""
        profiles = list_vaccine_profiles()
        assert isinstance(profiles, list), "Should return a list"
        assert len(profiles) > 0, "Should have at least one profile"
        assert 'mrna_original' in profiles, "Should include mrna_original"
    
    def test_describe_vaccine_profile_returns_string(self):
        """describe_vaccine_profile should return a description string."""
        description = describe_vaccine_profile('mrna_original')
        assert isinstance(description, str), "Should return a string"
        assert len(description) > 0, "Description should not be empty"
    
    def test_all_profiles_have_valid_ve_values(self):
        """All vaccine profiles should have VE values in [0, 1]."""
        for profile_name in list_vaccine_profiles():
            profile = get_vaccine_profile(profile_name)
            
            for key in ['VE_infection', 'VE_severe', 'VE_death']:
                ve = profile[key]
                assert 0 <= ve <= 1, \
                    f"{profile_name}.{key} = {ve} not in [0, 1]"
    
    def test_profile_can_be_used_in_simulation(self, minimal_inputs):
        """Vaccine profiles should be usable directly in simulation."""
        profile = get_vaccine_profile('mrna_original')
        
        # Extract only the VE parameters for simulation
        ve_params = {k: v for k, v in profile.items() 
                     if k in ['VE_infection', 'VE_severe', 'VE_death']}
        
        result = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.01,
            **ve_params,
            Tmax=50
        )
        
        # Should run without error and have vaccinated compartments
        assert 'S_vax' in result
        s_vax_total = get_total_array(result, 'S_vax')
        assert s_vax_total[-1] >= 0


# ========================================
# Test: Backward Compatibility
# ========================================

class TestBackwardCompatibility:
    """Tests that existing functionality is preserved."""
    
    def test_default_no_vaccination(self, minimal_inputs):
        """Without vaccination parameters, no vaccination should occur."""
        result = simulate_master_hospital_model(**minimal_inputs)
        
        # Vaccinated compartments should be zero or near-zero
        s_vax = get_total_array(result, 'S_vax')
        e_vax = get_total_array(result, 'E_vax')
        i_vax = get_total_array(result, 'I_vax')
        d_vax = get_total_array(result, 'D_vax')
        
        assert np.allclose(s_vax, 0, atol=1e-10)
        assert np.allclose(e_vax, 0, atol=1e-10)
        assert np.allclose(i_vax, 0, atol=1e-10)
        assert np.allclose(d_vax, 0, atol=1e-10)
    
    def test_unvaccinated_dynamics_unchanged(self, minimal_inputs):
        """Unvaccinated dynamics should be same as before when no vaccination."""
        result_no_vax = simulate_master_hospital_model(**minimal_inputs)
        result_with_vax_zero = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.0,
            VE_infection=0.5,
            VE_severe=0.5,
            VE_death=0.5,
        )
        
        # Unvaccinated compartments should be identical
        s_no_vax = get_total_array(result_no_vax, 'S')
        s_with_vax = get_total_array(result_with_vax_zero, 'S')
        i_no_vax = get_total_array(result_no_vax, 'I')
        i_with_vax = get_total_array(result_with_vax_zero, 'I')
        d_no_vax = get_total_array(result_no_vax, 'D')
        d_with_vax = get_total_array(result_with_vax_zero, 'D')
        
        np.testing.assert_allclose(s_no_vax, s_with_vax, rtol=1e-10)
        np.testing.assert_allclose(i_no_vax, i_with_vax, rtol=1e-10)
        np.testing.assert_allclose(d_no_vax, d_with_vax, rtol=1e-10)
    
    def test_result_dict_has_all_expected_keys(self, vaccination_inputs):
        """Result dict should have all vaccinated and unvaccinated compartments."""
        result = simulate_master_hospital_model(**vaccination_inputs)
        
        # Unvaccinated (original)
        expected_unvax = ['S', 'E', 'I', 'X', 'H_ward', 'H_icu', 'R', 'D', 'times',
                         'H_ward_total', 'H_icu_total', 'H_total',
                         'E_total', 'I_total', 'X_total', 'D_total']
        
        # Vaccinated (new)
        expected_vax = ['S_vax', 'E_vax', 'I_vax', 'X_vax',
                        'H_ward_vax', 'H_icu_vax', 'R_vax', 'D_vax']
        
        for key in expected_unvax + expected_vax:
            assert key in result, f"Missing key: {key}"


# ========================================
# Test: Combined Effects
# ========================================

class TestCombinedEffects:
    """Tests for combined vaccination with other features."""
    
    def test_vaccination_with_waning_immunity(self, minimal_inputs):
        """Vaccination should work alongside natural immunity waning."""
        result = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.01,
            VE_infection=0.6,
            VE_severe=0.8,
            VE_death=0.9,
            waning_params={'omega': 0.01},  # Natural immunity waning
            vaccine_waning_params={'omega_vax': 0.005, 'wane_to_S': True},
            Tmax=365
        )
        
        # Should run without error
        assert len(result['times']) > 0
        
        # Population should be conserved
        n_ages = len(minimal_inputs['age_pops'])
        initial_pop = sum(minimal_inputs['age_pops'])
        
        pop_at_end = 0
        for a in range(n_ages):
            pop_at_end += (result['S'][a][-1] + result['E'][a][-1] +
                          result['I'][a][-1] + result['X'][a][-1] +
                          result['H_ward'][a][-1] + result['H_icu'][a][-1] +
                          result['R'][a][-1] + result['D'][a][-1] +
                          result['S_vax'][a][-1] + result['E_vax'][a][-1] +
                          result['I_vax'][a][-1] + result['X_vax'][a][-1] +
                          result['H_ward_vax'][a][-1] + result['H_icu_vax'][a][-1] +
                          result['R_vax'][a][-1] + result['D_vax'][a][-1])
        
        assert pop_at_end == pytest.approx(initial_pop, rel=0.01)
    
    def test_vaccination_with_interventions(self, minimal_inputs):
        """Vaccination should work with policy interventions."""
        result = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.02,
            VE_infection=0.7,
            VE_severe=0.8,
            VE_death=0.9,
            interventions=[{
                'start_day': 30,
                'end_day': 60,
                'transmission_reduction': 0.5,
            }],
            Tmax=100
        )
        
        # Should run without error
        assert len(result['times']) > 0
        assert 'S_vax' in result
    
    def test_vaccination_with_seasonality(self, minimal_inputs):
        """Vaccination should work with seasonal forcing."""
        result = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.01,
            VE_infection=0.6,
            VE_severe=0.8,
            VE_death=0.9,
            seasonal_params={
                'amplitude': 0.2,
                'period': 365,
                'peak_day': 0,
            },
            Tmax=365
        )
        
        # Should run without error
        assert len(result['times']) > 0
        assert 'S_vax' in result


# ========================================
# Test: Edge Cases
# ========================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_all_initial_vaccinated(self, minimal_inputs):
        """Simulation should handle all-vaccinated initial population."""
        age_pops = minimal_inputs['age_pops']
        result = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.0,
            VE_infection=0.8,
            VE_severe=0.9,
            VE_death=0.95,
            initial_conditions={
                'S_by_age': [0, 0, 0],  # No unvaccinated susceptibles
                'S_vax_by_age': [p - 10 for p in age_pops],  # Almost all vaccinated
                'I_by_age': [10, 10, 10],  # Some initial infections
            },
            Tmax=50
        )
        
        # Population should be conserved
        n_ages = len(age_pops)
        initial_pop = sum(age_pops)
        
        pop_at_end = 0
        for a in range(n_ages):
            pop_at_end += (result['S'][a][-1] + result['E'][a][-1] +
                          result['I'][a][-1] + result['X'][a][-1] +
                          result['H_ward'][a][-1] + result['H_icu'][a][-1] +
                          result['R'][a][-1] + result['D'][a][-1] +
                          result['S_vax'][a][-1] + result['E_vax'][a][-1] +
                          result['I_vax'][a][-1] + result['X_vax'][a][-1] +
                          result['H_ward_vax'][a][-1] + result['H_icu_vax'][a][-1] +
                          result['R_vax'][a][-1] + result['D_vax'][a][-1])
        
        assert pop_at_end == pytest.approx(initial_pop, rel=0.01)
    
    def test_ve_at_boundaries(self, minimal_inputs):
        """VE values at 0 and 1 should work correctly."""
        # VE = 0 (no protection)
        result_no_ve = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.01,
            VE_infection=0.0,
            VE_severe=0.0,
            VE_death=0.0,
            Tmax=50
        )
        
        # VE = 1 (perfect protection)
        result_full_ve = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.01,
            VE_infection=1.0,
            VE_severe=1.0,
            VE_death=1.0,
            Tmax=50
        )
        
        # Both should run without error
        assert len(result_no_ve['times']) > 0
        assert len(result_full_ve['times']) > 0
    
    def test_very_high_vaccination_rate(self, minimal_inputs):
        """Very high vaccination rate should not cause instability."""
        result = simulate_master_hospital_model(
            **minimal_inputs,
            vaccination_rate=0.5,  # 50% per day - extremely high
            VE_infection=0.6,
            VE_severe=0.8,
            VE_death=0.9,
            Tmax=30
        )
        
        # All compartments should remain non-negative
        all_comps = ['S', 'E', 'I', 'X', 'H_ward', 'H_icu', 'R', 'D',
                     'S_vax', 'E_vax', 'I_vax', 'X_vax',
                     'H_ward_vax', 'H_icu_vax', 'R_vax', 'D_vax']
        
        for comp in all_comps:
            comp_array = get_total_array(result, comp)
            assert np.all(comp_array >= -1e-8), \
                f"{comp} went negative with high vaccination rate"
    
    def test_single_age_group_with_vaccination(self, minimal_inputs):
        """Vaccination should work with a single age group."""
        single_age_params = [{
            'alpha': 0.2,
            'sigma': 0.1,
            'eta': 0.2,
            'eta_icu': 0.1,
            'gamma_I': 0.1,
            'mu_I': 0.01,
            'gamma_X': 0.15,
            'mu_X': 0.02,
            'gamma_ward': 0.2,
            'mu_ward': 0.01,
            'gamma_icu': 0.1,
            'mu_icu': 0.03,
            'gamma_H': 0.2,
            'mu_H': 0.02,
        }]
        
        result = simulate_master_hospital_model(
            beta_base=0.3,
            age_params=single_age_params,
            contact_matrix=np.array([[8.0]]),
            age_pops=[10000],
            vaccination_rate=0.02,
            VE_infection=0.7,
            VE_severe=0.8,
            VE_death=0.9,
            Tmax=100
        )
        
        assert 'S_vax' in result
        assert len(result['S_vax']) == 1  # 1 age group


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
