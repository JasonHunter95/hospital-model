"""
Tests for demographic dynamics (births and background deaths).

This module tests the open population dynamics features:
- Birth rate functionality (births entering S compartment)
- Neonatal vaccination (births split between S and S_vax)
- Age-specific background mortality
- Population conservation with demographics
- Population drift warnings for long simulations
- Cumulative demographic tracking (cum_births, cum_background_deaths)
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from master_hospital_model import simulate_master_hospital_model
from simulation_helpers import (
    compute_birth_rate,
    compute_background_death_rate,
    validate_demographic_params,
)


class TestDemographicHelperFunctions:
    """Tests for demographic helper functions in simulation_helpers.py."""
    
    def test_compute_birth_rate_basic(self):
        """Test birth rate computation with default distribution."""
        age_pops = [1000, 2000, 500]
        birth_rate = 0.0001
        total_live_pop = 100000
        
        births = compute_birth_rate(total_live_pop, birth_rate, age_pops)
        
        # All births should go to first age group by default
        total_expected = birth_rate * total_live_pop
        assert births[0] == pytest.approx(total_expected, rel=1e-6)
        assert births[1] == 0.0
        assert births[2] == 0.0
    
    def test_compute_birth_rate_with_distribution(self):
        """Test birth rate computation with custom age distribution."""
        age_pops = [1000, 2000, 500]
        birth_rate = 0.0001
        total_live_pop = 100000
        birth_age_dist = [0.5, 0.3, 0.2]
        
        births = compute_birth_rate(total_live_pop, birth_rate, age_pops, birth_age_dist)
        
        total_births = birth_rate * total_live_pop
        assert births[0] == pytest.approx(0.5 * total_births, rel=1e-6)
        assert births[1] == pytest.approx(0.3 * total_births, rel=1e-6)
        assert births[2] == pytest.approx(0.2 * total_births, rel=1e-6)
    
    def test_compute_birth_rate_zero_rate(self):
        """Test birth rate returns zeros when rate is zero."""
        age_pops = [1000, 2000, 500]
        
        births = compute_birth_rate(100000, 0.0, age_pops)
        
        assert all(b == 0.0 for b in births)
    
    def test_compute_background_death_rate(self):
        """Test background death rate computation."""
        compartment_pop = np.array([1000.0, 2000.0, 500.0])
        mu_background = np.array([0.0001, 0.0002, 0.0005])
        
        bg_deaths = compute_background_death_rate(compartment_pop, mu_background)
        
        # Deaths should be proportional to mu_background * population
        for a in range(3):
            expected = mu_background[a] * compartment_pop[a]
            assert bg_deaths[a] == pytest.approx(expected, rel=1e-6)
    
    def test_validate_demographic_params_valid(self):
        """Test validation of valid demographic parameters."""
        params = {
            'birth_rate': 0.0001,
            'birth_age_distribution': [1.0, 0.0, 0.0],
            'mu_background': [0.0001, 0.0002, 0.0005],
            'neonatal_vaccination_rate': 0.5,
        }
        n_ages = 3
        
        # Should not raise
        result = validate_demographic_params(params, n_ages)
        assert result['birth_rate'] == 0.0001
    
    def test_validate_demographic_params_none(self):
        """Test validation returns defaults for None."""
        result = validate_demographic_params(None, 3)
        assert result['birth_rate'] == 0.0
        assert len(result['mu_background']) == 3
        assert all(m == 0.0 for m in result['mu_background'])
    
    def test_validate_demographic_params_negative_birth_rate(self):
        """Test validation rejects negative birth rate."""
        params = {
            'birth_rate': -0.0001,
            'birth_age_distribution': [1.0, 0.0, 0.0],
            'mu_background': [0.0001, 0.0002, 0.0005],
            'neonatal_vaccination_rate': 0.0,
        }
        
        with pytest.raises(ValueError, match="birth_rate"):
            validate_demographic_params(params, 3)
    
    def test_validate_demographic_params_invalid_mu_length(self):
        """Test validation rejects wrong length mu_background."""
        params = {
            'birth_rate': 0.0001,
            'birth_age_distribution': [1.0, 0.0, 0.0],
            'mu_background': [0.0001, 0.0002],  # Wrong length
            'neonatal_vaccination_rate': 0.0,
        }
        
        with pytest.raises(ValueError, match="mu_background"):
            validate_demographic_params(params, 3)
    
    def test_validate_demographic_params_neonatal_rate_bounds(self):
        """Test validation rejects out-of-bounds neonatal vaccination rate."""
        params = {
            'birth_rate': 0.0001,
            'birth_age_distribution': [1.0, 0.0, 0.0],
            'mu_background': [0.0001, 0.0002, 0.0005],
            'neonatal_vaccination_rate': 1.5,  # > 1
        }
        
        with pytest.raises(ValueError, match="neonatal_vaccination_rate"):
            validate_demographic_params(params, 3)


class TestDemographicIntegration:
    """Integration tests for demographic dynamics in full simulation."""
    
    def test_closed_population_without_demographics(self, minimal_inputs):
        """Test that population is conserved without demographic params."""
        results = simulate_master_hospital_model(**minimal_inputs, Tmax=100)
        
        # Sum all compartments at start and end
        n_ages = 3
        initial_pop = sum(minimal_inputs['age_pops'])
        
        # Final total should equal initial (closed population)
        final_live = results['live_population'][-1]
        final_dead = sum(results['D'][a][-1] + results['D_vax'][a][-1] for a in range(n_ages))
        
        assert final_live + final_dead == pytest.approx(initial_pop, rel=1e-4)
    
    def test_closed_population_with_zero_demographics(self, zero_demographic_inputs):
        """Test population is conserved with zero demographic rates."""
        results = simulate_master_hospital_model(**zero_demographic_inputs)
        
        n_ages = 3
        initial_pop = sum(zero_demographic_inputs['age_pops'])
        
        final_live = results['live_population'][-1]
        final_dead = sum(results['D'][a][-1] + results['D_vax'][a][-1] for a in range(n_ages))
        
        assert final_live + final_dead == pytest.approx(initial_pop, rel=1e-4)
        
        # Cumulative demographics should be zero
        assert results['cum_births_total'][-1] == pytest.approx(0.0, abs=1e-6)
        assert results['cum_background_deaths_total'][-1] == pytest.approx(0.0, abs=1e-6)
    
    def test_births_increase_population(self, high_birth_rate_inputs):
        """
        Verify that births increase the live population in an open population model.
        
        Open Population Dynamics:
        -------------------------
        Unlike closed population models where total population is constant,
        open population models allow births and background deaths.
        
        Birth Mechanism:
        ----------------
        Births enter the S compartment (or S_vax if neonatal vaccination is enabled).
        The birth rate is applied to the total live population:
        
        dS/dt += birth_rate * N_live * (1 - neonatal_vax_rate)
        dS_vax/dt += birth_rate * N_live * neonatal_vax_rate
        
        Population Balance:
        -------------------
        For an open population:
        N_final = N_initial + cumulative_births - cumulative_deaths
        
        Where cumulative_deaths = disease_deaths + background_deaths
        
        Test Validation:
        ----------------
        This test verifies:
        1. Cumulative births are tracked correctly
        2. Live population increases by approximately cumulative births
           (accounting for disease deaths)
        3. The population balance equation holds
        """
        results = simulate_master_hospital_model(**high_birth_rate_inputs)
        
        initial_pop = sum(high_birth_rate_inputs['age_pops'])
        final_live = results['live_population'][-1]
        cum_births = results['cum_births_total'][-1]
        
        # Live population should increase by approximately cum_births
        # (minus disease deaths)
        final_dead = sum(results['D'][a][-1] + results['D_vax'][a][-1] for a in range(3))
        
        expected_final = initial_pop + cum_births - final_dead
        assert final_live == pytest.approx(expected_final, rel=0.01)
        
        # Births should be positive
        assert cum_births > 0
    
    def test_background_deaths_decrease_population(self, high_mortality_inputs):
        """Test that background deaths decrease the live population."""
        results = simulate_master_hospital_model(**high_mortality_inputs)
        
        initial_pop = sum(high_mortality_inputs['age_pops'])
        final_live = results['live_population'][-1]
        cum_bg_deaths = results['cum_background_deaths_total'][-1]
        
        # Live population should decrease
        assert final_live < initial_pop
        
        # Background deaths should be tracked
        assert cum_bg_deaths > 0
    
    def test_neonatal_vaccination_distributes_births(self, neonatal_vaccination_inputs):
        """Test that neonatal vaccination splits births between S and S_vax."""
        results = simulate_master_hospital_model(**neonatal_vaccination_inputs)
        
        # With 80% neonatal vaccination, S_vax in young should have significant population
        # from births (even if disease dynamics also affect it)
        final_S_vax_young = results['S_vax'][0][-1]
        
        # S_vax should be non-trivial
        assert final_S_vax_young > 0
        
        # Verify births occurred
        assert results['cum_births_total'][-1] > 0
    
    def test_demographic_output_structure(self, demographic_inputs):
        """Test that demographic outputs are present and correctly structured."""
        results = simulate_master_hospital_model(**demographic_inputs)
        
        n_ages = 3
        n_times = len(results['times'])
        
        # Check per-age cumulative histories
        assert 'cum_births' in results
        assert 'cum_background_deaths' in results
        assert len(results['cum_births']) == n_ages
        assert len(results['cum_background_deaths']) == n_ages
        
        for a in range(n_ages):
            assert len(results['cum_births'][a]) == n_times
            assert len(results['cum_background_deaths'][a]) == n_times
        
        # Check aggregated totals
        assert 'cum_births_total' in results
        assert 'cum_background_deaths_total' in results
        assert 'live_population' in results
        
        assert len(results['cum_births_total']) == n_times
        assert len(results['cum_background_deaths_total']) == n_times
        assert len(results['live_population']) == n_times
    
    def test_cumulative_demographics_monotonic(self, demographic_inputs):
        """Test that cumulative demographic trackers are monotonically increasing."""
        results = simulate_master_hospital_model(**demographic_inputs)
        
        cum_births = results['cum_births_total']
        cum_bg_deaths = results['cum_background_deaths_total']
        
        # Both should be monotonically increasing
        for i in range(1, len(cum_births)):
            assert cum_births[i] >= cum_births[i-1]
            assert cum_bg_deaths[i] >= cum_bg_deaths[i-1]
    
    def test_per_age_demographic_consistency(self, demographic_inputs):
        """Test that per-age demographics sum to totals."""
        results = simulate_master_hospital_model(**demographic_inputs)
        
        n_ages = 3
        n_times = len(results['times'])
        
        for t in range(n_times):
            # Sum per-age cumulative births
            sum_births = sum(results['cum_births'][a][t] for a in range(n_ages))
            assert sum_births == pytest.approx(results['cum_births_total'][t], rel=1e-6)
            
            # Sum per-age cumulative background deaths
            sum_bg_deaths = sum(results['cum_background_deaths'][a][t] for a in range(n_ages))
            assert sum_bg_deaths == pytest.approx(results['cum_background_deaths_total'][t], rel=1e-6)


class TestPopulationConservation:
    """Tests for population conservation accounting with demographics."""
    
    def test_population_balance_equation(self, demographic_inputs):
        """
        Verify the fundamental population balance equation for open populations.
        
        The Population Balance Equation:
        ---------------------------------
        For an open population with births and background deaths:
        
        N_live(t) + D_disease(t) = N_initial + cumulative_births - cumulative_background_deaths
        
        Where:
        - N_live(t): Total living population at time t (all compartments except D, D_vax)
        - D_disease(t): Cumulative disease deaths (D + D_vax)
        - N_initial: Initial total population
        - cumulative_births: Total births from t=0 to t
        - cumulative_background_deaths: Total background deaths from t=0 to t
        
        Why This Matters:
        -----------------
        This equation is the conservation law for open populations. It ensures:
        1. All individuals are accounted for at all times
        2. Births and deaths are tracked correctly
        3. No 'leaks' exist in the population accounting
        
        Distinction from Closed Population:
        ------------------------------------
        Closed population: N_total = constant
        Open population: N_total = N_initial + net_migration + births - all_deaths
        
        In our model, we don't include migration, so:
        N_total = N_initial + births - background_deaths - disease_deaths
        
        Rearranging:
        N_live + D_disease = N_initial + births - background_deaths
        
        This test is critical for validating the demographic implementation.
        If it fails, there is a fundamental error in the population accounting.
        """
        results = simulate_master_hospital_model(**demographic_inputs)
        
        n_ages = 3
        initial_pop = sum(demographic_inputs['age_pops'])
        
        # Get final values
        final_live = results['live_population'][-1]
        final_disease_deaths = sum(results['D'][a][-1] + results['D_vax'][a][-1] for a in range(n_ages))
        cum_births = results['cum_births_total'][-1]
        cum_bg_deaths = results['cum_background_deaths_total'][-1]
        
        # Balance equation: N_live + D_disease = N_initial + births - background_deaths
        total_final = final_live + final_disease_deaths
        total_expected = initial_pop + cum_births - cum_bg_deaths
        
        assert total_final == pytest.approx(total_expected, rel=0.01)
    
    def test_live_population_calculation(self, demographic_inputs):
        """Test that live_population correctly sums all non-dead compartments."""
        results = simulate_master_hospital_model(**demographic_inputs)
        
        n_ages = 3
        
        # Check at multiple time points
        for t_idx in [0, len(results['times'])//2, -1]:
            manual_sum = 0.0
            for a in range(n_ages):
                # Unvaccinated
                manual_sum += results['S'][a][t_idx]
                manual_sum += results['E'][a][t_idx]
                manual_sum += results['I'][a][t_idx]
                manual_sum += results['X'][a][t_idx]  # Combined X
                manual_sum += results['H_ward'][a][t_idx]
                manual_sum += results['H_icu'][a][t_idx]
                manual_sum += results['R'][a][t_idx]
                # Vaccinated
                manual_sum += results['S_vax'][a][t_idx]
                manual_sum += results['E_vax'][a][t_idx]
                manual_sum += results['I_vax'][a][t_idx]
                manual_sum += results['X_vax'][a][t_idx]
                manual_sum += results['H_ward_vax'][a][t_idx]
                manual_sum += results['H_icu_vax'][a][t_idx]
                manual_sum += results['R_vax'][a][t_idx]
            
            assert results['live_population'][t_idx] == pytest.approx(manual_sum, rel=1e-4)


class TestDemographicEdgeCases:
    """Edge case tests for demographic dynamics."""
    
    def test_very_high_birth_rate(self, minimal_inputs):
        """Test simulation handles very high birth rate."""
        high_birth_params = {
            'birth_rate': 0.01,  # Unrealistically high, ~3650/1000/year
            'birth_age_distribution': [1.0, 0.0, 0.0],
            'mu_background': [0.0, 0.0, 0.0],
            'neonatal_vaccination_rate': 0.0,
        }
        
        results = simulate_master_hospital_model(
            **minimal_inputs,
            demographic_params=high_birth_params,
            Tmax=50,
        )
        
        # Population should grow significantly
        initial_pop = sum(minimal_inputs['age_pops'])
        final_live = results['live_population'][-1]
        assert final_live > initial_pop * 1.3  # At least 30% growth
    
    def test_very_high_mortality_rate(self, minimal_inputs):
        """Test simulation handles very high background mortality."""
        high_mort_params = {
            'birth_rate': 0.0,
            'birth_age_distribution': [1.0, 0.0, 0.0],
            'mu_background': [0.01, 0.01, 0.01],  # Unrealistically high
            'neonatal_vaccination_rate': 0.0,
        }
        
        results = simulate_master_hospital_model(
            **minimal_inputs,
            demographic_params=high_mort_params,
            Tmax=50,
        )
        
        # Population should decline significantly
        initial_pop = sum(minimal_inputs['age_pops'])
        final_live = results['live_population'][-1]
        assert final_live < initial_pop * 0.7  # At least 30% decline
    
    def test_full_neonatal_vaccination(self, minimal_inputs):
        """Test 100% neonatal vaccination rate."""
        full_neonatal_params = {
            'birth_rate': 0.0001,
            'birth_age_distribution': [1.0, 0.0, 0.0],
            'mu_background': [0.0, 0.0, 0.0],
            'neonatal_vaccination_rate': 1.0,  # 100%
        }
        
        results = simulate_master_hospital_model(
            **minimal_inputs,
            demographic_params=full_neonatal_params,
            VE_infection=0.9,
            VE_severe=0.95,
            VE_death=0.99,
            Tmax=100,
        )
        
        # All births should go to S_vax
        # S_vax in young age group should have increased
        initial_S_vax_young = 0  # Default initial
        final_S_vax_young = results['S_vax'][0][-1]
        
        # Should have some vaccinated susceptibles from births
        assert final_S_vax_young > initial_S_vax_young
    
    def test_zero_initial_population_fails_gracefully(self, minimal_inputs):
        """Test behavior with zero population in some age groups."""
        inputs = {**minimal_inputs, 'age_pops': [10000, 0, 0]}
        demo_params = {
            'birth_rate': 0.0001,
            'birth_age_distribution': [1.0, 0.0, 0.0],
            'mu_background': [0.0001, 0.0001, 0.0001],
            'neonatal_vaccination_rate': 0.0,
        }
        
        # Should not crash
        results = simulate_master_hospital_model(
            **inputs,
            demographic_params=demo_params,
            Tmax=50,
        )
        
        assert 'live_population' in results


class TestDemographicParametersInResults:
    """Test that demographic parameters are included in results metadata."""
    
    def test_demographic_params_in_parameters(self, demographic_inputs):
        """Test that demographic_params is stored in results['parameters']."""
        results = simulate_master_hospital_model(**demographic_inputs)
        
        assert 'parameters' in results
        assert 'demographic_params' in results['parameters']
        assert results['parameters']['demographic_params'] == demographic_inputs['demographic_params']
    
    def test_none_demographic_params_in_parameters(self, minimal_inputs):
        """Test that None demographic_params is stored correctly."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        assert 'parameters' in results
        assert 'demographic_params' in results['parameters']
        assert results['parameters']['demographic_params'] is None


class TestEquilibriumDemographics:
    """Tests for demographic equilibrium scenarios."""
    
    def test_equilibrium_population_stability(self, demographic_equilibrium_inputs):
        """Test that equilibrium demographics maintain roughly stable population."""
        results = simulate_master_hospital_model(**demographic_equilibrium_inputs)
        
        initial_pop = sum(demographic_equilibrium_inputs['age_pops'])
        
        # Check population at various times
        populations = results['live_population']
        n_times = len(populations)
        
        # Allow for disease-related changes, but population should remain within bounds
        # In a proper equilibrium with balanced births/deaths, drift should be minimal
        # (disease deaths will cause some decline)
        for t in range(0, n_times, n_times // 10):
            # Population should remain within 20% of initial (accounting for disease dynamics)
            assert populations[t] > initial_pop * 0.8
            assert populations[t] < initial_pop * 1.2
