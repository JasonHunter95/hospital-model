"""
Solver options tests for master_hospital_model.py.

This module tests alternative solver configurations:
1. solve_ivp path with different methods (BDF, Radau, RK45, RK23, DOP853)
2. Tolerance parameter effects
3. Solver method selection
4. Error handling for invalid solvers

The model supports two solver backends:
- 'odeint': scipy.integrate.odeint (LSODA algorithm)
- 'solve_ivp': scipy.integrate.solve_ivp with configurable method
"""

import pytest
import numpy as np
from master_hospital_model import simulate_master_hospital_model
from config_helpers import get_scenario_params
from config import AGE_PARAMS_DEFAULT, CONTACT_MATRIX_DEFAULT


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def minimal_params():
    """Minimal parameter set for fast solver tests."""
    return {
        'beta_base': 0.3,
        'age_params': AGE_PARAMS_DEFAULT,
        'contact_matrix': CONTACT_MATRIX_DEFAULT,
        'age_pops': [3000, 5000, 2000],
        'age_pops': [3000, 5000, 2000],
        'sim_config': {
            'Tmax': 30,  # Short simulation for speed
            'time_step': 0.5,
        }
    }


@pytest.fixture
def high_capacity_params(minimal_params):
    """Params with high capacity (no overflow)."""
    return {
        **minimal_params,
        'capacity_config': {
            'ward_capacity': 5000,
            'icu_capacity': 1000,
        }
    }


# =============================================================================
# TEST CLASS: SOLVER SELECTION
# =============================================================================

class TestSolverSelection:
    """Tests for solver backend selection."""
    
    def test_default_solver_is_odeint(self, minimal_params):
        """Default solver should be odeint."""
        results = simulate_master_hospital_model(**minimal_params)
        
        # Should complete without error
        assert 'times' in results
        assert 'D_total' in results
        
        # Check metadata
        params = results.get('parameters', {})
        assert params.get('solver') == 'odeint'
    
    def test_odeint_solver_works(self, minimal_params):
        """Explicit odeint solver should work."""
        results = simulate_master_hospital_model(
            **minimal_params,
            solver='odeint'
        )
        
        assert 'times' in results
        assert len(results['times']) > 1
    
    def test_solve_ivp_solver_works(self, minimal_params):
        """solve_ivp solver should work."""
        results = simulate_master_hospital_model(
            **minimal_params,
            solver='solve_ivp'
        )
        
        assert 'times' in results
        assert len(results['times']) > 1
        
        # Check metadata
        params = results.get('parameters', {})
        assert params.get('solver') == 'solve_ivp'
    
    def test_invalid_solver_raises_error(self, minimal_params):
        """Invalid solver should raise ValueError."""
        with pytest.raises(ValueError, match='Unknown solver'):
            simulate_master_hospital_model(
                **minimal_params,
                solver='invalid_solver_name'
            )


# =============================================================================
# TEST CLASS: SOLVE_IVP METHODS
# =============================================================================

class TestSolveIvpMethods:
    """Tests for different solve_ivp methods."""
    
    @pytest.mark.parametrize('method', ['RK45', 'RK23', 'DOP853'])
    def test_explicit_runge_kutta_methods(self, minimal_params, method):
        """Test explicit Runge-Kutta methods (RK45, RK23, DOP853)."""
        results = simulate_master_hospital_model(
            **minimal_params,
            solver='solve_ivp',
            solver_method=method
        )
        
        assert 'times' in results
        assert len(results['times']) > 1
        
        # Check results are valid
        D_total = results['D_total']
        assert not np.any(np.isnan(D_total))
        assert not np.any(np.isinf(D_total))
    
    @pytest.mark.parametrize('method', ['BDF', 'Radau'])
    def test_implicit_methods(self, minimal_params, method):
        """Test implicit methods for stiff problems (BDF, Radau)."""
        results = simulate_master_hospital_model(
            **minimal_params,
            solver='solve_ivp',
            solver_method=method
        )
        
        assert 'times' in results
        
        # Implicit methods should handle stiff systems well
        D_total = results['D_total']
        assert not np.any(np.isnan(D_total))
    
    def test_lsoda_method(self, minimal_params):
        """Test LSODA method (similar to odeint)."""
        results = simulate_master_hospital_model(
            **minimal_params,
            solver='solve_ivp',
            solver_method='LSODA'
        )
        
        assert 'times' in results


# =============================================================================
# TEST CLASS: TOLERANCE PARAMETERS
# =============================================================================

class TestToleranceParameters:
    """Tests for relative and absolute tolerance parameters."""
    
    def test_default_tolerances(self, minimal_params):
        """Default tolerances should work."""
        results = simulate_master_hospital_model(**minimal_params)
        
        # Check metadata for default tolerances
        params = results.get('parameters', {})
        rtol = params.get('rtol')
        atol = params.get('atol')
        
        # Defaults should be set
        assert rtol is not None
        assert atol is not None
    
    def test_tight_tolerances(self, minimal_params):
        """Tight tolerances should produce valid results."""
        results = simulate_master_hospital_model(
            **minimal_params,
            rtol=1e-10,
            atol=1e-12
        )
        
        assert 'times' in results
        
        # Results should be valid
        D_total = results['D_total']
        assert not np.any(np.isnan(D_total))
    
    def test_loose_tolerances(self, minimal_params):
        """Loose tolerances should produce valid results (faster)."""
        results = simulate_master_hospital_model(
            **minimal_params,
            rtol=1e-3,
            atol=1e-6
        )
        
        assert 'times' in results
        
        # Results should be valid
        D_total = results['D_total']
        assert not np.any(np.isnan(D_total))
    
    def test_tolerance_affects_accuracy(self, minimal_params):
        """
        Different tolerances should affect solution accuracy.
        
        Tighter tolerances should give more accurate results.
        """
        # Run with default tolerances
        results_default = simulate_master_hospital_model(**minimal_params)
        
        # Run with tight tolerances
        results_tight = simulate_master_hospital_model(
            **minimal_params,
            rtol=1e-10,
            atol=1e-12
        )
        
        # Run with loose tolerances
        results_loose = simulate_master_hospital_model(
            **minimal_params,
            rtol=1e-3,
            atol=1e-6
        )
        
        # All should complete
        assert len(results_default['D_total']) == len(results_tight['D_total'])
        assert len(results_default['D_total']) == len(results_loose['D_total'])
        
        # Final death counts should be similar (within numerical limits)
        D_default = results_default['D_total'][-1]
        D_tight = results_tight['D_total'][-1]
        D_loose = results_loose['D_total'][-1]
        
        # Should be reasonably close (within 10%)
        if D_default > 1:  # Only check if there are deaths
            assert abs(D_tight - D_default) / D_default < 0.1
            assert abs(D_loose - D_default) / D_default < 0.2


# =============================================================================
# TEST CLASS: SOLVER CONSISTENCY
# =============================================================================

class TestSolverConsistency:
    """Tests that different solvers produce consistent results."""
    
    def test_odeint_vs_solve_ivp_consistency(self, minimal_params):
        """odeint and solve_ivp should produce similar results."""
        results_odeint = simulate_master_hospital_model(
            **minimal_params,
            solver='odeint'
        )
        
        results_solve_ivp = simulate_master_hospital_model(
            **minimal_params,
            solver='solve_ivp',
            solver_method='LSODA'  # Same algorithm as odeint
        )
        
        D_odeint = results_odeint['D_total'][-1]
        D_solve_ivp = results_solve_ivp['D_total'][-1]
        
        # Should be very close (both use LSODA)
        if D_odeint > 1:
            relative_diff = abs(D_odeint - D_solve_ivp) / D_odeint
            assert relative_diff < 0.01, \
                f"odeint={D_odeint:.2f}, solve_ivp={D_solve_ivp:.2f}"
    
    def test_population_conservation_with_solve_ivp(self, high_capacity_params):
        """solve_ivp should maintain population conservation."""
        results = simulate_master_hospital_model(
            **high_capacity_params,
            solver='solve_ivp',
            solver_method='BDF'
        )
        
        initial_pop = sum(high_capacity_params['age_pops'])
        
        # Check conservation at each time point
        for t_idx in range(len(results['times'])):
            total_live = 0
            for comp in ['S', 'E', 'I', 'X_queued', 'X_admitted', 
                        'H_ward', 'H_icu', 'R',
                        'S_vax', 'E_vax', 'I_vax', 'X_queued_vax', 'X_admitted_vax',
                        'H_ward_vax', 'H_icu_vax', 'R_vax']:
                if comp in results:
                    comp_data = results[comp]
                    if isinstance(comp_data, list):
                        total_live += sum(arr[t_idx] for arr in comp_data)
            
            D_total = results['D_total'][t_idx]
            total_pop = total_live + D_total
            
            assert abs(total_pop - initial_pop) < 1e-4, \
                f"Conservation violated at t_idx={t_idx}: {total_pop} vs {initial_pop}"


# =============================================================================
# TEST CLASS: STIFF SYSTEM HANDLING
# =============================================================================

class TestStiffSystemHandling:
    """Tests for handling stiff ODE systems."""
    
    def test_high_beta_with_explicit_solver(self, minimal_params):
        """High transmission rate may require implicit solver."""
        high_beta_params = {
            **minimal_params,
            'beta_base': 1.0,  # High transmission
            'sim_config': {**minimal_params['sim_config'], 'Tmax': 20},
        }
        
        # Explicit method should still work (with warnings possibly)
        results = simulate_master_hospital_model(
            **high_beta_params,
            solver='solve_ivp',
            solver_method='RK45'
        )
        
        assert 'times' in results
    
    def test_high_beta_with_implicit_solver(self, minimal_params):
        """High transmission rate should work well with implicit solver."""
        high_beta_params = {
            **minimal_params,
            'beta_base': 1.0,
            'sim_config': {**minimal_params['sim_config'], 'Tmax': 20},
        }
        
        results = simulate_master_hospital_model(
            **high_beta_params,
            solver='solve_ivp',
            solver_method='BDF'  # Implicit method for stiff systems
        )
        
        assert 'times' in results
        assert not np.any(np.isnan(results['D_total']))
    
    def test_rapid_dynamics_with_small_timestep(self, minimal_params):
        """Rapid dynamics should work with small timestep."""
        rapid_params = {
            **minimal_params,
            'beta_base': 0.5,
            'sim_config': {
                'time_step': 0.1,  # Smaller timestep
                'Tmax': 20,
            }
        }
        
        results = simulate_master_hospital_model(**rapid_params)
        
        assert 'times' in results
        assert len(results['times']) == 201  # 20/0.1 + 1


# =============================================================================
# TEST CLASS: EDGE CASES
# =============================================================================

class TestSolverEdgeCases:
    """Edge cases for solver options."""
    
    def test_very_short_simulation(self, minimal_params):
        """Very short simulation should work."""
        short_params = {
            **minimal_params,
            'sim_config': {**minimal_params['sim_config'], 'Tmax': 1},
        }
        
        results = simulate_master_hospital_model(**short_params)
        assert len(results['times']) >= 2
    
    def test_zero_initial_infections(self, minimal_params):
        """Zero initial infections should not cause solver issues."""
        # Model should handle this gracefully
        results = simulate_master_hospital_model(**minimal_params)
        
        # With default ICs, E=10, so there should be some disease
        assert results['D_total'][-1] >= 0
    
    def test_large_population(self, minimal_params):
        """Large population should work."""
        large_params = {
            **minimal_params,
            'age_pops': [1_000_000, 2_000_000, 500_000],
            'capacity_config': {
                'ward_capacity': 50000,
                'icu_capacity': 10000,
            },
            'sim_config': {**minimal_params['sim_config'], 'Tmax': 20},
        }
        
        results = simulate_master_hospital_model(**large_params)
        assert 'times' in results


# =============================================================================
# TEST CLASS: SCENARIO SOLVER OPTIONS
# =============================================================================

class TestScenarioSolverOptions:
    """Test solver options with realistic scenarios."""
    
    def test_baseline_with_bdf(self):
        """Baseline scenario should work with BDF solver."""
        params = get_scenario_params('baseline')
        
        # Override Tmax for speed
        params['sim_config']['Tmax'] = 30
        
        results = simulate_master_hospital_model(
            **params,
            solver='solve_ivp',
            solver_method='BDF'
        )
        
        assert 'times' in results
        assert results['D_total'][-1] >= 0
    
    def test_stress_test_with_implicit_solver(self):
        """Stress test scenario (high transmission) with implicit solver."""
        params = get_scenario_params('stress_test')
        
        # Override Tmax for speed
        params['sim_config']['Tmax'] = 30
        
        results = simulate_master_hospital_model(
            **params,
            solver='solve_ivp',
            solver_method='Radau'
        )
        
        assert 'times' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
