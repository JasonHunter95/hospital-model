from manim import * # type: ignore
import sys
import os
import numpy as np

# Add parent directory to path to import model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from master_hospital_model import simulate_master_hospital_model
    from config_helpers import (
        get_scenario_params, compare_vaccine_profiles, compare_healthcare_systems,
        create_intervention_comparison, get_vaccine_profile, list_vaccine_profiles
    )
    from config import (
        INTERVENTION_NONE, INTERVENTION_EARLY_STRONG, INTERVENTION_DELAYED_STRONG,
        INTERVENTION_CYCLICAL, INTERVENTION_EARLY_MODERATE
    )
except ImportError:
    # Fallback for when running from different contexts
    sys.path.append("..")
    from master_hospital_model import simulate_master_hospital_model
    from config_helpers import (
        get_scenario_params, compare_vaccine_profiles, compare_healthcare_systems,
        create_intervention_comparison, get_vaccine_profile, list_vaccine_profiles
    )
    from config import (
        INTERVENTION_NONE, INTERVENTION_EARLY_STRONG, INTERVENTION_DELAYED_STRONG,
        INTERVENTION_CYCLICAL, INTERVENTION_EARLY_MODERATE
    )

class ModelData:
    """Helper class to run simulation and fetch data for animations."""
    def __init__(self, scenario='bad_covid_but_theres_a_vaccine'):
        try:
            self.params = get_scenario_params(scenario)
        except:
            # Fallback if scenario not found, use manual defaults
            print(f"Warning: Scenario '{scenario}' not found. Using default parameters.")
            from config import AGE_PARAMS_EMPIRICAL, CONTACT_MATRIX_DEFAULT
            self.params = {
                'beta_base': 0.35,
                'age_params': AGE_PARAMS_EMPIRICAL,
                'contact_matrix': CONTACT_MATRIX_DEFAULT,
                'age_pops': [120000, 200000, 80000],
                'Tmax': 100,
                'ward_capacity': 800,
                'icu_capacity': 200,
            }
        
        self.params['Tmax'] = 100  # Ensure long enough for visualization
        self.results = simulate_master_hospital_model(**self.params)
        
        # Store capacity values for visualization
        self.ward_capacity = self.params.get('ward_capacity', 800)
        self.icu_capacity = self.params.get('icu_capacity', 200)
        self.times = np.array(self.results['times'])
        
        # Pre-calculate totals if not present (though model usually returns them)
        if 'S_total' not in self.results:
            self.results['S_total'] = np.sum(self.results['S'], axis=0)
            self.results['E_total'] = np.sum(self.results['E'], axis=0)
            self.results['I_total'] = np.sum(self.results['I'], axis=0)
            self.results['X_admitted_total'] = np.sum(self.results['X_admitted'], axis=0)
            self.results['X_queued_total'] = np.sum(self.results['X_queued'], axis=0)
            self.results['H_ward_total'] = np.sum(self.results['H_ward'], axis=0)
            self.results['H_ICU_total'] = np.sum(self.results['H_icu'], axis=0)
            self.results['R_total'] = np.sum(self.results['R'], axis=0)
            self.results['D_total'] = np.sum(self.results['D'], axis=0)
            self.results['D_treated_total'] = np.sum(self.results['D_treated'], axis=0)
            self.results['D_untreated_total'] = np.sum(self.results['D_untreated'], axis=0)

    def get_times(self):
        return self.times

    def get_curve(self, key):
        return self.results[key]
    
    def get_value_at_time(self, key, t):
        """Interpolate value at specific time t."""
        return np.interp(t, self.times, self.results[key])

class EpidemicWaveScene(Scene):
    def construct(self):
        data = ModelData()
        max_time = 100 # Limit to 100 days for better pacing
        
        # Calculate max population for Y axis scaling
        total_pop = data.get_value_at_time('S_total', 0) + \
                    data.get_value_at_time('E_total', 0) + \
                    data.get_value_at_time('I_total', 0) + \
                    data.get_value_at_time('R_total', 0)
        
        # Logarithmic scaling setup
        # We plot log10(y + 1) to handle zeros and compress the scale
        # e.g., 200,000 -> log10 is ~5.3 -> ceil to 6 (1,000,000)
        y_max_exponent = int(np.ceil(np.log10(total_pop * 1.1)))

        # --- Layout ---
        # Axes on the left (65% of width)
        axes = Axes(
            x_range=[0, max_time, 100],
            y_range=[0, y_max_exponent, 1], # Y axis represents powers of 10
            x_length=8,
            y_length=6,
            axis_config={"color": WHITE, "include_numbers": False, "font_size": 20},
            tips=False,
        ).to_edge(LEFT, buff=0.8) # Increased buff for Y labels
        
        # Add X axis numbers manually
        axes.x_axis.add_numbers(range(0, max_time + 1, 20), font_size=16)
        
        # Custom Y axis labels (Powers of 10)
        y_labels = VGroup()
        for i in range(y_max_exponent + 1):
            val = 10**i
            if val >= 1000000:
                label_text = f"{val//1000000}M"
            elif val >= 1000:
                label_text = f"{val//1000}k"
            else:
                label_text = str(val)
            
            # Position label next to the tick
            label = Text(label_text, font_size=16).next_to(axes.c2p(0, i), LEFT, buff=0.15)
            y_labels.add(label)
        
        x_label = axes.get_x_axis_label("Time (Days)").scale(0.6).next_to(axes, DOWN, buff=0.1)
        y_label = axes.get_y_axis_label("Population (Log Scaled)").scale(0.6).rotate(90 * DEGREES).next_to(axes, LEFT, buff=0.4)

        # --- Dashboard ---
        dashboard_group = VGroup()
        title = Text("SEIXHRD Dynamics", font_size=32).to_edge(UP)
        dashboard_group.add(title)

        compartment_specs = [
            ("Susceptible", "S_total", BLUE),
            ("Exposed", "E_total", YELLOW),
            ("Infectious", "I_total", RED),
            ("Severe (Que)", "X_queued_total", ORANGE),
            ("Severe (Adm)", "X_admitted_total", PURPLE),
            ("Ward", "H_ward_total", TEAL),
            ("ICU", "H_ICU_total", MAROON),
            ("Recovered", "R_total", GREEN),
            ("Total Deaths", "D_total", GRAY),
            ("Deaths (Treated)", "D_treated_total", DARK_GRAY),
            ("Deaths (Untreated)", "D_untreated_total", LIGHT_GRAY),
        ]

        legend_items = VGroup()
        time_tracker = ValueTracker(0)
        
        # Pre-calculate points for efficiency
        curve_points = {}
        mask = data.times <= max_time
        times_clipped = data.times[mask]
        
        for name, key, color in compartment_specs:
            values_clipped = data.results[key][mask]
            # Log transform: log10(y + 1) to handle zeros safely
            log_values = np.log10(values_clipped + 1)
            points = [axes.c2p(x, y) for x, y in zip(times_clipped, log_values)]
            curve_points[key] = points

        # Create Legend Rows with Live Counters
        for name, key, color in compartment_specs:
            row = VGroup()
            indicator = Square(side_length=0.2, fill_color=color, fill_opacity=1, stroke_width=0)
            label = Text(name, font_size=18, color=WHITE).next_to(indicator, RIGHT, buff=0.2)
            
            # Fixed width container for number to prevent jitter
            num = DecimalNumber(0, num_decimal_places=0, include_sign=False, font_size=18, color=color)
            
            # Updater for the number
            def make_updater(k, n):
                return lambda m: m.set_value(data.get_value_at_time(k, time_tracker.get_value()))
            num.add_updater(make_updater(key, num))
            
            row.add(indicator, label, num)
            # Simple arrange
            row.arrange(RIGHT, aligned_edge=LEFT, buff=0.2)
            
            # Manually shift number to align columns (fixed offset from left of row)
            num.next_to(label, RIGHT, buff=2.5 - label.width) # Attempt to right-align
            
            legend_items.add(row)

        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        legend_items.next_to(axes, RIGHT, buff=0.5).to_edge(UP, buff=1.5)
        dashboard_group.add(legend_items)

        # --- Curves ---
        curves = VGroup()
        for name, key, color in compartment_specs:
            curve = VMobject().set_color(color).set_stroke(width=2)
            
            # Updater to draw curve up to current time
            def make_curve_updater(k, pts):
                def updater(mob):
                    t = time_tracker.get_value()
                    # Find index
                    idx = np.searchsorted(times_clipped, t)
                    if idx > 0:
                        mob.set_points_as_corners(pts[:idx+1])
                return updater

            curve.add_updater(make_curve_updater(key, curve_points[key]))
            curves.add(curve)

        # Scanning Line
        scan_line = always_redraw(lambda: Line(
            start=axes.c2p(time_tracker.get_value(), 0),
            end=axes.c2p(time_tracker.get_value(), y_max_exponent),
            color=WHITE, stroke_width=1, stroke_opacity=0.5
        ))
        
        # Hospital Capacity Lines (dotted)
        ward_capacity_log = np.log10(data.ward_capacity + 1)
        icu_capacity_log = np.log10(data.icu_capacity + 1)
        
        ward_capacity_line = DashedLine(
            start=axes.c2p(0, ward_capacity_log),
            end=axes.c2p(max_time, ward_capacity_log),
            color=TEAL,
            stroke_width=2,
            stroke_opacity=0.7,
            dash_length=0.15
        )
        ward_cap_label = Text(f"Ward Cap: {data.ward_capacity}", font_size=14, color=TEAL)
        ward_cap_label.next_to(ward_capacity_line, DOWN, buff=0.1)
        
        icu_capacity_line = DashedLine(
            start=axes.c2p(0, icu_capacity_log),
            end=axes.c2p(max_time, icu_capacity_log),
            color=MAROON,
            stroke_width=2,
            stroke_opacity=0.7,
            dash_length=0.15
        )
        icu_cap_label = Text(f"ICU Cap: {data.icu_capacity}", font_size=14, color=MAROON)
        icu_cap_label.next_to(icu_capacity_line, DOWN, buff=0.1)
        
        capacity_lines = VGroup(ward_capacity_line, ward_cap_label, icu_capacity_line, icu_cap_label)
        
        day_label = Text("Day: 0", font_size=24).next_to(axes, UP, aligned_edge=LEFT).shift(RIGHT * 0.5)
        day_label.add_updater(lambda m: m.become(Text(f"Day: {int(time_tracker.get_value())}", font_size=24).move_to(day_label.get_center())))
        
        # --- Animation ---
        self.add(axes, y_labels, x_label, y_label, dashboard_group, curves, scan_line, day_label, capacity_lines)
        
        self.play(
            time_tracker.animate.set_value(max_time),
            run_time=30,
            rate_func=linear
        )
        self.wait()


# =============================================================================
# COMPARISON DATA HELPER
# =============================================================================

def _to_array(data):
    """Convert list of scalar arrays or list to numpy array."""
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, list):
        # Could be list of 0-d arrays or list of scalars
        return np.array([float(x) for x in data])
    return np.array(data)


def _compute_totals(results):
    """Compute missing _total keys by summing age groups."""
    # Ensure times is array
    results['times'] = _to_array(results['times'])
    
    # Convert existing _total keys to proper arrays
    for key in list(results.keys()):
        if key.endswith('_total') and isinstance(results[key], list):
            results[key] = _to_array(results[key])
    
    # Compute S_total, R_total if missing (by summing age groups)
    if 'S_total' not in results and 'S' in results:
        results['S_total'] = np.sum([_to_array(s) for s in results['S']], axis=0)
    if 'R_total' not in results and 'R' in results:
        results['R_total'] = np.sum([_to_array(r) for r in results['R']], axis=0)
    
    # Compute X_admitted_total and X_queued_total
    if 'X_admitted_total' not in results and 'X_admitted' in results:
        results['X_admitted_total'] = np.sum([_to_array(x) for x in results['X_admitted']], axis=0)
    if 'X_queued_total' not in results and 'X_queued' in results:
        results['X_queued_total'] = np.sum([_to_array(x) for x in results['X_queued']], axis=0)
    
    # Convert beta_t, g_ward, g_icu, etc.
    for key in ['beta_t', 'g_ward', 'g_icu', 'policy_mult', 'seasonal_factor']:
        if key in results and isinstance(results[key], list):
            results[key] = _to_array(results[key])
    
    return results


class ComparisonData:
    """
    Helper class to run multiple scenario simulations for comparison animations.
    
    Usage:
        scenarios = compare_vaccine_profiles('covid_delta', ['mrna_original', 'inactivated'])
        data = ComparisonData(scenarios, Tmax=150)
        
        for name in data.scenario_names:
            deaths = data.get_curve(name, 'D_total')
    """
    def __init__(self, scenario_params: dict, Tmax: int = 200):
        """
        Args:
            scenario_params: Dict mapping scenario names to parameter dicts
            Tmax: Maximum simulation time (overrides individual scenario Tmax)
        """
        self.scenario_names = list(scenario_params.keys())
        self.results = {}
        self.Tmax = Tmax
        
        for name, params in scenario_params.items():
            params_copy = params.copy()
            params_copy['Tmax'] = Tmax
            raw_results = simulate_master_hospital_model(**params_copy)
            self.results[name] = _compute_totals(raw_results)
        
        # Store common time array from first scenario
        first_name = self.scenario_names[0]
        self.times = self.results[first_name]['times']
    
    def get_times(self):
        return self.times
    
    def get_curve(self, scenario_name: str, key: str):
        """Get time series data for a specific scenario and key."""
        return self.results[scenario_name][key]
    
    def get_value_at_time(self, scenario_name: str, key: str, t: float):
        """Interpolate value at specific time t."""
        return np.interp(t, self.times, self.results[scenario_name][key])
    
    def get_final_value(self, scenario_name: str, key: str):
        """Get final value of a time series."""
        return self.results[scenario_name][key][-1]
    
    def get_max_value(self, key: str):
        """Get maximum value across all scenarios for scaling."""
        return max(np.max(self.results[name][key]) for name in self.scenario_names)


# =============================================================================
# SCENE 2: VACCINE COMPARISON
# =============================================================================

class VaccineComparisonScene(Scene):
    """
    Animated comparison of different vaccine profiles showing:
    - Racing cumulative death curves
    - Three-factor efficacy breakdown (VE_infection, VE_severe, VE_death)
    - Final outcome comparison
    
    Duration: ~60 seconds for 150 days
    """
    
    # Color palette for different vaccine profiles (distinct, colorblind-friendly)
    VACCINE_COLORS = {
        'no_vaccine': GRAY,
        'minimal': RED_C,
        'inactivated': ORANGE,
        'adenovirus': YELLOW_C,
        'mrna_original': GREEN_C,
        'mrna_omicron': TEAL_C,
        'ideal': BLUE_C,
    }
    
    def construct(self):
        # === Setup: Load comparison data ===
        profiles_to_compare = ['minimal', 'inactivated', 'ideal']
        scenario_params = compare_vaccine_profiles(
            'bad_covid_but_theres_a_vaccine', 
            profiles_to_compare,
            include_no_vaccine=True
        )
        
        max_time = 150
        data = ComparisonData(scenario_params, Tmax=max_time)
        
        # === Phase 1: Title and Introduction (0-5s) ===
        title = Text("Vaccine Efficacy Comparison", font_size=36, weight=BOLD)
        title_group = VGroup(title).arrange(DOWN, buff=0.3)
        title_group.to_edge(UP, buff=0.4)
        
        self.play(FadeIn(title_group), run_time=1.5)
        self.wait(0.5)
        
        # === Build Main Axes (left 60% of screen) ===
        y_max = data.get_max_value('D_total') * 1.15
        
        axes = Axes(
            x_range=[0, max_time, 30],
            y_range=[0, y_max, y_max/5],
            x_length=7.5,
            y_length=4.5,
            axis_config={"color": WHITE, "include_numbers": False, "font_size": 18},
            tips=False,
        ).shift(LEFT * 1.5 + DOWN * 0.5)
        
        # X-axis labels
        x_nums = VGroup()
        for x in range(0, max_time + 1, 30):
            label = Text(str(x), font_size=14)
            label.next_to(axes.c2p(x, 0), DOWN, buff=0.15)
            x_nums.add(label)
        
        # Y-axis labels (formatted with K suffix)
        y_nums = VGroup()
        for y in np.linspace(0, y_max, 6):
            if y >= 1000:
                label_text = f"{y/1000:.1f}K"
            else:
                label_text = f"{int(y)}"
            label = Text(label_text, font_size=14)
            label.next_to(axes.c2p(0, y), LEFT, buff=0.15)
            y_nums.add(label)
        
        x_label = Text("Days", font_size=16).next_to(axes, DOWN, buff=0.4)
        y_label = Text("Cumulative Deaths", font_size=16).rotate(90 * DEGREES)
        y_label.next_to(axes, LEFT, buff=0.9)
        
        # === Build Legend (right side) ===
        legend = VGroup()
        time_tracker = ValueTracker(0)
        
        # Order: worst to best for visual clarity
        display_order = ['no_vaccine', 'minimal', 'inactivated', 'ideal']
        
        for name in display_order:
            if name not in data.scenario_names:
                continue
            color = self.VACCINE_COLORS.get(name, WHITE)
            
            # Row: colored square + label + live death count
            square = Square(side_length=0.22, fill_color=color, fill_opacity=1, stroke_width=0)
            
            # Clean up display name
            display_name = name.replace('_', ' ').title()
            if len(display_name) > 12:
                display_name = display_name[:12]
            label = Text(display_name, font_size=15, color=WHITE)
            if label.width > 1.4:
                label.width = 1.4  # Cap width using property setter
            
            # Death counter
            death_num = DecimalNumber(0, num_decimal_places=0, font_size=15, color=color)
            
            def make_death_updater(scenario_name, num_obj):
                return lambda m: m.set_value(
                    data.get_value_at_time(scenario_name, 'D_total', time_tracker.get_value())
                )
            death_num.add_updater(make_death_updater(name, death_num))
            
            row = VGroup(square, label, death_num).arrange(RIGHT, buff=0.15)
            legend.add(row)
        
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        legend.to_edge(RIGHT, buff=0.4).shift(UP * 0.3)
        
        # Day counter
        day_label = Text("Day: 0", font_size=20)
        day_label.next_to(axes, UP, buff=0.15)
        day_label.add_updater(lambda m: m.become(
            Text(f"Day: {int(time_tracker.get_value())}", font_size=20).move_to(m.get_center())
        ))
        
        # === Prepare Curve Data ===
        mask = data.times <= max_time
        times_clipped = data.times[mask]
        
        curve_points = {}
        for name in data.scenario_names:
            values = data.results[name]['D_total'][mask]
            points = [axes.c2p(x, y) for x, y in zip(times_clipped, values)]
            curve_points[name] = points
        
        # === Create Animated Curves ===
        curves = VGroup()
        for name in display_order:
            if name not in data.scenario_names:
                continue
            color = self.VACCINE_COLORS.get(name, WHITE)
            curve = VMobject().set_color(color).set_stroke(width=2.5)
            
            def make_curve_updater(scenario_name, pts):
                def updater(mob):
                    t = time_tracker.get_value()
                    idx = np.searchsorted(times_clipped, t)
                    if idx > 1:
                        mob.set_points_as_corners(pts[:idx])
                return updater
            
            curve.add_updater(make_curve_updater(name, curve_points[name]))
            curves.add(curve)
        
        # === Phase 2: Show axes and start animation (5-50s) ===
        axes_group = VGroup(axes, x_nums, y_nums, x_label, y_label)
        self.play(Create(axes_group), run_time=1.5)
        self.play(FadeIn(legend), FadeIn(day_label), run_time=1)
        
        self.add(curves)
        
        # Main animation: curves racing
        self.play(
            time_tracker.animate.set_value(max_time),
            run_time=40,
            rate_func=linear
        )