from manim import * # type: ignore
import sys
import os
import numpy as np

# adds parent directory to path to import model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from simulate_model import simulate_model
    from scenario_helpers import (
        get_scenario_params, compare_vaccine_profiles
    )
    from scenarios import (
        INTERVENTION_NONE, INTERVENTION_EARLY_STRONG, INTERVENTION_DELAYED_STRONG,
        INTERVENTION_CYCLICAL, INTERVENTION_EARLY_MODERATE
    )
except ImportError:
    # fallback for when running from different contexts
    sys.path.append("..")
    from simulate_model import simulate_model
    from scenario_helpers import (
        get_scenario_params, compare_vaccine_profiles
    )
    from scenarios import (
        INTERVENTION_NONE, INTERVENTION_EARLY_STRONG, INTERVENTION_DELAYED_STRONG,
        INTERVENTION_CYCLICAL, INTERVENTION_EARLY_MODERATE
    )

class ModelData:
    """Helper class to run simulations and fetch data for drawing the animations."""
    def __init__(self, scenario='seasonal_flu'):
        try:
            self.params = get_scenario_params(scenario)
        except:
            # fallback if scenario not found, use the manual defaults set here
            print(f"Warning: Scenario '{scenario}' not found. Using default parameters.")
            from scenarios import AGE_PARAMS_EMPIRICAL, CONTACT_MATRIX_DEFAULT
            self.params = {
                'beta_base': 0.35,
                'age_params': AGE_PARAMS_EMPIRICAL,
                'contact_matrix': CONTACT_MATRIX_DEFAULT,
                'age_pops': [120000, 200000, 80000],
                'sim_config': {'Tmax': 100},
                'capacity_config': {'ward_capacity': 800, 'icu_capacity': 200},
            }
        
        # override Tmax in sim_config
        if 'sim_config' in self.params:
            self.params['sim_config'] = {**self.params['sim_config'], 'Tmax': 100}
        else:
            self.params['sim_config'] = {'Tmax': 100}
        
        self.results = simulate_model(**self.params)
        
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
        
        # store capacity values for drawing in the animation
        capacity_config = self.params.get('capacity_config', {})
        self.ward_capacity = capacity_config.get('ward_capacity', 800)
        self.icu_capacity = capacity_config.get('icu_capacity', 200)
        self.times = np.array(self.results['times'])

    def get_times(self):
        return self.times

    def get_curve(self, key):
        return self.results[key]
    
    def get_value_at_time(self, key, t):
        """Interpolate value at specific time t."""
        return np.interp(t, self.times, self.results[key])
    

# =============================================================================
# COMPARISON DATA HELPER
# =============================================================================

def _to_array(data):
    """Convert list of scalar arrays or list to numpy array."""
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, list):
        # could be list of 0-d arrays or list of scalars
        return np.array([float(x) for x in data])
    return np.array(data)


def _compute_totals(results):
    """Computes any missing _total keys by summing age groups and converts all lists to arrays."""
    # ensure times is array
    results['times'] = _to_array(results['times'])
    
    # convert existing _total keys to proper arrays
    for key in list(results.keys()):
        if key.endswith('_total') and isinstance(results[key], list):
            results[key] = _to_array(results[key])
    
    # compute S_total, R_total if missing (by summing age groups)
    if 'S_total' not in results and 'S' in results:
        results['S_total'] = np.sum([_to_array(s) for s in results['S']], axis=0)
    if 'R_total' not in results and 'R' in results:
        results['R_total'] = np.sum([_to_array(r) for r in results['R']], axis=0)
    
    # compute X_admitted_total and X_queued_total
    if 'X_admitted_total' not in results and 'X_admitted' in results:
        results['X_admitted_total'] = np.sum([_to_array(x) for x in results['X_admitted']], axis=0)
    if 'X_queued_total' not in results and 'X_queued' in results:
        results['X_queued_total'] = np.sum([_to_array(x) for x in results['X_queued']], axis=0)
    
    # convert beta_t, g_ward, g_icu, etc.
    for key in ['beta_t', 'g_ward', 'g_icu', 'policy_mult', 'seasonal_factor']:
        if key in results and isinstance(results[key], list):
            results[key] = _to_array(results[key])
    
    return results


class ComparisonData:
    """
    Helper class to run multiple scenario simulations for comparison animations.
    Stores results and provides methods to access time series data.
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
            if 'sim_config' in params_copy:
                params_copy['sim_config'] = {**params_copy['sim_config'], 'Tmax': Tmax}
            else:
                params_copy['sim_config'] = {'Tmax': Tmax}
            raw_results = simulate_model(**params_copy)
            self.results[name] = _compute_totals(raw_results)
        
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

class EpidemicWaveScene(Scene):
    def construct(self):
        data = ModelData()
        max_time = 100 # limit to 100 days for better pacing

        
        
        # calculates max population to scale the Y-axis properly
        total_pop = data.get_value_at_time('S_total', 0) + \
                    data.get_value_at_time('E_total', 0) + \
                    data.get_value_at_time('I_total', 0) + \
                    data.get_value_at_time('R_total', 0)
        
        # logarithmic scaling setup
        # plot log10(y + 1) to handle zeros and compress the scale so that the smaller compartment behavior is still visible
        # e.g., 200,000 -> log10 is ~5.3 -> ceil to 6 (1,000,000)
        y_max_exponent = int(np.ceil(np.log10(total_pop * 1.1)))

        ## ===========================================================================
        ## Layout Setup
        ## ===========================================================================
        
        # axes on the left (65% of width)
        axes = Axes(
            x_range=[0, max_time, 100],
            y_range=[0, y_max_exponent, 1], # Y axis represents powers of 10
            x_length=8,
            y_length=6,
            axis_config={"color": WHITE, "include_numbers": False, "font_size": 20},
            tips=False,
        ).to_edge(LEFT, buff=0.8) # increased buff for Y labels
        
        # adds X axis numbers manually
        axes.x_axis.add_numbers(range(0, max_time + 1, 20), font_size=16)
        
        # custom Y axis labels for log scale (they were overcrowded before)
        y_labels = VGroup()
        for i in range(y_max_exponent + 1):
            val = 10**i
            if val >= 1000000:
                label_text = f"{val//1000000}M"
            elif val >= 1000:
                label_text = f"{val//1000}k"
            else:
                label_text = str(val)
            
            # position label next to the tick
            label = Text(label_text, font_size=16).next_to(axes.c2p(0, i), LEFT, buff=0.15)
            y_labels.add(label)
        
        x_label = axes.get_x_axis_label("Time (Days)").scale(0.6).next_to(axes, DOWN, buff=0.1)
        y_label = axes.get_y_axis_label("Population (Log Scaled)").scale(0.6).rotate(90 * DEGREES).next_to(axes, LEFT, buff=0.4)

        ## ===========================================================================
        ## Dashboard
        ## ===========================================================================
        
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
        
        # pre-calculate points for efficiency
        curve_points = {}
        mask = data.times <= max_time
        times_clipped = data.times[mask]
        
        for name, key, color in compartment_specs:
            values_clipped = data.results[key][mask]
            # log transform: log10(y + 1) to handle zeros safely
            log_values = np.log10(values_clipped + 1)
            points = [axes.c2p(x, y) for x, y in zip(times_clipped, log_values)]
            curve_points[key] = points

        # create legend rows with live counters for tracking each compartment
        for name, key, color in compartment_specs:
            row = VGroup()
            indicator = Square(side_length=0.2, fill_color=color, fill_opacity=1, stroke_width=0)
            label = Text(name, font_size=18, color=WHITE).next_to(indicator, RIGHT, buff=0.2)
            
            # fixed width container for number to prevent jitter during updates
            num = DecimalNumber(0, num_decimal_places=0, include_sign=False, font_size=18, color=color)
            
            # updater for the number
            def make_updater(k, n):
                return lambda m: m.set_value(data.get_value_at_time(k, time_tracker.get_value()))
            num.add_updater(make_updater(key, num))
            
            row.add(indicator, label, num)
            # simple arrange to set initial positions
            row.arrange(RIGHT, aligned_edge=LEFT, buff=0.2)
            
            # manually shift number to align columns (fixed offset from left of row)
            num.next_to(label, RIGHT, buff=2.5 - label.width) # attempt to right-align
            
            legend_items.add(row)

        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        legend_items.next_to(axes, RIGHT, buff=0.5).to_edge(UP, buff=1.5)
        dashboard_group.add(legend_items)

        ## ===========================================================================
        ## Curves
        ## ===========================================================================
        curves = VGroup()
        for name, key, color in compartment_specs:
            curve = VMobject().set_color(color).set_stroke(width=2)
            
            # updater to draw curve up to current time
            def make_curve_updater(k, pts):
                def updater(mob):
                    t = time_tracker.get_value()
                    # find index
                    idx = np.searchsorted(times_clipped, t)
                    if idx > 0:
                        mob.set_points_as_corners(pts[:idx+1])
                return updater

            curve.add_updater(make_curve_updater(key, curve_points[key]))
            curves.add(curve)

        # scanning Line to indicate current time
        scan_line = always_redraw(lambda: Line(
            start=axes.c2p(time_tracker.get_value(), 0),
            end=axes.c2p(time_tracker.get_value(), y_max_exponent),
            color=WHITE, stroke_width=1, stroke_opacity=0.5
        ))
        
        # hospital capacity lines (dotted) for ward and ICU
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
        
        ## ===========================================================================
        ## Animation Sequence
        ## ===========================================================================
        
        # add all elements to the scene
        self.add(axes, y_labels, x_label, y_label, dashboard_group, curves, scan_line, day_label, capacity_lines)
        
        # animate the scene over time for each frame
        self.play(
            time_tracker.animate.set_value(max_time),
            run_time=30,
            rate_func=linear
        )
        self.wait()





# =============================================================================
# SCENE 2: VACCINE COMPARISON
# =============================================================================

class VaccineComparisonScene(Scene):
    """
    Animated comparison of different vaccine profiles showing:
    - Racing cumulative death curves   
    """
    
    # color palette for different vaccine profiles
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
        profiles_to_compare = ['minimal', 'inactivated', 'ideal']
        scenario_params = compare_vaccine_profiles(
            'seasonal_flu', 
            profiles_to_compare,
            include_no_vaccine=True
        )
        
        max_time = 150
        data = ComparisonData(scenario_params, Tmax=max_time)
        
        ## ===========================================================================
        ## Title and Layout Setup
        ## ===========================================================================
        
        title = Text("Vaccine Comparison", font_size=36, weight=BOLD)
        title_group = VGroup(title).arrange(DOWN, buff=0.3)
        title_group.to_edge(UP, buff=0.4)
        
        self.play(FadeIn(title_group), run_time=1.5)
        self.wait(0.5)
        
        ## main axes setup
        y_max = data.get_max_value('D_total') * 1.15
        
        axes = Axes(
            x_range=[0, max_time, 30],
            y_range=[0, y_max, y_max/5],
            x_length=7.5,
            y_length=4.5,
            axis_config={"color": WHITE, "include_numbers": False, "font_size": 18},
            tips=False,
        ).shift(LEFT * 1.5 + DOWN * 0.5)
        
        # x-axis labels
        x_nums = VGroup()
        for x in range(0, max_time + 1, 30):
            label = Text(str(x), font_size=14)
            label.next_to(axes.c2p(x, 0), DOWN, buff=0.15)
            x_nums.add(label)
        
        # y-axis labels
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
        
        ## build legend
        legend = VGroup()
        time_tracker = ValueTracker(0)
        
        # Order: worst to best
        display_order = ['no_vaccine', 'minimal', 'inactivated', 'ideal']
        
        for name in display_order:
            if name not in data.scenario_names:
                continue
            color = self.VACCINE_COLORS.get(name, WHITE)
            
            # row: colored square + label + live death count
            square = Square(side_length=0.22, fill_color=color, fill_opacity=1, stroke_width=0)
            
            # clean up display name
            display_name = name.replace('_', ' ').title()
            if len(display_name) > 12:
                display_name = display_name[:12]
            label = Text(display_name, font_size=15, color=WHITE)
            if label.width > 1.4:
                label.width = 1.4
            
            # counter for deaths
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
        
        # counter for days
        day_label = Text("Day: 0", font_size=20)
        day_label.next_to(axes, UP, buff=0.15)
        day_label.add_updater(lambda m: m.become(
            Text(f"Day: {int(time_tracker.get_value())}", font_size=20).move_to(m.get_center())
        ))
        
        ## prepare curve data
        mask = data.times <= max_time
        times_clipped = data.times[mask]
        
        curve_points = {}
        for name in data.scenario_names:
            values = data.results[name]['D_total'][mask]
            points = [axes.c2p(x, y) for x, y in zip(times_clipped, values)]
            curve_points[name] = points
        
        ## create animated curves
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
        
        ## show axes and start animation sequence
        axes_group = VGroup(axes, x_nums, y_nums, x_label, y_label)
        self.play(Create(axes_group), run_time=1.5)
        self.play(FadeIn(legend), FadeIn(day_label), run_time=1)
        
        self.add(curves)
        
        # curves are drawn on the screen over time
        self.play(
            time_tracker.animate.set_value(max_time),
            run_time=40,
            rate_func=linear
        )