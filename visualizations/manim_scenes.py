from manim import *
import sys
import os
import numpy as np

# Add parent directory to path to import model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from master_hospital_model import simulate_master_hospital_model
    from config_helpers import get_scenario_params
except ImportError:
    # Fallback for when running from different contexts
    sys.path.append("..")
    from master_hospital_model import simulate_master_hospital_model
    from config_helpers import get_scenario_params

class ModelData:
    """Helper class to run simulation and fetch data for animations."""
    def __init__(self, scenario='covid_delta'):
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
                'age_pops': [30000, 50000, 20000],
                'Tmax': 100
            }
        
        self.params['Tmax'] = 100  # Ensure long enough for visualization
        self.results = simulate_master_hospital_model(**self.params)
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
        y_label = axes.get_y_axis_label("Population").scale(0.6).rotate(90 * DEGREES).next_to(axes, LEFT, buff=0.4)

        # --- Dashboard ---
        dashboard_group = VGroup()
        title = Text("SEIXHRD Dynamics", font_size=32).to_edge(UP)
        dashboard_group.add(title)

        compartment_specs = [
            ("Susceptible", "S_total", BLUE),
            ("Exposed", "E_total", YELLOW),
            ("Infectious", "I_total", RED),
            ("Severe (Adm)", "X_admitted_total", PURPLE),
            ("Severe (Que)", "X_queued_total", ORANGE),
            ("Ward", "H_ward_total", TEAL),
            ("ICU", "H_ICU_total", MAROON),
            ("Recovered", "R_total", GREEN),
            ("Deceased", "D_total", GRAY),
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
        
        day_label = Text("Day: 0", font_size=24).next_to(axes, UP, aligned_edge=LEFT).shift(RIGHT * 0.5)
        day_label.add_updater(lambda m: m.become(Text(f"Day: {int(time_tracker.get_value())}", font_size=24).move_to(day_label.get_center())))
        
        # --- Animation ---
        self.add(axes, y_labels, x_label, y_label, dashboard_group, curves, scan_line, day_label)
        
        self.play(
            time_tracker.animate.set_value(max_time),
            run_time=30,
            rate_func=linear
        )
        self.wait()