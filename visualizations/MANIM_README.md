# Manim Visualizations Guide for SEIXHRD Epidemic Model

Animated visualizations of the SEIXHRD epidemic model using [Manim](https://www.manim.community/).

The actual libary that 3b1b uses is his own fork, but the main libary is honestly just easier to use in my opinion.

## Prerequisites

1. **Install Manim**: (its possible theres another dependency called Ffmpeg that you might need to install separately depending on your system, but I'm pretty sure pip usually handles it)

   ```bash
   pip install manim
   ```

## Scenes I have for this repo:

| Scene | Description |
|-------|-------------|
| `EpidemicWaveScene` | Full model dynamics, live counters, and hospital capacity lines |
| `VaccineComparisonScene` | Graph of total death curves comparing vaccine profiles (no vaccine, minimal, inactivated, ideal) |

## Running Animations

From the `visualizations/` directory:

```bash
# Render EpidemicWaveScene (medium quality, preview)
manim -pql manim_scenes.py EpidemicWaveScene

# Render VaccineComparisonScene (medium quality, preview)
manim -pql manim_scenes.py VaccineComparisonScene

# High quality render (slower)
manim -pqh manim_scenes.py EpidemicWaveScene

# Production quality (1080p, 60fps)
manim -qh --fps 60 manim_scenes.py VaccineComparisonScene
```

### Customize Quality/Framerate Output

| Flag | Description |
|------|-------------|
| `-p` | Preview (opens video after rendering) |
| `-ql` | Low quality (480p, 15fps) - fastest |
| `-qm` | Medium quality (720p, 30fps) |
| `-qh` | High quality (1080p, 60fps) |
| `-s` | Save last frame as image only |

## Output Location

Videos end up save in their respective quality folder:

```
visualizations/media/videos/manim_scenes/<quality>/
```

## Customization

### Change Scenario

Edit the `ModelData` or `ComparisonData` initialization:

```python
# In EpidemicWaveScene
data = ModelData(scenario='covid_delta')

# In VaccineComparisonScene
profiles_to_compare = ['mrna_original', 'adenovirus', 'inactivated']
scenario_params = compare_vaccine_profiles('covid_delta', profiles_to_compare)
```

### Adjust Animation Duration

Modify `max_time` and `run_time` parameters:

```python
max_time = 200  # simulate 200 days instead of 100
self.play(time_tracker.animate.set_value(max_time), run_time=60)  # 60 second animation
```

### Add New Compartments to EpidemicWaveScene

Add entries to `compartment_specs`:

```python
compartment_specs = [
    # ... existing ...
    ("New Compartment", "new_key_total", PINK),
]
```

## Troubleshooting

**Missing scenarios**: Check available scenarios with:
```python
from scenarios import SCENARIO_REGISTRY
print(list(SCENARIO_REGISTRY.keys()))
```

**Slow rendering**: Use `-ql` for quick previews during development.