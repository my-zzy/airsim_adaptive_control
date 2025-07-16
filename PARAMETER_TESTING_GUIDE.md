# Parameter Testing Guide

This guide explains how to use the enhanced parameter testing system for the adaptive control simulation.

## Parameter Sets Available

The system now includes 5 different parameter sets for testing:

0. **baseline** - Original parameters (default)
1. **fast_adaptation** - Higher lambda values for faster adaptation
2. **aggressive_control** - Higher c values for more aggressive control
3. **conservative** - Lower values for better stability
4. **mixed_tuning** - Mixed tuning with different gains for different axes

## Usage Examples

### 1. List Available Parameter Sets
```bash
python run_att_in_analytical_model.py --list-params
```

### 2. Run with a Specific Parameter Set
```bash
# Run with parameter set 1 (fast_adaptation)
python run_att_in_analytical_model.py --param-set 1

# Run with parameter set 3 (conservative)
python run_att_in_analytical_model.py --param-set 3
```

### 3. Cycle to Next Parameter Set
```bash
# This will automatically move to the next parameter set and run
python run_att_in_analytical_model.py --cycle
```

### 4. Test All Parameter Sets Sequentially
```bash
# This will run simulations with all parameter sets and save results
python run_att_in_analytical_model.py --test-all
```

### 5. Run with Current Parameter Set (Default Behavior)
```bash
# Just run with whatever parameter set is currently selected
python run_att_in_analytical_model.py
```

## Manual Parameter Set Changes

You can also manually change the parameter set in code:

```python
import config as cfg

# List available sets
cfg.list_param_sets()

# Change to a specific set
cfg.set_param_set_index(2)  # Use aggressive_control

# Cycle to next set
cfg.cycle_param_set()

# Get current parameters
current_params = cfg.get_current_params()
print(f"Currently using: {current_params['name']}")
```

## Output Files

When running tests, the system will save plots with descriptive names:

- `pwm_signals_baseline.png` - PWM signals for baseline parameters
- `control_analysis_fast_adaptation.png` - Control analysis for fast adaptation
- etc.

This makes it easy to compare results between different parameter sets.

## Adding New Parameter Sets

To add new parameter sets, edit the `PARAM_SETS` list in `config.py`:

```python
# Add a new parameter set
{
    'lamphi': 0.12,
    'lamthe': 0.08,
    'lampsi': 0.015,
    'cphi': 6,
    'cthe': 4,
    'cpsi': 4,
    'cp': 0.12,
    'cq': 0.08,
    'cr': 0.015,
    'lamphi_star': 0.12,
    'lamthe_star': 0.08,
    'lampsi_star': 0.015,
    'name': 'custom_tuning'
}
```

## Parameter Descriptions

- **lamphi, lamthe, lampsi**: Adaptation gains for roll, pitch, yaw
- **cphi, cthe, cpsi**: Control gains for roll, pitch, yaw
- **cp, cq, cr**: Additional control parameters
- **lamphi_star, lamthe_star, lampsi_star**: Disturbance estimation gains
