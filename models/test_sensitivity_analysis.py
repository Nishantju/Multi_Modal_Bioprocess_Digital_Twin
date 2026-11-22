from .parameter_sweep import run_parameter_sweep

from visualization.sensitivity_visualization import (
    plot_sensitivity_results
)


result = run_parameter_sweep()

plot_sensitivity_results(result)