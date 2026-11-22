from .population_opto_metabolism import (
    simulate_opto_metabolism
)

from visualization.metabolic_visualization import (
    plot_metabolic_results
)


result = simulate_opto_metabolism(
    population_size=100,
    duration=10
)

plot_metabolic_results(result)