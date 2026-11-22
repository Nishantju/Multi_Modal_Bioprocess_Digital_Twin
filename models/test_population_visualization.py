from .population_opto_model import simulate_population_opto

from visualization.population_visualization import (
    plot_population_results
)


result = simulate_population_opto(
    population_size=100,
    duration=10
)

plot_population_results(result)