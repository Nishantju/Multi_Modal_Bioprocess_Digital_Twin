from .integrated_bioreactor import (
    simulate_integrated_bioreactor
)

from visualization.integrated_bioreactor_visualization import (
    plot_integrated_bioreactor
)


result = simulate_integrated_bioreactor(
    population_size=100,
    duration=20
)

plot_integrated_bioreactor(result)