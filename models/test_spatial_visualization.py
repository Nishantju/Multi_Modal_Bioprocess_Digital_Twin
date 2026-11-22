from .spatial_optogenetics import (
    calculate_spatial_activation
)

from visualization.opto_heatmap import (
    plot_opto_heatmaps
)


width = 1.0
height = 2.0

result = calculate_spatial_activation(
    width=width,
    height=height,
    surface_intensity=100,
    attenuation_coefficient=2,
    half_activation_intensity=30,
    hill_coefficient=2,
    resolution=50
)

plot_opto_heatmaps(
    result,
    width,
    height
)