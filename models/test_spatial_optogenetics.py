from .spatial_optogenetics import (
    calculate_spatial_activation
)

result = calculate_spatial_activation(
    width=1.0,
    height=2.0,
    surface_intensity=100,
    attenuation_coefficient=2,
    half_activation_intensity=30,
    hill_coefficient=2,
    resolution=5
)

print("SPATIAL GENE ACTIVATION GRID")
print(result["activation"])

print("\nSPATIAL LIGHT INTENSITY GRID")
print(result["intensity"])