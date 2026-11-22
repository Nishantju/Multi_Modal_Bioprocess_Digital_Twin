from .rheology import (
    calculate_shear_stress,
    calculate_apparent_viscosity,
    classify_fluid
)


K = 0.5
shear_rate = 100

flow_indices = [0.6, 1.0, 1.4]

for n in flow_indices:

    stress = calculate_shear_stress(
        K, shear_rate, n
    )

    viscosity = calculate_apparent_viscosity(
        K, shear_rate, n
    )

    fluid_type = classify_fluid(n)

    print("\nFlow behaviour index:", n)
    print("Fluid type:", fluid_type)
    print("Shear stress:", stress, "Pa")
    print("Apparent viscosity:", viscosity, "Pa.s")