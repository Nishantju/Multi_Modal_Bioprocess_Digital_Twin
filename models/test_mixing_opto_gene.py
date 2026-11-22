from .mixing_opto_gene import (
    simulate_moving_microorganism
)

from .opto_gene_coupling import (
    simulate_opto_gene_expression
)


common_parameters = {
    "initial_mRNA": 0.2,
    "initial_protein": 0.1,
    "duration": 10,

    "surface_intensity": 100,
    "attenuation_coefficient": 2,

    "half_activation_intensity": 30,
    "hill_coefficient": 2,

    "basal_activation": 0.1,

    "alpha_m": 2.0,
    "delta_m": 0.2,
    "alpha_e": 0.5,
    "delta_e": 0.1
}


# Case 1: Stationary near the surface

surface = simulate_opto_gene_expression(
    **common_parameters,
    depth=0.1
)


# Case 2: Stationary deep inside the reactor

deep = simulate_opto_gene_expression(
    **common_parameters,
    depth=1.9
)


# Case 3: Moving microorganism

moving = simulate_moving_microorganism(
    **common_parameters,
    mean_depth=1.0,
    movement_amplitude=0.9,
    movement_frequency=0.2
)


print("COMPARISON OF MICROORGANISM MOVEMENT")

print("\nCONDITION | FINAL mRNA | FINAL PROTEIN")

print(
    "Surface |",
    round(surface["mRNA"][-1], 4),
    "|",
    round(surface["protein"][-1], 4)
)

print(
    "Deep |",
    round(deep["mRNA"][-1], 4),
    "|",
    round(deep["protein"][-1], 4)
)

print(
    "Moving |",
    round(moving["mRNA"][-1], 4),
    "|",
    round(moving["protein"][-1], 4)
)


print("\nMOVING MICROORGANISM")

print(
    "Average light intensity:",
    round(moving["light_intensity"].mean(), 4)
)

print(
    "Average gene activation:",
    round(moving["activation"].mean(), 4)
)

print(
    "Minimum depth:",
    round(moving["depth"].min(), 4)
)

print(
    "Maximum depth:",
    round(moving["depth"].max(), 4)
)