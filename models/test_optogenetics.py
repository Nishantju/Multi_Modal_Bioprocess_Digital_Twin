from .optogenetics import (
    calculate_light_intensity,
    calculate_opto_activation
)


surface_intensity = 100
attenuation_coefficient = 2
half_activation_intensity = 30
hill_coefficient = 2

depths = [0, 0.25, 0.5, 0.75, 1.0]

print("DEPTH | LIGHT INTENSITY | GENE ACTIVATION")

for depth in depths:

    intensity = calculate_light_intensity(
        surface_intensity,
        attenuation_coefficient,
        depth
    )

    activation = calculate_opto_activation(
        intensity,
        half_activation_intensity,
        hill_coefficient
    )

    print(
        depth,
        "|",
        round(intensity, 3),
        "|",
        round(activation, 4)
    )