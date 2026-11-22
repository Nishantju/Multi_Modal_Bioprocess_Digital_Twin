import numpy as np


def calculate_light_intensity(
    surface_intensity,
    attenuation_coefficient,
    depth
):
    intensity = surface_intensity * np.exp(
        -attenuation_coefficient * depth
    )

    return intensity


def calculate_opto_activation(
    light_intensity,
    half_activation_intensity,
    hill_coefficient
):

    light_intensity = np.asarray(
        light_intensity,
        dtype=float
    )

    if np.any(light_intensity < 0):
        raise ValueError(
            "Light intensity cannot be negative."
        )

    if half_activation_intensity <= 0:
        raise ValueError(
            "Half activation intensity must be positive."
        )

    if hill_coefficient <= 0:
        raise ValueError(
            "Hill coefficient must be positive."
        )

    numerator = light_intensity ** hill_coefficient

    denominator = (
        half_activation_intensity ** hill_coefficient
        + numerator
    )

    activation = numerator / denominator

    return activation