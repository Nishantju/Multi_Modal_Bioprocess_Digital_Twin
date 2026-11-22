import numpy as np

from .optogenetics import (
    calculate_light_intensity,
    calculate_opto_activation
)


def calculate_spatial_activation(
    width,
    height,
    surface_intensity,
    attenuation_coefficient,
    half_activation_intensity,
    hill_coefficient,
    resolution=50
):

    x = np.linspace(0, width, resolution)
    z = np.linspace(0, height, resolution)

    X, Z = np.meshgrid(x, z)

    intensity = calculate_light_intensity(
        surface_intensity,
        attenuation_coefficient,
        Z
    )

    activation = calculate_opto_activation(
        intensity,
        half_activation_intensity,
        hill_coefficient
    )

    return {
        "X": X,
        "Z": Z,
        "intensity": intensity,
        "activation": activation
    }