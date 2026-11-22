import numpy as np
from scipy.integrate import solve_ivp

from .optogenetics import (
    calculate_light_intensity,
    calculate_opto_activation
)


def simulate_moving_microorganism(
    duration,
    mean_depth,
    movement_amplitude,
    movement_frequency,
    surface_intensity,
    attenuation_coefficient,
    half_activation_intensity,
    hill_coefficient,
    basal_activation,
    initial_mRNA,
    initial_protein,
    alpha_m,
    delta_m,
    alpha_e,
    delta_e
):

    def calculate_depth(time):

        depth = (
            mean_depth
            + movement_amplitude
            * np.sin(2 * np.pi * movement_frequency * time)
        )

        return depth

    def calculate_activation(time):

        depth = calculate_depth(time)

        intensity = calculate_light_intensity(
            surface_intensity,
            attenuation_coefficient,
            depth
        )

        field_signal = calculate_opto_activation(
            intensity,
            half_activation_intensity,
            hill_coefficient
        )

        activation = (
            basal_activation
            + (1 - basal_activation) * field_signal
        )

        return activation

    def model(time, state):

        mRNA, protein = state

        activation = calculate_activation(time)

        dmRNA_dt = (
            alpha_m * activation
            - delta_m * mRNA
        )

        dProtein_dt = (
            alpha_e * mRNA
            - delta_e * protein
        )

        return [dmRNA_dt, dProtein_dt]

    time_points = np.linspace(0, duration, 500)

    solution = solve_ivp(
        model,
        [0, duration],
        [initial_mRNA, initial_protein],
        t_eval=time_points,
        method="RK45",
        rtol=1e-7,
        atol=1e-9
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    depths = np.array([
        calculate_depth(t) for t in solution.t
    ])

    intensities = calculate_light_intensity(
        surface_intensity,
        attenuation_coefficient,
        depths
    )

    activations = np.array([
        calculate_activation(t) for t in solution.t
    ])

    return {
        "time": solution.t,
        "depth": depths,
        "light_intensity": intensities,
        "activation": activations,
        "mRNA": solution.y[0],
        "protein": solution.y[1]
    }