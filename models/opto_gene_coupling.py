import numpy as np
from scipy.integrate import solve_ivp

from .optogenetics import (
    calculate_light_intensity,
    calculate_opto_activation
)


def simulate_opto_gene_expression(
    initial_mRNA,
    initial_protein,
    duration,
    surface_intensity,
    attenuation_coefficient,
    depth,
    half_activation_intensity,
    hill_coefficient,
    basal_activation,
    alpha_m,
    delta_m,
    alpha_e,
    delta_e
):

    light_intensity = calculate_light_intensity(
        surface_intensity,
        attenuation_coefficient,
        depth
    )

    field_activation = calculate_opto_activation(
        light_intensity,
        half_activation_intensity,
        hill_coefficient
    )

    basal_activation = np.clip(
        basal_activation, 0, 1
    )

    activation = (
        basal_activation
        + (1 - basal_activation) * field_activation
    )

    def model(time, state):

        mRNA, protein = state

        dmRNA_dt = (
            alpha_m * activation
            - delta_m * mRNA
        )

        dProtein_dt = (
            alpha_e * mRNA
            - delta_e * protein
        )

        return [dmRNA_dt, dProtein_dt]

    time_points = np.linspace(0, duration, 300)

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

    return {
        "time": solution.t,
        "mRNA": solution.y[0],
        "protein": solution.y[1],
        "light_intensity": light_intensity,
        "field_activation": field_activation,
        "gene_activation": activation
    }