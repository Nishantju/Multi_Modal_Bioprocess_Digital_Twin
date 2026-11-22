import numpy as np
from scipy.integrate import solve_ivp

from .bioreactor import estimate_kla


def oxygen_derivative(
    time,
    state,
    kla,
    oxygen_saturation,
    specific_oxygen_uptake,
    biomass
):

    dissolved_oxygen = state[0]

    otr = kla * (
        oxygen_saturation - dissolved_oxygen
    )

    our = specific_oxygen_uptake * biomass

    dDO_dt = otr - our

    return [dDO_dt]


def simulate_dynamic_oxygen(
    initial_oxygen,
    oxygen_saturation,
    kla,
    specific_oxygen_uptake,
    biomass,
    duration
):

    time_points = np.linspace(
        0,
        duration,
        200
    )

    solution = solve_ivp(
        oxygen_derivative,
        [0, duration],
        [initial_oxygen],
        t_eval=time_points,
        args=(
            kla,
            oxygen_saturation,
            specific_oxygen_uptake,
            biomass
        )
    )

    return {
        "time": solution.t,
        "dissolved_oxygen": solution.y[0]
    }