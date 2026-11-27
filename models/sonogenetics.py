import numpy as np
from scipy.integrate import solve_ivp


def steady_state_sonogenetic_activation(
    ultrasound_intensity,
    half_activation_intensity,
    hill_coefficient
):
    """
    Calculate steady-state activation using a Hill equation.
    """

    if ultrasound_intensity < 0:
        raise ValueError("Ultrasound intensity cannot be negative.")

    if half_activation_intensity <= 0:
        raise ValueError("Half-activation intensity must be positive.")

    if hill_coefficient <= 0:
        raise ValueError("Hill coefficient must be positive.")

    numerator = ultrasound_intensity ** hill_coefficient

    denominator = (
        half_activation_intensity ** hill_coefficient
        + numerator
    )

    return numerator / denominator


def sonogenetic_rhs(
    time,
    state,
    ultrasound_intensity,
    half_activation_intensity,
    hill_coefficient,
    response_time
):
    """
    Dynamic sonogenetic activation model.

    dA/dt = (A_ss - A)/tau
    """

    activation = state[0]

    target_activation = steady_state_sonogenetic_activation(
        ultrasound_intensity,
        half_activation_intensity,
        hill_coefficient
    )

    d_activation_dt = (
        target_activation - activation
    ) / response_time

    return [d_activation_dt]


def simulate_sonogenetics(
    time,
    ultrasound_intensity,
    half_activation_intensity=5.0,
    hill_coefficient=2.0,
    response_time=0.5,
    initial_activation=0.0
):
    """
    Simulate time-dependent ultrasound activation.
    """

    time = np.asarray(time, dtype=float)
    ultrasound_intensity = np.asarray(
        ultrasound_intensity,
        dtype=float
    )

    if time.ndim != 1 or len(time) < 2:
        raise ValueError("Time must contain at least two points.")

    if ultrasound_intensity.ndim != 1:
        raise ValueError("Ultrasound intensity must be a 1D array.")

    if len(ultrasound_intensity) != len(time):
        raise ValueError(
            "Ultrasound intensity and time arrays must have equal lengths."
        )

    if not np.all(np.isfinite(time)):
        raise ValueError("Time must contain finite values.")

    if not np.all(np.isfinite(ultrasound_intensity)):
        raise ValueError("Ultrasound intensity must contain finite values.")

    if np.any(np.diff(time) <= 0):
        raise ValueError("Time values must be strictly increasing.")

    if np.any(ultrasound_intensity < 0):
        raise ValueError("Ultrasound intensity cannot be negative.")

    if half_activation_intensity <= 0:
        raise ValueError("Half-activation intensity must be positive.")

    if hill_coefficient <= 0 or response_time <= 0:
        raise ValueError("Hill coefficient and response time must be positive.")

    if not 0 <= initial_activation <= 1:
        raise ValueError("Initial activation must be between 0 and 1.")

    def rhs(t, state):

        current_intensity = np.interp(
            t,
            time,
            ultrasound_intensity
        )

        return sonogenetic_rhs(
            t,
            state,
            current_intensity,
            half_activation_intensity,
            hill_coefficient,
            response_time
        )

    solution = solve_ivp(
        rhs,
        (time[0], time[-1]),
        [initial_activation],
        t_eval=time,
        method="RK45",
        rtol=1e-7,
        atol=1e-9
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    activation = np.clip(solution.y[0], 0.0, 1.0)

    steady_state = np.array([
        steady_state_sonogenetic_activation(
            intensity,
            half_activation_intensity,
            hill_coefficient
        )
        for intensity in ultrasound_intensity
    ])

    return {
        "time": time,
        "ultrasound_intensity": ultrasound_intensity,
        "activation": activation,
        "steady_state_activation": steady_state
    }