import numpy as np
from scipy.integrate import solve_ivp


def steady_state_magnetic_activation(
    magnetic_field,
    half_activation_field,
    hill_coefficient
):
    """
    Calculate steady-state magnetic-field activation.
    """

    if magnetic_field < 0:
        raise ValueError("Magnetic field cannot be negative.")

    if half_activation_field <= 0:
        raise ValueError("Half-activation field must be positive.")

    if hill_coefficient <= 0:
        raise ValueError("Hill coefficient must be positive.")

    numerator = magnetic_field ** hill_coefficient

    denominator = (
        half_activation_field ** hill_coefficient
        + numerator
    )

    return numerator / denominator


def magnetogenetic_rhs(
    time,
    state,
    magnetic_field,
    half_activation_field,
    hill_coefficient,
    response_time
):
    """
    Dynamic magnetogenetic activation model.
    """

    activation = state[0]

    target_activation = steady_state_magnetic_activation(
        magnetic_field,
        half_activation_field,
        hill_coefficient
    )

    d_activation_dt = (
        target_activation - activation
    ) / response_time

    return [d_activation_dt]


def simulate_magnetogenetics(
    time,
    magnetic_field,
    half_activation_field=5.0,
    hill_coefficient=2.0,
    response_time=0.5,
    initial_activation=0.0
):
    """
    Simulate time-dependent magnetogenetic activation.
    """

    time = np.asarray(time, dtype=float)
    magnetic_field = np.asarray(magnetic_field, dtype=float)

    if time.ndim != 1 or len(time) < 2:
        raise ValueError("Time must contain at least two points.")

    if magnetic_field.ndim != 1 or len(magnetic_field) != len(time):
        raise ValueError(
            "Magnetic field and time arrays must have equal lengths."
        )

    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(magnetic_field)):
        raise ValueError("Inputs must contain finite values.")

    if np.any(np.diff(time) <= 0):
        raise ValueError("Time values must be strictly increasing.")

    if np.any(magnetic_field < 0):
        raise ValueError("Magnetic field cannot be negative.")

    if half_activation_field <= 0 or hill_coefficient <= 0:
        raise ValueError("Hill parameters must be positive.")

    if response_time <= 0:
        raise ValueError("Response time must be positive.")

    if not 0 <= initial_activation <= 1:
        raise ValueError("Initial activation must be between 0 and 1.")

    def rhs(t, state):

        current_field = np.interp(
            t,
            time,
            magnetic_field
        )

        return magnetogenetic_rhs(
            t,
            state,
            current_field,
            half_activation_field,
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
        steady_state_magnetic_activation(
            field,
            half_activation_field,
            hill_coefficient
        )
        for field in magnetic_field
    ])

    return {
        "time": time,
        "magnetic_field": magnetic_field,
        "activation": activation,
        "steady_state_activation": steady_state
    }