import numpy as np
from scipy.integrate import solve_ivp


def steady_state_electrical_activation(
    electric_field,
    half_activation_field,
    hill_coefficient
):
    """
    Calculate steady-state activation produced by an
    applied electrical field using a Hill equation.
    """

    if electric_field < 0:
        raise ValueError("Electric field cannot be negative.")

    if half_activation_field <= 0:
        raise ValueError("Half-activation field must be positive.")

    if hill_coefficient <= 0:
        raise ValueError("Hill coefficient must be positive.")

    numerator = electric_field ** hill_coefficient

    denominator = (
        half_activation_field ** hill_coefficient
        + numerator
    )

    return numerator / denominator


def electrogenetic_rhs(
    time,
    state,
    electric_field,
    half_activation_field,
    hill_coefficient,
    response_time
):
    """
    Dynamic electrogenetic activation model.

    dA/dt = (A_ss - A)/tau
    """

    activation = state[0]

    target_activation = steady_state_electrical_activation(
        electric_field,
        half_activation_field,
        hill_coefficient
    )

    d_activation_dt = (
        target_activation - activation
    ) / response_time

    return [d_activation_dt]


def simulate_electrogenetics(
    time,
    electric_field,
    half_activation_field=5.0,
    hill_coefficient=2.0,
    response_time=0.5,
    initial_activation=0.0
):
    """
    Simulate time-dependent electrogenetic activation.

    Parameters
    ----------
    time : array
        Simulation time points.

    electric_field : array
        Applied electrical-field strength at each time point.

    half_activation_field : float
        Field strength giving half-maximal activation.

    hill_coefficient : float
        Hill coefficient.

    response_time : float
        Activation response time constant.

    initial_activation : float
        Initial activation level.
    """

    time = np.asarray(time, dtype=float)
    electric_field = np.asarray(electric_field, dtype=float)

    if time.ndim != 1 or len(time) < 2:
        raise ValueError("Time must contain at least two points.")

    if electric_field.ndim != 1 or len(electric_field) != len(time):
        raise ValueError("Electric field and time arrays must have equal lengths.")

    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(electric_field)):
        raise ValueError("Inputs must contain finite values.")

    if np.any(np.diff(time) <= 0):
        raise ValueError("Time values must be strictly increasing.")

    if np.any(electric_field < 0):
        raise ValueError("Electric field cannot be negative.")

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
            electric_field
        )

        return electrogenetic_rhs(
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
        steady_state_electrical_activation(
            field,
            half_activation_field,
            hill_coefficient
        )
        for field in electric_field
    ])

    return {
        "time": time,
        "electric_field": electric_field,
        "activation": activation,
        "steady_state_activation": steady_state
    }


