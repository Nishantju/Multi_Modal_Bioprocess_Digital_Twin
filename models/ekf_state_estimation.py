import numpy as np


def calculate_reactor_derivatives(state, qP, params):

    X, S, DO, P = state

    # Prevent negative concentrations in rate calculations
    X = max(X, 0.0)
    S = max(S, 0.0)
    DO = max(DO, 0.0)

    # Biomass growth
    substrate_factor = S / (params["K_S"] + S)

    oxygen_factor = DO / (params["K_DO"] + DO)

    mu = (
        params["mu_max"]
        * substrate_factor
        * oxygen_factor
    )

    dX_dt = mu * X

    # Product formation
    dP_dt = X * max(qP, 0.0)

    # Substrate consumption
    substrate_growth = (
        mu * X / params["Y_XS"]
    )

    substrate_product = (
        dP_dt / params["Y_PS"]
    )

    dS_dt = -(
        substrate_growth + substrate_product
    )

    # Oxygen transfer and uptake
    OTR = params["kla"] * (
        params["oxygen_saturation"] - DO
    )

    OUR = (
        params["qO2"]
        * X
        * oxygen_factor
    )

    dDO_dt = OTR - OUR

    return np.array([
        dX_dt,
        dS_dt,
        dDO_dt,
        dP_dt
    ])


def rk4_prediction(state, dt, qP, params):

    """
    Fourth-order Runge-Kutta integration.
    """

    k1 = calculate_reactor_derivatives(
        state, qP, params
    )

    k2 = calculate_reactor_derivatives(
        state + 0.5 * dt * k1,
        qP,
        params
    )

    k3 = calculate_reactor_derivatives(
        state + 0.5 * dt * k2,
        qP,
        params
    )

    k4 = calculate_reactor_derivatives(
        state + dt * k3,
        qP,
        params
    )

    predicted = state + (
        dt / 6.0
    ) * (
        k1 + 2*k2 + 2*k3 + k4
    )

    return np.maximum(predicted, 0.0)


def calculate_jacobian(state, dt, qP, params):

    """
    Numerically calculate the process-model Jacobian.
    """

    n = len(state)

    A = np.zeros((n, n))

    epsilon = 1e-6

    for j in range(n):

        perturbation = np.zeros(n)

        perturbation[j] = epsilon * max(
            abs(state[j]), 1.0
        )

        upper = rk4_prediction(
            state + perturbation,
            dt,
            qP,
            params
        )

        lower = rk4_prediction(
            np.maximum(state - perturbation, 0.0),
            dt,
            qP,
            params
        )

        denominator = (
            (state[j] + perturbation[j])
            - max(state[j] - perturbation[j], 0.0)
        )

        if denominator > 0:

            A[:, j] = (
                upper - lower
            ) / denominator

    return A


def extended_kalman_filter(
    measurements,
    time,
    qP_profile,
    initial_state,
    params,
    measurement_variance,
    process_variance
):

    measurements = np.asarray(
        measurements, dtype=float
    )

    time = np.asarray(time, dtype=float)

    qP_profile = np.asarray(
        qP_profile, dtype=float
    )

    if measurements.ndim != 2 or measurements.shape[1] != 4:
        raise ValueError(
            "Measurements must have four columns: X, S, DO, P."
        )

    if len(time) != len(measurements):
        raise ValueError("Time and measurements must have equal lengths.")

    if len(qP_profile) != len(time):
        raise ValueError("qP profile and time must have equal lengths.")

    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("Time must contain increasing values.")

    if np.any(measurement_variance <= 0):
        raise ValueError("Measurement variances must be positive.")

    if np.any(process_variance < 0):
        raise ValueError("Process variances cannot be negative.")

    n = 4

    H = np.eye(n)

    R = np.diag(measurement_variance)

    Q = np.diag(process_variance)

    estimate = np.maximum(
        np.asarray(initial_state, dtype=float),
        0.0
    )

    P_cov = np.diag(
        np.maximum(measurement_variance, 1e-12)
    )

    estimates = np.zeros_like(measurements)

    uncertainties = np.zeros((len(time), n))

    for k in range(len(time)):

        if k > 0:

            dt = time[k] - time[k-1]

            qP = 0.5 * (
                qP_profile[k] + qP_profile[k-1]
            )

            # STEP 1: Nonlinear process prediction

            predicted_state = rk4_prediction(
                estimate,
                dt,
                qP,
                params
            )

            # STEP 2: Calculate Jacobian

            A = calculate_jacobian(
                estimate,
                dt,
                qP,
                params
            )

            # STEP 3: Predict uncertainty

            predicted_covariance = (
                A @ P_cov @ A.T + Q
            )

        else:

            predicted_state = estimate.copy()

            predicted_covariance = P_cov.copy()

        # STEP 4: Sensor correction

        measurement = measurements[k]

        innovation = (
            measurement - H @ predicted_state
        )

        innovation_covariance = (
            H @ predicted_covariance @ H.T + R
        )

        K = (
            predicted_covariance @ H.T
            @ np.linalg.inv(innovation_covariance)
        )

        estimate = (
            predicted_state + K @ innovation
        )

        estimate = np.maximum(estimate, 0.0)

        # Joseph-form covariance update
        identity = np.eye(n)

        P_cov = (
            (identity - K @ H)
            @ predicted_covariance
            @ (identity - K @ H).T
            + K @ R @ K.T
        )

        estimates[k] = estimate

        uncertainties[k] = np.diag(P_cov)

    return {
        "estimates": estimates,
        "uncertainties": uncertainties
    }


def estimate_integrated_bioreactor(
    simulation,
    sensor_data
):

    time = simulation["time"]

    measurements = np.column_stack([

        sensor_data["measured_biomass"],

        sensor_data["measured_substrate"],

        sensor_data["measured_oxygen"],

        sensor_data["measured_product"]

    ])

    # Use the mechanistic gene-expression model's
    # predicted metabolic rate as a known model input.

    qP_profile = simulation[
        "mean_specific_product_rate"
    ]

    params = {

        "mu_max": 0.5,

        "K_S": 1.0,

        "K_DO": 0.001,

        "Y_XS": 0.5,

        "Y_PS": 0.8,

        "kla": 50,

        "oxygen_saturation": 0.008,

        "qO2": 0.2

    }

    # Illustrative measurement variances.
    # These should eventually be calibrated against sensor data.

    scales = np.array([

        max(np.max(sensor_data["measured_biomass"]), 1e-12),

        max(np.max(sensor_data["measured_substrate"]), 1e-12),

        max(np.max(sensor_data["measured_oxygen"]), 1e-12),

        max(np.max(sensor_data["measured_product"]), 1e-12)

    ])

    noise_fractions = np.array([
        0.02,
        0.025,
        0.03,
        0.04
    ])

    measurement_variance = (
        noise_fractions * scales
    ) ** 2

    # Initial state from the first sensor measurement

    initial_state = measurements[0].copy()

    # Illustrative process uncertainty

    process_variance = np.array([
        1e-5,
        1e-4,
        1e-8,
        1e-6
    ])

    result = extended_kalman_filter(

        measurements=measurements,

        time=time,

        qP_profile=qP_profile,

        initial_state=initial_state,

        params=params,

        measurement_variance=measurement_variance,

        process_variance=process_variance

    )

    return result