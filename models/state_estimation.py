import numpy as np


def dynamic_kalman_filter(
    measurements,
    dt,
    measurement_variance,
    acceleration_variance,
    initial_estimate=None
):

    measurements = np.asarray(
        measurements,
        dtype=float
    )

    if measurements.ndim != 1 or len(measurements) == 0:
        raise ValueError("Measurements must be a non-empty 1D array.")

    if dt <= 0:
        raise ValueError("Time step must be positive.")

    if measurement_variance <= 0:
        raise ValueError("Measurement variance must be positive.")

    if acceleration_variance < 0:
        raise ValueError("Acceleration variance cannot be negative.")

    # State: [concentration, rate of change]

    estimate = np.array([
        measurements[0] if initial_estimate is None
        else float(initial_estimate),
        0.0
    ])

    P = np.diag([measurement_variance, 1.0])

    F = np.array([
        [1.0, dt],
        [0.0, 1.0]
    ])

    H = np.array([[1.0, 0.0]])

    Q = acceleration_variance * np.array([
        [dt**4 / 4, dt**3 / 2],
        [dt**3 / 2, dt**2]
    ])

    R = measurement_variance

    estimates = np.zeros(len(measurements))
    rates = np.zeros(len(measurements))

    uncertainties = np.zeros(len(measurements))

    for k, measurement in enumerate(measurements):

        # STEP 1: Predict concentration and rate

        predicted_state = F @ estimate

        predicted_P = (
            F @ P @ F.T + Q
        )

        # STEP 2: Calculate Kalman gain

        innovation = (
            measurement - (H @ predicted_state)[0]
        )

        innovation_variance = (
            H @ predicted_P @ H.T
        )[0, 0] + R

        K = (
            predicted_P @ H.T
        ) / innovation_variance

        # STEP 3: Correct the state

        estimate = (
            predicted_state + K.flatten() * innovation
        )

        P = (
            np.eye(2) - K @ H
        ) @ predicted_P

        # Store results

        estimates[k] = max(estimate[0], 0.0)

        rates[k] = estimate[1]

        uncertainties[k] = max(P[0, 0], 0.0)

    return {
        "estimate": estimates,
        "rate": rates,
        "uncertainty": uncertainties
    }


def estimate_bioreactor_states(sensor_data):

    estimated_states = {}

    time = sensor_data["time"]

    dt = float(np.mean(np.diff(time)))

    variables = {

        "biomass": ("measured_biomass", 0.02),

        "oxygen": ("measured_oxygen", 0.03),

        "substrate": ("measured_substrate", 0.025),

        "product": ("measured_product", 0.04)

    }

    for name, (measurement_key, noise_fraction) in variables.items():

        measurements = np.asarray(
            sensor_data[measurement_key],
            dtype=float
        )

        scale = max(
            np.max(np.abs(measurements)),
            1e-12
        )

        # Normalize measurements for numerical stability

        normalized = measurements / scale

        R = noise_fraction**2

        # Initial tuning value for the dynamic filter

        Q_acceleration = 1.0

        result = dynamic_kalman_filter(

            normalized,

            dt=dt,

            measurement_variance=R,

            acceleration_variance=Q_acceleration

        )

        estimated_states[name] = (
            result["estimate"] * scale
        )

        estimated_states[name + "_rate"] = (
            result["rate"] * scale
        )

        estimated_states[name + "_uncertainty"] = (
            result["uncertainty"] * scale**2
        )

    estimated_states["time"] = time

    return estimated_states