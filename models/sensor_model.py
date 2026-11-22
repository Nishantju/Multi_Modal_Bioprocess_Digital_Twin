import numpy as np


def add_sensor_noise(
    true_values,
    relative_noise,
    rng,
    minimum_scale=1e-12
):
    """
    Add illustrative Gaussian measurement noise
    to a simulated process variable.
    """

    true_values = np.asarray(
        true_values,
        dtype=float
    )

    signal_scale = max(
        np.max(np.abs(true_values)),
        minimum_scale
    )

    noise_std = relative_noise * signal_scale

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=true_values.shape
    )

    measured_values = true_values + noise

    return np.maximum(measured_values, 0.0)


def simulate_sensor_measurements(
    simulation_result,
    seed=42
):
    """
    Generate noisy measurements from the
    integrated bioreactor simulation.
    """

    rng = np.random.default_rng(seed)

    # True process states

    biomass = simulation_result["biomass"]

    oxygen = simulation_result["dissolved_oxygen"]

    substrate = simulation_result["substrate"]

    product = simulation_result["product"]

    # Illustrative relative measurement noise

    measured_biomass = add_sensor_noise(
        biomass,
        relative_noise=0.02,
        rng=rng
    )

    measured_oxygen = add_sensor_noise(
        oxygen,
        relative_noise=0.03,
        rng=rng
    )

    measured_substrate = add_sensor_noise(
        substrate,
        relative_noise=0.025,
        rng=rng
    )

    measured_product = add_sensor_noise(
        product,
        relative_noise=0.04,
        rng=rng
    )

    return {

        "time": simulation_result["time"],

        "true_biomass": biomass,
        "measured_biomass": measured_biomass,

        "true_oxygen": oxygen,
        "measured_oxygen": measured_oxygen,

        "true_substrate": substrate,
        "measured_substrate": measured_substrate,

        "true_product": product,
        "measured_product": measured_product

    }