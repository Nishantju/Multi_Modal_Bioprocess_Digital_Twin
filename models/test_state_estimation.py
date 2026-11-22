from .integrated_bioreactor import (
    simulate_integrated_bioreactor
)

from .sensor_model import (
    simulate_sensor_measurements
)

from .state_estimation import (
    estimate_bioreactor_states
)

from visualization.state_estimation_visualization import (
    plot_state_estimation
)


# STEP 1: Simulate the reactor

simulation = simulate_integrated_bioreactor(

    population_size=100,

    duration=20,

    surface_intensity=100,

    kla=50,

    seed=42

)


# STEP 2: Generate noisy sensor measurements

sensor_data = simulate_sensor_measurements(

    simulation,

    seed=42

)


# STEP 3: Estimate the process states

estimates = estimate_bioreactor_states(
    sensor_data
)


# STEP 4: Display final comparisons

print("BIOREACTOR STATE ESTIMATION")

print("-" * 60)

for name in ["biomass", "oxygen", "substrate", "product"]:

    true_key = {
        "biomass": "biomass",
        "oxygen": "dissolved_oxygen",
        "substrate": "substrate",
        "product": "product"
    }[name]

    true_value = simulation[true_key][-1]

    measured_value = sensor_data[
        "measured_" + name
    ][-1]

    estimated_value = estimates[name][-1]

    print(f"\n{name.upper()}")

    print("True value:", true_value)

    print("Sensor measurement:", measured_value)

    print("Estimated value:", estimated_value)


# STEP 5: Visualize

plot_state_estimation(
    sensor_data,
    estimates
)