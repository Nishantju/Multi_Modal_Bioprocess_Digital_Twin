from .integrated_bioreactor import (
    simulate_integrated_bioreactor
)

from .sensor_model import (
    simulate_sensor_measurements
)

from .ekf_state_estimation import (
    estimate_integrated_bioreactor
)

from visualization.state_estimation_visualization import (
    plot_state_estimation
)


# STEP 1: Simulate integrated bioreactor

simulation = simulate_integrated_bioreactor(

    population_size=100,

    duration=20,

    surface_intensity=100,

    kla=50,

    seed=42

)


# STEP 2: Generate sensor measurements

sensor_data = simulate_sensor_measurements(

    simulation,

    seed=42

)


# STEP 3: Run the Extended Kalman Filter

ekf_result = estimate_integrated_bioreactor(

    simulation,

    sensor_data

)


# Adapt output to the existing visualization format

estimates = {

    "time": simulation["time"],

    "biomass": ekf_result["estimates"][:, 0],

    "substrate": ekf_result["estimates"][:, 1],

    "oxygen": ekf_result["estimates"][:, 2],

    "product": ekf_result["estimates"][:, 3]

}


# STEP 4: Print comparison

true_states = {

    "biomass": simulation["biomass"],

    "substrate": simulation["substrate"],

    "oxygen": simulation["dissolved_oxygen"],

    "product": simulation["product"]

}

sensor_states = {

    "biomass": sensor_data["measured_biomass"],

    "substrate": sensor_data["measured_substrate"],

    "oxygen": sensor_data["measured_oxygen"],

    "product": sensor_data["measured_product"]

}


print("EXTENDED KALMAN FILTER STATE ESTIMATION")

print("=" * 65)


for name in ["biomass", "substrate", "oxygen", "product"]:

    print("\n" + name.upper())

    print("True state:", true_states[name][-1])

    print("Sensor measurement:", sensor_states[name][-1])

    print("EKF estimate:", estimates[name][-1])


# STEP 5: Calculate estimation errors

print("\nESTIMATION PERFORMANCE")

print("=" * 65)

for name in ["biomass", "substrate", "oxygen", "product"]:

    rmse = (
        (estimates[name] - true_states[name]) ** 2
    ).mean() ** 0.5

    print(name.upper(), "RMSE:", rmse)


# STEP 6: Visualize

plot_state_estimation(

    sensor_data,

    estimates

)