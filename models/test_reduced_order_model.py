import numpy as np

from control.reduced_order_model import (
    make_population_phases,
    predict_reduced_order
)


# --------------------------------------------------
# Model parameters
# --------------------------------------------------

params = {

    # Reactor geometry and cell movement

    "height": 2.0,
    "mean_depth": 1.0,
    "movement_amplitude": 0.9,
    "movement_frequency": 0.2,

    # Optogenetics

    "attenuation_coefficient": 2.0,
    "half_activation_intensity": 30.0,
    "hill_coefficient": 2.0,
    "basal_activation": 0.1,

    # Gene expression

    "alpha_m": 2.0,
    "delta_m": 0.2,
    "alpha_e": 0.5,
    "delta_e": 0.1,

    # Product formation

    "qP_max": 0.05,
    "K_E": 2.0,
    "K_SP": 1.0,
    "K_OP": 0.001,
    "Y_PS": 0.8,

    # Biomass growth

    "mu_max": 0.5,
    "K_S": 1.0,
    "K_DO": 0.001,
    "Y_XS": 0.5,

    # Oxygen transfer

    "oxygen_saturation": 0.008,
    "qO2": 0.2
}


# --------------------------------------------------
# Initial state
# --------------------------------------------------

initial_state = np.array([

    0.1,    # Biomass
    20.0,   # Substrate
    0.007,  # Dissolved oxygen
    0.0,    # Product
    0.2,    # Average mRNA
    0.1     # Average enzyme

])


# --------------------------------------------------
# MPC control inputs
# --------------------------------------------------

light_intensity = 100
kla = 100

control = (
    light_intensity,
    kla
)


# --------------------------------------------------
# Population
# --------------------------------------------------

phases = make_population_phases(
    population_size=100,
    seed=42
)


# --------------------------------------------------
# Run prediction
# --------------------------------------------------

result = predict_reduced_order(

    initial_state=initial_state,

    control=control,

    params=params,

    phases=phases,

    start_time=0,

    horizon=20,

    n_points=100

)


# --------------------------------------------------
# Display results
# --------------------------------------------------

trajectory = result["trajectory"]

print("\nREDUCED-ORDER BIOPROCESS MODEL")
print("--------------------------------")

print(
    "Final biomass:",
    trajectory[0, -1],
    "g/L"
)

print(
    "Final substrate:",
    trajectory[1, -1],
    "g/L"
)

print(
    "Final dissolved oxygen:",
    trajectory[2, -1],
    "g/L"
)

print(
    "Final product:",
    trajectory[3, -1],
    "g/L"
)

print(
    "Final average mRNA:",
    trajectory[4, -1]
)

print(
    "Final average enzyme:",
    trajectory[5, -1]
)

print(
    "Average gene activation:",
    np.mean(result["mean_activation"])
)

print("\nSimulation completed successfully.")