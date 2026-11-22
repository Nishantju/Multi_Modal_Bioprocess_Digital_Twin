import numpy as np

from control.reduced_order_model import (
    make_population_phases
)

from control.mpc_gene_expression import (
    run_gene_expression_mpc
)


# --------------------------------------------------
# Model parameters
# --------------------------------------------------

params = {

    "height": 2.0,
    "mean_depth": 1.0,
    "movement_amplitude": 0.9,
    "movement_frequency": 0.2,

    "attenuation_coefficient": 2.0,
    "half_activation_intensity": 30.0,
    "hill_coefficient": 2.0,
    "basal_activation": 0.1,

    "alpha_m": 2.0,
    "delta_m": 0.2,
    "alpha_e": 0.5,
    "delta_e": 0.1,

    "qP_max": 0.05,
    "K_E": 2.0,
    "K_SP": 1.0,
    "K_OP": 0.001,
    "Y_PS": 0.8,

    "mu_max": 0.5,
    "K_S": 1.0,
    "K_DO": 0.001,
    "Y_XS": 0.5,

    "oxygen_saturation": 0.008,
    "qO2": 0.2,

    # Control limits

    "max_light": 100.0,

    "min_kla": 30.0,

    "max_kla": 150.0,

    # Oxygen target

    "oxygen_target": 0.002,

    # Optimization weights

    "weight_light": 0.02,

    "weight_kla": 0.02,

    "weight_oxygen": 0.05,

    "weight_control_change": 0.02

}


# --------------------------------------------------
# Initial process state
# --------------------------------------------------

initial_state = np.array([

    0.1,

    20.0,

    0.007,

    0.0,

    0.2,

    0.1

])


# --------------------------------------------------
# Population
# --------------------------------------------------

phases = make_population_phases(

    population_size=100,

    seed=42

)


# --------------------------------------------------
# Run gene-expression-aware MPC
# --------------------------------------------------

result = run_gene_expression_mpc(

    initial_state=initial_state,

    params=params,

    phases=phases,

    duration=20,

    control_interval=0.5,

    prediction_horizon=6

)


# --------------------------------------------------
# Display final results
# --------------------------------------------------

print("\nGENE-EXPRESSION-AWARE MPC RESULTS")

print("--------------------------------------")

print(

    "Final biomass:",

    result["biomass"][-1],

    "g/L"

)

print(

    "Final substrate:",

    result["substrate"][-1],

    "g/L"

)

print(

    "Final dissolved oxygen:",

    result["dissolved_oxygen"][-1],

    "g/L"

)

print(

    "Final product:",

    result["product"][-1],

    "g/L"

)

print(

    "Final average mRNA:",

    result["mRNA"][-1]

)

print(

    "Final average enzyme:",

    result["enzyme"][-1]

)

print("\nCONTROL PERFORMANCE")

print("--------------------------------------")

print(

    "Average light intensity:",

    np.mean(result["light"])

)

print(

    "Average kLa:",

    np.mean(result["kla"]),

    "h^-1"

)

print(

    "Number of control updates:",

    len(result["light"])

)

print("\nSimulation completed successfully.")