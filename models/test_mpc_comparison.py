import numpy as np
from visualization.mpc_analysis import plot_mpc_analysis


from control.reduced_order_model import (
    make_population_phases,
    predict_reduced_order
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

    "max_light": 100.0,
    "min_kla": 30.0,
    "max_kla": 150.0,

    "oxygen_target": 0.002,

    "weight_light": 0.02,
    "weight_kla": 0.02,
    "weight_oxygen": 0.05,
    "weight_control_change": 0.02

}


# --------------------------------------------------
# Initial conditions
# --------------------------------------------------

initial_state = np.array([
    0.1,
    20.0,
    0.007,
    0.0,
    0.2,
    0.1
])


phases = make_population_phases(
    population_size=100,
    seed=42
)


# --------------------------------------------------
# EXPERIMENT A
# Fixed baseline conditions
# Light = 100, kLa = 100
# --------------------------------------------------

fixed_baseline = predict_reduced_order(

    initial_state=initial_state,

    control=(100, 100),

    params=params,

    phases=phases,

    start_time=0,

    horizon=20,

    n_points=100

)


# --------------------------------------------------
# EXPERIMENT B
# Fixed MPC-average conditions
# Light = 39.7630, kLa = 102.8419
# --------------------------------------------------

fixed_average = predict_reduced_order(

    initial_state=initial_state,

    control=(39.76304466395525, 102.84189368407942),

    params=params,

    phases=phases,

    start_time=0,

    horizon=20,

    n_points=100

)


# --------------------------------------------------
# EXPERIMENT C
# Dynamic MPC control
# --------------------------------------------------

mpc = run_gene_expression_mpc(

    initial_state=initial_state,

    params=params,

    phases=phases,

    duration=20,

    control_interval=0.5,

    prediction_horizon=6

)


# --------------------------------------------------
# Extract final states
# --------------------------------------------------

baseline_state = fixed_baseline["final_state"]

average_state = fixed_average["final_state"]


# --------------------------------------------------
# Three-way comparison
# --------------------------------------------------

print("\nTHREE-WAY CONTROL STRATEGY COMPARISON")

print("=" * 85)

print(
    f"{'Parameter':<25}"
    f"{'Fixed Baseline':>20}"
    f"{'Fixed MPC Avg':>20}"
    f"{'Dynamic MPC':>20}"
)

print("-" * 85)


comparison = [

    (
        "Final biomass (g/L)",
        baseline_state[0],
        average_state[0],
        mpc["biomass"][-1]
    ),

    (
        "Final substrate (g/L)",
        baseline_state[1],
        average_state[1],
        mpc["substrate"][-1]
    ),

    (
        "Final oxygen (g/L)",
        baseline_state[2],
        average_state[2],
        mpc["dissolved_oxygen"][-1]
    ),

    (
        "Final product (g/L)",
        baseline_state[3],
        average_state[3],
        mpc["product"][-1]
    ),

    (
        "Final mRNA",
        baseline_state[4],
        average_state[4],
        mpc["mRNA"][-1]
    ),

    (
        "Final enzyme",
        baseline_state[5],
        average_state[5],
        mpc["enzyme"][-1]
    )

]


for name, a, b, c in comparison:

    print(
        f"{name:<25}"
        f"{a:>20.6f}"
        f"{b:>20.6f}"
        f"{c:>20.6f}"
    )


# --------------------------------------------------
# Control input comparison
# --------------------------------------------------

print("\nCONTROL INPUT COMPARISON")

print("=" * 85)

print(
    f"{'Control':<25}"
    f"{'Fixed Baseline':>20}"
    f"{'Fixed MPC Avg':>20}"
    f"{'Dynamic MPC':>20}"
)

print("-" * 85)


print(
    f"{'Average light':<25}"
    f"{100:>20.4f}"
    f"{63.6254:>20.4f}"
    f"{np.mean(mpc['light']):>20.4f}"
)

print(
    f"{'Average kLa':<25}"
    f"{100:>20.4f}"
    f"{136.4246:>20.4f}"
    f"{np.mean(mpc['kla']):>20.4f}"
)


print("\nAll three experiments completed successfully.")

plot_mpc_analysis(mpc)