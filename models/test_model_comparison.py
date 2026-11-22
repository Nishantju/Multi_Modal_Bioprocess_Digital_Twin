import numpy as np

from models.integrated_bioreactor import (
    simulate_integrated_bioreactor
)

from control.reduced_order_model import (
    make_population_phases,
    predict_reduced_order
)


# --------------------------------------------------
# 1. Run the original full population model
# --------------------------------------------------

full = simulate_integrated_bioreactor(
    population_size=100,
    duration=20,
    surface_intensity=100,
    kla=100,
    seed=42
)


# --------------------------------------------------
# 2. Parameters for reduced-order model
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
    "qO2": 0.2
}


# --------------------------------------------------
# 3. Initial conditions
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
# 4. Generate population phases
# --------------------------------------------------

phases = make_population_phases(
    population_size=100,
    seed=42
)


# --------------------------------------------------
# 5. Run reduced-order model
# --------------------------------------------------

reduced = predict_reduced_order(

    initial_state=initial_state,

    control=(100, 100),

    params=params,

    phases=phases,

    start_time=0,

    horizon=20,

    n_points=100
)


# --------------------------------------------------
# 6. Extract final results
# --------------------------------------------------

full_biomass = full["biomass"][-1]
full_substrate = full["substrate"][-1]
full_oxygen = full["dissolved_oxygen"][-1]
full_product = full["product"][-1]

full_mRNA = np.mean(full["mRNA"][:, -1])
full_enzyme = np.mean(full["enzyme"][:, -1])

reduced_biomass = reduced["final_state"][0]
reduced_substrate = reduced["final_state"][1]
reduced_oxygen = reduced["final_state"][2]
reduced_product = reduced["final_state"][3]
reduced_mRNA = reduced["final_state"][4]
reduced_enzyme = reduced["final_state"][5]


# --------------------------------------------------
# 7. Display comparison
# --------------------------------------------------

print("\nFULL MODEL VS REDUCED-ORDER MODEL")
print("---------------------------------------------")

print(
    f"{'Parameter':<20}"
    f"{'Full Model':>15}"
    f"{'Reduced Model':>18}"
)

print("-" * 53)

results = [
    ("Biomass", full_biomass, reduced_biomass),
    ("Substrate", full_substrate, reduced_substrate),
    ("Dissolved oxygen", full_oxygen, reduced_oxygen),
    ("Product", full_product, reduced_product),
    ("Average mRNA", full_mRNA, reduced_mRNA),
    ("Average enzyme", full_enzyme, reduced_enzyme)
]

for name, full_value, reduced_value in results:

    print(
        f"{name:<20}"
        f"{full_value:>15.6f}"
        f"{reduced_value:>18.6f}"
    )

print("\nComparison completed.")