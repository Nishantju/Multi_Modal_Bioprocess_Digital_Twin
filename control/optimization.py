import numpy as np
from scipy.optimize import minimize

from models.integrated_bioreactor import (
    simulate_integrated_bioreactor
)


def optimize_bioreactor():

    # Operating limits

    light_bounds = (10, 100)

    kla_bounds = (30, 150)

    # Count model evaluations

    evaluation_count = 0

    # Objective function

    def objective(parameters):

        nonlocal evaluation_count

        light_intensity, kla = parameters

        result = simulate_integrated_bioreactor(

            population_size=100,

            duration=20,

            surface_intensity=light_intensity,

            kla=kla,

            seed=42

        )

        product = result["product"][-1]

        evaluation_count += 1

        print(
            f"Evaluation {evaluation_count}: "
            f"Light={light_intensity:.3f}, "
            f"kLa={kla:.3f}, "
            f"Product={product:.6f}"
        )

        # Minimization algorithm:
        # Minimize negative product to maximize product.

        return -product

    # Initial operating conditions

    initial_guess = [80, 75]

    # Run optimization

    optimization_result = minimize(

        objective,

        initial_guess,

        method="L-BFGS-B",

        bounds=[
            light_bounds,
            kla_bounds
        ],

        options={
            "maxiter": 15,
            "ftol": 1e-6
        }

    )

    # Extract optimized parameters

    best_light = optimization_result.x[0]

    best_kla = optimization_result.x[1]

    # Verify the optimized operating conditions

    final_result = simulate_integrated_bioreactor(

        population_size=100,

        duration=20,

        surface_intensity=best_light,

        kla=best_kla,

        seed=42

    )

    best_product = final_result["product"][-1]

    print("\nOPTIMIZATION COMPLETED")

    print("------------------------------")

    print(
        "Optimization successful:",
        optimization_result.success
    )

    print(
        "Best light intensity:",
        best_light
    )

    print(
        "Best kLa:",
        best_kla,
        "h-1"
    )

    print(
        "Maximum predicted product:",
        best_product,
        "g/L"
    )

    print(
        "Number of evaluations:",
        evaluation_count
    )

    return {

        "best_light": best_light,

        "best_kla": best_kla,

        "best_product": best_product,

        "simulation": final_result,

        "optimizer": optimization_result

    }