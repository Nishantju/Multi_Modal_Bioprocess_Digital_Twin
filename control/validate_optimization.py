import numpy as np

from models.integrated_bioreactor import (
    simulate_integrated_bioreactor
)


def validate_optimized_conditions():

    optimized_light = 100.0

    optimized_kla = 44.640548285448524

    # Test nearby kLa values

    kla_values = [
        30,
        35,
        40,
        optimized_kla,
        50,
        55,
        60,
        75
    ]

    # Test nearby light intensities

    light_values = [
        80,
        90,
        95,
        100
    ]

    print("OPTIMIZATION VALIDATION")

    print("=" * 65)

    print("\nPART A: kLa VALIDATION")

    kla_results = []

    for kla in kla_values:

        result = simulate_integrated_bioreactor(

            population_size=100,

            duration=20,

            surface_intensity=optimized_light,

            kla=kla,

            seed=42

        )

        product = result["product"][-1]

        kla_results.append(product)

        print(
            f"kLa={kla:.4f} h-1 | "
            f"Product={product:.8f} g/L"
        )

    print("\nPART B: LIGHT INTENSITY VALIDATION")

    light_results = []

    for light in light_values:

        result = simulate_integrated_bioreactor(

            population_size=100,

            duration=20,

            surface_intensity=light,

            kla=optimized_kla,

            seed=42

        )

        product = result["product"][-1]

        light_results.append(product)

        print(
            f"Light={light:.2f} | "
            f"Product={product:.8f} g/L"
        )

    # Reproducibility check

    print("\nPART C: REPRODUCIBILITY")

    first = simulate_integrated_bioreactor(

        population_size=100,

        duration=20,

        surface_intensity=optimized_light,

        kla=optimized_kla,

        seed=42

    )["product"][-1]

    second = simulate_integrated_bioreactor(

        population_size=100,

        duration=20,

        surface_intensity=optimized_light,

        kla=optimized_kla,

        seed=42

    )["product"][-1]

    print("First simulation:", first)

    print("Second simulation:", second)

    print(
        "Reproducible:",
        np.isclose(first, second)
    )

    return {

        "kla_values": kla_values,

        "kla_products": kla_results,

        "light_values": light_values,

        "light_products": light_results

    }