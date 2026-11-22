import numpy as np

from models.integrated_bioreactor import (
    simulate_integrated_bioreactor
)


def run_pareto_analysis():

    light_values = [20, 40, 60, 80, 100]

    kla_values = [30, 50, 75, 100, 150]

    records = []

    print("MULTI-OBJECTIVE PARETO ANALYSIS")

    print("-" * 70)

    for light in light_values:

        for kla in kla_values:

            result = simulate_integrated_bioreactor(

                population_size=100,

                duration=20,

                surface_intensity=light,

                kla=kla,

                seed=42

            )

            product = result["product"][-1]

            oxygen = result["dissolved_oxygen"][-1]

            # Normalized operating-burden proxy.
            # This is NOT actual energy consumption.

            burden = (
                0.5 * (light / 100)
                + 0.5 * (kla / 150)
            )

            records.append({

                "light": light,

                "kla": kla,

                "product": product,

                "oxygen": oxygen,

                "burden": burden

            })

            print(
                f"Light={light:3}, "
                f"kLa={kla:3}, "
                f"Product={product:.5f}, "
                f"Burden={burden:.4f}"
            )

    # Identify Pareto-optimal conditions.
    # Objective 1: maximize product.
    # Objective 2: minimize burden.

    pareto_indices = []

    for i, candidate in enumerate(records):

        dominated = False

        for j, other in enumerate(records):

            if i == j:
                continue

            better_or_equal_product = (
                other["product"] >= candidate["product"]
            )

            better_or_equal_burden = (
                other["burden"] <= candidate["burden"]
            )

            strictly_better = (
                other["product"] > candidate["product"]
                or other["burden"] < candidate["burden"]
            )

            if (
                better_or_equal_product
                and better_or_equal_burden
                and strictly_better
            ):

                dominated = True

                break

        if not dominated:

            pareto_indices.append(i)

    pareto_records = [
        records[i] for i in pareto_indices
    ]

    print("\nPARETO-OPTIMAL CONDITIONS")

    print("-" * 70)

    for item in pareto_records:

        print(
            f"Light={item['light']}, "
            f"kLa={item['kla']}, "
            f"Product={item['product']:.5f}, "
            f"Burden={item['burden']:.4f}"
        )

    return {

        "records": records,

        "pareto_records": pareto_records

    }