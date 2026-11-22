import numpy as np

from .integrated_bioreactor import (
    simulate_integrated_bioreactor
)


def run_parameter_sweep():

    light_values = [10, 30, 50, 75, 100]

    kla_values = [30, 50, 75, 100, 150]

    product_matrix = np.zeros(
        (len(kla_values), len(light_values))
    )

    biomass_matrix = np.zeros_like(product_matrix)

    oxygen_matrix = np.zeros_like(product_matrix)

    print("STARTING PARAMETER SENSITIVITY ANALYSIS")

    for i, kla in enumerate(kla_values):

        for j, light in enumerate(light_values):

            result = simulate_integrated_bioreactor(

                population_size=100,

                duration=20,

                surface_intensity=light,

                kla=kla,

                seed=42

            )

            product_matrix[i, j] = (
                result["product"][-1]
            )

            biomass_matrix[i, j] = (
                result["biomass"][-1]
            )

            oxygen_matrix[i, j] = (
                result["dissolved_oxygen"][-1]
            )

            print(
                f"Light={light:3}, "
                f"kLa={kla:3} -> "
                f"Product={product_matrix[i, j]:.5f} g/L"
            )

    best_index = np.unravel_index(
        np.argmax(product_matrix),
        product_matrix.shape
    )

    best_kla = kla_values[best_index[0]]

    best_light = light_values[best_index[1]]

    best_product = product_matrix[best_index]

    print("\nSENSITIVITY ANALYSIS COMPLETE")

    print("\nBest sampled operating condition:")

    print("Light intensity:", best_light)

    print("kLa:", best_kla, "h-1")

    print("Product:", best_product, "g/L")

    return {

        "light_values": light_values,

        "kla_values": kla_values,

        "product_matrix": product_matrix,

        "biomass_matrix": biomass_matrix,

        "oxygen_matrix": oxygen_matrix

    }