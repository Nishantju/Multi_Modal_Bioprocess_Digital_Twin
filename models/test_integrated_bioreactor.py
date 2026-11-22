from .integrated_bioreactor import (
    simulate_integrated_bioreactor
)


result = simulate_integrated_bioreactor(
    population_size=100,
    duration=20
)


print("INTEGRATED OPTOGENETIC BIOREACTOR MODEL")

print("\nFINAL BIOREACTOR CONDITIONS")

print(
    "Biomass:",
    result["biomass"][-1],
    "g/L"
)

print(
    "Substrate:",
    result["substrate"][-1],
    "g/L"
)

print(
    "Dissolved Oxygen:",
    result["dissolved_oxygen"][-1],
    "g/L"
)

print(
    "Product:",
    result["product"][-1],
    "g/L"
)

print("\nFINAL CELLULAR CONDITIONS")

print(
    "Average Gene Activation:",
    result["mean_activation"][-1]
)

print(
    "Average mRNA:",
    result["average_mRNA"][-1]
)

print(
    "Average Enzyme:",
    result["mean_enzyme"][-1]
)

print("\nOXYGEN TRANSFER")

print(
    "OTR:",
    result["OTR"][-1],
    "g/L/h"
)

print(
    "OUR:",
    result["OUR"][-1],
    "g/L/h"
)

print("\nMETABOLIC PRODUCTIVITY")

print(
    "Specific Product Formation Rate:",
    result["mean_specific_product_rate"][-1]
)

print(
    "Volumetric Product Formation Rate:",
    result["biomass"][-1]
    * result["mean_specific_product_rate"][-1]
)