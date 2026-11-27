from models.multimodal_metabolism import (
    calculate_multimodal_metabolic_rates
)


# Example fermentation conditions

enzyme = 32.0659
substrate = 20.0
dissolved_oxygen = 0.007
biomass = 0.1

# Illustrative growth rate
mu = 0.2


results = calculate_multimodal_metabolic_rates(
    enzyme=enzyme,
    substrate=substrate,
    dissolved_oxygen=dissolved_oxygen,
    biomass=biomass,
    qP_max=0.05,
    K_E=2.0,
    K_SP=1.0,
    K_OP=0.001,
    Y_PS=0.8,
    mu=mu,
    Y_XS=0.5
)


print("\nMULTI-MODAL METABOLIC RESULTS")
print("--------------------------------")

print(
    "Enzyme factor:",
    results["enzyme_factor"]
)

print(
    "Substrate factor:",
    results["substrate_factor"]
)

print(
    "Oxygen factor:",
    results["oxygen_factor"]
)

print(
    "Specific product formation rate:",
    results["specific_product_rate"]
)

print(
    "Product formation rate:",
    results["product_formation_rate"]
)

print(
    "Substrate consumption rate:",
    results["substrate_consumption_rate"]
)