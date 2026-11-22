from .oxygen_transfer import calculate_oxygen_balance


result = calculate_oxygen_balance(
    kla=100,
    oxygen_saturation=0.008,
    dissolved_oxygen=0.004,
    specific_oxygen_uptake=0.2,
    biomass=2
)

print("Oxygen Transfer Rate:", result["OTR"], "g/L/h")

print("Oxygen Uptake Rate:", result["OUR"], "g/L/h")

print(
    "Rate of oxygen change:",
    result["oxygen_change"],
    "g/L/h"
)