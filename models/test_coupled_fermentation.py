from .coupled_fermentation import (
    simulate_coupled_fermentation
)


params = {

    "mu_max": 0.5,
    "Ks": 1.0,
    "KDO": 0.001,

    "kla": 100,
    "oxygen_saturation": 0.008,

    "qO2": 0.2,

    "Yxs": 0.5,
    "qP": 0.05,
    "Yps": 0.8
}


result = simulate_coupled_fermentation(

    initial_biomass=0.1,
    initial_substrate=20,
    initial_oxygen=0.007,

    duration=20,

    params=params
)


print("TIME | BIOMASS | SUBSTRATE | DISSOLVED OXYGEN")

for i in range(0, len(result["time"]), 30):

    print(

        round(result["time"][i], 2),

        "|",

        round(result["biomass"][i], 4),

        "|",

        round(result["substrate"][i], 4),

        "|",

        round(result["dissolved_oxygen"][i], 6)

    )


print("\nFINAL RESULTS")

print("Biomass:", result["biomass"][-1], "g/L")

print("Substrate:", result["substrate"][-1], "g/L")

print(
    "Dissolved oxygen:",
    result["dissolved_oxygen"][-1],
    "g/L"
)
print(
    "Product:",
    result["product"][-1],
    "g/L"
)