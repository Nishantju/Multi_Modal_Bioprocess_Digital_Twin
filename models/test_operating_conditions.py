from .integrated_bioreactor import (
    simulate_integrated_bioreactor
)


conditions = {

    "A: Low Light + Low Oxygen Transfer": {
        "surface_intensity": 20,
        "kla": 50
    },

    "B: High Light + Low Oxygen Transfer": {
        "surface_intensity": 100,
        "kla": 50
    },

    "C: Low Light + High Oxygen Transfer": {
        "surface_intensity": 20,
        "kla": 150
    },

    "D: High Light + High Oxygen Transfer": {
        "surface_intensity": 100,
        "kla": 150
    }

}


results = {}

print("FOUR-CONDITION BIOREACTOR COMPARISON")

print("-" * 90)

for name, parameters in conditions.items():

    result = simulate_integrated_bioreactor(

        population_size=100,

        duration=20,

        surface_intensity=parameters["surface_intensity"],

        kla=parameters["kla"]

    )

    results[name] = result

    print("\n", name)

    print(
        "Final Biomass:",
        result["biomass"][-1],
        "g/L"
    )

    print(
        "Final Product:",
        result["product"][-1],
        "g/L"
    )

    print(
        "Final Dissolved Oxygen:",
        result["dissolved_oxygen"][-1],
        "g/L"
    )

    print(
        "Average Gene Activation:",
        result["mean_activation"][-1]
    )

    print(
        "Average Enzyme:",
        result["mean_enzyme"][-1]
    )

    print("-" * 90)