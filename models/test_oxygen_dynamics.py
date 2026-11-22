from .oxygen_dynamics import simulate_dynamic_oxygen


result = simulate_dynamic_oxygen(
    initial_oxygen=0.007,
    oxygen_saturation=0.008,
    kla=100,
    specific_oxygen_uptake=0.2,
    biomass=3,
    duration=5
)


print("Time (hours) | Dissolved Oxygen (g/L)")

for t, oxygen in zip(
    result["time"][::20],
    result["dissolved_oxygen"][::20]
):

    print(
        round(t, 2),
        "|",
        round(oxygen, 6)
    )


print(
    "\nFinal dissolved oxygen:",
    result["dissolved_oxygen"][-1],
    "g/L"
)