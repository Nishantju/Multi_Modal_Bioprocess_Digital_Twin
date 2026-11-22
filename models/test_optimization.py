from control.optimization import optimize_bioreactor


result = optimize_bioreactor()


print("\nFINAL OPTIMIZED OPERATING CONDITIONS")

print(
    "Light intensity:",
    result["best_light"]
)

print(
    "Oxygen transfer coefficient:",
    result["best_kla"]
)

print(
    "Predicted product concentration:",
    result["best_product"],
    "g/L"
)