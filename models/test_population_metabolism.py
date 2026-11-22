from .population_metabolism import (
    simulate_population_metabolism
)


result = simulate_population_metabolism(
    population_size=100,
    duration=10
)

print("POPULATION-LEVEL METABOLIC MODEL")

print("\nFinal average enzyme abundance:")

print(result["mean_enzyme"][-1])

print("\nFinal average product formation:")

print(result["mean_product"][-1])

print("\nFinal total simulated product:")

print(result["total_product"][-1])