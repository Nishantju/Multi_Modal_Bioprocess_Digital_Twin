from .population_opto_metabolism import (
    simulate_opto_metabolism
)

result = simulate_opto_metabolism(
    population_size=100,
    duration=10
)

print("INTEGRATED OPTOGENETIC METABOLIC MODEL")

print("\nAverage gene activation:")
print(result["mean_activation"][-1])

print("\nAverage enzyme abundance:")
print(result["mean_enzyme"][-1])

print("\nAverage accumulated product per cell:")
print(result["mean_product"][-1])

print("\nTotal accumulated product:")
print(result["total_product"][-1])

print("\nVariation in product across cells:")
print(result["product_std"][-1])

print("\nIndividual cell product distribution:")

print("Minimum:", result["product"][:, -1].min())

print("Maximum:", result["product"][:, -1].max())