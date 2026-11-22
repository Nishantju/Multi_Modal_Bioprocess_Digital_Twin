from .population_opto_model import simulate_population_opto

result = simulate_population_opto(
    population_size=100,
    duration=10
)

print("POPULATION-LEVEL OPTOGENETIC MODEL")

print("\nNumber of microorganisms:")
print(result["protein"].shape[0])

print("\nFinal population statistics:")

print(
    "Average gene activation:",
    result["mean_activation"][-1]
)

print(
    "Average mRNA:",
    result["mean_mRNA"][-1]
)

print(
    "Average protein:",
    result["mean_protein"][-1]
)

print(
    "Protein standard deviation:",
    result["std_protein"][-1]
)

print("\nFinal protein distribution:")

print(
    "10th percentile:",
    result["protein_percentiles"][0, -1]
)

print(
    "50th percentile:",
    result["protein_percentiles"][1, -1]
)

print(
    "90th percentile:",
    result["protein_percentiles"][2, -1]
)