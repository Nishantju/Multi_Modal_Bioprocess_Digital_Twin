import numpy as np
import pandas as pd

from models.multimodal_bioreactor import (
    simulate_multimodal_bioreactor
)


# Simulation settings
time = np.linspace(0, 20, 400)

# Physical-field input profiles
opto = np.where(time >= 2, 100.0, 0.0)
electro = np.where(time >= 4, 8.0, 0.0)
sono = np.where(time >= 6, 8.0, 0.0)
magneto = np.where(time >= 8, 8.0, 0.0)

zero = np.zeros_like(time)


# Define six experimental scenarios
scenarios = {
    "No activation": (zero, zero, zero, zero),
    "Optogenetics": (opto, zero, zero, zero),
    "Electrogenetics": (zero, electro, zero, zero),
    "Sonogenetics": (zero, zero, sono, zero),
    "Magnetogenetics": (zero, zero, zero, magneto),
    "All four modalities": (opto, electro, sono, magneto)
}


results_table = []


for name, inputs in scenarios.items():

    result = simulate_multimodal_bioreactor(
        time=time,
        opto_input=inputs[0],
        electro_input=inputs[1],
        sono_input=inputs[2],
        magneto_input=inputs[3]
    )

    results_table.append({
        "Scenario": name,
        "Final Biomass": result["biomass"][-1],
        "Final Substrate": result["substrate"][-1],
        "Final Oxygen": result["dissolved_oxygen"][-1],
        "Final mRNA": result["mRNA"][-1],
        "Final Enzyme": result["enzyme"][-1],
        "Final Product": result["product"][-1],
        "Combined Activation": result["total_activation"][-1]
    })


df = pd.DataFrame(results_table)

print("\nMULTI-MODAL CONTROL COMPARISON")
print("--------------------------------")

print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


import matplotlib.pyplot as plt

# Product comparison
plt.figure(figsize=(11, 6))

plt.bar(
    df["Scenario"],
    df["Final Product"]
)

plt.xlabel("Control Strategy")
plt.ylabel("Final Product Concentration")
plt.title("Product Formation Under Different Physical-Field Strategies")

plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# Enzyme comparison
plt.figure(figsize=(11, 6))

plt.bar(
    df["Scenario"],
    df["Final Enzyme"]
)

plt.xlabel("Control Strategy")
plt.ylabel("Final Enzyme Level")
plt.title("Enzyme Production Under Different Physical-Field Strategies")

plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()