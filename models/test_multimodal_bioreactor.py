import numpy as np
import matplotlib.pyplot as plt

from models.multimodal_bioreactor import (
    simulate_multimodal_bioreactor
)


# Simulation duration: 20 hours
time = np.linspace(0, 20, 400)

# Four physical fields switch on at different times
opto_input = np.where(time >= 2, 100.0, 0.0)

electro_input = np.where(time >= 4, 8.0, 0.0)

sono_input = np.where(time >= 6, 8.0, 0.0)

magneto_input = np.where(time >= 8, 8.0, 0.0)


# Run integrated simulation
results = simulate_multimodal_bioreactor(
    time=time,
    opto_input=opto_input,
    electro_input=electro_input,
    sono_input=sono_input,
    magneto_input=magneto_input
)


# Display final results
print("\nINTEGRATED MULTI-MODAL BIOREACTOR RESULTS")
print("------------------------------------------")

print("Final biomass:", results["biomass"][-1])
print("Final substrate:", results["substrate"][-1])
print("Final dissolved oxygen:", results["dissolved_oxygen"][-1])
print("Final product:", results["product"][-1])

print("Final mRNA:", results["mRNA"][-1])
print("Final enzyme:", results["enzyme"][-1])

print("Final optogenetic activation:", results["opto_activation"][-1])
print("Final electrogenetic activation:", results["electro_activation"][-1])
print("Final sonogenetic activation:", results["sono_activation"][-1])
print("Final magnetogenetic activation:", results["magneto_activation"][-1])

print("Final combined activation:", results["total_activation"][-1])


# Plot 1: Individual and combined activations
plt.figure(figsize=(10, 5))

plt.plot(time, results["opto_activation"], label="Optogenetic")
plt.plot(time, results["electro_activation"], label="Electrogenetic")
plt.plot(time, results["sono_activation"], label="Sonogenetic")
plt.plot(time, results["magneto_activation"], label="Magnetogenetic")
plt.plot(
    time,
    results["total_activation"],
    label="Combined activation",
    linewidth=2
)

plt.xlabel("Time (hours)")
plt.ylabel("Activation")
plt.title("Multi-Modal Physical-Field Activation")
plt.legend()
plt.grid()
plt.tight_layout()


# Plot 2: Gene expression
plt.figure(figsize=(10, 5))

plt.plot(time, results["mRNA"], label="mRNA")
plt.plot(time, results["enzyme"], label="Enzyme")

plt.xlabel("Time (hours)")
plt.ylabel("Model-level concentration")
plt.title("Multi-Modal Gene Expression Dynamics")
plt.legend()
plt.grid()
plt.tight_layout()


# Plot 3: Fermentation variables
plt.figure(figsize=(10, 6))

plt.plot(time, results["biomass"], label="Biomass")
plt.plot(time, results["substrate"], label="Substrate")
plt.plot(time, results["dissolved_oxygen"], label="Dissolved oxygen")
plt.plot(time, results["product"], label="Product")

plt.xlabel("Time (hours)")
plt.ylabel("Concentration (model units)")
plt.title("Integrated Batch Fermentation Dynamics")
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()