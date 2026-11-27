import numpy as np
import matplotlib.pyplot as plt

from models.multimodal_activation import (
    calculate_multimodal_activation
)

from models.multimodal_gene_expression import (
    simulate_multimodal_gene_expression
)


time = np.linspace(0, 20, 400)

# Four physical fields are switched on after 2 hours.

opto = np.where(time < 2, 0.0, 0.4)
electro = np.where(time < 2, 0.0, 0.5)
sono = np.where(time < 2, 0.0, 0.3)
magneto = np.where(time < 2, 0.0, 0.6)


# Combine the four activation signals.

activation_results = calculate_multimodal_activation(
    opto_activation=opto,
    electro_activation=electro,
    sono_activation=sono,
    magneto_activation=magneto
)


total_activation = activation_results["total_activation"]


# Simulate gene expression.

results = simulate_multimodal_gene_expression(
    time=time,
    total_activation=total_activation,
    alpha_m=2.0,
    delta_m=0.2,
    alpha_e=0.5,
    delta_e=0.1,
    initial_mRNA=0.2,
    initial_enzyme=0.1
)


print("\nMULTI-MODAL GENE EXPRESSION RESULTS")
print("-----------------------------------")

print("Final combined activation:",
      results["total_activation"][-1])

print("Final mRNA:",
      results["mRNA"][-1])

print("Final enzyme:",
      results["enzyme"][-1])


# Plot combined activation.

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    total_activation,
    label="Combined activation"
)

plt.xlabel("Time")
plt.ylabel("Normalized activation")
plt.title("Multi-Modal Physical-Field Activation")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()


# Plot mRNA and enzyme dynamics.

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    results["mRNA"],
    label="mRNA"
)

plt.plot(
    time,
    results["enzyme"],
    label="Enzyme"
)

plt.xlabel("Time")
plt.ylabel("Model concentration")
plt.title("Multi-Modal Gene Expression Dynamics")
plt.grid(True)
plt.legend()
plt.show()