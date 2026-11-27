import numpy as np

from models.multimodal_activation import (
    calculate_multimodal_activation
)


# Four illustrative activation signals

opto = np.array([0.4])
electro = np.array([0.5])
sono = np.array([0.3])
magneto = np.array([0.6])


results = calculate_multimodal_activation(
    opto_activation=opto,
    electro_activation=electro,
    sono_activation=sono,
    magneto_activation=magneto
)


print("\nMULTI-MODAL ACTIVATION RESULTS")
print("--------------------------------")

print("Optogenetic activation:",
      results["opto_activation"][0])

print("Electrogenetic activation:",
      results["electro_activation"][0])

print("Sonogenetic activation:",
      results["sono_activation"][0])

print("Magnetogenetic activation:",
      results["magneto_activation"][0])

print("Combined activation:",
      results["total_activation"][0])


# Verify the mathematical result

expected = 0.916

assert np.isclose(
    results["total_activation"][0],
    expected
)

print("\nMathematical verification successful.")