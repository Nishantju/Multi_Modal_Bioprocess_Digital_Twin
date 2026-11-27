import numpy as np
import matplotlib.pyplot as plt

from models.magnetogenetics import simulate_magnetogenetics


time = np.linspace(0, 10, 200)

# Magnetic field switches on after 2 hours
magnetic_field = np.where(time < 2, 0.0, 8.0)

results = simulate_magnetogenetics(
    time=time,
    magnetic_field=magnetic_field,
    half_activation_field=5.0,
    hill_coefficient=2.0,
    response_time=0.5
)

print("\nMAGNETOGENETIC SIMULATION")
print("-------------------------")

print("Initial activation:",
      results["activation"][0])

print("Final activation:",
      results["activation"][-1])

print("Final steady-state activation:",
      results["steady_state_activation"][-1])

print("Maximum activation:",
      np.max(results["activation"]))


plt.figure(figsize=(10, 5))

plt.plot(
    results["time"],
    results["magnetic_field"],
    label="Magnetic field"
)

plt.xlabel("Time")
plt.ylabel("Magnetic field strength")
plt.title("Applied Magnetic Field")
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    results["time"],
    results["activation"],
    label="Actual activation"
)

plt.plot(
    results["time"],
    results["steady_state_activation"],
    "--",
    label="Steady-state activation"
)

plt.xlabel("Time")
plt.ylabel("Normalized activation")
plt.title("Dynamic Magnetogenetic Response")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()