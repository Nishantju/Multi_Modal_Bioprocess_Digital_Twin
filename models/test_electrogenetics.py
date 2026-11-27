import numpy as np

from models.electrogenetics import simulate_electrogenetics
import matplotlib.pyplot as plt

time = np.linspace(0, 10, 200)

# Electrical field increases after 2 hours
electric_field = np.where(time < 2, 0.0, 8.0)

results = simulate_electrogenetics(
    time=time,
    electric_field=electric_field,
    half_activation_field=5.0,
    hill_coefficient=2.0,
    response_time=0.5
)

print("\nELECTROGENETIC SIMULATION")
print("------------------------")

print("Initial activation:",
      results["activation"][0])

print("Final activation:",
      results["activation"][-1])

print("Final steady-state activation:",
      results["steady_state_activation"][-1])

print("Maximum activation:",
      np.max(results["activation"]))

print("Simulation successful.")


plt.figure(figsize=(10, 5))

plt.plot(
    results["time"],
    results["electric_field"],
    label="Electrical field",
    linewidth=2
)

plt.xlabel("Time")
plt.ylabel("Electrical field strength")
plt.title("Applied Electrical Field")
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    results["time"],
    results["activation"],
    label="Actual activation",
    linewidth=2
)

plt.plot(
    results["time"],
    results["steady_state_activation"],
    "--",
    label="Steady-state activation"
)

plt.xlabel("Time")
plt.ylabel("Normalized activation")
plt.title("Dynamic Electrogenetic Response")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()