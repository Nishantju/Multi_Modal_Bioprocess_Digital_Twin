import numpy as np
import matplotlib.pyplot as plt

from models.sonogenetics import simulate_sonogenetics


time = np.linspace(0, 10, 200)

# Ultrasound is switched on after 2 hours
ultrasound_intensity = np.where(time < 2, 0.0, 8.0)

results = simulate_sonogenetics(
    time=time,
    ultrasound_intensity=ultrasound_intensity,
    half_activation_intensity=5.0,
    hill_coefficient=2.0,
    response_time=0.5
)

print("\nSONOGENETIC SIMULATION")
print("----------------------")

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
    results["ultrasound_intensity"],
    label="Ultrasound intensity"
)

plt.xlabel("Time")
plt.ylabel("Ultrasound intensity")
plt.title("Applied Ultrasound Signal")
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
plt.title("Dynamic Sonogenetic Response")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()