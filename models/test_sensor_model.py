from .integrated_bioreactor import (
    simulate_integrated_bioreactor
)

from .sensor_model import (
    simulate_sensor_measurements
)

from visualization.sensor_visualization import (
    plot_sensor_comparison
)


# Run the integrated bioreactor simulation

simulation = simulate_integrated_bioreactor(

    population_size=100,

    duration=20,

    surface_intensity=100,

    kla=50,

    seed=42

)


# Generate simulated sensor measurements

sensor_data = simulate_sensor_measurements(

    simulation,

    seed=42

)


print("SENSOR MEASUREMENT SIMULATION")

print("-" * 55)


print(
    "Final true biomass:",
    simulation["biomass"][-1]
)

print(
    "Final measured biomass:",
    sensor_data["measured_biomass"][-1]
)


print(
    "\nFinal true dissolved oxygen:",
    simulation["dissolved_oxygen"][-1]
)

print(
    "Final measured dissolved oxygen:",
    sensor_data["measured_oxygen"][-1]
)


print(
    "\nFinal true substrate:",
    simulation["substrate"][-1]
)

print(
    "Final measured substrate:",
    sensor_data["measured_substrate"][-1]
)


print(
    "\nFinal true product:",
    simulation["product"][-1]
)

print(
    "Final measured product:",
    sensor_data["measured_product"][-1]
)


# Visualize sensor measurements

plot_sensor_comparison(sensor_data)