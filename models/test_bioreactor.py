from .bioreactor import (
    simulate_oxygen_transfer_with_rheology
)


flow_indices = [0.6, 1.0, 1.4]

for n in flow_indices:

    result = simulate_oxygen_transfer_with_rheology(
        K=0.5,
        shear_rate=100,
        n=n,
        kla_reference=100,
        reference_viscosity=0.5,
        sensitivity=0.5,
        oxygen_saturation=0.008,
        dissolved_oxygen=0.004
    )

    print("\nFlow behaviour index:", n)

    print(
        "Viscosity:",
        result["viscosity"],
        "Pa.s"
    )

    print(
        "Estimated kLa:",
        result["kLa"],
        "1/hour"
    )

    print(
        "Oxygen Transfer Rate:",
        result["OTR"],
        "g/L/hour"
    )