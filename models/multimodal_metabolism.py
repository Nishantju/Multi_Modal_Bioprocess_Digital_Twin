import numpy as np


def calculate_multimodal_metabolic_rates(
    enzyme,
    substrate,
    dissolved_oxygen,
    biomass,
    qP_max=0.05,
    K_E=2.0,
    K_SP=1.0,
    K_OP=0.001,
    Y_PS=0.8,
    mu=0.0,
    Y_XS=0.5
):
    """
    Calculate enzyme-dependent product formation
    and substrate consumption rates.

    All concentrations and kinetic parameters use
    consistent model units.
    """

    # Convert inputs to numerical values
    enzyme = np.asarray(enzyme, dtype=float)
    substrate = np.asarray(substrate, dtype=float)
    dissolved_oxygen = np.asarray(
        dissolved_oxygen, dtype=float
    )
    biomass = np.asarray(biomass, dtype=float)
    mu = np.asarray(mu, dtype=float)

    # Validate inputs
    inputs = [
        enzyme,
        substrate,
        dissolved_oxygen,
        biomass,
        mu
    ]

    for value in inputs:
        if not np.all(np.isfinite(value)):
            raise ValueError("Inputs must contain finite values.")

        if np.any(value < 0):
            raise ValueError("Inputs cannot be negative.")

    parameters = [qP_max, K_E, K_SP, K_OP, Y_PS, Y_XS]

    if not all(np.isfinite(p) and p > 0 for p in parameters):
        raise ValueError("All kinetic parameters and yields must be positive.")

    # Enzyme-dependent activation of product formation
    enzyme_factor = enzyme / (K_E + enzyme)

    # Substrate availability
    substrate_factor = substrate / (K_SP + substrate)

    # Oxygen availability
    oxygen_factor = dissolved_oxygen / (
        K_OP + dissolved_oxygen
    )

    # Specific product formation rate
    qP = (
        qP_max
        * enzyme_factor
        * substrate_factor
        * oxygen_factor
    )

    # Product formation rate
    dP_dt = qP * biomass

    # Substrate consumption for product formation
    substrate_for_product = dP_dt / Y_PS

    # Substrate consumption for biomass growth
    substrate_for_growth = mu * biomass / Y_XS

    # Total substrate consumption rate
    substrate_consumption = (
        substrate_for_product
        + substrate_for_growth
    )

    return {
        "enzyme_factor": enzyme_factor,
        "substrate_factor": substrate_factor,
        "oxygen_factor": oxygen_factor,
        "specific_product_rate": qP,
        "product_formation_rate": dP_dt,
        "substrate_consumption_rate": substrate_consumption
    }