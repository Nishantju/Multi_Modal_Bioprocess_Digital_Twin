import numpy as np


def calculate_otr(kla, oxygen_saturation, dissolved_oxygen):

    """
    Calculate Oxygen Transfer Rate.

    OTR = kLa * (Cstar - CL)

    Units:
    kLa: 1/hour
    Oxygen concentrations: g/L
    OTR: g/L/hour
    """

    oxygen_deficit = max(
        oxygen_saturation - dissolved_oxygen,
        0.0
    )

    otr = kla * oxygen_deficit

    return otr


def calculate_our(specific_oxygen_uptake, biomass):

    """
    Calculate Oxygen Uptake Rate.

    OUR = qO2 * X

    Units:
    qO2: g oxygen / g biomass / hour
    X: g biomass / L
    OUR: g oxygen / L / hour
    """

    our = specific_oxygen_uptake * biomass

    return our


def calculate_oxygen_balance(
    kla,
    oxygen_saturation,
    dissolved_oxygen,
    specific_oxygen_uptake,
    biomass
):

    """
    Calculate the dynamic dissolved oxygen balance.

    dCL/dt = OTR - OUR
    """

    otr = calculate_otr(
        kla,
        oxygen_saturation,
        dissolved_oxygen
    )

    our = calculate_our(
        specific_oxygen_uptake,
        biomass
    )

    oxygen_change = otr - our

    return {
        "OTR": otr,
        "OUR": our,
        "oxygen_change": oxygen_change
    }


def calculate_oxygen_limitation(
    dissolved_oxygen,
    oxygen_half_saturation
):

    """
    Calculate oxygen limitation using a Monod-type relationship.

    fDO = CL / (KDO + CL)
    """

    dissolved_oxygen = max(dissolved_oxygen, 0.0)

    factor = dissolved_oxygen / (
        oxygen_half_saturation + dissolved_oxygen + 1e-12
    )

    return factor