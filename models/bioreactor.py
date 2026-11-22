from .rheology import calculate_apparent_viscosity
from .oxygen_transfer import calculate_otr


def estimate_kla(
    kla_reference,
    viscosity,
    reference_viscosity,
    sensitivity
):
    """
    Illustrative viscosity-dependent kLa model.

    kLa = kLa_reference *
          (reference_viscosity / viscosity)^sensitivity

    This is a simplified assumed relationship.
    """

    kla = kla_reference * (
        reference_viscosity / viscosity
    ) ** sensitivity

    return kla


def simulate_oxygen_transfer_with_rheology(
    K,
    shear_rate,
    n,
    kla_reference,
    reference_viscosity,
    sensitivity,
    oxygen_saturation,
    dissolved_oxygen
):

    viscosity = calculate_apparent_viscosity(
        K,
        shear_rate,
        n
    )

    kla = estimate_kla(
        kla_reference,
        viscosity,
        reference_viscosity,
        sensitivity
    )

    otr = calculate_otr(
        kla,
        oxygen_saturation,
        dissolved_oxygen
    )

    return {
        "viscosity": viscosity,
        "kLa": kla,
        "OTR": otr
    }