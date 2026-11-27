import numpy as np


def calculate_multimodal_activation(
    opto_activation,
    electro_activation,
    sono_activation,
    magneto_activation
):
    """
    Combine four physical-field activation signals.

    Proposed phenomenological combination rule:

    A_total = 1 - (1-A_opto)
                  (1-A_electro)
                  (1-A_sono)
                  (1-A_magneto)

    All activation inputs must lie between 0 and 1.
    """

    activations = {
        "Optogenetic": opto_activation,
        "Electrogenetic": electro_activation,
        "Sonogenetic": sono_activation,
        "Magnetogenetic": magneto_activation
    }

    for name, activation in activations.items():

        if not np.all(np.isfinite(activation)):
            raise ValueError(
                f"{name} activation must contain finite values."
            )

        if np.any(np.asarray(activation) < 0) or np.any(
            np.asarray(activation) > 1
        ):
            raise ValueError(
                f"{name} activation must be between 0 and 1."
            )

    opto_activation = np.asarray(opto_activation, dtype=float)
    electro_activation = np.asarray(electro_activation, dtype=float)
    sono_activation = np.asarray(sono_activation, dtype=float)
    magneto_activation = np.asarray(magneto_activation, dtype=float)

    if not (
        opto_activation.shape
        == electro_activation.shape
        == sono_activation.shape
        == magneto_activation.shape
    ):
        raise ValueError(
            "All activation signals must have the same shape."
        )

    total_activation = 1.0 - (
        (1.0 - opto_activation)
        * (1.0 - electro_activation)
        * (1.0 - sono_activation)
        * (1.0 - magneto_activation)
    )

    return {
        "opto_activation": opto_activation,
        "electro_activation": electro_activation,
        "sono_activation": sono_activation,
        "magneto_activation": magneto_activation,
        "total_activation": total_activation
    }