def calculate_shear_stress(K, shear_rate, n):

    """
    Power-law model:

    tau = K * shear_rate^n
    """

    shear_stress = K * (shear_rate ** n)

    return shear_stress


def calculate_apparent_viscosity(K, shear_rate, n):

    """
    Apparent viscosity:

    mu = K * shear_rate^(n-1)
    """

    if shear_rate <= 0:
        raise ValueError("Shear rate must be positive.")

    viscosity = K * (shear_rate ** (n - 1))

    return viscosity


def classify_fluid(n):

    if n < 1:
        return "Shear-thinning"

    elif n == 1:
        return "Newtonian"

    else:
        return "Shear-thickening"