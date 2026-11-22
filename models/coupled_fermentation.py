import numpy as np
from scipy.integrate import solve_ivp


def fermentation_oxygen_model(time, state, params):

    # Extract state variables
    X, S, DO, P = state

    # Prevent negative concentrations in rate calculations
    X = max(X, 0.0)
    S = max(S, 0.0)
    DO = max(DO, 0.0)

    # Extract parameters
    mu_max = params["mu_max"]
    Ks = params["Ks"]
    KDO = params["KDO"]

    kla = params["kla"]
    oxygen_saturation = params["oxygen_saturation"]
    qO2 = params["qO2"]

    Yxs = params["Yxs"]
    qP = params["qP"]
    Yps = params["Yps"]

    # Substrate limitation
    substrate_factor = S / (Ks + S + 1e-12)

    # Oxygen limitation
    oxygen_factor = DO / (KDO + DO + 1e-12)

    # Actual microbial growth rate
    mu = mu_max * substrate_factor * oxygen_factor

    # Biomass growth
    dX_dt = mu * X

    # Substrate consumption for biomass growth
    substrate_for_growth = (mu * X) / Yxs

    # Product formation
    dP_dt = qP * X

    # Substrate consumption for product formation
    substrate_for_product = dP_dt / Yps

    # Total substrate consumption
    dS_dt = -(
        substrate_for_growth
        + substrate_for_product
    )

    # Oxygen transfer
    OTR = kla * (oxygen_saturation - DO)

    # Oxygen uptake
    OUR = qO2 * X * oxygen_factor

    # Dissolved oxygen balance
    dDO_dt = OTR - OUR

    return [
        dX_dt,
        dS_dt,
        dDO_dt,
        dP_dt
    ]


def simulate_coupled_fermentation(
    initial_biomass,
    initial_substrate,
    initial_oxygen,
    duration,
    params
):

    initial_state = [
        initial_biomass,
        initial_substrate,
        initial_oxygen,
        0.0
    ]

    time_points = np.linspace(0, duration, 300)

    solution = solve_ivp(
        fermentation_oxygen_model,
        [0, duration],
        initial_state,
        t_eval=time_points,
        args=(params,),
        method="RK45",
        rtol=1e-7,
        atol=1e-9
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return {
        "time": solution.t,
        "biomass": np.maximum(solution.y[0], 0.0),
        "substrate": np.maximum(solution.y[1], 0.0),
        "dissolved_oxygen": np.maximum(solution.y[2], 0.0),
        "product": np.maximum(solution.y[3], 0.0)
    }