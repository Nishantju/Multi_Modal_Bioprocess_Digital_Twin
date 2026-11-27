import numpy as np
from scipy.integrate import solve_ivp


def multimodal_bioreactor_rhs(
    t,
    state,
    time,
    opto_input,
    electro_input,
    sono_input,
    magneto_input,
    params
):
    """
    Integrated multi-modal batch bioreactor model.

    Dynamic states:
    X, S, DO, P, mRNA, enzyme,
    optogenetic activation,
    electrogenetic activation,
    sonogenetic activation,
    magnetogenetic activation.
    """

    (
        X, S, DO, P, mRNA, enzyme,
        A_opto, A_electro, A_sono, A_magneto
    ) = state

    # Prevent numerical integration from using negative concentrations
    X = max(X, 0.0)
    S = max(S, 0.0)
    DO = max(DO, 0.0)
    mRNA = max(mRNA, 0.0)
    enzyme = max(enzyme, 0.0)

    # Interpolate physical-field inputs at current time
    light = np.interp(t, time, opto_input)
    electric = np.interp(t, time, electro_input)
    ultrasound = np.interp(t, time, sono_input)
    magnetic = np.interp(t, time, magneto_input)

    # Retrieve parameters
    p = params

    # Calculate steady-state activation for each modality
    def hill_activation(value, half_activation, hill):
        value = max(value, 0.0)

        numerator = value ** hill
        denominator = half_activation ** hill + numerator

        return numerator / denominator

    A_opto_ss = hill_activation(
        light,
        p["K_light"],
        p["hill_opto"]
    )

    A_electro_ss = hill_activation(
        electric,
        p["K_electric"],
        p["hill_electro"]
    )

    A_sono_ss = hill_activation(
        ultrasound,
        p["K_ultrasound"],
        p["hill_sono"]
    )

    A_magneto_ss = hill_activation(
        magnetic,
        p["K_magnetic"],
        p["hill_magneto"]
    )

    # Dynamic physical-field responses
    dA_opto_dt = (
        A_opto_ss - A_opto
    ) / p["tau_opto"]

    dA_electro_dt = (
        A_electro_ss - A_electro
    ) / p["tau_electro"]

    dA_sono_dt = (
        A_sono_ss - A_sono
    ) / p["tau_sono"]

    dA_magneto_dt = (
        A_magneto_ss - A_magneto
    ) / p["tau_magneto"]

    # Combine four activation signals
    total_activation = 1.0 - (
        (1.0 - A_opto)
        * (1.0 - A_electro)
        * (1.0 - A_sono)
        * (1.0 - A_magneto)
    )

    # Gene expression dynamics
    dmRNA_dt = (
        p["alpha_m"] * total_activation
        - p["delta_m"] * mRNA
    )

    denzyme_dt = (
        p["alpha_e"] * mRNA
        - p["delta_e"] * enzyme
    )

    # Substrate and oxygen availability
    substrate_factor = S / (p["K_S"] + S)

    oxygen_factor = DO / (p["K_DO"] + DO)

    # Biomass growth
    mu = (
        p["mu_max"]
        * substrate_factor
        * oxygen_factor
    )

    dX_dt = mu * X

    # Enzyme-dependent product formation
    enzyme_factor = enzyme / (p["K_E"] + enzyme)

    product_substrate_factor = (
        S / (p["K_SP"] + S)
    )

    product_oxygen_factor = (
        DO / (p["K_OP"] + DO)
    )

    qP = (
        p["qP_max"]
        * enzyme_factor
        * product_substrate_factor
        * product_oxygen_factor
    )

    dP_dt = qP * X

    # Batch substrate balance
    dS_dt = -(
        mu * X / p["Y_XS"]
        + dP_dt / p["Y_PS"]
    )

    # Oxygen transfer and consumption
    OTR = p["kla"] * (
        p["oxygen_saturation"] - DO
    )

    OUR = p["qO2"] * X * oxygen_factor

    dDO_dt = OTR - OUR

    return [
        dX_dt,
        dS_dt,
        dDO_dt,
        dP_dt,
        dmRNA_dt,
        denzyme_dt,
        dA_opto_dt,
        dA_electro_dt,
        dA_sono_dt,
        dA_magneto_dt
    ]


def simulate_multimodal_bioreactor(
    time,
    opto_input,
    electro_input,
    sono_input,
    magneto_input,
    params=None,
    initial_conditions=None
):
    """
    Simulate integrated multi-modal batch fermentation.
    """

    time = np.asarray(time, dtype=float)

    opto_input = np.asarray(opto_input, dtype=float)
    electro_input = np.asarray(electro_input, dtype=float)
    sono_input = np.asarray(sono_input, dtype=float)
    magneto_input = np.asarray(magneto_input, dtype=float)

    if time.ndim != 1 or len(time) < 2:
        raise ValueError("Time must be a 1D array with at least two points.")

    if not np.all(np.isfinite(time)):
        raise ValueError("Time must contain finite values.")

    if np.any(np.diff(time) <= 0):
        raise ValueError("Time must be strictly increasing.")

    inputs = [
        opto_input,
        electro_input,
        sono_input,
        magneto_input
    ]

    for values in inputs:
        if values.ndim != 1 or len(values) != len(time):
            raise ValueError("All field inputs must match the time array.")

        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Field inputs must be finite and nonnegative.")

    if params is None:
        params = {
            "K_light": 30.0,
            "K_electric": 5.0,
            "K_ultrasound": 5.0,
            "K_magnetic": 5.0,

            "hill_opto": 2.0,
            "hill_electro": 2.0,
            "hill_sono": 2.0,
            "hill_magneto": 2.0,

            "tau_opto": 0.5,
            "tau_electro": 0.5,
            "tau_sono": 0.5,
            "tau_magneto": 0.5,

            "alpha_m": 2.0,
            "delta_m": 0.2,
            "alpha_e": 0.5,
            "delta_e": 0.1,

            "mu_max": 0.5,
            "K_S": 1.0,
            "K_DO": 0.001,
            "Y_XS": 0.5,

            "qP_max": 0.05,
            "K_E": 2.0,
            "K_SP": 1.0,
            "K_OP": 0.001,
            "Y_PS": 0.8,

            "kla": 100.0,
            "oxygen_saturation": 0.008,
            "qO2": 0.2
        }

    required_parameters = [
        "K_light", "K_electric", "K_ultrasound", "K_magnetic",
        "hill_opto", "hill_electro", "hill_sono", "hill_magneto",
        "tau_opto", "tau_electro", "tau_sono", "tau_magneto",
        "alpha_m", "delta_m", "alpha_e", "delta_e",
        "mu_max", "K_S", "K_DO", "Y_XS",
        "qP_max", "K_E", "K_SP", "K_OP", "Y_PS",
        "kla", "oxygen_saturation", "qO2"
    ]

    for key in required_parameters:
        if key not in params:
            raise ValueError(f"Missing parameter: {key}")

        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"Parameter {key} must be positive.")

    if initial_conditions is None:
        initial_conditions = {
            "biomass": 0.1,
            "substrate": 20.0,
            "dissolved_oxygen": 0.007,
            "product": 0.0,
            "mRNA": 0.2,
            "enzyme": 0.1
        }

    required_initial = [
        "biomass",
        "substrate",
        "dissolved_oxygen",
        "product",
        "mRNA",
        "enzyme"
    ]

    for key in required_initial:
        if key not in initial_conditions:
            raise ValueError(f"Missing initial condition: {key}")

        value = initial_conditions[key]

        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"Initial condition {key} must be finite and nonnegative."
            )

    if (
        initial_conditions["dissolved_oxygen"]
        > params["oxygen_saturation"]
    ):
        raise ValueError(
            "Initial dissolved oxygen cannot exceed oxygen saturation."
        )

    # Initial activation states
    initial_state = [
        initial_conditions["biomass"],
        initial_conditions["substrate"],
        initial_conditions["dissolved_oxygen"],
        initial_conditions["product"],
        initial_conditions["mRNA"],
        initial_conditions["enzyme"],
        0.0,
        0.0,
        0.0,
        0.0
    ]

    solution = solve_ivp(
        fun=lambda t, y: multimodal_bioreactor_rhs(
            t,
            y,
            time,
            opto_input,
            electro_input,
            sono_input,
            magneto_input,
            params
        ),
        t_span=(time[0], time[-1]),
        y0=initial_state,
        t_eval=time,
        method="RK45",
        rtol=1e-7,
        atol=1e-9
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    # Extract simulation results
    X = solution.y[0]
    S = solution.y[1]
    DO = solution.y[2]
    P = solution.y[3]
    mRNA = solution.y[4]
    enzyme = solution.y[5]

    A_opto = solution.y[6]
    A_electro = solution.y[7]
    A_sono = solution.y[8]
    A_magneto = solution.y[9]

    total_activation = 1.0 - (
        (1.0 - A_opto)
        * (1.0 - A_electro)
        * (1.0 - A_sono)
        * (1.0 - A_magneto)
    )

    return {
        "time": solution.t,
        "biomass": X,
        "substrate": S,
        "dissolved_oxygen": DO,
        "product": P,
        "mRNA": mRNA,
        "enzyme": enzyme,
        "opto_activation": A_opto,
        "electro_activation": A_electro,
        "sono_activation": A_sono,
        "magneto_activation": A_magneto,
        "total_activation": total_activation
    }