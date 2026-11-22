import numpy as np
from scipy.integrate import solve_ivp


def make_population_phases(population_size=100, seed=42):

    if population_size < 1:
        raise ValueError("Population size must be at least 1.")

    rng = np.random.default_rng(seed)

    phases = rng.uniform(
        0,
        2 * np.pi,
        population_size
    )

    return phases


def mean_optogenetic_activation(
    time,
    light_intensity,
    params,
    phases
):

    if light_intensity < 0:
        raise ValueError("Light intensity cannot be negative.")

    mean_depth = params["mean_depth"]
    amplitude = params["movement_amplitude"]
    frequency = params["movement_frequency"]

    attenuation = params["attenuation_coefficient"]
    half_intensity = params["half_activation_intensity"]
    hill = params["hill_coefficient"]
    basal = params["basal_activation"]

    # Individual cell positions

    depths = (
        mean_depth
        + amplitude
        * np.sin(2 * np.pi * frequency * time + phases)
    )

    depths = np.clip(
        depths,
        0,
        params["height"]
    )

    # Light intensity received by each cell

    cell_light = (
        light_intensity
        * np.exp(-attenuation * depths)
    )

    # Hill-function gene activation

    numerator = cell_light ** hill

    denominator = (
        half_intensity ** hill
        + numerator
        + 1e-12
    )

    cell_activation = (
        basal
        + (1 - basal) * numerator / denominator
    )

    # Average activation across the population

    return float(np.mean(cell_activation))


def reduced_order_rhs(
    time,
    state,
    control,
    params,
    phases
):

    X, S, DO, P, mRNA, enzyme = state

    # Prevent negative concentrations in rate calculations

    X = max(X, 0)
    S = max(S, 0)
    DO = max(DO, 0)
    mRNA = max(mRNA, 0)
    enzyme = max(enzyme, 0)

    light_intensity, kla = control

    if light_intensity < 0 or kla <= 0:
        raise ValueError("Invalid control inputs.")

    # --------------------------------------------------
    # 1. Population-average optical activation
    # --------------------------------------------------

    activation = mean_optogenetic_activation(
        time,
        light_intensity,
        params,
        phases
    )

    # --------------------------------------------------
    # 2. mRNA dynamics
    # --------------------------------------------------

    dmRNA_dt = (
        params["alpha_m"] * activation
        - params["delta_m"] * mRNA
    )

    # --------------------------------------------------
    # 3. Enzyme dynamics
    # --------------------------------------------------

    denzyme_dt = (
        params["alpha_e"] * mRNA
        - params["delta_e"] * enzyme
    )

    # --------------------------------------------------
    # 4. Substrate and oxygen limitations
    # --------------------------------------------------

    substrate_factor = (
        S / (params["K_S"] + S + 1e-12)
    )

    oxygen_factor = (
        DO / (params["K_DO"] + DO + 1e-12)
    )

    # --------------------------------------------------
    # 5. Biomass growth
    # --------------------------------------------------

    mu = (
        params["mu_max"]
        * substrate_factor
        * oxygen_factor
    )

    dX_dt = mu * X

    # --------------------------------------------------
    # 6. Enzyme-dependent product formation
    # --------------------------------------------------

    enzyme_factor = (
        enzyme / (params["K_E"] + enzyme + 1e-12)
    )

    product_substrate_factor = (
        S / (params["K_SP"] + S + 1e-12)
    )

    product_oxygen_factor = (
        DO / (params["K_OP"] + DO + 1e-12)
    )

    qP = (
        params["qP_max"]
        * enzyme_factor
        * product_substrate_factor
        * product_oxygen_factor
    )

    dP_dt = qP * X

    # --------------------------------------------------
    # 7. Substrate consumption
    # --------------------------------------------------

    dS_dt = -(
        mu * X / params["Y_XS"]
        + dP_dt / params["Y_PS"]
    )

    # --------------------------------------------------
    # 8. Oxygen transfer and consumption
    # --------------------------------------------------

    OTR = (
        kla
        * (params["oxygen_saturation"] - DO)
    )

    OUR = (
        params["qO2"]
        * X
        * oxygen_factor
    )

    dDO_dt = OTR - OUR

    return [
        dX_dt,
        dS_dt,
        dDO_dt,
        dP_dt,
        dmRNA_dt,
        denzyme_dt
    ]


def predict_reduced_order(
    initial_state,
    control,
    params,
    phases,
    start_time=0,
    horizon=2,
    n_points=30
):

    if horizon <= 0:
        raise ValueError("Prediction horizon must be positive.")

    if n_points < 2:
        raise ValueError("At least two prediction points are required.")

    initial_state = np.asarray(
        initial_state,
        dtype=float
    )

    if initial_state.shape != (6,):
        raise ValueError(
            "Initial state must contain exactly six values."
        )

    if np.any(initial_state < 0):
        raise ValueError(
            "Initial concentrations and states cannot be negative."
        )

    if len(control) != 2:
        raise ValueError(
            "Control must contain light intensity and kLa."
        )

    if len(phases) < 1:
        raise ValueError("Population phases cannot be empty.")

    if not np.all(np.isfinite(initial_state)):
        raise ValueError("Initial state contains invalid values.")

    time_points = np.linspace(
        start_time,
        start_time + horizon,
        n_points
    )

    solution = solve_ivp(
        fun=lambda t, y: reduced_order_rhs(
            t,
            y,
            control,
            params,
            phases
        ),
        t_span=(
            start_time,
            start_time + horizon
        ),
        y0=initial_state,
        t_eval=time_points,
        method="RK45",
        rtol=1e-6,
        atol=1e-9
    )

    if not solution.success:
        raise RuntimeError(
            f"Prediction failed: {solution.message}"
        )

    trajectory = solution.y

    final_state = trajectory[:, -1]

    mean_activation = np.array([
        mean_optogenetic_activation(
            t,
            control[0],
            params,
            phases
        )
        for t in time_points
    ])

    return {
        "time": time_points,
        "trajectory": trajectory,
        "final_state": final_state,
        "mean_activation": mean_activation
    }