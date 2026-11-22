import numpy as np
from scipy.integrate import solve_ivp


def simulate_integrated_bioreactor(

    # Population parameters
    population_size=100,
    duration=20,

    # Reactor geometry and movement
    height=2.0,
    mean_depth=1.0,
    movement_amplitude=0.9,
    movement_frequency=0.2,

    # Optogenetic parameters
    surface_intensity=100,
    attenuation_coefficient=2,
    half_activation_intensity=30,
    hill_coefficient=2,
    basal_activation=0.1,

    # Gene expression parameters
    alpha_m=2.0,
    delta_m=0.2,
    alpha_e=0.5,
    delta_e=0.1,

    # Metabolic parameters
    qP_max=0.05,
    K_E=2.0,
    K_SP=1.0,
    K_OP=0.001,
    Y_PS=0.8,

    # Biomass parameters
    mu_max=0.5,
    K_S=1.0,
    K_DO=0.001,
    Y_XS=0.5,

    # Oxygen parameters
    kla=100,
    oxygen_saturation=0.008,
    qO2=0.2,

    # Initial conditions
    initial_mRNA=0.2,
    initial_enzyme=0.1,
    initial_biomass=0.1,
    initial_substrate=20.0,
    initial_oxygen=0.007,

    # Reproducibility
    seed=42

):

    # -----------------------------------
    # 1. INPUT VALIDATION
    # -----------------------------------

    if population_size < 1:
        raise ValueError("Population size must be positive.")

    if duration <= 0:
        raise ValueError("Duration must be positive.")

    if height <= 0:
        raise ValueError("Reactor height must be positive.")

    if not (0 <= mean_depth <= height):
        raise ValueError("Mean depth must be inside the reactor.")

    if movement_amplitude < 0 or movement_amplitude > min(
        mean_depth, height - mean_depth
    ):
        raise ValueError("Movement exceeds reactor boundaries.")

    if min(
        attenuation_coefficient,
        alpha_m,
        delta_m,
        alpha_e,
        delta_e,
        qP_max,
        mu_max,
        kla,
        qO2
    ) < 0:
        raise ValueError("Rate parameters cannot be negative.")

    if min(
        half_activation_intensity,
        hill_coefficient,
        K_E,
        K_SP,
        K_OP,
        K_S,
        K_DO,
        Y_PS,
        Y_XS
    ) <= 0:
        raise ValueError("Half-saturation constants and yields must be positive.")

    if min(
        initial_mRNA,
        initial_enzyme,
        initial_biomass,
        initial_substrate,
        initial_oxygen
    ) < 0:
        raise ValueError("Initial states cannot be negative.")

    if oxygen_saturation <= 0 or initial_oxygen > oxygen_saturation:
        raise ValueError("Initial oxygen must be between zero and saturation.")

    # -----------------------------------
    # 2. POPULATION MOVEMENT
    # -----------------------------------

    rng = np.random.default_rng(seed)

    phases = rng.uniform(
        0,
        2 * np.pi,
        population_size
    )

    def calculate_activation(time):

        depth = mean_depth + movement_amplitude * np.sin(
            2 * np.pi * movement_frequency * time + phases
        )

        intensity = surface_intensity * np.exp(
            -attenuation_coefficient * depth
        )

        response = intensity ** hill_coefficient / (
            half_activation_intensity ** hill_coefficient
            + intensity ** hill_coefficient
        )

        activation = (
            basal_activation
            + (1 - basal_activation) * response
        )

        return activation, intensity, depth

    # -----------------------------------
    # 3. INTEGRATED DIFFERENTIAL EQUATIONS
    # -----------------------------------

    def reactor_model(time, state):

        N = population_size

        # Extract individual cellular states
        mRNA = state[:N]

        enzyme = state[N:2 * N]

        # Extract reactor states
        biomass = max(state[2 * N], 0.0)

        substrate = max(state[2 * N + 1], 0.0)

        oxygen = max(state[2 * N + 2], 0.0)

        # --------------------------------
        # OPTICAL AND GENETIC RESPONSE
        # --------------------------------

        activation, intensity, depth = (
            calculate_activation(time)
        )

        dmRNA_dt = (
            alpha_m * activation
            - delta_m * mRNA
        )

        denzyme_dt = (
            alpha_e * mRNA
            - delta_e * enzyme
        )

        # --------------------------------
        # BIOMASS GROWTH
        # --------------------------------

        substrate_factor = substrate / (
            K_S + substrate
        )

        oxygen_factor = oxygen / (
            K_DO + oxygen
        )

        mu = (
            mu_max
            * substrate_factor
            * oxygen_factor
        )

        dX_dt = mu * biomass

        # --------------------------------
        # ENZYME-DEPENDENT METABOLISM
        # --------------------------------

        enzyme_factor = enzyme / (
            K_E + enzyme
        )

        product_substrate_factor = substrate / (
            K_SP + substrate
        )

        product_oxygen_factor = oxygen / (
            K_OP + oxygen
        )

        specific_product_rate = (
            qP_max
            * enzyme_factor
            * product_substrate_factor
            * product_oxygen_factor
        )

        average_specific_product_rate = np.mean(
            specific_product_rate
        )

        # Volumetric product formation rate
        dP_dt = (
            biomass
            * average_specific_product_rate
        )

        # --------------------------------
        # SUBSTRATE CONSUMPTION
        # --------------------------------

        substrate_for_growth = (
            mu * biomass / Y_XS
        )

        substrate_for_product = (
            dP_dt / Y_PS
        )

        dS_dt = -(
            substrate_for_growth
            + substrate_for_product
        )

        # --------------------------------
        # OXYGEN TRANSFER AND CONSUMPTION
        # --------------------------------

        OTR = kla * (
            oxygen_saturation - oxygen
        )

        OUR = qO2 * biomass * oxygen_factor

        dDO_dt = OTR - OUR

        # --------------------------------
        # RETURN ALL DERIVATIVES
        # --------------------------------

        return np.concatenate([

            dmRNA_dt,

            denzyme_dt,

            [dX_dt, dS_dt, dDO_dt, dP_dt]

        ])

    # -----------------------------------
    # 4. INITIAL CONDITIONS
    # -----------------------------------

    initial_state = np.concatenate([

        np.full(population_size, initial_mRNA),

        np.full(population_size, initial_enzyme),

        [
            initial_biomass,
            initial_substrate,
            initial_oxygen,
            0.0
        ]

    ])

    # -----------------------------------
    # 5. NUMERICAL SOLUTION
    # -----------------------------------

    time_points = np.linspace(
        0,
        duration,
        400
    )

    solution = solve_ivp(

        reactor_model,

        [0, duration],

        initial_state,

        t_eval=time_points,

        method="RK45",

        rtol=1e-7,

        atol=1e-9

    )

    if not solution.success:
        raise RuntimeError(solution.message)

    # -----------------------------------
    # 6. EXTRACT RESULTS
    # -----------------------------------

    N = population_size

    mRNA = solution.y[:N]

    enzyme = solution.y[N:2 * N]

    biomass = np.maximum(
        solution.y[2 * N],
        0
    )

    substrate = np.maximum(
        solution.y[2 * N + 1],
        0
    )

    oxygen = np.maximum(
        solution.y[2 * N + 2],
        0
    )

    product = np.maximum(
        solution.y[2 * N + 3],
        0
    )

    # -----------------------------------
    # 7. CALCULATE POPULATION STATISTICS
    # -----------------------------------

    mean_activation = []

    mean_enzyme = []

    mean_specific_product_rate = []

    for index, time in enumerate(solution.t):

        activation, intensity, depth = (
            calculate_activation(time)
        )

        mean_activation.append(
            np.mean(activation)
        )

        mean_enzyme.append(
            np.mean(enzyme[:, index])
        )

        current_enzyme = enzyme[:, index]

        current_substrate = substrate[index]

        current_oxygen = oxygen[index]

        qP = (
            qP_max
            * current_enzyme / (K_E + current_enzyme)
            * current_substrate / (K_SP + current_substrate)
            * current_oxygen / (K_OP + current_oxygen)
        )

        mean_specific_product_rate.append(
            np.mean(qP)
        )

    # -----------------------------------
    # 8. RETURN RESULTS
    # -----------------------------------

    return {

        "time": solution.t,

        "mRNA": mRNA,

        "enzyme": enzyme,

        "biomass": biomass,

        "substrate": substrate,

        "dissolved_oxygen": oxygen,

        "product": product,

        "mean_activation": np.array(mean_activation),

        "mean_enzyme": np.array(mean_enzyme),

        "mean_specific_product_rate": np.array(
            mean_specific_product_rate
        ),

        "OTR": kla * (
            oxygen_saturation - oxygen
        ),

        "OUR": qO2 * biomass * (
            oxygen / (K_DO + oxygen)
        ),

        "average_mRNA": np.mean(
            mRNA,
            axis=0
        )

    }