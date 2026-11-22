import numpy as np
from scipy.integrate import solve_ivp


def simulate_opto_metabolism(
    population_size=100,
    duration=10,
    height=2.0,
    mean_depth=1.0,
    movement_amplitude=0.9,
    movement_frequency=0.2,

    surface_intensity=100,
    attenuation_coefficient=2,
    half_activation_intensity=30,
    hill_coefficient=2,
    basal_activation=0.1,

    alpha_m=2.0,
    delta_m=0.2,
    alpha_e=0.5,
    delta_e=0.1,

    qP_max=0.5,
    K_E=2.0,

    initial_mRNA=0.2,
    initial_enzyme=0.1,

    seed=42
):

    if population_size < 1:
        raise ValueError("Population size must be positive.")

    if not (0 <= mean_depth <= height):
        raise ValueError("Mean depth must be inside the reactor.")

    if movement_amplitude < 0 or movement_amplitude > min(
        mean_depth, height - mean_depth
    ):
        raise ValueError("Movement exceeds reactor boundaries.")

    if attenuation_coefficient < 0:
        raise ValueError("Attenuation cannot be negative.")

    if half_activation_intensity <= 0 or hill_coefficient <= 0:
        raise ValueError("Hill parameters must be positive.")

    if K_E <= 0:
        raise ValueError("K_E must be positive.")

    if min(alpha_m, delta_m, alpha_e, delta_e, qP_max) < 0:
        raise ValueError("Kinetic parameters cannot be negative.")

    if initial_mRNA < 0 or initial_enzyme < 0:
        raise ValueError("Initial concentrations cannot be negative.")

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

    def model(time, state):

        mRNA = state[:population_size]

        enzyme = state[
            population_size:2 * population_size
        ]

        product = state[
            2 * population_size:
        ]

        activation, _, _ = calculate_activation(time)

        dmRNA_dt = (
            alpha_m * activation
            - delta_m * mRNA
        )

        denzyme_dt = (
            alpha_e * mRNA
            - delta_e * enzyme
        )

        enzyme_activity = enzyme / (K_E + enzyme)

        dproduct_dt = qP_max * enzyme_activity

        return np.concatenate([
            dmRNA_dt,
            denzyme_dt,
            dproduct_dt
        ])

    initial_state = np.concatenate([
        np.full(population_size, initial_mRNA),
        np.full(population_size, initial_enzyme),
        np.zeros(population_size)
    ])

    time_points = np.linspace(0, duration, 300)

    solution = solve_ivp(
        model,
        [0, duration],
        initial_state,
        t_eval=time_points,
        method="RK45",
        rtol=1e-7,
        atol=1e-9
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    mRNA = solution.y[:population_size]

    enzyme = solution.y[
        population_size:2 * population_size
    ]

    product = solution.y[
        2 * population_size:
    ]

    activation_history = []
    intensity_history = []
    depth_history = []

    for time in solution.t:

        activation, intensity, depth = (
            calculate_activation(time)
        )

        activation_history.append(activation)
        intensity_history.append(intensity)
        depth_history.append(depth)

    activation_history = np.array(
        activation_history
    ).T

    intensity_history = np.array(
        intensity_history
    ).T

    depth_history = np.array(
        depth_history
    ).T

    return {

        "time": solution.t,

        "depth": depth_history,

        "light_intensity": intensity_history,

        "activation": activation_history,

        "mRNA": mRNA,

        "enzyme": enzyme,

        "product": product,

        "mean_activation": np.mean(
            activation_history,
            axis=0
        ),

        "mean_enzyme": np.mean(
            enzyme,
            axis=0
        ),

        "mean_product": np.mean(
            product,
            axis=0
        ),

        "total_product": np.sum(
            product,
            axis=0
        ),

        "product_std": np.std(
            product,
            axis=0
        )
    }