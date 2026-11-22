import numpy as np
from scipy.integrate import solve_ivp


def simulate_population_opto(
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
    initial_mRNA=0.2,
    initial_protein=0.1,
    alpha_m=2.0,
    delta_m=0.2,
    alpha_e=0.5,
    delta_e=0.1,
    seed=42
):

    rng = np.random.default_rng(seed)

    # Generate a different starting movement phase
    # for every microorganism.
    phases = rng.uniform(
        0,
        2 * np.pi,
        population_size
    )

    # Check that movement stays inside the reactor.
    if movement_amplitude > min(
        mean_depth,
        height - mean_depth
    ):
        raise ValueError(
            "Movement amplitude exceeds reactor boundaries."
        )

    def calculate_population_depth(time):

        depth = mean_depth + movement_amplitude * np.sin(
            2 * np.pi * movement_frequency * time + phases
        )

        return depth

    def calculate_population_activation(time):

        depth = calculate_population_depth(time)

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

    def population_model(time, state):

        mRNA = state[:population_size]
        protein = state[population_size:]

        activation, _, _ = calculate_population_activation(
            time
        )

        dmRNA_dt = (
            alpha_m * activation
            - delta_m * mRNA
        )

        dProtein_dt = (
            alpha_e * mRNA
            - delta_e * protein
        )

        return np.concatenate([
            dmRNA_dt,
            dProtein_dt
        ])

    initial_state = np.concatenate([
        np.full(population_size, initial_mRNA),
        np.full(population_size, initial_protein)
    ])

    time_points = np.linspace(0, duration, 300)

    solution = solve_ivp(
        population_model,
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
    protein = solution.y[population_size:]

    depths = []
    intensities = []
    activations = []

    for time in solution.t:

        activation, intensity, depth = (
            calculate_population_activation(time)
        )

        depths.append(depth)
        intensities.append(intensity)
        activations.append(activation)

    depths = np.array(depths).T
    intensities = np.array(intensities).T
    activations = np.array(activations).T

    return {

        "time": solution.t,

        "depth": depths,
        "light_intensity": intensities,
        "activation": activations,

        "mRNA": mRNA,
        "protein": protein,

        "mean_activation": np.mean(
            activations, axis=0
        ),

        "mean_mRNA": np.mean(
            mRNA, axis=0
        ),

        "mean_protein": np.mean(
            protein, axis=0
        ),

        "std_activation": np.std(
            activations, axis=0
        ),

        "std_protein": np.std(
            protein, axis=0
        ),

        "protein_percentiles": np.percentile(
            protein,
            [10, 50, 90],
            axis=0
        )
    }