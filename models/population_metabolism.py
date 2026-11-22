import numpy as np
from scipy.integrate import solve_ivp


def simulate_population_metabolism(
    population_size=100,
    duration=10,

    # Gene expression parameters
    alpha_m=2.0,
    delta_m=0.2,
    alpha_e=0.5,
    delta_e=0.1,

    # Metabolic parameters
    qP_max=0.5,
    K_E=2.0,

    # Initial conditions
    initial_mRNA=0.2,
    initial_enzyme=0.1,
    initial_product=0.0
):

    def model(time, state):

        mRNA = state[:population_size]

        enzyme = state[
            population_size:2 * population_size
        ]

        product = state[
            2 * population_size:
        ]

        # Illustrative time-varying activation
        activation = (
            0.1 + 0.9 * (1 - np.exp(-0.3 * time))
        )

        # Gene expression dynamics
        dmRNA_dt = (
            alpha_m * activation
            - delta_m * mRNA
        )

        # Enzyme production
        denzyme_dt = (
            alpha_e * mRNA
            - delta_e * enzyme
        )

        # Enzyme-dependent metabolic activity
        enzyme_activity = (
            enzyme / (K_E + enzyme)
        )

        # Product formation
        dproduct_dt = (
            qP_max * enzyme_activity
        )

        return np.concatenate([
            dmRNA_dt,
            denzyme_dt,
            dproduct_dt
        ])

    initial_state = np.concatenate([

        np.full(population_size, initial_mRNA),

        np.full(population_size, initial_enzyme),

        np.full(population_size, initial_product)

    ])

    time_points = np.linspace(
        0,
        duration,
        300
    )

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

    mRNA = solution.y[
        :population_size
    ]

    enzyme = solution.y[
        population_size:2 * population_size
    ]

    product = solution.y[
        2 * population_size:
    ]

    return {

        "time": solution.t,

        "mRNA": mRNA,

        "enzyme": enzyme,

        "product": product,

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
        )
    }