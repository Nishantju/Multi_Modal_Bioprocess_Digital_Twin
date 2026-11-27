import numpy as np
from scipy.integrate import solve_ivp


def multimodal_gene_expression_rhs(
    time,
    state,
    total_activation,
    alpha_m,
    delta_m,
    alpha_e,
    delta_e
):
    """
    Coupled mRNA and enzyme dynamics.

    dmRNA/dt = alpha_m * A_total - delta_m * mRNA

    dE/dt = alpha_e * mRNA - delta_e * E
    """

    mrna, enzyme = state

    dmrna_dt = (
        alpha_m * total_activation
        - delta_m * mrna
    )

    denzyme_dt = (
        alpha_e * mrna
        - delta_e * enzyme
    )

    return [dmrna_dt, denzyme_dt]


def simulate_multimodal_gene_expression(
    time,
    total_activation,
    alpha_m=2.0,
    delta_m=0.2,
    alpha_e=0.5,
    delta_e=0.1,
    initial_mRNA=0.2,
    initial_enzyme=0.1
):
    """
    Simulate mRNA and enzyme production driven by
    combined multi-modal activation.
    """

    time = np.asarray(time, dtype=float)
    total_activation = np.asarray(
        total_activation,
        dtype=float
    )

    if time.ndim != 1 or len(time) < 2:
        raise ValueError("Time must contain at least two points.")

    if total_activation.ndim != 1:
        raise ValueError("Activation must be a one-dimensional array.")

    if len(time) != len(total_activation):
        raise ValueError(
            "Time and activation arrays must have equal lengths."
        )

    if not np.all(np.isfinite(time)):
        raise ValueError("Time must contain finite values.")

    if not np.all(np.isfinite(total_activation)):
        raise ValueError("Activation must contain finite values.")

    if np.any(np.diff(time) <= 0):
        raise ValueError("Time must be strictly increasing.")

    if np.any(total_activation < 0) or np.any(total_activation > 1):
        raise ValueError("Activation must be between 0 and 1.")

    rates = [alpha_m, delta_m, alpha_e, delta_e]

    if not all(np.isfinite(value) and value > 0 for value in rates):
        raise ValueError("All gene-expression rate parameters must be positive.")

    if not np.isfinite(initial_mRNA) or initial_mRNA < 0:
        raise ValueError("Initial mRNA must be non-negative.")

    if not np.isfinite(initial_enzyme) or initial_enzyme < 0:
        raise ValueError("Initial enzyme must be non-negative.")

    def rhs(t, state):

        current_activation = np.interp(
            t,
            time,
            total_activation
        )

        return multimodal_gene_expression_rhs(
            t,
            state,
            current_activation,
            alpha_m,
            delta_m,
            alpha_e,
            delta_e
        )

    solution = solve_ivp(
        rhs,
        (time[0], time[-1]),
        [initial_mRNA, initial_enzyme],
        t_eval=time,
        method="RK45",
        rtol=1e-7,
        atol=1e-9
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return {
        "time": time,
        "total_activation": total_activation,
        "mRNA": solution.y[0],
        "enzyme": solution.y[1]
    }