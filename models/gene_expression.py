import numpy as np


def gene_expression_dynamics(
    mRNA,
    protein,
    activation,
    alpha_m,
    delta_m,
    alpha_e,
    delta_e
):

    """
    Dynamic gene expression model.

    dmRNA/dt = alpha_m * activation - delta_m * mRNA

    dProtein/dt = alpha_e * mRNA - delta_e * protein
    """

    # Transcription
    dmRNA_dt = (
        alpha_m * activation
        - delta_m * mRNA
    )

    # Translation
    dProtein_dt = (
        alpha_e * mRNA
        - delta_e * protein
    )

    return dmRNA_dt, dProtein_dt


def calculate_gene_activation(
    field_signal,
    basal_activation
):

    """
    Calculate normalized gene activation.

    Both inputs are assumed to be between 0 and 1.
    """

    field_signal = np.clip(field_signal, 0, 1)

    basal_activation = np.clip(
        basal_activation,
        0,
        1
    )

    activation = (
        basal_activation
        + (1 - basal_activation) * field_signal
    )

    return activation


from scipy.integrate import solve_ivp
import numpy as np


def simulate_gene_expression(
    initial_mRNA,
    initial_protein,
    duration,
    switch_time,
    field_signal,
    basal_activation,
    alpha_m,
    delta_m,
    alpha_e,
    delta_e
):

    def model(time, state):

        mRNA, protein = state

        # Switch field signal on at the specified time
        if time < switch_time:
            current_field = 0.0
        else:
            current_field = field_signal

        # Calculate gene activation
        activation = calculate_gene_activation(
            current_field,
            basal_activation
        )

        # Calculate gene expression rates
        dmRNA_dt, dProtein_dt = gene_expression_dynamics(
            mRNA,
            protein,
            activation,
            alpha_m,
            delta_m,
            alpha_e,
            delta_e
        )

        return [dmRNA_dt, dProtein_dt]

    time_points = np.linspace(0, duration, 300)

    solution = solve_ivp(
        model,
        [0, duration],
        [initial_mRNA, initial_protein],
        t_eval=time_points,
        method="RK45",
        rtol=1e-7,
        atol=1e-9
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return {
        "time": solution.t,
        "mRNA": solution.y[0],
        "protein": solution.y[1]
    }
