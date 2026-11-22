import numpy as np

from scipy.optimize import minimize

from control.reduced_order_model import (
    predict_reduced_order
)


# --------------------------------------------------
# MPC objective function
# --------------------------------------------------

def gene_expression_mpc_objective(

    control,
    state,
    params,
    phases,
    current_time,
    prediction_horizon,
    previous_control=None

):

    light, kla = control

    # ----------------------------------------------
    # Predict future biological behaviour
    # ----------------------------------------------

    prediction = predict_reduced_order(

        initial_state=state,

        control=(light, kla),

        params=params,

        phases=phases,

        start_time=current_time,

        horizon=prediction_horizon,

        n_points=10

    )

    trajectory = prediction["trajectory"]

    predicted_state = prediction["final_state"]


    # ----------------------------------------------
    # 1. Product formation reward
    # ----------------------------------------------

    product_gain = (
        predicted_state[3] - state[3]
    )


    # ----------------------------------------------
    # 2. Operating burden
    # ----------------------------------------------

    light_burden = (

        light / params["max_light"]

    ) ** 2


    kla_burden = (

        kla / params["max_kla"]

    ) ** 2


    # ----------------------------------------------
    # 3. Oxygen limitation penalty
    # ----------------------------------------------

    oxygen_target = params["oxygen_target"]

    oxygen_deficit = np.maximum(

        oxygen_target - trajectory[2],

        0

    )

    oxygen_penalty = np.mean(

        (oxygen_deficit / oxygen_target) ** 2

    )


    # ----------------------------------------------
    # 4. Control movement penalty
    # ----------------------------------------------

    control_change_penalty = 0.0

    if previous_control is not None:

        light_change = (

            (light - previous_control[0])
            / params["max_light"]

        ) ** 2

        kla_change = (

            (kla - previous_control[1])
            / params["max_kla"]

        ) ** 2

        control_change_penalty = (

            light_change + kla_change

        )


    # ----------------------------------------------
    # 5. Substrate-aware operating penalty
    # ----------------------------------------------

    initial_substrate = max(

        state[1],

        1e-9

    )

    predicted_substrate = np.clip(

        trajectory[1],

        0,

        initial_substrate

    )

    substrate_depletion_fraction = (

        1 - predicted_substrate / initial_substrate

    )

    average_substrate_depletion = np.mean(

        substrate_depletion_fraction

    )

    late_stage_control_penalty = (

        average_substrate_depletion

        * (

            light_burden + kla_burden

        )

    )


    # ----------------------------------------------
    # 6. Total optimization objective
    # ----------------------------------------------

    objective = (

        -product_gain

        + params["weight_light"] * light_burden

        + params["weight_kla"] * kla_burden

        + params["weight_oxygen"] * oxygen_penalty

        + params["weight_control_change"]
        * control_change_penalty

        + params.get("weight_late_stage_control", 0.02)
        * late_stage_control_penalty

    )

    return objective


# --------------------------------------------------
# Optimize operating conditions
# --------------------------------------------------

def optimize_gene_expression_control(

    state,
    params,
    phases,
    current_time,
    prediction_horizon,
    previous_control=None

):

    if previous_control is not None:

        initial_guess = list(previous_control)

    else:

        initial_guess = [50.0, 50.0]


    bounds = [

        (0, params["max_light"]),

        (params["min_kla"], params["max_kla"])

    ]


    result = minimize(

        gene_expression_mpc_objective,

        x0=initial_guess,

        args=(

            state,

            params,

            phases,

            current_time,

            prediction_horizon,

            previous_control

        ),

        method="SLSQP",

        bounds=bounds,

        options={

            "maxiter": 15,

            "ftol": 1e-7

        }

    )


    if not result.success:

        raise RuntimeError(

            "MPC optimization failed: "

            + result.message

        )


    optimized_light = result.x[0]

    optimized_kla = result.x[1]


    return {

        "light": optimized_light,

        "kla": optimized_kla,

        "objective": result.fun,

        "success": result.success

    }


# --------------------------------------------------
# Run receding-horizon MPC
# --------------------------------------------------

def run_gene_expression_mpc(

    initial_state,

    params,

    phases,

    duration=5,

    control_interval=0.5,

    prediction_horizon=2

):

    state = np.array(

        initial_state,

        dtype=float

    )

    current_time = 0.0

    time_history = [current_time]

    state_history = [state.copy()]

    light_history = []

    kla_history = []

    objective_history = []

    previous_control = None


    while current_time < duration:


        # ------------------------------------------
        # 1. Optimize current control inputs
        # ------------------------------------------

        control_result = optimize_gene_expression_control(

            state=state,

            params=params,

            phases=phases,

            current_time=current_time,

            prediction_horizon=prediction_horizon,

            previous_control=previous_control

        )


        light = control_result["light"]

        kla = control_result["kla"]

        control = (light, kla)


        # ------------------------------------------
        # 2. Apply optimized control
        # ------------------------------------------

        actual_interval = min(

            control_interval,

            duration - current_time

        )


        process_prediction = predict_reduced_order(

            initial_state=state,

            control=control,

            params=params,

            phases=phases,

            start_time=current_time,

            horizon=actual_interval,

            n_points=10

        )


        # ------------------------------------------
        # 3. Update process state
        # ------------------------------------------

        state = process_prediction["final_state"]

        current_time += actual_interval


        # ------------------------------------------
        # 4. Store results
        # ------------------------------------------

        time_history.append(current_time)

        state_history.append(state.copy())

        light_history.append(light)

        kla_history.append(kla)

        objective_history.append(

            control_result["objective"]

        )

        previous_control = control


    state_history = np.array(state_history)


    return {

        "time": np.array(time_history),

        "biomass": state_history[:, 0],

        "substrate": state_history[:, 1],

        "dissolved_oxygen": state_history[:, 2],

        "product": state_history[:, 3],

        "mRNA": state_history[:, 4],

        "enzyme": state_history[:, 5],

        "light": np.array(light_history),

        "kla": np.array(kla_history),

        "objective": np.array(objective_history)

    }