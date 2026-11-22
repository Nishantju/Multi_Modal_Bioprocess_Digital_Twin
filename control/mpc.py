import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def fermentation_prediction_model(time, state, control, params):

    X, S, DO, P = state

    X = max(X, 0.0)
    S = max(S, 0.0)
    DO = max(DO, 0.0)

    light_intensity, kla = control

    # 1. Optogenetic activation

    light_response = (
        light_intensity ** params["hill"]
        /
        (
            params["half_light"] ** params["hill"]
            + light_intensity ** params["hill"]
            + 1e-12
        )
    )

    activation = (
        params["basal"]
        + (1 - params["basal"]) * light_response
    )

    # 2. Substrate and oxygen limitation

    substrate_factor = S / (params["Ks"] + S + 1e-12)

    oxygen_factor = DO / (params["KDO"] + DO + 1e-12)

    # 3. Biomass growth

    mu = (
        params["mu_max"]
        * substrate_factor
        * oxygen_factor
    )

    dX_dt = mu * X

    # 4. Product formation

    qP = (
        params["qP_max"]
        * activation
        * substrate_factor
        * oxygen_factor
    )

    dP_dt = qP * X

    # 5. Substrate consumption

    substrate_growth = mu * X / params["Yxs"]

    substrate_product = dP_dt / params["Yps"]

    dS_dt = -(substrate_growth + substrate_product)

    # 6. Oxygen transfer and uptake

    OTR = kla * (params["Csat"] - DO)

    OUR = params["qO2"] * X * oxygen_factor

    dDO_dt = OTR - OUR

    return [dX_dt, dS_dt, dDO_dt, dP_dt]


def predict_future(
    initial_state,
    control,
    params,
    prediction_horizon
):

    solution = solve_ivp(

        lambda t, y: fermentation_prediction_model(
            t, y, control, params
        ),

        [0, prediction_horizon],

        initial_state,

        t_eval=np.linspace(0, prediction_horizon, 30),

        method="RK45",

        rtol=1e-6,

        atol=1e-8
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y[:, -1], solution.y


def mpc_objective(control, state, params, prediction_horizon):

    light, kla = control

    predicted_state, trajectory = predict_future(
        state,
        control,
        params,
        prediction_horizon
    )

    predicted_product = predicted_state[3]

    # Product reward

    product_reward = predicted_product - state[3]

    # Operating burden proxies

    light_burden = (light / params["max_light"]) ** 2

    kla_burden = (kla / params["max_kla"]) ** 2

    # Oxygen limitation penalty

    predicted_oxygen = trajectory[2]

    oxygen_deficit = np.maximum(
        params["DO_target"] - predicted_oxygen,
        0
    )

    oxygen_penalty = np.mean(oxygen_deficit ** 2)

    # Total objective

    objective = (

        -product_reward

        + params["weight_light"] * light_burden

        + params["weight_kla"] * kla_burden

        + params["weight_oxygen"] * oxygen_penalty

    )

    return objective


def optimize_operating_conditions(
    state,
    params,
    prediction_horizon=2.0
):

    initial_guess = [

        50,

        50

    ]

    bounds = [

        (0, params["max_light"]),

        (params["min_kla"], params["max_kla"])

    ]

    result = minimize(

        mpc_objective,

        initial_guess,

        args=(state, params, prediction_horizon),

        method="SLSQP",

        bounds=bounds

    )

    if not result.success:
        raise RuntimeError(result.message)

    return {

        "light": result.x[0],

        "kla": result.x[1],

        "objective": result.fun,

        "predicted_state": predict_future(
            state,
            result.x,
            params,
            prediction_horizon
        )[0]

    }


def run_mpc(
    initial_state,
    params,
    duration=20,
    control_interval=0.5,
    prediction_horizon=2.0
):

    state = np.array(initial_state, dtype=float)

    time_history = []

    state_history = []

    light_history = []

    kla_history = []

    current_time = 0.0

    while current_time < duration:

        # Optimize current operating conditions

        result = optimize_operating_conditions(
            state,
            params,
            prediction_horizon
        )

        light = result["light"]

        kla = result["kla"]

        # Apply the selected controls for one interval

        step = min(control_interval, duration - current_time)

        _, trajectory = predict_future(
            state,
            [light, kla],
            params,
            step
        )

        state = trajectory[:, -1]

        state[0:2] = np.maximum(state[0:2], 0)

        state[2] = np.clip(
            state[2],
            0,
            params["Csat"]
        )

        state[3] = max(state[3], 0)

        current_time += step

        time_history.append(current_time)

        state_history.append(state.copy())

        light_history.append(light)

        kla_history.append(kla)

    state_history = np.array(state_history)

    return {

        "time": np.array(time_history),

        "biomass": state_history[:, 0],

        "substrate": state_history[:, 1],

        "oxygen": state_history[:, 2],

        "product": state_history[:, 3],

        "light": np.array(light_history),

        "kla": np.array(kla_history)

    }