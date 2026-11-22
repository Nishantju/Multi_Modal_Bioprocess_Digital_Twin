import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from models.integrated_bioreactor import (
    simulate_integrated_bioreactor
)

from control.reduced_order_model import (
    make_population_phases,
    predict_reduced_order
)

from control.mpc_gene_expression import (
    run_gene_expression_mpc
)


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Multi-Modal Bioprocess Digital Twin",
    page_icon="🧫",
    layout="wide"
)


# ==========================================================
# 2. CUSTOM STYLING
# ==========================================================

st.markdown("""
<style>

.main {
    background: #0b1220;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(148,163,184,0.22);
    padding: 14px 16px;
    border-radius: 12px;
}

.hero {
    padding: 1.4rem;
    border-radius: 16px;
    background: linear-gradient(110deg, #10243b, #153b4b);
    border: 1px solid #2b6570;
    margin-bottom: 1rem;
}

.small-note {
    color: #aab8ca;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. HEADER
# ==========================================================

st.markdown("""
<div class="hero">

<h1>🧫 Multi-Modal Bioprocess Digital Twin</h1>

<div style="font-size:1.08rem;color:#c6e7ef;">

Physical-field-mediated metabolic steering
through integrated biological simulation and predictive control

</div>

<div class="small-note">

MULTISCALE BIOPROCESS MODEL • DIGITAL TWIN RESEARCH PROTOTYPE

</div>

</div>
""", unsafe_allow_html=True)


st.warning(
    "Research prototype: model outputs are simulated using illustrative "
    "parameters. The application is not experimentally validated and "
    "does not establish the effectiveness of any physical-field actuator "
    "in a real fermentation process."
)


# ==========================================================
# 4. HELPER FUNCTION: FIXED-CONTROL SIMULATION
# ==========================================================

def simulate_fixed_control(
    initial_state,
    params,
    phases,
    duration,
    control_interval,
    light,
    kla
):

    state = np.array(initial_state, dtype=float)

    current_time = 0.0

    time_history = [current_time]

    state_history = [state.copy()]

    while current_time < duration:

        actual_interval = min(
            control_interval,
            duration - current_time
        )

        prediction = predict_reduced_order(

            initial_state=state,

            control=(light, kla),

            params=params,

            phases=phases,

            start_time=current_time,

            horizon=actual_interval,

            n_points=10

        )

        state = prediction["final_state"]

        current_time += actual_interval

        time_history.append(current_time)

        state_history.append(state.copy())

    state_history = np.array(state_history)

    return {

        "time": np.array(time_history),

        "biomass": state_history[:, 0],

        "substrate": state_history[:, 1],

        "dissolved_oxygen": state_history[:, 2],

        "product": state_history[:, 3],

        "mRNA": state_history[:, 4],

        "enzyme": state_history[:, 5]

    }


# ==========================================================
# 5. SIDEBAR: REACTOR CONFIGURATION
# ==========================================================

st.sidebar.header("⚙️ Reactor Configuration")

duration = st.sidebar.slider(
    "Simulation duration (hours)",
    5, 40, 20, 1
)

population_size = st.sidebar.slider(
    "Cell population size",
    10, 500, 100, 10
)

height = st.sidebar.number_input(
    "Reactor height (m)",
    0.5, 5.0, 2.0, 0.1
)

mean_depth = st.sidebar.slider(
    "Mean cell depth (m)",
    0.0,
    float(height),
    min(1.0, float(height)),
    0.05
)

movement_amplitude = st.sidebar.slider(
    "Cell movement amplitude (m)",
    0.0,
    float(min(mean_depth, height - mean_depth)),
    min(0.9, float(min(mean_depth, height - mean_depth))),
    0.05
)

movement_frequency = st.sidebar.slider(
    "Cell movement frequency (1/h)",
    0.01, 1.0, 0.2, 0.01
)


# ==========================================================
# 6. SIDEBAR: INITIAL CONDITIONS
# ==========================================================

st.sidebar.header("🧪 Initial Conditions")

initial_biomass = st.sidebar.number_input(
    "Initial biomass (g/L)",
    0.01, 5.0, 0.1, 0.01
)

initial_substrate = st.sidebar.number_input(
    "Initial substrate (g/L)",
    1.0, 100.0, 20.0, 1.0
)

initial_oxygen = st.sidebar.number_input(
    "Initial dissolved oxygen (g/L)",
    0.0001, 0.008, 0.007, 0.0001,
    format="%.4f"
)

initial_mRNA = st.sidebar.number_input(
    "Initial mRNA (model units)",
    0.0, 10.0, 0.2, 0.1
)

initial_enzyme = st.sidebar.number_input(
    "Initial enzyme (model units)",
    0.0, 10.0, 0.1, 0.1
)


# ==========================================================
# 7. SIDEBAR: OPTOGENETIC PARAMETERS
# ==========================================================

st.sidebar.header("💡 Optogenetic Parameters")

surface_intensity = st.sidebar.slider(
    "Surface light intensity",
    0.0, 200.0, 100.0, 5.0
)

attenuation_coefficient = st.sidebar.slider(
    "Optical attenuation coefficient (1/m)",
    0.1, 5.0, 2.0, 0.1
)

half_activation_intensity = st.sidebar.slider(
    "Half-activation intensity",
    1.0, 100.0, 30.0, 1.0
)

hill_coefficient = st.sidebar.slider(
    "Hill coefficient",
    1.0, 5.0, 2.0, 0.1
)

basal_activation = st.sidebar.slider(
    "Basal gene activation",
    0.0, 0.5, 0.1, 0.01
)


# ==========================================================
# 8. SIDEBAR: GENE EXPRESSION
# ==========================================================

st.sidebar.header("🧬 Gene Expression")

alpha_m = st.sidebar.slider(
    "mRNA production rate",
    0.1, 5.0, 2.0, 0.1
)

delta_m = st.sidebar.slider(
    "mRNA degradation rate (1/h)",
    0.01, 1.0, 0.2, 0.01
)

alpha_e = st.sidebar.slider(
    "Enzyme production rate",
    0.1, 2.0, 0.5, 0.1
)

delta_e = st.sidebar.slider(
    "Enzyme degradation rate (1/h)",
    0.01, 1.0, 0.1, 0.01
)


# ==========================================================
# 9. SIDEBAR: METABOLISM
# ==========================================================

st.sidebar.header("⚗️ Metabolic Parameters")

qP_max = st.sidebar.slider(
    "Maximum specific product rate",
    0.001, 0.2, 0.05, 0.001
)

K_E = st.sidebar.slider(
    "Enzyme saturation constant",
    0.1, 10.0, 2.0, 0.1
)

K_SP = st.sidebar.slider(
    "Product substrate constant",
    0.1, 10.0, 1.0, 0.1
)

K_OP = st.sidebar.slider(
    "Product oxygen constant",
    0.0001, 0.01, 0.001, 0.0001,
    format="%.4f"
)

Y_PS = st.sidebar.slider(
    "Product yield",
    0.1, 2.0, 0.8, 0.05
)


# ==========================================================
# 10. SIDEBAR: BIOMASS PARAMETERS
# ==========================================================

st.sidebar.header("🦠 Biomass Parameters")

mu_max = st.sidebar.slider(
    "Maximum growth rate (1/h)",
    0.05, 1.0, 0.5, 0.01
)

K_S = st.sidebar.slider(
    "Substrate saturation constant",
    0.1, 10.0, 1.0, 0.1
)

K_DO = st.sidebar.slider(
    "Oxygen saturation constant",
    0.0001, 0.01, 0.001, 0.0001,
    format="%.4f"
)

Y_XS = st.sidebar.slider(
    "Biomass yield",
    0.1, 1.0, 0.5, 0.05
)


# ==========================================================
# 11. SIDEBAR: OXYGEN TRANSFER
# ==========================================================

st.sidebar.header("💨 Oxygen Transfer")

kla = st.sidebar.slider(
    "Volumetric mass transfer coefficient kLa (1/h)",
    10.0, 200.0, 100.0, 5.0
)

oxygen_saturation = 0.008

qO2 = st.sidebar.slider(
    "Specific oxygen uptake rate",
    0.01, 1.0, 0.2, 0.01
)


# ==========================================================
# 12. SIDEBAR: MPC SETTINGS
# ==========================================================

st.sidebar.header("🤖 Predictive Control")

fixed_light = st.sidebar.slider(
    "Fixed-control light intensity",
    0.0, 100.0, 100.0, 5.0
)

fixed_kla = st.sidebar.slider(
    "Fixed-control kLa",
    30.0, 150.0, 100.0, 5.0
)

control_interval = st.sidebar.slider(
    "MPC control interval (h)",
    0.25, 2.0, 0.5, 0.25
)

prediction_horizon = st.sidebar.slider(
    "MPC prediction horizon (h)",
    1.0, 10.0, 6.0, 0.5
)

oxygen_target = st.sidebar.slider(
    "MPC oxygen target (g/L)",
    0.0005, 0.005, 0.002, 0.0001,
    format="%.4f"
)

weight_light = st.sidebar.slider(
    "Light operating penalty",
    0.0, 0.1, 0.02, 0.005
)

weight_kla = st.sidebar.slider(
    "kLa operating penalty",
    0.0, 0.1, 0.02, 0.005
)

weight_oxygen = st.sidebar.slider(
    "Oxygen deficit penalty",
    0.0, 0.2, 0.05, 0.005
)

weight_control_change = st.sidebar.slider(
    "Control movement penalty",
    0.0, 0.1, 0.01, 0.005
)

weight_late_stage_control = st.sidebar.slider(
    "Late-stage operating penalty",
    0.0, 0.1, 0.02, 0.005
)


# ==========================================================
# 13. MODEL PARAMETER DICTIONARIES
# ==========================================================

integrated_params = {

    "population_size": population_size,

    "duration": duration,

    "height": height,

    "mean_depth": mean_depth,

    "movement_amplitude": movement_amplitude,

    "movement_frequency": movement_frequency,

    "surface_intensity": surface_intensity,

    "attenuation_coefficient": attenuation_coefficient,

    "half_activation_intensity": half_activation_intensity,

    "hill_coefficient": hill_coefficient,

    "basal_activation": basal_activation,

    "alpha_m": alpha_m,

    "delta_m": delta_m,

    "alpha_e": alpha_e,

    "delta_e": delta_e,

    "qP_max": qP_max,

    "K_E": K_E,

    "K_SP": K_SP,

    "K_OP": K_OP,

    "Y_PS": Y_PS,

    "mu_max": mu_max,

    "K_S": K_S,

    "K_DO": K_DO,

    "Y_XS": Y_XS,

    "kla": kla,

    "oxygen_saturation": oxygen_saturation,

    "qO2": qO2,

    "initial_mRNA": initial_mRNA,

    "initial_enzyme": initial_enzyme,

    "initial_biomass": initial_biomass,

    "initial_substrate": initial_substrate,

    "initial_oxygen": initial_oxygen,

    "seed": 42

}


mpc_params = {

    "height": height,

    "mean_depth": mean_depth,

    "movement_amplitude": movement_amplitude,

    "movement_frequency": movement_frequency,

    "attenuation_coefficient": attenuation_coefficient,

    "half_activation_intensity": half_activation_intensity,

    "hill_coefficient": hill_coefficient,

    "basal_activation": basal_activation,

    "alpha_m": alpha_m,

    "delta_m": delta_m,

    "alpha_e": alpha_e,

    "delta_e": delta_e,

    "qP_max": qP_max,

    "K_E": K_E,

    "K_SP": K_SP,

    "K_OP": K_OP,

    "Y_PS": Y_PS,

    "mu_max": mu_max,

    "K_S": K_S,

    "K_DO": K_DO,

    "Y_XS": Y_XS,

    "oxygen_saturation": oxygen_saturation,

    "qO2": qO2,

    "max_light": 100.0,

    "min_kla": 30.0,

    "max_kla": 150.0,

    "oxygen_target": oxygen_target,

    "weight_light": weight_light,

    "weight_kla": weight_kla,

    "weight_oxygen": weight_oxygen,

    "weight_control_change": weight_control_change,

    "weight_late_stage_control": weight_late_stage_control

}


initial_state = [

    initial_biomass,

    initial_substrate,

    initial_oxygen,

    0.0,

    initial_mRNA,

    initial_enzyme

]


# ==========================================================
# 14. RUN SIMULATIONS
# ==========================================================

st.sidebar.divider()

run_simulation = st.sidebar.button(
    "▶ Run complete simulation",
    use_container_width=True,
    type="primary"
)


if run_simulation:

    try:

        with st.spinner(
            "Running integrated bioreactor and predictive-control simulations..."
        ):

            # ----------------------------------------------
            # A. Integrated population-level simulation
            # ----------------------------------------------

            integrated_results = simulate_integrated_bioreactor(
                **integrated_params
            )

            # ----------------------------------------------
            # B. Population phases for reduced-order model
            # ----------------------------------------------

            phases = make_population_phases(
                population_size=population_size,
                seed=42
            )

            # ----------------------------------------------
            # C. Fixed-control simulation
            # ----------------------------------------------

            fixed_results = simulate_fixed_control(

                initial_state=initial_state,

                params=mpc_params,

                phases=phases,

                duration=duration,

                control_interval=control_interval,

                light=fixed_light,

                kla=fixed_kla

            )

            # ----------------------------------------------
            # D. Dynamic MPC simulation
            # ----------------------------------------------

            mpc_results = run_gene_expression_mpc(

                initial_state=initial_state,

                params=mpc_params,

                phases=phases,

                duration=duration,

                control_interval=control_interval,

                prediction_horizon=prediction_horizon

            )

        # ----------------------------------------------
        # Save outputs for display and download
        # ----------------------------------------------

        st.session_state["integrated_results"] = integrated_results

        st.session_state["integrated_population_size"] = population_size

        st.session_state["fixed_results"] = fixed_results

        st.session_state["mpc_results"] = mpc_results

        st.session_state["simulation_params"] = {
            "duration": duration,
            "population_size": population_size
        }

        st.success("All simulations completed successfully.")

    except Exception as exc:

        st.error(f"Simulation failed: {exc}")

        st.stop()


# ==========================================================
# 15. DISPLAY INSTRUCTIONS
# ==========================================================

if "integrated_results" not in st.session_state:

    st.info(
        "Configure the reactor using the sidebar and click "
        "'Run complete simulation' to generate results."
    )

    st.stop()


integrated = st.session_state["integrated_results"]

if st.session_state.get("integrated_population_size") != population_size:
    st.warning(
        "Population size has changed. "
        "Click 'Run complete simulation' to generate updated results."
    )
    st.stop()

fixed = st.session_state["fixed_results"]

mpc = st.session_state["mpc_results"]


# ==========================================================
# 16. MAIN DASHBOARD TABS
# ==========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "📊 Bioreactor",

    "🧬 Gene Expression",

    "💨 Oxygen Transfer",

    "🤖 Predictive Control",

    "📁 Research Data"

])


# ==========================================================
# TAB 1: BIOREACTOR PERFORMANCE
# ==========================================================

with tab1:

    st.header("Integrated Bioreactor Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Final biomass",
        f"{integrated['biomass'][-1]:.3f} g/L"
    )

    c2.metric(
        "Final substrate",
        f"{integrated['substrate'][-1]:.3f} g/L"
    )

    c3.metric(
        "Final product",
        f"{integrated['product'][-1]:.3f} g/L"
    )

    c4.metric(
        "Final dissolved oxygen",
        f"{integrated['dissolved_oxygen'][-1]:.5f} g/L"
    )


    # ----------------------------------------------
    # Biomass, substrate and product
    # ----------------------------------------------

    fig = make_subplots(

        rows=2,

        cols=2,

        subplot_titles=(

            "Biomass growth",

            "Substrate consumption",

            "Product formation",

            "Oxygen concentration"

        )

    )

    fig.add_trace(

        go.Scatter(
            x=integrated["time"],
            y=integrated["biomass"],
            name="Biomass",
            line=dict(width=3)
        ),

        row=1,
        col=1

    )

    fig.add_trace(

        go.Scatter(
            x=integrated["time"],
            y=integrated["substrate"],
            name="Substrate",
            line=dict(width=3)
        ),

        row=1,
        col=2

    )

    fig.add_trace(

        go.Scatter(
            x=integrated["time"],
            y=integrated["product"],
            name="Product",
            line=dict(width=3)
        ),

        row=2,
        col=1

    )

    fig.add_trace(

        go.Scatter(
            x=integrated["time"],
            y=integrated["dissolved_oxygen"],
            name="Dissolved oxygen",
            line=dict(width=3)
        ),

        row=2,
        col=2

    )

    fig.update_layout(

        height=650,

        template="plotly_dark",

        showlegend=False

    )

    fig.update_xaxes(title_text="Time (hours)")

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ----------------------------------------------
    # Population and reactor summary
    # ----------------------------------------------

    st.subheader("Final simulation summary")

    summary = pd.DataFrame({

        "Parameter": [

            "Biomass",

            "Substrate",

            "Dissolved oxygen",

            "Product",

            "Average mRNA",

            "Average enzyme",

            "Mean gene activation"

        ],

        "Final value": [

            integrated["biomass"][-1],

            integrated["substrate"][-1],

            integrated["dissolved_oxygen"][-1],

            integrated["product"][-1],

            integrated["average_mRNA"][-1],

            integrated["mean_enzyme"][-1],

            integrated["mean_activation"][-1]

        ]

    })

    st.dataframe(
        summary.style.format({"Final value": "{:.6f}"}),
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# TAB 2: GENE EXPRESSION
# ==========================================================

with tab2:

    st.header("Population-Level Genetic Dynamics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Mean gene activation",
        f"{integrated['mean_activation'][-1]:.3f}"
    )

    c2.metric(
        "Final mean mRNA",
        f"{integrated['average_mRNA'][-1]:.3f}"
    )

    c3.metric(
        "Final mean enzyme",
        f"{integrated['mean_enzyme'][-1]:.3f}"
    )


    # ----------------------------------------------
    # Gene expression dynamics
    # ----------------------------------------------

    fig = make_subplots(

        rows=1,

        cols=2,

        subplot_titles=(

            "mRNA dynamics",

            "Enzyme dynamics"

        )

    )

    fig.add_trace(

        go.Scatter(
            x=integrated["time"],
            y=integrated["average_mRNA"],
            name="Mean mRNA",
            line=dict(width=3)
        ),

        row=1,
        col=1

    )

    fig.add_trace(

        go.Scatter(
            x=integrated["time"],
            y=integrated["mean_enzyme"],
            name="Mean enzyme",
            line=dict(width=3)
        ),

        row=1,
        col=2

    )

    fig.update_layout(
        height=430,
        template="plotly_dark"
    )

    fig.update_xaxes(title_text="Time (hours)")

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ----------------------------------------------
    # Individual cell enzyme distribution
    # ----------------------------------------------

    enzyme_matrix = integrated["enzyme"]

    final_enzyme = enzyme_matrix[:, -1]

    st.subheader("Individual-cell enzyme distribution")

    enzyme_fig = go.Figure()

    enzyme_fig.add_trace(

        go.Histogram(

            x=final_enzyme,

            nbinsx=25,

            name="Cell enzyme levels"

        )

    )

    enzyme_fig.update_layout(

        template="plotly_dark",

        height=380,

        xaxis_title="Enzyme concentration (model units)",

        yaxis_title="Number of cells"

    )

    st.plotly_chart(
        enzyme_fig,
        use_container_width=True
    )


    st.write(
        "Population enzyme standard deviation:",
        f"{np.std(final_enzyme):.4f}"
    )


    # ----------------------------------------------
    # Gene activation
    # ----------------------------------------------

    activation_fig = go.Figure()

    activation_fig.add_trace(

        go.Scatter(

            x=integrated["time"],

            y=integrated["mean_activation"],

            name="Mean activation",

            fill="tozeroy"

        )

    )

    activation_fig.update_layout(

        template="plotly_dark",

        height=350,

        xaxis_title="Time (hours)",

        yaxis_title="Normalized activation"

    )

    st.plotly_chart(
        activation_fig,
        use_container_width=True
    )


# ==========================================================
# TAB 3: OXYGEN TRANSFER
# ==========================================================

with tab3:

    st.header("Oxygen Transfer and Consumption")

    final_otr = integrated["OTR"][-1]

    final_our = integrated["OUR"][-1]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Final OTR",
        f"{final_otr:.5f} g/L/h"
    )

    c2.metric(
        "Final OUR",
        f"{final_our:.5f} g/L/h"
    )

    c3.metric(
        "Final oxygen",
        f"{integrated['dissolved_oxygen'][-1]:.5f} g/L"
    )


    # ----------------------------------------------
    # OTR and OUR
    # ----------------------------------------------

    oxygen_fig = go.Figure()

    oxygen_fig.add_trace(

        go.Scatter(

            x=integrated["time"],

            y=integrated["OTR"],

            name="Oxygen transfer rate",

            line=dict(width=3)

        )

    )

    oxygen_fig.add_trace(

        go.Scatter(

            x=integrated["time"],

            y=integrated["OUR"],

            name="Oxygen uptake rate",

            line=dict(width=3)

        )

    )

    oxygen_fig.update_layout(

        template="plotly_dark",

        height=420,

        xaxis_title="Time (hours)",

        yaxis_title="Oxygen rate (g/L/h)"

    )

    st.plotly_chart(
        oxygen_fig,
        use_container_width=True
    )


    # ----------------------------------------------
    # Oxygen balance explanation
    # ----------------------------------------------

    st.latex(
        r"\\frac{dDO}{dt}=k_La(C^*-DO)-q_{O_2}Xf_{O_2}"
    )

    st.markdown("""
    **Interpretation**

    - OTR represents oxygen transferred from gas to liquid.
    - OUR represents oxygen consumed by microorganisms.
    - When OTR exceeds OUR, dissolved oxygen increases.
    - When OUR exceeds OTR, dissolved oxygen decreases.
    - At steady state, OTR approximately equals OUR.
    """)


# ==========================================================
# TAB 4: PREDICTIVE CONTROL
# ==========================================================

with tab4:

    st.header("Model Predictive Control")

    st.markdown("""
    The MPC controller optimizes light intensity and kLa
    using a reduced-order prediction model.
    """)


    # ----------------------------------------------
    # Final performance comparison
    # ----------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Fixed-control product",
        f"{fixed['product'][-1]:.4f} g/L"
    )

    c2.metric(
        "Dynamic MPC product",
        f"{mpc['product'][-1]:.4f} g/L"
    )

    improvement = (

        (mpc["product"][-1] / max(fixed["product"][-1], 1e-12) - 1)
        * 100

    )

    c3.metric(
        "MPC product difference",
        f"{improvement:+.2f}%"
    )


    # ----------------------------------------------
    # Product comparison
    # ----------------------------------------------

    product_fig = go.Figure()

    product_fig.add_trace(

        go.Scatter(

            x=fixed["time"],

            y=fixed["product"],

            name="Fixed control",

            line=dict(width=3, dash="dash")

        )

    )

    product_fig.add_trace(

        go.Scatter(

            x=mpc["time"],

            y=mpc["product"],

            name="Dynamic MPC",

            line=dict(width=3)

        )

    )

    product_fig.update_layout(

        template="plotly_dark",

        height=400,

        xaxis_title="Time (hours)",

        yaxis_title="Product concentration (g/L)"

    )

    st.plotly_chart(
        product_fig,
        use_container_width=True
    )


    # ----------------------------------------------
    # Biomass and oxygen comparison
    # ----------------------------------------------

    comparison_fig = make_subplots(

        rows=1,

        cols=2,

        subplot_titles=(

            "Biomass comparison",

            "Dissolved oxygen comparison"

        )

    )

    for result, name in [
        (fixed, "Fixed control"),
        (mpc, "Dynamic MPC")
    ]:

        comparison_fig.add_trace(

            go.Scatter(
                x=result["time"],
                y=result["biomass"],
                name=name
            ),

            row=1,
            col=1

        )

        comparison_fig.add_trace(

            go.Scatter(
                x=result["time"],
                y=result["dissolved_oxygen"],
                name=name,
                showlegend=False
            ),

            row=1,
            col=2

        )

    comparison_fig.update_layout(

        template="plotly_dark",

        height=420

    )

    comparison_fig.update_xaxes(
        title_text="Time (hours)"
    )

    st.plotly_chart(
        comparison_fig,
        use_container_width=True
    )


    # ----------------------------------------------
    # MPC control inputs
    # ----------------------------------------------

    control_fig = make_subplots(

        rows=1,

        cols=2,

        subplot_titles=(

            "Optimized light intensity",

            "Optimized kLa"

        )

    )

    control_fig.add_trace(

        go.Scatter(

            x=mpc["time"][1:],

            y=mpc["light"],

            name="MPC light",

            mode="lines+markers",

            line=dict(shape="hv")

        ),

        row=1,
        col=1

    )

    control_fig.add_trace(

        go.Scatter(

            x=mpc["time"][1:],

            y=mpc["kla"],

            name="MPC kLa",

            mode="lines+markers",

            line=dict(shape="hv")

        ),

        row=1,
        col=2

    )

    control_fig.update_layout(

        template="plotly_dark",

        height=420,

        showlegend=False

    )

    control_fig.update_xaxes(
        title_text="Time (hours)"
    )

    st.plotly_chart(
        control_fig,
        use_container_width=True
    )


    # ----------------------------------------------
    # Comparison table
    # ----------------------------------------------

    st.subheader("Control strategy comparison")

    comparison = pd.DataFrame({

        "Metric": [

            "Final biomass (g/L)",

            "Final substrate (g/L)",

            "Final dissolved oxygen (g/L)",

            "Final product (g/L)",

            "Final mRNA (model units)",

            "Final enzyme (model units)"

        ],

        "Fixed control": [

            fixed["biomass"][-1],

            fixed["substrate"][-1],

            fixed["dissolved_oxygen"][-1],

            fixed["product"][-1],

            fixed["mRNA"][-1],

            fixed["enzyme"][-1]

        ],

        "Dynamic MPC": [

            mpc["biomass"][-1],

            mpc["substrate"][-1],

            mpc["dissolved_oxygen"][-1],

            mpc["product"][-1],

            mpc["mRNA"][-1],

            mpc["enzyme"][-1]

        ]

    })

    st.dataframe(

        comparison.style.format({

            "Fixed control": "{:.5f}",

            "Dynamic MPC": "{:.5f}"

        }),

        use_container_width=True,

        hide_index=True

    )


    st.info(
        "MPC and fixed-control comparisons use the same reduced-order "
        "model and initial state. They are not direct experimental "
        "performance comparisons. The MPC operating penalties are "
        "illustrative, not measured energy costs."
    )


# ==========================================================
# TAB 5: RESEARCH DATA AND EXPORT
# ==========================================================

with tab5:

    st.header("Research Data and Export")

    # ----------------------------------------------
    # Integrated model data
    # ----------------------------------------------

    integrated_df = pd.DataFrame({

        "Time (h)": integrated["time"],

        "Biomass (g/L)": integrated["biomass"],

        "Substrate (g/L)": integrated["substrate"],

        "Dissolved oxygen (g/L)": integrated["dissolved_oxygen"],

        "Product (g/L)": integrated["product"],

        "Mean mRNA": integrated["average_mRNA"],

        "Mean enzyme": integrated["mean_enzyme"],

        "Mean activation": integrated["mean_activation"],

        "OTR (g/L/h)": integrated["OTR"],

        "OUR (g/L/h)": integrated["OUR"]

    })


    st.subheader("Integrated bioreactor data")

    st.dataframe(
        integrated_df.head(15),
        use_container_width=True,
        hide_index=True
    )


    st.download_button(
        "⬇️ Download integrated simulation CSV",
        integrated_df.to_csv(index=False).encode("utf-8"),
        file_name="integrated_bioreactor_results.csv",
        mime="text/csv",
        key="download_integrated_results"
    )


    # ----------------------------------------------
    # MPC data
    # ----------------------------------------------

    mpc_df = pd.DataFrame({

        "Time (h)": mpc["time"],

        "Biomass": mpc["biomass"],

        "Substrate": mpc["substrate"],

        "Dissolved oxygen": mpc["dissolved_oxygen"],

        "Product": mpc["product"],

        "mRNA": mpc["mRNA"],

        "Enzyme": mpc["enzyme"]

    })

    st.subheader("Predictive-control data")

    st.dataframe(
        mpc_df.head(15),
        use_container_width=True,
        hide_index=True
    )



    # ----------------------------------------------
    # Export individual-cell data
    # ----------------------------------------------

    # ----------------------------------------------
    # Export individual-cell enzyme data
    # ----------------------------------------------

    enzyme_matrix = integrated["enzyme"]

    # Get the actual population size from simulation results
    actual_population_size = enzyme_matrix.shape[0]

    # Create column names using the actual simulated population
    cell_df = pd.DataFrame(

        enzyme_matrix.T,

        columns=[
            f"Cell_{i+1}"
            for i in range(actual_population_size)
        ]

    )

    cell_df.insert(
        0,
        "Time (h)",
        integrated["time"]
    )

    st.download_button(
        "⬇️ Download individual-cell enzyme data",
        cell_df.to_csv(index=False).encode("utf-8"),
        file_name="individual_cell_enzyme_data.csv",
        mime="text/csv",
        key="download_cell_data"
    )



# ==========================================================
# 17. SCIENTIFIC SCOPE AND LIMITATIONS
# ==========================================================

st.divider()

with st.expander("Model equations and scientific assumptions"):

    st.subheader("1. Biomass Growth")

    st.latex(r"""
    \frac{dX}{dt}
    =
    \mu_{\max}
    \frac{S}{K_S+S}
    \frac{DO}{K_{DO}+DO}X
    """)

    st.subheader("2. Gene Expression")

    st.latex(r"""
    \frac{dmRNA_i}{dt}
    =
    \alpha_m A_i-\delta_m mRNA_i
    """)

    st.latex(r"""
    \frac{dE_i}{dt}
    =
    \alpha_e mRNA_i-\delta_e E_i
    """)

    st.subheader("3. Optical Attenuation")

    st.latex(r"""
    I_i(t)=I_0 e^{-\kappa z_i(t)}
    """)

    st.subheader("4. Product Formation")

    st.latex(r"""
    \frac{dP}{dt}
    =
    X\overline{q_P}
    """)

    st.subheader("5. Oxygen Balance")

    st.latex(r"""
    \frac{dDO}{dt}
    =
    k_La(C^*-DO)-q_{O_2}Xf_{O_2}
    """)


    st.markdown("""
    **Current implementation**

    - Population-level optogenetic activation.
    - Individual-cell mRNA and enzyme dynamics.
    - Shared batch-fermentation environment.
    - Monod-based biomass growth.
    - Enzyme-dependent product formation.
    - Dynamic dissolved-oxygen balance.
    - Reduced-order predictive-control model.

    **Not yet fully implemented**

    - Electrogenetic dynamic actuator model.
    - Sonogenetic dynamic actuator model.
    - Magnetogenetic dynamic actuator model.
    - Experimental calibration and validation.
    - CFD-derived spatial mixing.
    - Mechanistic metabolic network.
    - Hardware-connected sensor feedback.

    **Important:** The MPC model uses average mRNA and enzyme states.
    It is an approximation of the individual-cell population model.
    """)


st.caption(
    "Multi-Modal Bioprocess Digital Twin | Computational research prototype | "
    "Batch fermentation | Simulated results only"
)