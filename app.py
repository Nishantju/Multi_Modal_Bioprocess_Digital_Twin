import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from models.integrated_bioreactor import (
    simulate_integrated_bioreactor
)

from models.multimodal_bioreactor import (
    simulate_multimodal_bioreactor
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
# 13. SIDEBAR: MULTI-MODAL PHYSICAL-FIELD CONTROL
# ==========================================================

st.sidebar.divider()

st.sidebar.header("🔬 Multi-Modal Physical-Field Control")

st.sidebar.subheader("💡 Optogenetics")
multimodal_light = st.sidebar.slider(
    "Multi-modal light intensity",
    0.0, 200.0, 100.0, 5.0
)

st.sidebar.subheader("⚡ Electrogenetics")

multimodal_electric = st.sidebar.slider(
    "Electric field magnitude",
    0.0, 20.0, 8.0, 0.5
)

st.sidebar.subheader("🔊 Sonogenetics")

multimodal_ultrasound = st.sidebar.slider(
    "Ultrasound intensity",
    0.0, 20.0, 8.0, 0.5
)

st.sidebar.subheader("🧲 Magnetogenetics")

multimodal_magnetic = st.sidebar.slider(
    "Magnetic field magnitude",
    0.0, 20.0, 8.0, 0.5
)

st.sidebar.subheader("⏱️ Actuator response times")

tau_opto = st.sidebar.slider(
    "Optogenetic response time (h)",
    0.1, 5.0, 0.5, 0.1
)

tau_electro = st.sidebar.slider(
    "Electrogenetic response time (h)",
    0.1, 5.0, 0.5, 0.1
)

tau_sono = st.sidebar.slider(
    "Sonogenetic response time (h)",
    0.1, 5.0, 0.5, 0.1
)

tau_magneto = st.sidebar.slider(
    "Magnetogenetic response time (h)",
    0.1, 5.0, 0.5, 0.1
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

multimodal_params = {

    "K_light": 30.0,
    "K_electric": 5.0,
    "K_ultrasound": 5.0,
    "K_magnetic": 5.0,

    "hill_opto": 2.0,
    "hill_electro": 2.0,
    "hill_sono": 2.0,
    "hill_magneto": 2.0,

    "tau_opto": tau_opto,
    "tau_electro": tau_electro,
    "tau_sono": tau_sono,
    "tau_magneto": tau_magneto,

    "alpha_m": alpha_m,
    "delta_m": delta_m,
    "alpha_e": alpha_e,
    "delta_e": delta_e,

    "mu_max": mu_max,
    "K_S": K_S,
    "K_DO": K_DO,
    "Y_XS": Y_XS,

    "qP_max": qP_max,
    "K_E": K_E,
    "K_SP": K_SP,
    "K_OP": K_OP,
    "Y_PS": Y_PS,

    "kla": kla,
    "oxygen_saturation": oxygen_saturation,
    "qO2": qO2
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
            # E. Integrated multi-modal simulation
            # ----------------------------------------------

            multimodal_time = np.linspace(
                0,
                duration,
                400
            )

            multimodal_results = simulate_multimodal_bioreactor(

                time=multimodal_time,

                opto_input=np.full(
                    400,
                    multimodal_light
                ),

                electro_input=np.full(
                    400,
                    multimodal_electric
                ),

                sono_input=np.full(
                    400,
                    multimodal_ultrasound
                ),

                magneto_input=np.full(
                    400,
                    multimodal_magnetic
                ),

                params=multimodal_params,

                initial_conditions={
                    "biomass": initial_biomass,
                    "substrate": initial_substrate,
                    "dissolved_oxygen": initial_oxygen,
                    "product": 0.0,
                    "mRNA": initial_mRNA,
                    "enzyme": initial_enzyme
                }

            )

                        # ----------------------------------------------
            # F. MULTI-MODAL SCENARIO COMPARISON
            # ----------------------------------------------

            zero_input = np.zeros(400)

            light_input = np.full(400, multimodal_light)

            electric_input = np.full(400, multimodal_electric)

            ultrasound_input = np.full(400, multimodal_ultrasound)

            magnetic_input = np.full(400, multimodal_magnetic)

            scenario_inputs = {

                "No activation": (
                    zero_input,
                    zero_input,
                    zero_input,
                    zero_input
                ),

                "Optogenetics only": (
                    light_input,
                    zero_input,
                    zero_input,
                    zero_input
                ),

                "Electrogenetics only": (
                    zero_input,
                    electric_input,
                    zero_input,
                    zero_input
                ),

                "Sonogenetics only": (
                    zero_input,
                    zero_input,
                    ultrasound_input,
                    zero_input
                ),

                "Magnetogenetics only": (
                    zero_input,
                    zero_input,
                    zero_input,
                    magnetic_input
                )

            }

            scenario_results = {}

            common_initial_conditions = {

                "biomass": initial_biomass,

                "substrate": initial_substrate,

                "dissolved_oxygen": initial_oxygen,

                "product": 0.0,

                "mRNA": initial_mRNA,

                "enzyme": initial_enzyme

            }

            for scenario_name, inputs in scenario_inputs.items():

                scenario_results[scenario_name] = (
                    simulate_multimodal_bioreactor(

                        time=multimodal_time,

                        opto_input=inputs[0],

                        electro_input=inputs[1],

                        sono_input=inputs[2],

                        magneto_input=inputs[3],

                        params=multimodal_params,

                        initial_conditions=common_initial_conditions

                    )
                )

            # Reuse the already calculated all-four simulation

            scenario_results["All four modalities"] = multimodal_results



        # ----------------------------------------------
        # Save outputs for display and download
        # ----------------------------------------------

        st.session_state["integrated_results"] = integrated_results

        st.session_state["integrated_population_size"] = population_size

        st.session_state["fixed_results"] = fixed_results

        st.session_state["mpc_results"] = mpc_results

        st.session_state["multimodal_results"] = multimodal_results

        st.session_state["scenario_results"] = scenario_results

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

multimodal = st.session_state["multimodal_results"]

scenarios = st.session_state["scenario_results"]

# ==========================================================
# 16. MAIN DASHBOARD TABS
# ==========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([

    "📊 Bioreactor",

    "🧬 Gene Expression",

    "💨 Oxygen Transfer",

    "🤖 Predictive Control",

    "📁 Research Data",

    "🔬 Multi-Modal Control"

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
# TAB 6: MULTI-MODAL PHYSICAL-FIELD CONTROL
# ==========================================================

with tab6:

    st.header("Multi-Modal Physical-Field Control")

    st.markdown("""
    This module simulates the combined effects of four physical-field
    inputs on actuator activation, gene expression, and fermentation.

    The four modalities are:

    - Optogenetics: Light stimulation
    - Electrogenetics: Electrical-field stimulation
    - Sonogenetics: Ultrasound stimulation
    - Magnetogenetics: Magnetic-field stimulation
    """)

    # ------------------------------------------------------
    # 1. FINAL PERFORMANCE METRICS
    # ------------------------------------------------------

    st.subheader("Final Multi-Modal Simulation Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Final biomass",
        f"{multimodal['biomass'][-1]:.3f} g/L"
    )

    c2.metric(
        "Final product",
        f"{multimodal['product'][-1]:.3f} g/L"
    )

    c3.metric(
        "Final mRNA",
        f"{multimodal['mRNA'][-1]:.3f}"
    )

    c4.metric(
        "Final enzyme",
        f"{multimodal['enzyme'][-1]:.3f}"
    )

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "Final dissolved oxygen",
        f"{multimodal['dissolved_oxygen'][-1]:.5f} g/L"
    )

    c6.metric(
        "Final substrate",
        f"{multimodal['substrate'][-1]:.3f} g/L"
    )

    c7.metric(
        "Combined activation",
        f"{multimodal['total_activation'][-1]:.3f}"
    )

    st.divider()

    # ------------------------------------------------------
    # 2. INDIVIDUAL ACTUATOR ACTIVATION
    # ------------------------------------------------------

    st.subheader("Individual Physical-Field Activation Dynamics")

    activation_fig = go.Figure()

    activation_data = [

        ("Optogenetics", "opto_activation"),

        ("Electrogenetics", "electro_activation"),

        ("Sonogenetics", "sono_activation"),

        ("Magnetogenetics", "magneto_activation")

    ]

    for name, key in activation_data:

        activation_fig.add_trace(

            go.Scatter(

                x=multimodal["time"],

                y=multimodal[key],

                name=name,

                mode="lines",

                line=dict(width=3)

            )

        )

    activation_fig.update_layout(

        template="plotly_dark",

        height=450,

        title="Dynamic Response of the Four Actuators",

        xaxis_title="Time (hours)",

        yaxis_title="Normalized activation",

        yaxis=dict(range=[0, 1.05])

    )

    st.plotly_chart(
        activation_fig,
        use_container_width=True
    )

    st.caption(
        "Each activation state follows its own first-order response "
        "dynamics. The response times can be adjusted in the sidebar."
    )

    # ------------------------------------------------------
    # 3. COMBINED ACTIVATION
    # ------------------------------------------------------

    st.subheader("Combined Multi-Modal Activation")

    total_activation_fig = go.Figure()

    total_activation_fig.add_trace(

        go.Scatter(

            x=multimodal["time"],

            y=multimodal["total_activation"],

            name="Combined activation",

            fill="tozeroy",

            line=dict(width=3)

        )

    )

    total_activation_fig.update_layout(

        template="plotly_dark",

        height=350,

        title="Overall Physical-Field Activation",

        xaxis_title="Time (hours)",

        yaxis_title="Combined activation",

        yaxis=dict(range=[0, 1.05])

    )

    st.plotly_chart(
        total_activation_fig,
        use_container_width=True
    )

    st.latex(
        r"""
        A_{\mathrm{total}}
        =
        1-\prod_{i=1}^{4}(1-A_i)
        """
    )

    # ------------------------------------------------------
    # 4. GENE EXPRESSION DYNAMICS
    # ------------------------------------------------------

    st.subheader("Multi-Modal Gene Expression Dynamics")

    gene_fig = make_subplots(

        rows=1,

        cols=2,

        subplot_titles=(

            "mRNA production",

            "Enzyme production"

        )

    )

    gene_fig.add_trace(

        go.Scatter(

            x=multimodal["time"],

            y=multimodal["mRNA"],

            name="mRNA",

            line=dict(width=3)

        ),

        row=1,

        col=1

    )

    gene_fig.add_trace(

        go.Scatter(

            x=multimodal["time"],

            y=multimodal["enzyme"],

            name="Enzyme",

            line=dict(width=3)

        ),

        row=1,

        col=2

    )

    gene_fig.update_layout(

        template="plotly_dark",

        height=430,

        showlegend=False

    )

    gene_fig.update_xaxes(
        title_text="Time (hours)"
    )

    st.plotly_chart(
        gene_fig,
        use_container_width=True
    )

    st.latex(
        r"""
        \frac{dmRNA}{dt}
        =
        \alpha_m A_{\mathrm{total}}
        -
        \delta_m mRNA
        """
    )

    st.latex(
        r"""
        \frac{dE}{dt}
        =
        \alpha_e mRNA
        -
        \delta_e E
        """
    )

    # ------------------------------------------------------
    # 5. FERMENTATION PERFORMANCE
    # ------------------------------------------------------

    st.subheader("Multi-Modal Fermentation Performance")

    fermentation_fig = make_subplots(

        rows=2,

        cols=2,

        subplot_titles=(

            "Biomass growth",

            "Substrate consumption",

            "Product formation",

            "Dissolved oxygen"

        )

    )

    fermentation_variables = [

        ("Biomass", "biomass", 1, 1),

        ("Substrate", "substrate", 1, 2),

        ("Product", "product", 2, 1),

        ("Dissolved oxygen", "dissolved_oxygen", 2, 2)

    ]

    for name, key, row, col in fermentation_variables:

        fermentation_fig.add_trace(

            go.Scatter(

                x=multimodal["time"],

                y=multimodal[key],

                name=name,

                line=dict(width=3)

            ),

            row=row,

            col=col

        )

    fermentation_fig.update_layout(

        template="plotly_dark",

        height=650,

        showlegend=False

    )

    fermentation_fig.update_xaxes(
        title_text="Time (hours)"
    )

    st.plotly_chart(
        fermentation_fig,
        use_container_width=True
    )

    # ------------------------------------------------------
    # 6. MULTI-MODAL INPUT CONFIGURATION
    # ------------------------------------------------------

    st.subheader("Applied Physical-Field Inputs")

    field_summary = pd.DataFrame({

        "Modality": [

            "Optogenetics",

            "Electrogenetics",

            "Sonogenetics",

            "Magnetogenetics"

        ],

        "Applied input": [

            multimodal_light,

            multimodal_electric,

            multimodal_ultrasound,

            multimodal_magnetic

        ],

        "Response time (h)": [

            tau_opto,

            tau_electro,

            tau_sono,

            tau_magneto

        ]

    })

    st.dataframe(

        field_summary,

        use_container_width=True,

        hide_index=True

    )

    # ------------------------------------------------------
    # 7. MULTI-MODAL RESULTS DATA
    # ------------------------------------------------------

    st.subheader("Multi-Modal Simulation Dataset")

    multimodal_df = pd.DataFrame({

        "Time (h)": multimodal["time"],

        "Optogenetic activation": multimodal["opto_activation"],

        "Electrogenetic activation": multimodal["electro_activation"],

        "Sonogenetic activation": multimodal["sono_activation"],

        "Magnetogenetic activation": multimodal["magneto_activation"],

        "Combined activation": multimodal["total_activation"],

        "mRNA": multimodal["mRNA"],

        "Enzyme": multimodal["enzyme"],

        "Biomass (g/L)": multimodal["biomass"],

        "Substrate (g/L)": multimodal["substrate"],

        "Dissolved oxygen (g/L)": multimodal["dissolved_oxygen"],

        "Product (g/L)": multimodal["product"]

    })

    st.dataframe(

        multimodal_df.head(15),

        use_container_width=True,

        hide_index=True

    )

    st.download_button(

        "⬇️ Download Multi-Modal Simulation CSV",

        multimodal_df.to_csv(index=False).encode("utf-8"),

        file_name="multimodal_bioreactor_results.csv",

        mime="text/csv",

        key="download_multimodal_results"

    )

    # ------------------------------------------------------
    # 8. SCIENTIFIC INTERPRETATION
    # ------------------------------------------------------

    st.info("""
    **Scientific interpretation**

    The four actuator models generate individual activation states.
    These states are combined using a proposed phenomenological
    activation equation.

    Combined activation drives mRNA production, followed by enzyme
    production. The enzyme-dependent metabolic model then determines
    product formation.

    The simulation represents a computational hypothesis. It does
    not establish experimentally validated biological synergy,
    actuator orthogonality, or industrial performance.
    """)


    # ======================================================
    # 9. SIX-SCENARIO PERFORMANCE COMPARISON
    # ======================================================

    st.divider()

    st.header("Six-Scenario Performance Comparison")

    st.markdown("""
    Compare fermentation performance under individual physical-field
    stimulation and combined multi-modal stimulation.
    """)

    scenario_names = list(scenarios.keys())

    comparison_df = pd.DataFrame({

        "Scenario": scenario_names,

        "Final Biomass (g/L)": [
            scenarios[name]["biomass"][-1]
            for name in scenario_names
        ],

        "Final Product (g/L)": [
            scenarios[name]["product"][-1]
            for name in scenario_names
        ],

        "Final Enzyme": [
            scenarios[name]["enzyme"][-1]
            for name in scenario_names
        ],

        "Final mRNA": [
            scenarios[name]["mRNA"][-1]
            for name in scenario_names
        ],

        "Final Activation": [
            scenarios[name]["total_activation"][-1]
            for name in scenario_names
        ]

    })

    st.dataframe(

        comparison_df.style.format({

            "Final Biomass (g/L)": "{:.4f}",

            "Final Product (g/L)": "{:.4f}",

            "Final Enzyme": "{:.4f}",

            "Final mRNA": "{:.4f}",

            "Final Activation": "{:.4f}"

        }),

        use_container_width=True,

        hide_index=True

    )

        # ======================================================
    # 10. PRODUCT FORMATION COMPARISON
    # ======================================================

    st.subheader("Product Formation Across Six Scenarios")

    product_comparison_fig = go.Figure()

    for name in scenario_names:

        result = scenarios[name]

        product_comparison_fig.add_trace(

            go.Scatter(

                x=result["time"],

                y=result["product"],

                name=name,

                mode="lines",

                line=dict(width=3)

            )

        )

    product_comparison_fig.update_layout(

        template="plotly_dark",

        height=480,

        xaxis_title="Time (hours)",

        yaxis_title="Product concentration (g/L)",

        title="Product Formation Under Different Physical-Field Strategies"

    )

    st.plotly_chart(

        product_comparison_fig,

        use_container_width=True

    )


        # ======================================================
    # 11. FINAL PRODUCT COMPARISON
    # ======================================================

    st.subheader("Final Product Concentration")

    product_bar = go.Figure()

    product_bar.add_trace(

        go.Bar(

            x=comparison_df["Scenario"],

            y=comparison_df["Final Product (g/L)"],

            text=comparison_df["Final Product (g/L)"].round(4),

            textposition="outside"

        )

    )

    product_bar.update_layout(

        template="plotly_dark",

        height=450,

        xaxis_title="Physical-field strategy",

        yaxis_title="Final product concentration (g/L)",

        title="Final Product Yield Comparison"

    )

    st.plotly_chart(

        product_bar,

        use_container_width=True

    )

        # ======================================================
    # 12. ENZYME PRODUCTION COMPARISON
    # ======================================================

    st.subheader("Enzyme Production Across Scenarios")

    enzyme_comparison_fig = go.Figure()

    for name in scenario_names:

        result = scenarios[name]

        enzyme_comparison_fig.add_trace(

            go.Scatter(

                x=result["time"],

                y=result["enzyme"],

                name=name,

                mode="lines",

                line=dict(width=3)

            )

        )

    enzyme_comparison_fig.update_layout(

        template="plotly_dark",

        height=450,

        xaxis_title="Time (hours)",

        yaxis_title="Enzyme (model units)",

        title="Enzyme Expression Under Different Stimulation Strategies"

    )

    st.plotly_chart(

        enzyme_comparison_fig,

        use_container_width=True

    )

        # ======================================================
    # 13. PRODUCT IMPROVEMENT ANALYSIS
    # ======================================================

    st.subheader("Product Improvement Relative to Baseline")

    baseline_product = scenarios["No activation"]["product"][-1]

    improvement_values = []

    for name in scenario_names:

        final_product = scenarios[name]["product"][-1]

        improvement = (

            (final_product - baseline_product)

            / max(abs(baseline_product), 1e-12)

        ) * 100

        improvement_values.append(improvement)

    improvement_df = pd.DataFrame({

        "Scenario": scenario_names,

        "Final Product (g/L)": [

            scenarios[name]["product"][-1]

            for name in scenario_names

        ],

        "Improvement vs Baseline (%)": improvement_values

    })

    st.dataframe(

        improvement_df.style.format({

            "Final Product (g/L)": "{:.5f}",

            "Improvement vs Baseline (%)": "{:+.2f}%"

        }),

        use_container_width=True,

        hide_index=True

    )

    improvement_fig = go.Figure()

    improvement_fig.add_trace(

        go.Bar(

            x=improvement_df["Scenario"],

            y=improvement_df["Improvement vs Baseline (%)"],

            text=improvement_df["Improvement vs Baseline (%)"].round(2),

            textposition="outside"

        )

    )

    improvement_fig.update_layout(

        template="plotly_dark",

        height=450,

        xaxis_title="Physical-field strategy",

        yaxis_title="Product improvement (%)",

        title="Relative Improvement Compared with No Activation"

    )

    st.plotly_chart(

        improvement_fig,

        use_container_width=True

    )

        # ======================================================
    # 14. COMBINED RESPONSE VS ADDITIVE REFERENCE
    # ======================================================

    st.subheader("Combined Response vs Additive Reference")

    individual_scenarios = [

        "Optogenetics only",

        "Electrogenetics only",

        "Sonogenetics only",

        "Magnetogenetics only"

    ]

    additive_product = baseline_product

    for name in individual_scenarios:

        additive_product += (

            scenarios[name]["product"][-1]

            - baseline_product

        )

    combined_product = scenarios["All four modalities"]["product"][-1]

    difference = combined_product - additive_product

    c1, c2, c3 = st.columns(3)

    c1.metric(

        "Additive reference",

        f"{additive_product:.4f} g/L"

    )

    c2.metric(

        "Simulated combined product",

        f"{combined_product:.4f} g/L"

    )

    c3.metric(

        "Difference from reference",

        f"{difference:+.4f} g/L"

    )

    st.info("""
    The additive reference is a mathematical comparison constructed
    from the four single-modality simulations.

    A positive difference indicates that the combined model prediction
    exceeds this reference. A negative difference indicates that it
    falls below the reference.

    This comparison does not establish biological synergy or antagonism.
    The result depends on the assumed actuator response functions,
    combined-activation equation, and downstream metabolic model.
    """)

        # ======================================================
    # 15. EXPORT SIX-SCENARIO DATA
    # ======================================================

    st.subheader("Export Scenario Comparison Data")

    st.download_button(

        "⬇️ Download six-scenario comparison CSV",

        comparison_df.to_csv(index=False).encode("utf-8"),

        file_name="multimodal_scenario_comparison.csv",

        mime="text/csv",

        key="download_scenario_comparison"

    )



    

# ==========================================================
# 17. SCIENTIFIC SCOPE AND LIMITATIONS
# ==========================================================

st.divider()

with st.expander("Model equations and scientific assumptions"):

    # 1. Biomass Growth
    st.subheader("1. Biomass Growth")

    st.latex(r"""
    \frac{dX}{dt}
    =
    \mu_{\max}
    \frac{S}{K_S+S}
    \frac{DO}{K_{DO}+DO}
    X
    """)

    # 2. Gene Expression
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

    # 3. Optical Attenuation
    st.subheader("3. Optogenetics - Optical Attenuation")

    st.latex(r"""
    I_i(t)=I_0 e^{-\kappa z_i(t)}
    """)

    # 4. Electrogenetics
    st.subheader("4. Electrogenetics - Electrical Field Activation")

    st.latex(r"""
    A_{\mathrm{elec}}(t)
    =
    \frac{\mathcal{E}(t)^n}
    {K_{\mathcal{E}}^n+\mathcal{E}(t)^n}
    """)

    st.markdown("""
    **Parameters:**

    - `E(t)`: Applied electrical-field magnitude (V/m).
    - `K_E`: Half-activation electrical-field magnitude.
    - `n`: Hill coefficient.
    - `A_elec`: Normalized electrical activation.
    """)

    # 5. Magnetogenetics
    st.subheader("5. Magnetogenetics - Magnetic Field Activation")

    st.latex(r"""
    A_{\mathrm{mag}}(t)
    =
    \frac{B(t)^n}
    {K_B^n+B(t)^n}
    """)

    st.markdown("""
    **Parameters:**

    - `B(t)`: Magnetic-field magnitude (T).
    - `K_B`: Half-activation magnetic-field magnitude.
    - `n`: Hill coefficient.
    - `A_mag`: Normalized magnetic activation.
    """)

    # 6. Sonogenetics
    st.subheader("6. Sonogenetics - Ultrasound Activation")

    st.latex(r"""
    A_{\mathrm{sono}}(t)
    =
    \frac{I_{\mathrm{US}}(t)^n}
    {K_{\mathrm{US}}^n+I_{\mathrm{US}}(t)^n}
    """)

    st.markdown("""
    **Parameters:**

    - `I_US`: Ultrasound intensity (W/m²).
    - `K_US`: Half-activation ultrasound intensity.
    - `n`: Hill coefficient.
    - `A_sono`: Normalized ultrasound activation.
    """)

    # 7. Combined Physical-Field Activation
    st.subheader("7. Combined Physical-Field Activation")

    st.latex(r"""
    A_{\mathrm{total}}
    =
    1-
    (1-A_{\mathrm{opto}})
    (1-A_{\mathrm{elec}})
    (1-A_{\mathrm{mag}})
    (1-A_{\mathrm{sono}})
    """)

    st.markdown("""
    This is a proposed phenomenological combination rule.

    It assumes independent contributions to normalized activation.
    It does not establish biological synergy or prove that the
    physical-field responses are independent.
    """)

    # 8. Multi-Modal Gene Expression
    st.subheader("8. Multi-Modal Gene Expression Dynamics")

    st.latex(r"""
    \frac{dmRNA}{dt}
    =
    \alpha_m A_{\mathrm{total}}
    -
    \delta_m mRNA
    """)

    st.latex(r"""
    \frac{dE}{dt}
    =
    \alpha_e mRNA
    -
    \delta_e E
    """)

    # 9. Product Formation
    st.subheader("9. Product Formation")

    st.latex(r"""
    \frac{dP}{dt}
    =
    X\overline{q_P}
    """)

    # 10. Oxygen Balance
    st.subheader("10. Oxygen Balance")

    st.latex(r"""
    \frac{dDO}{dt}
    =
    k_La(C^*-DO)
    -
    q_{O_2}Xf_{O_2}
    """)

    # Scientific scope
    st.markdown("---")

    st.markdown("""
    **Currently implemented**

    - Population-level optogenetic simulation.
    - Integrated multi-modal electrical, ultrasound, magnetic,
    and optical actuator dynamics.
    - Coupled multi-modal gene-expression and fermentation model.
    - Individual-cell mRNA and enzyme dynamics.
    - Shared batch-fermentation environment.
    - Monod-based biomass growth.
    - Enzyme-dependent product formation.
    - Dynamic dissolved-oxygen balance.
    - Reduced-order predictive-control model.

    **Not yet fully implemented**

    - Dynamic electrical-field actuator model.
    - Dynamic magnetic-field actuator model.
    - Dynamic ultrasound actuator model.
    - Experimental calibration and validation.
    - CFD-derived spatial mixing and field distributions.
    - Mechanistic metabolic network.
    - Hardware-connected sensor feedback.
    - Experimentally established interactions between modalities.

    **Important:** The MPC model uses average mRNA and enzyme states.
    It is an approximation of the individual-cell population model.
    """)


st.caption(
    "Multi-Modal Bioprocess Digital Twin | Computational research prototype | "
    "Batch fermentation | Simulated results only"
)