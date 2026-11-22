from .mixing_opto_gene import (
    simulate_moving_microorganism
)

from visualization.mixing_visualization import (
    plot_mixing_dynamics
)


result = simulate_moving_microorganism(

    duration=10,

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

    delta_e=0.1
)


plot_mixing_dynamics(result)