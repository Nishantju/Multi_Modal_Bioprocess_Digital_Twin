from .integrated_bioreactor import (
    simulate_integrated_bioreactor
)

from visualization.operating_comparison import (
    plot_operating_comparison
)


conditions = {

    "A: Low Light + Low O2": {
        "surface_intensity": 20,
        "kla": 50
    },

    "B: High Light + Low O2": {
        "surface_intensity": 100,
        "kla": 50
    },

    "C: Low Light + High O2": {
        "surface_intensity": 20,
        "kla": 150
    },

    "D: High Light + High O2": {
        "surface_intensity": 100,
        "kla": 150
    }

}


results = {}

for name, parameters in conditions.items():

    results[name] = simulate_integrated_bioreactor(

        population_size=100,

        duration=20,

        surface_intensity=parameters["surface_intensity"],

        kla=parameters["kla"]

    )


plot_operating_comparison(results)