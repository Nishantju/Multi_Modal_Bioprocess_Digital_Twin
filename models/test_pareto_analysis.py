from control.pareto_analysis import (
    run_pareto_analysis
)

from visualization.pareto_visualization import (
    plot_pareto_results
)


result = run_pareto_analysis()

plot_pareto_results(result)