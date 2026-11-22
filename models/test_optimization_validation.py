from control.validate_optimization import (
    validate_optimized_conditions
)


from visualization.optimization_validation import (
    plot_optimization_validation
)


result = validate_optimized_conditions()

plot_optimization_validation(result)