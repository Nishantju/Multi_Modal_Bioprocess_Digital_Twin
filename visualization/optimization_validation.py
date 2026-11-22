import matplotlib.pyplot as plt


def plot_optimization_validation(result):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5)
    )

    # kLa validation

    axes[0].plot(
        result["kla_values"],
        result["kla_products"],
        marker="o"
    )

    axes[0].set_title(
        "Product vs Oxygen Transfer"
    )

    axes[0].set_xlabel(
        "kLa (h-1)"
    )

    axes[0].set_ylabel(
        "Final Product (g/L)"
    )

    axes[0].grid(True)

    # Light validation

    axes[1].plot(
        result["light_values"],
        result["light_products"],
        marker="o"
    )

    axes[1].set_title(
        "Product vs Light Intensity"
    )

    axes[1].set_xlabel(
        "Light Intensity"
    )

    axes[1].set_ylabel(
        "Final Product (g/L)"
    )

    axes[1].grid(True)

    plt.tight_layout()

    plt.show()