import matplotlib.pyplot as plt


def plot_sensor_comparison(result):

    time = result["time"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9)
    )

    variables = [

        (
            "true_biomass",
            "measured_biomass",
            "Biomass",
            "Biomass (g/L)"
        ),

        (
            "true_oxygen",
            "measured_oxygen",
            "Dissolved Oxygen",
            "DO (g/L)"
        ),

        (
            "true_substrate",
            "measured_substrate",
            "Substrate",
            "Substrate (g/L)"
        ),

        (
            "true_product",
            "measured_product",
            "Product",
            "Product (g/L)"
        )

    ]

    for ax, variable in zip(
        axes.flat,
        variables
    ):

        true_key = variable[0]

        measured_key = variable[1]

        title = variable[2]

        ylabel = variable[3]

        ax.plot(
            time,
            result[true_key],
            label="True state"
        )

        ax.scatter(
            time,
            result[measured_key],
            label="Sensor measurement",
            s=10
        )

        ax.set_title(title)

        ax.set_xlabel("Time (hours)")

        ax.set_ylabel(ylabel)

        ax.grid(True)

        ax.legend()

    plt.suptitle(
        "True Bioreactor States vs Simulated Sensor Measurements"
    )

    plt.tight_layout()

    plt.show()