import matplotlib.pyplot as plt


def plot_state_estimation(sensor_data, estimates):

    time = sensor_data["time"]

    variables = [

        ("biomass", "true_biomass", "measured_biomass",
         "Biomass (g/L)"),

        ("oxygen", "true_oxygen", "measured_oxygen",
         "Dissolved Oxygen (g/L)"),

        ("substrate", "true_substrate", "measured_substrate",
         "Substrate (g/L)"),

        ("product", "true_product", "measured_product",
         "Product (g/L)")

    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9)
    )

    for ax, variable in zip(axes.flat, variables):

        name, true_key, measured_key, ylabel = variable

        ax.plot(
            time,
            sensor_data[true_key],
            label="True state"
        )

        ax.scatter(
            time,
            sensor_data[measured_key],
            label="Noisy sensor",
            s=9
        )

        ax.plot(
            time,
            estimates[name],
            label="Kalman estimate"
        )

        ax.set_title(name.capitalize())

        ax.set_xlabel("Time (hours)")

        ax.set_ylabel(ylabel)

        ax.grid(True)

        ax.legend()

    plt.suptitle(
        "Bioreactor State Estimation Using Kalman Filtering"
    )

    plt.tight_layout()

    plt.show()