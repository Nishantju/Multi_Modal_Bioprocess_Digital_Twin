import matplotlib.pyplot as plt
import numpy as np


def plot_operating_comparison(results):

    names = list(results.keys())

    products = [
        results[name]["product"][-1]
        for name in names
    ]

    biomass = [
        results[name]["biomass"][-1]
        for name in names
    ]

    oxygen = [
        results[name]["dissolved_oxygen"][-1]
        for name in names
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 10),
        constrained_layout=True
    )

    # 1. Product comparison

    axes[0, 0].bar(names, products)

    axes[0, 0].set_title(
        "Final Product Concentration"
    )

    axes[0, 0].set_ylabel(
        "Product (g/L)"
    )

    axes[0, 0].tick_params(
        axis="x",
        rotation=25
    )

    axes[0, 0].grid(axis="y")


    # 2. Biomass comparison

    axes[0, 1].bar(names, biomass)

    axes[0, 1].set_title(
        "Final Biomass Concentration"
    )

    axes[0, 1].set_ylabel(
        "Biomass (g/L)"
    )

    axes[0, 1].tick_params(
        axis="x",
        rotation=25
    )

    axes[0, 1].grid(axis="y")


    # 3. Dissolved oxygen comparison

    axes[1, 0].bar(names, oxygen)

    axes[1, 0].set_title(
        "Final Dissolved Oxygen"
    )

    axes[1, 0].set_ylabel(
        "DO (g/L)"
    )

    axes[1, 0].tick_params(
        axis="x",
        rotation=25
    )

    axes[1, 0].grid(axis="y")


    # 4. Product accumulation over time

    for name in names:

        axes[1, 1].plot(
            results[name]["time"],
            results[name]["product"],
            label=name
        )

    axes[1, 1].set_title(
        "Product Accumulation Over Time"
    )

    axes[1, 1].set_xlabel(
        "Time (hours)"
    )

    axes[1, 1].set_ylabel(
        "Product (g/L)"
    )

    axes[1, 1].legend(
        fontsize=8
    )

    axes[1, 1].grid(True)

    plt.show()