import matplotlib.pyplot as plt
import numpy as np


def plot_metabolic_results(result):

    time = result["time"]

    enzyme = result["enzyme"]

    product = result["product"]

    mean_enzyme = result["mean_enzyme"]

    mean_product = result["mean_product"]

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 15),
        constrained_layout=True
    )

    # Graph 1: Average enzyme abundance

    axes[0].plot(
        time,
        mean_enzyme,
        label="Average Enzyme"
    )

    axes[0].set_title(
        "Population Average Enzyme Abundance"
    )

    axes[0].set_xlabel("Time (hours)")

    axes[0].set_ylabel("Enzyme (model units)")

    axes[0].grid(True)

    axes[0].legend()


    # Graph 2: Average product accumulation

    axes[1].plot(
        time,
        mean_product,
        label="Average Product"
    )

    axes[1].set_title(
        "Population Average Product Accumulation"
    )

    axes[1].set_xlabel("Time (hours)")

    axes[1].set_ylabel("Product (model units)")

    axes[1].grid(True)

    axes[1].legend()


    # Graph 3: Enzyme versus final product

    final_enzyme = enzyme[:, -1]

    final_product = product[:, -1]

    axes[2].scatter(
        final_enzyme,
        final_product
    )

    correlation = np.corrcoef(
        final_enzyme,
        final_product
    )[0, 1]

    axes[2].set_title(
        f"Enzyme Abundance vs Product Formation "
        f"(Correlation = {correlation:.3f})"
    )

    axes[2].set_xlabel(
        "Final Enzyme Abundance"
    )

    axes[2].set_ylabel(
        "Final Accumulated Product"
    )

    axes[2].grid(True)

    plt.show()