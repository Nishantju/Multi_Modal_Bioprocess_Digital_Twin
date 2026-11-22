import matplotlib.pyplot as plt
import numpy as np


def plot_sensitivity_results(result):

    light = result["light_values"]

    kla = result["kla_values"]

    product = result["product_matrix"]

    fig, ax = plt.subplots(figsize=(11, 8))

    image = ax.imshow(
        product,
        origin="lower",
        aspect="auto"
    )

    ax.set_title(
        "Sensitivity Analysis: Light Intensity vs Oxygen Transfer"
    )

    ax.set_xlabel("Surface Light Intensity")

    ax.set_ylabel("Oxygen Transfer Coefficient (kLa, h-1)")

    ax.set_xticks(np.arange(len(light)))
    ax.set_xticklabels(light)

    ax.set_yticks(np.arange(len(kla)))
    ax.set_yticklabels(kla)

    colorbar = fig.colorbar(image, ax=ax)

    colorbar.set_label(
        "Final Product Concentration (g/L)"
    )

    for i in range(len(kla)):

        for j in range(len(light)):

            ax.text(
                j,
                i,
                f"{product[i, j]:.3f}",
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.show()