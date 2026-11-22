import numpy as np
import matplotlib.pyplot as plt


def plot_population_results(result):

    time = result["time"]

    protein = result["protein"]

    mean_protein = result["mean_protein"]

    std_protein = result["std_protein"]

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 15),
        constrained_layout=True
    )

    # GRAPH 1: Final protein distribution

    final_protein = protein[:, -1]

    axes[0].hist(
        final_protein,
        bins=15,
        edgecolor="black"
    )

    axes[0].set_title(
        "Protein Distribution Across Microbial Population"
    )

    axes[0].set_xlabel("Final Protein Level (model units)")

    axes[0].set_ylabel("Number of Microorganisms")

    axes[0].grid(True)


    # GRAPH 2: Average protein with standard deviation

    axes[1].plot(
        time,
        mean_protein,
        label="Average Protein"
    )

    axes[1].fill_between(
        time,
        mean_protein - std_protein,
        mean_protein + std_protein,
        alpha=0.25,
        label="Mean ± 1 Standard Deviation"
    )

    axes[1].set_title(
        "Population-Level Protein Production"
    )

    axes[1].set_xlabel("Time (hours)")

    axes[1].set_ylabel("Protein (model units)")

    axes[1].legend()

    axes[1].grid(True)


    # GRAPH 3: Protein production heatmap

    image = axes[2].imshow(
        protein,
        aspect="auto",
        origin="lower",
        extent=[
            time[0],
            time[-1],
            0,
            protein.shape[0]
        ]
    )

    axes[2].set_title(
        "Protein Production Across Individual Microorganisms"
    )

    axes[2].set_xlabel("Time (hours)")

    axes[2].set_ylabel("Microorganism Index")

    fig.colorbar(
        image,
        ax=axes[2],
        label="Protein Level"
    )

    plt.show()