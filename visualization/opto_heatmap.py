import matplotlib.pyplot as plt


def plot_opto_heatmaps(result, width, height):

    X = result["X"]
    Z = result["Z"]

    intensity = result["intensity"]
    activation = result["activation"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Light intensity heatmap

    light_plot = axes[0].imshow(
        intensity,
        origin="lower",
        extent=[0, width, 0, height],
        aspect="auto",
        cmap="inferno"
    )

    axes[0].set_title("Spatial Light Intensity")
    axes[0].set_xlabel("Horizontal Position (m)")
    axes[0].set_ylabel("Depth (m)")

    fig.colorbar(
        light_plot,
        ax=axes[0],
        label="Light Intensity"
    )

    # Gene activation heatmap

    activation_plot = axes[1].imshow(
        activation,
        origin="lower",
        extent=[0, width, 0, height],
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    axes[1].set_title("Spatial Gene Activation")
    axes[1].set_xlabel("Horizontal Position (m)")
    axes[1].set_ylabel("Depth (m)")

    fig.colorbar(
        activation_plot,
        ax=axes[1],
        label="Gene Activation"
    )
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()