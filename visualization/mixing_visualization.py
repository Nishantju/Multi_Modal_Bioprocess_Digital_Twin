import matplotlib.pyplot as plt


def plot_mixing_dynamics(result):

    time = result["time"]
    depth = result["depth"]
    intensity = result["light_intensity"]
    activation = result["activation"]

    fig, axes = plt.subplots(
        3, 1,
        figsize=(11, 10),
        sharex=True
    )

    # 1. Microorganism movement

    axes[0].plot(time, depth)

    axes[0].set_title(
        "Microorganism Movement Inside Bioreactor"
    )

    axes[0].set_ylabel("Depth (m)")

    axes[0].grid(True)

    # 2. Light exposure

    axes[1].plot(time, intensity)

    axes[1].set_title(
        "Changing Light Exposure"
    )

    axes[1].set_ylabel("Light Intensity")

    axes[1].grid(True)

    # 3. Gene activation

    axes[2].plot(time, activation)

    axes[2].set_title(
        "Dynamic Gene Activation"
    )

    axes[2].set_xlabel("Time (hours)")

    axes[2].set_ylabel("Activation")

    axes[2].set_ylim(0, 1)

    axes[2].grid(True)

    plt.tight_layout()

    plt.show()