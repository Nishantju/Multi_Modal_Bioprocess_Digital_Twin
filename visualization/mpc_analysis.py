import matplotlib.pyplot as plt


def plot_mpc_analysis(mpc):

    time = mpc["time"]

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(11, 18)
    )

    # 1. Light control

    axes[0].plot(time[:-1], mpc["light"])

    axes[0].set_title("MPC Light Control")
    axes[0].set_ylabel("Light Intensity")
    axes[0].grid(True)


    # 2. Oxygen transfer control

    axes[1].plot(time[:-1], mpc["kla"])

    axes[1].set_title("MPC Oxygen Transfer Control")
    axes[1].set_ylabel("kLa (h⁻¹)")
    axes[1].grid(True)


    # 3. Product formation

    axes[2].plot(time, mpc["product"])

    axes[2].set_title("Product Formation")
    axes[2].set_ylabel("Product (g/L)")
    axes[2].grid(True)


    # 4. Enzyme dynamics

    axes[3].plot(time, mpc["enzyme"])

    axes[3].set_title("Enzyme Dynamics")
    axes[3].set_ylabel("Enzyme (model units)")
    axes[3].grid(True)


    # 5. Substrate consumption

    axes[4].plot(time, mpc["substrate"])

    axes[4].set_title("Substrate Consumption")
    axes[4].set_ylabel("Substrate (g/L)")
    axes[4].grid(True)


    # 6. Dissolved oxygen

    axes[5].plot(time, mpc["dissolved_oxygen"])

    axes[5].set_title("Dissolved Oxygen Dynamics")
    axes[5].set_ylabel("DO (g/L)")
    axes[5].set_xlabel("Time (hours)")
    axes[5].grid(True)


    plt.tight_layout()

    plt.show()