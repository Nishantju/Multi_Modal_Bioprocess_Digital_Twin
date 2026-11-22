import matplotlib.pyplot as plt


def plot_integrated_bioreactor(result):

    time = result["time"]

    biomass = result["biomass"]

    substrate = result["substrate"]

    oxygen = result["dissolved_oxygen"]

    product = result["product"]

    OTR = result["OTR"]

    OUR = result["OUR"]

    fig, axes = plt.subplots(
        5, 1,
        figsize=(12, 18),
        constrained_layout=True
    )

    # GRAPH 1: Biomass growth

    axes[0].plot(
        time,
        biomass,
        label="Biomass"
    )

    axes[0].set_title(
        "Microbial Biomass Growth"
    )

    axes[0].set_ylabel(
        "Biomass (g/L)"
    )

    axes[0].grid(True)

    axes[0].legend()


    # GRAPH 2: Substrate consumption

    axes[1].plot(
        time,
        substrate,
        label="Substrate"
    )

    axes[1].set_title(
        "Substrate Consumption"
    )

    axes[1].set_ylabel(
        "Substrate (g/L)"
    )

    axes[1].grid(True)

    axes[1].legend()


    # GRAPH 3: Dissolved oxygen

    axes[2].plot(
        time,
        oxygen,
        label="Dissolved Oxygen"
    )

    axes[2].set_title(
        "Dissolved Oxygen Dynamics"
    )

    axes[2].set_ylabel(
        "DO (g/L)"
    )

    axes[2].grid(True)

    axes[2].legend()


    # GRAPH 4: Product accumulation

    axes[3].plot(
        time,
        product,
        label="Product"
    )

    axes[3].set_title(
        "Product Accumulation"
    )

    axes[3].set_ylabel(
        "Product (g/L)"
    )

    axes[3].grid(True)

    axes[3].legend()


    # GRAPH 5: OTR versus OUR

    axes[4].plot(
        time,
        OTR,
        label="OTR"
    )

    axes[4].plot(
        time,
        OUR,
        label="OUR"
    )

    axes[4].set_title(
        "Oxygen Transfer Rate vs Oxygen Uptake Rate"
    )

    axes[4].set_xlabel(
        "Time (hours)"
    )

    axes[4].set_ylabel(
        "Oxygen Rate (g/L/h)"
    )

    axes[4].grid(True)

    axes[4].legend()

    plt.show()