import matplotlib.pyplot as plt


def plot_pareto_results(result):

    records = result["records"]

    pareto = result["pareto_records"]

    burden = [
        item["burden"]
        for item in records
    ]

    product = [
        item["product"]
        for item in records
    ]

    oxygen = [
        item["oxygen"]
        for item in records
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(

        burden,

        product,

        c=oxygen,

        s=65

    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax
    )

    colorbar.set_label(
        "Final Dissolved Oxygen (g/L)"
    )

    pareto_burden = [
        item["burden"]
        for item in pareto
    ]

    pareto_product = [
        item["product"]
        for item in pareto
    ]

    ax.scatter(

        pareto_burden,

        pareto_product,

        marker="*",

        s=180,

        label="Pareto-optimal conditions"

    )

    ax.set_title(
        "Pareto Analysis: Product vs Operating Burden"
    )

    ax.set_xlabel(
        "Normalized Operating-Burden Proxy"
    )

    ax.set_ylabel(
        "Final Product Concentration (g/L)"
    )

    ax.grid(True)

    ax.legend()

    plt.tight_layout()

    plt.show()