from .gene_expression import simulate_gene_expression


result = simulate_gene_expression(

    initial_mRNA=0.2,
    initial_protein=0.1,

    duration=10,
    switch_time=2,

    field_signal=0.8,
    basal_activation=0.1,

    alpha_m=2.0,
    delta_m=0.2,

    alpha_e=0.5,
    delta_e=0.1
)


print("TIME | mRNA | PROTEIN")

for i in range(0, len(result["time"]), 30):

    print(

        round(result["time"][i], 2),

        "|",

        round(result["mRNA"][i], 4),

        "|",

        round(result["protein"][i], 4)

    )


print("\nFINAL RESULTS")

print("Final mRNA:", result["mRNA"][-1])

print("Final protein:", result["protein"][-1])