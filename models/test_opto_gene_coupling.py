from .opto_gene_coupling import (
    simulate_opto_gene_expression
)


depths = [0, 0.25, 0.5, 0.75, 1.0]

print(
    "DEPTH | LIGHT | ACTIVATION | FINAL mRNA | FINAL PROTEIN"
)

for depth in depths:

    result = simulate_opto_gene_expression(
        initial_mRNA=0.2,
        initial_protein=0.1,
        duration=10,

        surface_intensity=100,
        attenuation_coefficient=2,
        depth=depth,

        half_activation_intensity=30,
        hill_coefficient=2,

        basal_activation=0.1,

        alpha_m=2.0,
        delta_m=0.2,
        alpha_e=0.5,
        delta_e=0.1
    )

    print(
        depth,
        "|",
        round(result["light_intensity"], 3),
        "|",
        round(result["gene_activation"], 4),
        "|",
        round(result["mRNA"][-1], 4),
        "|",
        round(result["protein"][-1], 4)
    )