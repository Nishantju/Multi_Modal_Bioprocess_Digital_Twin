from .gene_expression import (
    gene_expression_dynamics,
    calculate_gene_activation
)


# Initial concentrations
mRNA = 0.2
protein = 0.1

# Physical-field activation signal
field_signal = 0.8

# Basal gene activation
basal_activation = 0.1

# Calculate gene activation
activation = calculate_gene_activation(
    field_signal,
    basal_activation
)

# Model parameters
alpha_m = 2.0
delta_m = 0.2

alpha_e = 0.5
delta_e = 0.1

# Calculate rates of change
dmRNA_dt, dProtein_dt = gene_expression_dynamics(

    mRNA,
    protein,
    activation,
    alpha_m,
    delta_m,
    alpha_e,
    delta_e
)


print("Gene activation:", activation)

print("mRNA concentration:", mRNA)

print("Protein concentration:", protein)

print("Rate of mRNA change:", dmRNA_dt)

print("Rate of protein change:", dProtein_dt)