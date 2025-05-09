import pandas as pd

# Generates valid 4-component combinations where the sum equals 100%.
# Three components are iterated explicitly; the fourth is computed as residual.
# Only combinations within specified bounds for all components are retained.
# Useful for constructing input sets for convex hulls in compositional analysis.

# Range limits for each soil component
sand_min, sand_max = 2, 15
humus_min, humus_max = 82, 96
ton_min, ton_max = 0, 1
schluff_min, schluff_max = 1, 9

# List to store valid combinations
combinations = []

# Loop over all possible clay percentages
for ton in range(ton_min, ton_max + 1):
    # Loop over all possible silt percentages
    for schluff in range(schluff_min, schluff_max + 1):
        # Loop over all possible sand percentages
        for sand in range(sand_min, sand_max + 1):
            # Calculate the humus percentage as residual to 100%
            humus = 100 - ton - schluff - sand
            # Check if humus is within the valid range
            if humus_min <= humus <= humus_max:
                # Add valid combination to the list
                combinations.append({
                    'Silt (%)': schluff,
                    'Humus (%)': humus,
                    'Sand (%)': sand,
                    'Clay (%)': ton
                })

# Create a DataFrame from the list of valid combinations
df = pd.DataFrame(combinations)

# Reorder the columns
df = df[['Sand (%)', 'Silt (%)', 'Humus (%)', 'Clay (%)']]

# Save output to the 'data' subfolder in the repository
filepath = "data/valid_compositions.xlsx"


print(df.columns)
print(f"Data was successfully saved to '{filepath}'.")
