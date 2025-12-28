# Generates valid four-component integer compositions summing to 100% by
# computing a constrained Cartesian product of sand, silt, clay
# and humus within predefined bonuds. bounds.

import pandas as pd

# Range limits for each component (example soil texture)
sand_min, sand_max = 10, 45
silt_min, silt_max = 55, 75
clay_min, clay_max = 0, 15
humus_min, humus_max = 0, 15

# List to store valid combinations
combinations = []

# Loop over all possible humus contents
for humus in range(humus_min, humus_max + 1):
    # Dynamic scaling based on remaining percentage (100 - humus)
    scale_factor = (100 - humus) / 100

    sand_min_scaled = int(sand_min * scale_factor)
    sand_max_scaled = int(sand_max * scale_factor)
    silt_min_scaled = int(silt_min * scale_factor)
    silt_max_scaled = int(silt_max * scale_factor)
    clay_min_scaled = int(clay_min * scale_factor)
    clay_max_scaled = int(clay_max * scale_factor)

    # Loop over all possible clay values
    for clay in range(clay_min_scaled, clay_max_scaled + 1):
        # Loop over all possible silt values
        for silt in range(silt_min_scaled, silt_max_scaled + 1):
            # Loop over all possible sand values
            for sand in range(sand_min_scaled, sand_max_scaled + 1):
                # Sum of mineral components
                sum_components = sand + silt + clay

                # Remaining percentage is assigned to humus
                calculated_humus = 100 - sum_components

                # Check if humus is within valid range
                if humus_min <= calculated_humus <= humus_max:
                    combinations.append({
                        'Sand (%)': sand,
                        'Silt (%)': silt,
                        'Clay (%)': clay,
                        'Humus (%)': calculated_humus
                    })

# Create DataFrame
df = pd.DataFrame(combinations)

# Set column order
df = df[['Sand (%)', 'Silt (%)', 'Clay (%)', 'Humus (%)']]


# Convert results into a DataFrame
output_file = "data/valid_compositions.xlsx"
df.to_excel(output_file, index=False)

# Output result
print(f"Data has been saved to '{output_file}'.")
print(df.head())



