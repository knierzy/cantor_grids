
# Generates valid 4-component integer compositions that sum to 100%.
# For each humus level ("Stufe"), the code iterates over sand, silt (schluff), and clay (ton)
# within defined bounds. Humus is calculated as the residual (100 - sum of others)
# and checked against its specific allowed interval per step.
# The result is a constrained Cartesian product with sum and bound restrictions,
# useful for input generation in compositional or soil texture analyses.

# 
import pandas as pd
# Normalized computation of tuples as humus content increases
stufen = [
    {'humus_min': 0, 'humus_max': 1,  'sand_min': 20, 'sand_max': 74, 'schluff_min': 10, 'schluff_max': 54, 'ton_min': 15, 'ton_max': 25},
    {'humus_min': 1, 'humus_max': 2,  'sand_min': 20, 'sand_max': 74, 'schluff_min': 10, 'schluff_max': 54, 'ton_min': 15, 'ton_max': 25},
    {'humus_min': 2, 'humus_max': 3,  'sand_min': 19, 'sand_max': 73, 'schluff_min': 10, 'schluff_max': 53, 'ton_min': 15, 'ton_max': 24},
    {'humus_min': 3, 'humus_max': 4,  'sand_min': 19, 'sand_max': 72, 'schluff_min': 10, 'schluff_max': 53, 'ton_min': 14, 'ton_max': 24},
    {'humus_min': 4, 'humus_max': 5,  'sand_min': 19, 'sand_max': 71, 'schluff_min': 10, 'schluff_max': 52, 'ton_min': 14, 'ton_max': 24},
    {'humus_min': 5, 'humus_max': 6,  'sand_min': 19, 'sand_max': 71, 'schluff_min': 9, 'schluff_max': 52, 'ton_min': 14, 'ton_max': 24},
    {'humus_min': 6, 'humus_max': 7,  'sand_min': 19, 'sand_max': 70, 'schluff_min': 9, 'schluff_max': 51, 'ton_min': 14, 'ton_max': 23},
    {'humus_min': 7, 'humus_max': 8,  'sand_min': 18, 'sand_max': 69, 'schluff_min': 9, 'schluff_max': 51, 'ton_min': 14, 'ton_max': 23},
    {'humus_min': 8, 'humus_max': 9,  'sand_min': 18, 'sand_max': 68, 'schluff_min': 9, 'schluff_max': 50, 'ton_min': 14, 'ton_max': 23},
    {'humus_min': 9, 'humus_max': 10,  'sand_min': 18, 'sand_max': 68, 'schluff_min': 9, 'schluff_max': 50, 'ton_min': 14, 'ton_max': 23},
    {'humus_min': 10, 'humus_max': 11, 'sand_min': 18, 'sand_max': 67, 'schluff_min': 9, 'schluff_max': 48, 'ton_min': 13, 'ton_max': 22},
    {'humus_min': 11, 'humus_max': 12, 'sand_min': 18, 'sand_max': 66, 'schluff_min': 9, 'schluff_max': 48, 'ton_min': 13, 'ton_max': 22},
    {'humus_min': 12, 'humus_max': 13, 'sand_min': 17, 'sand_max': 65, 'schluff_min': 9, 'schluff_max': 48, 'ton_min': 13, 'ton_max': 22},
    {'humus_min': 13, 'humus_max': 14, 'sand_min': 17, 'sand_max': 65, 'schluff_min': 9, 'schluff_max': 47, 'ton_min': 13, 'ton_max': 22},
    {'humus_min': 14, 'humus_max': 15, 'sand_min': 17, 'sand_max': 64, 'schluff_min': 9, 'schluff_max': 47, 'ton_min': 13, 'ton_max': 21},
]

#  List to store all valid combinations
alle_kombis = []

# process each humus level based on the specified ranges
for stufen_nummer, stufe in enumerate(stufen):
    for sand in range(stufe['sand_min'], stufe['sand_max'] + 1):
        for schluff in range(int(stufe['schluff_min']), stufe['schluff_max'] + 1):
            for ton in range(int(stufe['ton_min']), stufe['ton_max'] + 1):
                humus = 100 - sand - schluff - ton
                if stufe['humus_min'] <= humus <= stufe['humus_max']:
                    alle_kombis.append({
                        'Stufe': stufen_nummer + 1,
                        'Sand (%)': sand,
                        'Schluff (%)': schluff,
                        'Ton (%)': ton,
                        'Humus (%)': humus
                    })

# convert the results into a DataFrame
df = pd.DataFrame(alle_kombis)
output_path = "data/valid_compositions.xlsx"
df.to_excel(output_path, index=False)
print(f"✅ Exakt nach Plan erzeugt und gespeichert unter:\n{output_path}")

