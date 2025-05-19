import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull
import numpy as np

# file path (relative to repo root)
convex_hulls_file_1 = "data/convex_hull_SandigerTon_STEPS_Anpassung.xlsx"
convex_hulls_file_2 = "data/convex_hull_TonigerSand_STEPS_Anpassung.xlsx"
convex_hulls_file_3 = "data/convex_hull_SandigerLehnm_STEPS_Anpassung.xlsx"
convex_hulls_file_4 = "data/convex_hull_LehmigerSchluff_STEPS_Anpassung.xlsx"
convex_hulls_file_5 = "data/convex_hull_Schluff_STEPS_allereduziert.xlsx"
convex_hulls_file_6 = "data/convex_hull_sandigerSchluff_STEPS_Anpassung.xlsx"
convex_hulls_file_7 = "data/convex_hull_Lehm_STEPS_Anpassung.xlsx"
convex_hulls_file_8 = "data/convex_hull_lehmigerSand_STEPS_Anpassung.xlsx"
convex_hulls_file_9 = "data/convex_hull_schluffigerSand_STEPS_Anpassung.xlsx"
convex_hulls_file_10 = "data/convex_hull_lehmigerTon_STEPS_Anpassung1.xlsx"
convex_hulls_file_11 = "data/convex_hull_Ton_STEPS_Anpassung.xlsx"
convex_hulls_file_12 = "data/convex_hull_Sand1_STEPS_Anpassung.xlsx"
convex_hulls_file_14 = "data/convex_hulls_humicgleysoils.xlsx"
convex_hulls_file_15 = "data/convex_hull_schluffigerLehm_STEPS_ANPASSUNG.xlsx"

# color mappings

color_mapping_files = {
    convex_hulls_file_1: "rgba(160, 82, 45, 0.95)",     # Sandiger Ton
    convex_hulls_file_2: "rgba(204, 121, 167, 0.95)",     # Toniger Sand
    convex_hulls_file_3: "rgba(178, 34, 34, 0.95)",       # Sandiger Lehm
    convex_hulls_file_4: "rgba(253, 192, 134, 0.95)",     # Lehmiger Schluff
    convex_hulls_file_5: "rgba(139, 139, 139, 0.95)",     # Schluff
    convex_hulls_file_6: "rgba(94, 60, 153, 0.95)",       # Sandiger Schluff
    convex_hulls_file_7: "rgba(123, 204, 196, 0.95)",     # Lehm
    convex_hulls_file_8: "rgba(240, 228, 66, 0.95)",      # Lehmiger Sand
    convex_hulls_file_9: "rgba(0, 158, 115, 0.95)",       # Schluffiger Sand
    convex_hulls_file_10: "rgba(0, 90, 160, 0.95)",       # Lehmiger Ton
    convex_hulls_file_11: "rgba(51, 51, 51, 0.95)",       # Ton
    convex_hulls_file_12: "rgba(86, 180, 233, 0.95)",     # Sand
    convex_hulls_file_14: "rgba(101, 67, 33, 0.50)",    # Humic Soils
    convex_hulls_file_15: "rgba(160, 180, 80, 0.95)",        # Schluffiger Lehm
}


legend_mapping = {
    convex_hulls_file_1: "Sandy Clay",               # Sandiger Ton
    convex_hulls_file_2: "Clayey Sand",              # Toniger Sand
    convex_hulls_file_3: "Sandy Loam",               # Sandiger Lehm
    convex_hulls_file_4: "Loamy Silt",               # Lehmiger Schluff
    convex_hulls_file_5: "Silt",                     # Schluff
    convex_hulls_file_6: "Sandy Silt",               # Sandiger Schluff
    convex_hulls_file_7: "Loam",                     # Lehm
    convex_hulls_file_8: "Loamy Sand",               # Lehmiger Sand
    convex_hulls_file_9: "Silty Sand",               # Schluffiger Sand
    convex_hulls_file_10: "Clay Loam",               # Lehmiger Ton
    convex_hulls_file_11: "Clay",                    # Ton
    convex_hulls_file_12: "Sand",                    # Sand
    convex_hulls_file_14: "Humic Soils",             # Humose Böden
    convex_hulls_file_15: "Silty Loam",  # Schluffiger Lehm

}



# Rectangle data with the classification system up to AB1
rechtecke = [
    (0, 100, "AB99"), (100, 99, "AB98"), (199, 98, "AB97"), (297, 97, "AB96"), (394, 96, "AB95"),
    (490, 95, "AB94"), (585, 94, "AB93"), (679, 93, "AB92"), (772, 92, "AB91"), (864, 91, "AB90"),
    (955, 90, "AB89"), (1045, 89, "AB88"), (1134, 88, "AB87"), (1222, 87, "AB86"), (1309, 86, "AB85"),
    (1395, 85, "AB84"), (1480, 84, "AB83"), (1564, 83, "AB82"), (1647, 82, "AB81"), (1729, 81, "AB80"),
    (1810, 80, "AB79"), (1890, 79, "AB78"), (1969, 78, "AB77"), (2047, 77, "AB76"), (2124, 76, "AB75"),
    (2200, 75, "AB74"), (2275, 74, "AB73"), (2349, 73, "AB72"), (2422, 72, "AB71"), (2494, 71, "AB70"),
    (2565, 70, "AB69"), (2635, 69, "AB68"), (2704, 68, "AB67"), (2772, 67, "AB66"), (2839, 66, "AB65"),
    (2905, 65, "AB64"), (2970, 64, "AB63"), (3034, 63, "AB62"), (3097, 62, "AB61"), (3159, 61, "AB60"),
    (3220, 60, "AB59"), (3280, 59, "AB58"), (3339, 58, "AB57"), (3397, 57, "AB56"), (3454, 56, "AB55"),
    (3510, 55, "AB54"), (3565, 54, "AB53"), (3619, 53, "AB52"), (3672, 52, "AB51"), (3724, 51, "AB50"),
    (3775, 50, "AB49"), (3825, 49, "AB48"), (3874, 48, "AB47"), (3922, 47, "AB46"), (3969, 46, "AB45"),
    (4015, 45, "AB44"), (4060, 44, "AB43"), (4104, 43, "AB42"), (4147, 42, "AB41"), (4189, 41, "AB40"),
    (4230, 40, "AB39"), (4270, 39, "AB38"), (4309, 38, "AB37"), (4347, 37, "AB36"), (4384, 36, "AB35"),
    (4420, 35, "AB34"), (4455, 34, "AB33"), (4489, 33, "AB32"), (4522, 32, "AB31"), (4554, 31, "AB30"),
    (4585, 30, "AB29"), (4615, 29, "AB28"), (4644, 28, "AB27"), (4672, 27, "AB26"), (4699, 26, "AB25"),
    (4725, 25, "AB24"), (4750, 24, "AB23"), (4774, 23, "AB22"), (4797, 22, "AB21"), (4819, 21, "AB20"),
    (4840, 20, "AB19"), (4860, 19, "AB18"), (4879, 18, "AB17"), (4897, 17, "AB16"), (4914, 16, "AB15"),
    (4930, 15, "AB14"), (4945, 14, "AB13"), (4959, 13, "AB12"), (4972, 12, "AB11"), (4984, 11, "AB10"),
    (4995, 10, "AB9"), (5005, 9, "AB8"), (5014, 8, "AB7"), (5022, 7, "AB6"), (5029, 6, "AB5"),
    (5035, 5, "AB4"), (5040, 4, "AB3"), (5044, 3, "AB2"), (5047, 2, "AB1")
]

# Set up diagram
fig = go.Figure()


def ensure_transparency(color, alpha=0.7):
    if "rgba" in color:
        return color[:color.rfind(",")] + f", {alpha})"
    elif color.startswith("#"):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    else:
        return f"rgba(0, 0, 0, {alpha})"

def strip_alpha(rgba):
    return rgba.rsplit(",", 1)[0] + ")" if "rgba" in rgba else rgba


# Add rectangles with color gradients along the new x-axis (after rotation)
def add_rechtecke_mit_farbverlauf(rechtecke, x_offset, spiegeln=False):
    for i, (y_position, hoehe, label) in enumerate(rechtecke):
        breite = i + 1
        gradient_steps = 10  # Number of steps in the color gradient


        grau_start = 180  # Dark gray tone (RGB value 80)
        grau_ende = 220  # Light gray tone (RGB value 200)

        for step in range(gradient_steps):
            # Calculate the gray value within an AB-rectanglce
            grau_wert = int(grau_start + (grau_ende - grau_start) * (step / (gradient_steps - 1)))

            # Variation of transparency to achieve a smoother effect
            alpha = 0.8 - (0.6 * (step / (gradient_steps - 1)))
            color = f'rgba({grau_wert}, {grau_wert}, {grau_wert}, {alpha})'

            # Determine the coordinates for the gradient along the new x-axis (sum A + B)
            y_start = y_position + (step / gradient_steps) * hoehe
            y_end = y_position + ((step + 1) / gradient_steps) * hoehe

            if spiegeln:
                x_start, x_end = x_offset - breite, x_offset
            else:
                x_start, x_end = x_offset, x_offset + breite

            # Add rectangle for the current step
            fig.add_trace(go.Scatter(
                x=[x_start, x_start, x_end, x_end, x_start],
                y=[y_start, y_end, y_end, y_start, y_start],
                fill="toself",
                mode="lines",
                fillcolor=color,
                line=dict(color="gray", width=0)  # Optional: schmale Umrandung
            ))

        for x_pos in range(1, breite):
            x_val = x_offset - x_pos if spiegeln else x_offset + x_pos
            fig.add_trace(go.Scatter(
                x=[x_val, x_val],
                y=[y_position, y_position + hoehe],
                mode="lines",
                line=dict(color="gray", width=2),
                showlegend=False
            ))

    # Create  axis labels
    x_labels = {50: "AB99",440: "AB95",  915: "AB90", 1350: "AB85", 1760: "AB80", 2158: "AB75", 2540: "AB70",
                 2870: "AB65", 3195: "AB60", 3480: "AB55", 3755: "AB50", 3995: "AB45", 4209: "AB40", 4405: "AB35", 4570: "AB30", 4830: "AB20", 4990: "AB10" }


    # Update X-axis with labels
    fig.update_layout(
    xaxis=dict(
        title=dict(
            text="Summe A + B (%)",
            font=dict(size=24, color="black")
        ),
        tickvals=list(x_labels.keys()),
        ticktext=list(x_labels.values()),
        tickangle=0
    ))



# Add rectangles
add_rechtecke_mit_farbverlauf(rechtecke, 0)


# Load data from Excel file (relative path for GitHub)
file_path_gilgen = "data/Komp_Pub.xlsx"
df = pd.read_excel(file_path_gilgen, sheet_name='Boden_O')


# Remove rows with NaN in columns "Unnamed: 1" to "Unnamed: 4"; filter only rows where the sum is >= 98
df_parameters = df[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].dropna()
df_parameters = df_parameters[df_parameters.apply(lambda row: row[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].sum() >= 98, axis=1)]


def round_to_100(row):
    values = {
        'Unnamed: 1': row['Unnamed: 1'],
        'Unnamed: 2': row['Unnamed: 2'],
        'Unnamed: 3': row['Unnamed: 3'],
        'Unnamed: 4': row['Unnamed: 4']
    }

    # Round down all values and calculate rest
    floored = {k: int(np.floor(v)) for k, v in values.items()}
    decimal_parts = {k: values[k] - floored[k] for k in values}

    total = sum(floored.values())
    missing = 100 - total

    # Distribute the missing percentage points to the largest decimal parts
    for k in sorted(decimal_parts, key=decimal_parts.get, reverse=True):
        if missing <= 0:
            break
        floored[k] += 1
        missing -= 1

    # Aktualisiere die Werte im Original-Row
    for k in floored:
        row[k] = floored[k]
    return row


# Apply the function to the DataFrame
df_parameters = df_parameters.apply(round_to_100, axis=1)

#  Load origin and index number (with the filtered rows without NaN
df_parameters['Herkunft'] = df.loc[df_parameters.index, 'Unnamed: 5'].values
df_parameters['Index'] = df.loc[df_parameters.index, 'Unnamed: 6'].values
df_parameters['location'] = df.loc[df_parameters.index, 'Unnamed: 7'].values


# Update the values in the original row
def adjust_sum_to_100(row):
    total = row['Unnamed: 1'] + row['Unnamed: 2'] + row['Unnamed: 3'] + row['Unnamed: 4']
    difference = 100 - total
    if difference != 0:
        row['Unnamed: 4'] += difference  # Passe den letzten Parameter an, um die Summe auf 100 zu bringen
    return row

# Apply the adjustment only to rows where the sum is between 98 and 100
df_parameters = df_parameters.apply(
    lambda row: adjust_sum_to_100(row) if 98 <= row[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].sum() <= 100 else row,
    axis=1
)


# Calculate AB (A + B) for the y-position
df_parameters['AB'] = df_parameters['Unnamed: 1'] + df_parameters['Unnamed: 2']

# Compute y-position based on AB
def calculate_y_position(ab_value, b_value):
    ab_index = 99 - int(ab_value)
    if ab_index >= 0 and ab_index < len(rechtecke):
        start_zeile = rechtecke[ab_index][0]
        hoehe = rechtecke[ab_index][1]
        y_position = start_zeile + b_value + 0.5
        return y_position
    return None

herkunfts_list = df_parameters['Herkunft'].unique()
color_palette = px.colors.qualitative.Plotly  # Plotly-Farben
color_mapping = {herkunft: color_palette[i % len(color_palette)] for i, herkunft in enumerate(herkunfts_list)}



#  Manually ordered list of convex hulls in the legend
ordered_hulls = [
    convex_hulls_file_11,  # Clay
    convex_hulls_file_10,  # Loamy Clay
    convex_hulls_file_1,   # Sandy Clay
    convex_hulls_file_2,   # Clayey Sand
    convex_hulls_file_7,   # Loam
    convex_hulls_file_4,   # Silty Loam 
    convex_hulls_file_15,  # Loamy Silt 
    convex_hulls_file_5,   # Silt
    convex_hulls_file_3,   # Sandy Loam
    convex_hulls_file_8,   # Loamy Sand
    convex_hulls_file_6,   # Sandy Silt
    convex_hulls_file_9,   # Silty Sand
    convex_hulls_file_12,  # Sand
    convex_hulls_file_14,  # Humic Soils

]


# Mapping of RGBA colors to sub-classes
farbe_to_subklasse = {
    "rgba(86, 180, 233)": "Sand",
    "rgba(0, 158, 115)": "Silty Sand",
    "rgba(240, 228, 66)": "Loamy Sand",
    "rgba(94, 60, 153)": "Sandy Silt",
    "rgba(139, 139, 139)": "Silt",
    "rgba(204, 121, 167)": "Clayey Sand",
    "rgba(178, 34, 34)": "Sandy Loam",
    "rgba(253, 192, 134)": "Loamy Silt",
    "rgba(160, 82, 45)": "Sandy Clay",
    "rgba(123, 204, 196)": "Loam",
    "rgba(0, 114, 178)": "Clay Loam",
    "rgba(51, 51, 51)": "Clay",
    "rgba(160, 180, 80)": "Silty Loam",
    "rgba(101, 67, 33)": "Humic Soils",
}




#  Recreate legend text as HTML with fixed display order
legende_text = (
    "<span style='font-size:24px; font-weight:bold;'>Soil texture classes</span><br>"

)

# Predefined display order of the legend
ordered_legende = [
    "Clay",
    "Clay Loam",
    "Sandy Clay",
    "Clayey Sand",
    "Loam",
    "Loamy Silt",
    "Silty Loam",
    "Silt",
    "Sandy Loam",
    "Loamy Sand",
    "Sandy Silt",
    "Silty Sand",
    "Sand",
    "Humic Soils"
]

for name in ordered_legende:
    farbe = next((k for k, v in farbe_to_subklasse.items() if v == name), None)
    if farbe:
        legende_text += f'<span style="color:{farbe}; font-size:40px;">■</span> {name}<br>'


# Set to track already added class names
already_added = set()

for file_path in ordered_hulls:
    hull_name = legend_mapping.get(file_path, file_path.split("\\")[-1].split(".")[0])
    if hull_name not in already_added:
        # Einheitliches Mapping für alle Klassen
        regel_farb_mapping = {
            legend_mapping[file]: color_mapping_files[file]
            for file in legend_mapping
        }


#List to group points by origin and rectangle (AB)
grouped_points = {}

df_hulls_combined = pd.concat([
    pd.read_excel(path).assign(file_source=path)
    for path in color_mapping_files
])

# Group the hull data based on origin and AB_Value
grouped_hulls_combined = df_hulls_combined.groupby(["Soil texture class", "AB_Value"])


# Function to plot convex hulls with different colors based on source file
def plot_imported_hulls_with_file_colors(grouped_hulls, file_color_mapping):
    for (soil_class, ab_value), group in grouped_hulls:
        # Extract X and Y coordinates of hull points
        hull_x = group["X"].values
        hull_y = group["Y"].values

        # Close the hull by appending the first point to the end
        hull_x = np.append(hull_x, hull_x[0])
        hull_y = np.append(hull_y, hull_y[0])

        # Determine the color based on the source file
        file_source = group["file_source"].iloc[0]
        color = file_color_mapping.get(file_source, "rgba(0, 0, 0, 0.5)")

        # Plot  convex hull
        fig.add_trace(go.Scatter(
            x=hull_x,
            y=hull_y,
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="toself",
            fillcolor=color if file_source in [convex_hulls_file_14]
            else ensure_transparency(color, alpha=0.60),
            name=f"Class: {soil_class}, AB: {ab_value}"
        ))

# Plot the imported convex hulls using file-specific colors
plot_imported_hulls_with_file_colors(grouped_hulls_combined, color_mapping_files)

color_count = {}

for idx, row in df_parameters.iterrows():
    a = row['Unnamed: 1']
    b = row['Unnamed: 2']
    c = row['Unnamed: 3']
    d = row['Unnamed: 4']
    herkunft = row['Herkunft']
    index = row['Index']
    location = row['location']
    ab_value = row['AB']
    color = None

    # Normalization and Texture Classification of Soil Data; Color-coded point plotting based on color of subfields
    multiplikator = (100 - c) / 100


    # Sand 1 – Sky blue (part 1 of a concave region)
    if (80 * multiplikator <= a <= 100 * multiplikator) and (0 <= b <= 20 * multiplikator) and (
            0 <= d <= 10 * multiplikator):
        color = "rgba(86, 180, 233, 0.75)"  # Sky blue

    # Sand 2 – Sky Blue (Part 2 of the concave area)
    elif (65 * multiplikator <= a <= 80 * multiplikator) and (15 * multiplikator <= b <= 30 * multiplikator) and (
            0 <= d <= 5 * multiplikator):
        color = "rgba(86, 180, 233, 0.75)"  # Sky Blue

    # Silty Sand – Green (convex_hulls_file_9)
    elif (40 * multiplikator <= a <= 70 * multiplikator) and (30 * multiplikator <= b <= 55 * multiplikator) and (
            0 <= d <= 5 * multiplikator):
        color = "rgba(0, 158, 115, 0.75)"

    # Loamy Sand – Yellow (convex_hulls_file_8)
    elif (30 * multiplikator <= a <= 80 * multiplikator) and (10 * multiplikator <= b <= 55 * multiplikator) and (
            5 * multiplikator <= d <= 15 * multiplikator):
        color = "rgba(240, 228, 66, 0.75)"

    # Sandy Silt – Violet (convex_hulls_file_6)
    elif (10 * multiplikator <= a <= 45 * multiplikator) and (55 * multiplikator <= b <= 75 * multiplikator) and (
            0 <= d <= 15 * multiplikator):
        color = "rgba(94, 60, 153, 0.75)"

    # Silt – Silver (convex_hulls_file_5)
    elif (0 <= a <= 25 * multiplikator) and (75 * multiplikator <= b <= 100 * multiplikator) and (
            0 <= d <= 25 * multiplikator):
        color = "rgba(139, 139, 139, 0.75)"  # NEW!

    # Clayey Sand – Turquoise Green (convex_hulls_file_2)
    elif (65 * multiplikator <= a <= 90 * multiplikator) and (0 <= b <= 10 * multiplikator) and (
            10 * multiplikator <= d <= 25 * multiplikator):
        color = "rgba(204, 121, 167, 0.75)"

    # Sandy Loam – Firebrick (convex_hulls_file_3)
    elif (20 * multiplikator <= a <= 75 * multiplikator) and (10 * multiplikator <= b <= 55 * multiplikator) and (
            15 * multiplikator <= d <= 25 * multiplikator):
        color = "rgba(178, 34, 34, 0.75)"

    # Loamy Silt – Light Orange (convex_hulls_file_4)
    elif (0 <= a <= 30 * multiplikator) and (55 * multiplikator <= b <= 75 * multiplikator) and (
            15 * multiplikator <= d <= 25 * multiplikator):
        color = "rgba(253, 192, 134, 0.75)"

    # Sandy Clay – Sienna (convex_hulls_file_1)
    elif (50 * multiplikator <= a <= 75 * multiplikator) and (0 <= b <= 10 * multiplikator) and (
            25 * multiplikator <= d <= 40 * multiplikator):
        color = "rgba(160, 82, 45, 0.75)"

    # Loam – Turquoise Gray (convex_hulls_file_7)
    elif (5 * multiplikator <= a <= 65 * multiplikator) and (10 * multiplikator <= b <= 55 * multiplikator) and (
            25 * multiplikator <= d <= 40 * multiplikator):
        color = "rgba(123, 204, 196, 0.70)"

    # Silty Loam – Dusty Pink (convex_hulls_file_15)
    elif (0 <= a <= 20 * multiplikator) and (55 * multiplikator <= b <= 75 * multiplikator) and (
            25 * multiplikator <= d <= 45 * multiplikator):
        color = "rgba(160, 180, 80, 0.75)"

    # Loamy Clay – Blue (convex_hulls_file_10)
    elif (0 <= a <= 60 * multiplikator) and (0 <= b <= 55 * multiplikator) and (
            40 * multiplikator <= d <= 50 * multiplikator):
        color = "rgba(0, 114, 178, 0.75)"

    # Clay – Dark Gray (convex_hulls_file_11)
    elif (0 <= a <= 50 * multiplikator) and (0 <= b <= 50 * multiplikator) and (
            50 * multiplikator <= d <= 100 * multiplikator):
        color = "rgba(51, 51, 51, 0.75)"

    # Special case: if humus content (C) is very high, assign specific color
    if c > 30:
        color = "rgba(80, 80, 80, 0.9)"  # Dark gray tone for very high humus
    elif c > 15:
        color = "rgba(101, 67, 33, 0.9)"  # Brown for high humus


    if color is None:
        color = find_nearest_class(a, b, c, d)

    # ✅ Increment counter – regardless of whether assigned by rule or fallback
    if color not in color_count:
        color_count[color] = 1
    else:
        color_count[color] += 1

    # **DEBUG output**
    y_position_punkt = calculate_y_position(ab_value, b)
    print(f"DEBUG -> Index: {index}, A={a}, B={b}, D={d}, C={c}, AB={ab_value}, y={y_position_punkt}, Farbe={color}")

    # Mapping colors to English soil texture classes (for hover text)
    color_to_soil_class = {
        "rgba(86, 180, 233, 0.75)": "Sand",
        "rgba(0, 158, 115, 0.75)": "Silty Sand",
        "rgba(240, 228, 66, 0.75)": "Loamy Sand",
        "rgba(94, 60, 153, 0.75)": "Sandy Silt",
        "rgba(139, 139, 139, 0.75)": "Silt",
        "rgba(204, 121, 167, 0.75)": "Clayey Sand",
        "rgba(178, 34, 34, 0.75)": "Sandy Loam",
        "rgba(253, 192, 134, 0.75)": "Loamy Silt",
        "rgba(160, 82, 45, 0.75)": "Sandy Clay",
        "rgba(123, 204, 196, 0.70)": "Loam",
        "rgba(0, 114, 178, 0.75)": "Clay Loam",
        "rgba(51, 51, 51, 0.75)": "Clay",
        "rgba(245, 222, 179, 0.5)": "Humic Soils",
        "rgba(160, 180, 80, 0.75)": "Silty Loam"
    }
    soil_class = color_to_soil_class.get(color, "Unknown")

    # **Plot point**
    fig.add_trace(go.Scatter(
        x=[c],
        y=[y_position_punkt],
        mode='markers',
        marker=dict(
            symbol='circle',
            size=17,
            color=color,
            opacity=1,
            line=dict(color="black", width=3)

    ),
        showlegend=False,
        hovertemplate=(
            f"<b>Index:</b> {index}<br>"
            f"<b>Soil texture class:</b> {soil_class}<br>"
            f"<b>Location:</b> {location}<br>"  
            f"Sand (A): {a}%<br>"
            f"Silt (B): {b}%<br>"
            f"Humus (C): {c}%<br>"
            f"Clay (D): {d}%<extra></extra>"
        )

    ))



#  Output class distribution summary
print("\n--- Soil texture class distribution based on number of points ---")
total_points = sum(color_count.values())

for color_code, count in sorted(color_count.items(), key=lambda x: x[1], reverse=True):
    base_color = strip_alpha(color_code)
    soil_class = farbe_to_subklasse.get(base_color, "Unknown")
    percent = (count / total_points) * 100
    print(f"{soil_class}: {count} points ({percent:.1f} %)")




# Adjust layout to center and rotate plot by 90 degrees
fig.update_layout(
plot_bgcolor="white",  # Set plot background to white
    paper_bgcolor="white",  # Set whole figure background to white
   xaxis=dict(
    title=dict(
        text="Sum of Sand % (A) and Silt % (B)",
        font=dict(size=35, color="black", family="Arial Black")
    ),
    range=[0, rechtecke[-1][0] + rechtecke[-1][1]],
    tickformat=".0f",
    tickfont=dict(size=25, color="black")
),

   yaxis=dict(
    title=dict(
        text="Humus (%) /// Difference between height of AB rectangle and the Humus content (%) equals Clay content (%)",
        font=dict(size=22, color="black", family="Arial Black")
    ),
    range=[0, 16],
    tickformat=".0f",
    dtick=5,
    color="black",
    linecolor="gray",
    tickfont=dict(size=25, color="black")

    ),
    autosize=False,  # Disable automatic sizing
    width=2260,  # Set plot width
    height=1210,  # Set plot height
    margin=dict(l=0, r=5, t=20, b=5),  # Center the plot by minimizing margins
    showlegend=False  # Disable the legend
)
# **Add legend as an annotation**
fig.update_layout(
    annotations=[
        dict(
            x=15,
            y=15.5,
            text=legende_text,
            showarrow=False,
            font=dict(size=28, color="black"),
            bgcolor="rgba(249, 249, 249,1)",
            bordercolor="black",
            borderwidth=2,
            xanchor="left",
            yanchor="top",
            align="left"
        )
    ]
)


# Y-values where dashed horizontal lines should be added
y_values = [ 10, 20, 30, 40, 50, 60, 70, 80, 90 ]

# Add horizontal dashed lines
for y in y_values:
    fig.add_shape(
        type="line",
        x0=0,   # Start point on the X-axis (left edge of the chart)
        x1=max(df_parameters['Unnamed: 3']) + rechtecke[-1][0] + rechtecke[-1][1],  # End point on the X-axis (right edge)
        y0=y,   # Y-position where the line is drawn
        y1=y,   # Constant Y (horizontal line)
        line=dict(
            color="black",  # Line color
            width=0.5,        # Line width
            dash="dash"    # Dashed line style
        )
    )

# X-values for vertical dashed lines
x_values = [909.5, 3749.5, 4569.5, 1769.5, 2529.5, 3189.5, 4209.5, 4850.5, 4995.5, 442, 1352, 2162, 2872, 3482, 3992, 4402, 4712, 4922, 5037.5]  # Mittlere Positionen von AB90, AB50, AB30

# Add vertical dashed lines
for x in x_values:
    fig.add_shape(
        type="line",
        x0=x,  # X-position where the line is drawn
        x1=x,  # Constant X (vertical line)
        y0=0,  # Start point on the Y-axis
        y1=100,  # End point on the Y-axis (maximum)
        line=dict(
            color="black",  # Line color
            width=1,       # Line width
            dash="dash"     # Dashed line style
        )
    )

# Rotate the plot 90 degrees by swapping X and Y data
for trace in fig.data:
    trace.x, trace.y = trace.y, trace.x

# Update hover label styling
fig.update_layout(
    hoverlabel=dict(
        font_size=22,
        font_family="Arial"
    )
)

# Show plot
fig.show()
