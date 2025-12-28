# Cantor diagram of Austrian soil texture classes with humus as 4th component (data Kirchbach)
# -------------------------------------------------------------------------
# Creates an interactive Cantor plot showing:
#   • AB-rectangles (AB99–AB1)
#   • Austrian soil texture classes via convex hulls
#   • Classification of samples based on sand, silt, clay and humus
#   • AWC (Available Water Capacity) displayed as color halo
#   • Inner points colored by soil texture class
#   • Export to HTML, PNG (Playwright), and TIFF (400 dpi)
#   • Plots 108 points from the district Kirchdorf

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


# Largest Remainder Method
def normalize_to_100_with_remainders(row):
    cols = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    values = row[cols].astype(float).to_numpy()

    ints = np.floor(values).astype(int)
    remainders = values - np.floor(values)
    missing = 100 - ints.sum()

    if missing > 0:
        order = np.argsort(-remainders)
        for i in range(missing):
            ints[order[i]] += 1
    elif missing < 0:
        order = np.argsort(remainders)
        for i in range(abs(missing)):
            ints[order[i]] -= 1

    row[cols] = ints
    return row


# file paths

convex_hulls_file_1 = "data/convex_hull_sandy_clay.xlsx"
convex_hulls_file_2 = "data/convex_hull_clayey_sand.xlsx"
convex_hulls_file_3 = "data/convex_hull_sandy_loam.xlsx"
convex_hulls_file_4 = "data/convex_hull_loamy_silt.xlsx"
convex_hulls_file_5 = "data/convex_hull_silt.xlsx"
convex_hulls_file_6 = "data/convex_hull_sandy_silt.xlsx"
convex_hulls_file_7 = "data/convex_hull_loam.xlsx"
convex_hulls_file_8 = "data/convex_hull_loamy_sand.xlsx"
convex_hulls_file_9 = "data/convex_hull_silty_sand.xlsx"
convex_hulls_file_10 = "data/convex_hull_clay_loam.xlsx"
convex_hulls_file_11 = "data/convex_hull_clay.xlsx"
convex_hulls_file_12 = "data/convex_hull_sand.xlsx"
convex_hulls_file_14 = "data/convex_hull_organo_mineral_soils.xlsx"
convex_hulls_file_15 = "data/convex_hull_silty_loam.xlsx"
convex_hulls_file_16 = "data/convex_hull_organicsoils.xlsx"


# Transparency
ALPHA_HULL = 0.50
ALPHA_POINT = 0.5
ALPHA_LEGEND = 0.5




# color mappings
color_mapping_files = {
    convex_hulls_file_1: "rgba(160, 82, 45, 0.75)",       
    convex_hulls_file_2: "rgba(57, 255, 20, 0.85)",    
    convex_hulls_file_3: "rgba(178, 34, 34, 0.75)",       
    convex_hulls_file_4: "rgba(253, 192, 134, 0.75)",     
    convex_hulls_file_5: "rgba(70, 70, 70, 0.75)",      
    convex_hulls_file_6: "rgba(94, 60, 153, 0.75)",       
    convex_hulls_file_7: "rgba(110, 165, 160, 0.75)",     
    convex_hulls_file_8: "rgba(225, 195, 65, 0.75)",      
    convex_hulls_file_9: "rgba(0, 158, 115, 0.75)",       
    convex_hulls_file_10: "rgba(0, 60, 140, 0.75)",      
    convex_hulls_file_11: "rgba(17, 17, 17, 0.85)",       
    convex_hulls_file_12: "rgba(86, 180, 233, 0.75)",     
    convex_hulls_file_14: "rgba(100, 95, 90, 0.2)",  
    convex_hulls_file_15: "rgba(204, 121, 167, 0.75)",  
    convex_hulls_file_16: "rgba(100, 95, 90, 0.2)",
}




legend_mapping = {
    convex_hulls_file_1: "Sandy Clay",
    convex_hulls_file_2: "Clayey Sand",
    convex_hulls_file_3: "Sandy Loam",
    convex_hulls_file_4: "Loamy Silt",
    convex_hulls_file_5: "Silt",
    convex_hulls_file_6: "Sandy Silt",
    convex_hulls_file_7: "Loam",
    convex_hulls_file_8: "Loamy Sand",
    convex_hulls_file_9: "Silty Sand",
    convex_hulls_file_10: "Clay Loam",
    convex_hulls_file_11: "Clay",
    convex_hulls_file_12: "Sand",
    convex_hulls_file_14: "Organo-Mineral Soils",
    convex_hulls_file_15: "Silty Loam",
    convex_hulls_file_16: "Organic Soils"
}


# Soil class → color
soilclass_to_color = {}

for file_path, soil_class in legend_mapping.items():
    soilclass_to_color[soil_class] = color_mapping_files[file_path]


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


def apply_alpha(color, alpha):
    """
    Force a specific alpha value on any rgba / hex color.
    This is the ONLY place where transparency is controlled.
    """
    if color.startswith("rgba"):
        r, g, b, _ = color.replace("rgba(", "").replace(")", "").split(",")
        return f"rgba({r.strip()},{g.strip()},{b.strip()},{alpha})"
    elif color.startswith("#"):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return color


# Add rectangles with color gradients along the X-axis
def add_rechtecke_mit_farbverlauf(rechtecke, x_offset, spiegeln=False):
    for i, (y_position, hoehe, label) in enumerate(rechtecke):
        breite = i + 1
        gradient_steps = 14  # Number of steps in the color gradient


        grau_start = 180  # Dark gray tone
        grau_ende = 220  # Light gray tone

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

        # add rectangle polygon for this gradient step
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
            title="Summe A + B (%)",
            tickvals=list(x_labels.keys()),
            ticktext=list(x_labels.values()),
            tickangle=0,

        ))


# Add rectangles
add_rechtecke_mit_farbverlauf(rechtecke, 0)

# Load data from Excel file (relative path for GitHub)
file_path_gilgen = "data/compendium.xlsx"
df = pd.read_excel(file_path_gilgen, sheet_name='Soil_Kirchdorf')

# 🔒 Original decimal texture values (before LRM)
df_tex_raw = df[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].copy()


# Load origin and index number
df_parameters = df[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].dropna()
df_parameters = df_parameters[df_parameters.apply(lambda row: row[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].sum() >= 98, axis=1)]


#  Function to adjust so that the sum equals 100
def normalize_to_100_LRM(row):
    cols = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    values = row[cols].astype(float).to_numpy()

    # 1. Abrunden (Floor)
    ints = np.floor(values).astype(int)

    # 2. Reste berechnen
    remainders = values - ints

    # 3. Differenz zur Zielsumme 100
    missing = 100 - ints.sum()

    if missing > 0:
        order = np.argsort(-remainders)
        for i in range(missing):
            ints[order[i]] += 1

    elif missing < 0:
        order = np.argsort(remainders)
        for i in range(-missing):
            ints[order[i]] -= 1

    row[cols] = ints
    return row

df_parameters = df_parameters.apply(normalize_to_100_LRM, axis=1)


#  Load origin and index number
df_parameters['Herkunft'] = df.loc[df_parameters.index, 'Unnamed: 5'].values
df_parameters['Index'] = df.loc[df_parameters.index, 'Unnamed: 6'].values
df_parameters['location'] = df.loc[df_parameters.index, 'Unnamed: 7'].values

# Check contents and possible inconsistencies in 'location'
duplicates = df_parameters["location"].value_counts()
print("\n=== Locations with multiple entries (Top 10) ===")
print(duplicates[duplicates > 1].head(10))

print("\n=== Example variations of location names ===")
print(df_parameters["location"].drop_duplicates().sort_values().head(20))


# Calculate AB = A + B for the y-position
df_parameters['AB'] = df_parameters['Unnamed: 1'] + df_parameters['Unnamed: 2']

def calculate_y_position_exact(a, b, c, d):
    # Compute AB index for selecting the correct rectangle
    ab_value = a + b
    ab_index = 99 - int(ab_value)

    # Check index validity
    if 0 <= ab_index < len(rechtecke):
        start_zeile = rechtecke[ab_index][0]   # Top of the rectangle
        hoehe = rechtecke[ab_index][1]         # Height of the rectangle

        # Split C and D into integer and fractional parts
        c_int, d_int = int(c), int(d)
        c_frac, d_frac = c - c_int, d - d_int

        # Total fractional distribution across the field
        field_sum = c + d
        field_int = int(round(field_sum))
        field_frac = field_sum - field_int

        total_frac = c_frac + d_frac
        if total_frac > 0:
            frac_c = c_frac / total_frac
            frac_d = d_frac / total_frac
        else:
            frac_c = frac_d = 0.5

        # Fractional offset within the field
        offset_ratio = field_frac * frac_c

        # Final y-position inside the rectangle
        y_position = start_zeile + b + offset_ratio * hoehe
        return y_position

    return None


def classify_soil(a, b, c, d):
    # Mineral fraction
    m = (100 - c) / 100.0

    # Organic soil classes
    if c > 35: return "Organic Soils"
    if c > 15: return "Organo-Mineral Soils"

    # Texture classes
    if (80*m <= a <= 100*m) and (0 <= b <= 20*m) and (0 <= d <= 10*m): return "Sand"
    if (65*m <= a <= 80*m) and (15*m <= b <= 30*m) and (0 <= d <= 5*m): return "Sand"
    if (40*m <= a <= 70*m) and (30*m <= b <= 55*m) and (0 <= d <= 5*m): return "Silty Sand"
    if (30*m <= a <= 80*m) and (10*m <= b <= 55*m) and (5*m <= d <= 15*m): return "Loamy Sand"
    if (10*m <= a <= 45*m) and (55*m <= b <= 75*m) and (0 <= d <= 15*m): return "Sandy Silt"
    if (0 <= a <= 25*m)  and (75*m <= b <= 100*m) and (0 <= d <= 25*m): return "Silt"
    if (65*m <= a <= 90*m) and (0 <= b <= 10*m)  and (10*m <= d <= 25*m): return "Clayey Sand"
    if (20*m <= a <= 75*m) and (10*m <= b <= 55*m) and (15*m <= d <= 25*m): return "Sandy Loam"
    if (0 <= a <= 30*m)  and (55*m <= b <= 75*m) and (15*m <= d <= 25*m): return "Loamy Silt"
    if (50*m <= a <= 75*m) and (0 <= b <= 10*m)  and (25*m <= d <= 40*m): return "Sandy Clay"
    if (5*m <= a <= 65*m)  and (10*m <= b <= 55*m) and (25*m <= d <= 40*m): return "Loam"
    if (0 <= a <= 20*m)  and (55*m <= b <= 75*m) and (25*m <= d <= 45*m): return "Silty Loam"
    if (0 <= a <= 60*m)  and (0 <= b <= 55*m)  and (40*m <= d <= 50*m): return "Clay Loam"
    if (0 <= a <= 50*m)  and (0 <= b <= 50*m)  and (50*m <= d <= 100*m): return "Clay"
    return "Other"


# Unique origin categories and color mapping
herkunfts_list = df_parameters['Herkunft'].unique()
color_palette = px.colors.qualitative.Plotly
color_mapping = {herkunft: color_palette[i % len(color_palette)] for i, herkunft in enumerate(herkunfts_list)}



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

        hull_x = group["X"].values
        hull_y = group["Y"].values

        hull_x = np.append(hull_x, hull_x[0])
        hull_y = np.append(hull_y, hull_y[0])

        file_source = group["file_source"].iloc[0]
        color = file_color_mapping.get(file_source, "rgba(0, 0, 0, 0.5)")

        # Plot convex hull
        fig.add_trace(go.Scatter(
            x=hull_x,
            y=hull_y,
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="toself",
            fillcolor=(
                apply_alpha(color, 0.20)
                if file_source in [convex_hulls_file_14, convex_hulls_file_16]
                else apply_alpha(color, 0.50)
            ),
            name=f"Class: {soil_class}, AB: {ab_value}"
        ))

# Plot the imported convex hulls using file-specific colors
plot_imported_hulls_with_file_colors(grouped_hulls_combined, color_mapping_files)


# AWC calculation using the original decimal texture values (Saxton & Rawls 2006 with OM)
# Reference: Saxton & Rawls (2006), Soil Sci. Soc. Am. J. 70:1569–1578

import numpy as np

# --- AWC calculation from ORIGINAL (unrounded) values ---

tex = df_tex_raw.loc[df_parameters.index]


sand = tex['Unnamed: 1'] / 100.0
clay = tex['Unnamed: 4'] / 100.0
om_pct_raw = tex['Unnamed: 3']          # OM in %
om_pct = np.clip(om_pct_raw, 0, 8) / 100.0

# OM capped at 8% (Saxton & Rawls requirement)
om_pct = np.clip(om_pct_raw, 0, 8) / 100.0



#  Field capacity (33 kPa) according to Saxton & Rawls (2006)
theta33_t = (
    -0.251 * sand
    + 0.195 * clay
    + 0.011 * om_pct
    + 0.006 * sand * om_pct
    - 0.027 * clay * om_pct
    + 0.452 * sand * clay
    + 0.299
)
theta33 = theta33_t + (1.283 * theta33_t**2 - 0.374 * theta33_t - 0.015)

#  Wilting point (1500 kPa) according to Saxton & Rawls (2006)
theta1500_t = (
    -0.024 * sand
    + 0.487 * clay
    + 0.006 * om_pct
    + 0.005 * sand * om_pct
    - 0.013 * clay * om_pct
    + 0.068 * sand * clay
    + 0.031
)
theta1500 = theta1500_t + (0.14 * theta1500_t - 0.02)

# Insert values into df_parameters
df_parameters.loc[:, "theta33"] = theta33
df_parameters.loc[:, "theta1500"] = theta1500

# AWC in percentage of volumetric water content
df_parameters.loc[:, "AWC"] = (theta33 - theta1500).clip(lower=0) * 100


# Output: AWC minimum and maximum

awc_filtered = df_parameters[df_parameters["Unnamed: 3"] <= 8]["AWC"]

awc_min = awc_filtered.min()
awc_max = awc_filtered.max()

print(f"📊 AWC (realistic range): min={awc_min:.1f} %, max={awc_max:.1f} %")



# Debug table
dbg = df_parameters.copy()

# Store raw AWC and clipped
dbg["AWC_raw"] = dbg["AWC"]
dbg["AWC_clipped"] = dbg["AWC"].clip(lower=0)
# Rename columns for readability
dbg = dbg.rename(columns={
    'Unnamed: 1': 'Sand_%',
    'Unnamed: 2': 'Silt_%',
    'Unnamed: 3': 'Humus_%',
    'Unnamed: 4': 'Clay_%'
})

#  Mapping class → RGB color for outline (rings)
outline_colors = {
    "rgba(86, 180, 233, 0.75)": "Sand",
    "rgba(0, 158, 115, 0.75)": "Silty Sand",
    "rgba(225, 195, 65, 0.75)": "Loamy Sand",
    "rgba(94, 60, 153, 0.75)": "Sandy Silt",
    "rgba(70, 70, 70, 0.75)": "Silt",
    "rgba(57, 255, 20, 0.85)": "Clayey Sand",
    "rgba(178, 34, 34, 0.75)": "Sandy Loam",
    "rgba(253, 192, 134, 0.75)": "Loamy Silt",
    "rgba(160, 82, 45, 0.75)": "Sandy Clay",
    "rgba(123, 204, 196, 0.75)": "Loam",
    "rgba(0, 90, 160, 0.75)": "Clay Loam",
    "rgba(17, 17, 17, 0.85)": "Clay",
    "rgba(204, 121, 167, 0.75)": "Silty Loam",
    "Organo-Mineral Soils": "rgba(184, 115, 51, 0.95)",
    "Organic Soils": "rgba(120, 85, 60, 0.95)"
}

# Prepare points and rings together
points_x, points_y, awc_values, symbols, hover_texts = [], [], [], [], []
ring_x, ring_y, ring_colors = [], [], []
color_count = {}

for idx, row in df_parameters.iterrows():
    # Extract texture components
    a, b, c, d = row['Unnamed: 1'], row['Unnamed: 2'], row['Unnamed: 3'], row['Unnamed: 4']
    index, location = row['Index'], row['location']
    # Retrieve start and end depths (if columns exist)
    start_tiefe = df.loc[row.name, 'Unnamed: 16'] if 'Unnamed: 16' in df.columns else None
    end_tiefe   = df.loc[row.name, 'Unnamed: 15'] if 'Unnamed: 15' in df.columns else None
    # Skip if depth information is missing
    if pd.isna(start_tiefe) or pd.isna(end_tiefe):
        continue
    plot_ranges = [(0, 20), (70, 400)]
    marker_symbol = None
    for (low, high) in plot_ranges:
        if not (end_tiefe < low or start_tiefe > high):  # Überlappung
            if low == 0 and high == 20:
                marker_symbol = "circle"
            else:
                marker_symbol = "square"
            break

    if marker_symbol is None:
        continue
    # Classify the soil sample based on a, b, c, d
    bodenklasse = classify_soil(a, b, c, d)
    df_parameters.loc[idx, "Bodenklasse"] = bodenklasse
    color_count[bodenklasse] = color_count.get(bodenklasse, 0) + 1

    y_position_punkt = calculate_y_position_exact(a, b, c, d)
    if y_position_punkt is None:
        continue

    # Apply a small jitter to avoid point overlap
    jitter_y = np.random.uniform(-0.0, 0.0)
    jitter_x = np.random.uniform(-0.12, 0.12)
    x_val = c + jitter_x
    y_val = y_position_punkt + jitter_y


    # Handle samples with high organic matter (OM > 8%)

    humus_val = c
    is_high_om = humus_val > 8

    if is_high_om:
        if "highom_x" not in globals():
            highom_x, highom_y, highom_inner_color, highom_hovertexts = [], [], [], []

        highom_x.append(x_val)
        highom_y.append(y_val)
        highom_inner_color.append(
            soilclass_to_color.get(bodenklasse, "rgba(0,0,0,1)")
        )
        highom_hovertexts.append(
            f"<b>Index:</b> {index}<br>"
            f"<b>Location:</b> {location}<br>"
            f"Sand (A): {a:.1f}%<br>"
            f"Silt (B): {b:.1f}%<br>"
            f"Humus (C): {c:.1f}%<br>"
            f"Clay (D): {d:.1f}%<br>"
            f"<b>Class:</b> {bodenklasse}<br>"
            f"<b>AWC:</b> not applicable"
        )

        continue



    # Store coordinates for AWC-colored point
    points_x.append(x_val)
    points_y.append(y_val)
    awc_values.append(row["AWC"])
    symbols.append(marker_symbol)

    # Store hover text for the normal AWC point
    hover_texts.append(
        f"<b>Index:</b> {index}<br>"
        f"<b>Location:</b> {location}<br>"
        f"Sand (A): {a:.1f}%<br>"
        f"Silt (B): {b:.1f}%<br>"
        f"Humus (C): {c:.1f}%<br>"
        f"Clay (D): {d:.1f}%<br>"
        f"<b>Class:</b> {bodenklasse}<br>"
        f"<b>AWC:</b> {row['AWC']:.2f}"
    )

    # Outer halo shows AWC (Available Water Capacity) using a Jet colorscale

    ring_x.append(x_val)
    ring_y.append(y_val)

    # Soil-texture color for the inner core (fallback = black)
    subfeldfarbe = soilclass_to_color.get(bodenklasse, "rgba(0,0,0,0.8)")
    ring_colors.append(subfeldfarbe)


#Grid lines

for i, (start, hoehe, label) in enumerate(rechtecke):
    if i == 0:
        continue  # AB99 braucht keine Linie links davon

    x_pos = start

    fig.add_trace(go.Scatter(
        y=[x_pos, x_pos],
        x=[0, i],              # bis zur jeweiligen AB-Stufe
        mode="lines",
        line=dict(
            color="black",
            width=2,
            dash="dash"
        ),
        hoverinfo="skip",
        showlegend=False
    ))



# Add marker traces (circles and squares)
# Create Boolean masks for circle and square markers
mask_circle = np.array([s == "circle" for s in symbols])
mask_square = np.array([s == "square" for s in symbols])


# 1) CIRCLES

# 1.1 Black outer ring
fig.add_trace(go.Scatter(
    x=np.array(points_x)[mask_circle],
    y=np.array(points_y)[mask_circle],
    mode="markers",
    marker=dict(
        symbol="circle",
        size=28,
        color="rgba(0,0,0,0)",   # transparent
        line=dict(color="black", width=2)
    ),
    hoverinfo="skip",
    showlegend=False
))


# 1.2 AWC halo outer color ring + white cutout


# Outer ring (AWC color)
fig.add_trace(go.Scatter(
    x=np.array(points_x)[mask_circle],
    y=np.array(points_y)[mask_circle],
    mode="markers",
    marker=dict(
        symbol="circle",
        size=27,
        color=np.array(awc_values)[mask_circle],   # AWC values → Jet colorscale
        colorscale="Jet",
        coloraxis="coloraxis",
        opacity=0.90,
        line=dict(width=0)
    ),
    hoverinfo="skip",
    showlegend=False
))

# Inner white cutout
fig.add_trace(go.Scatter(
    x=np.array(points_x)[mask_circle],
    y=np.array(points_y)[mask_circle],
    mode="markers",
    marker=dict(
        symbol="circle",
        size=13,
        color="white",
        line=dict(width=0),
        opacity=1
    ),
    hoverinfo="skip",
    showlegend=False
))



# 1.3 Inner core (soil texture color)

fig.add_trace(go.Scatter(
    x=np.array(points_x)[mask_circle],
    y=np.array(points_y)[mask_circle],
    mode="markers",
    marker=dict(
        symbol="circle",
        size=13,
        color=np.array(ring_colors)[mask_circle],   # soil texture class color
        line=dict(color="black", width=1),
        opacity=0.8
    ),
    text=np.array(hover_texts)[mask_circle],
    hovertemplate="%{text}<extra></extra>",
    showlegend=False
))

#1.4 Black central dot

fig.add_trace(go.Scatter(
    x=np.array(points_x)[mask_circle],
    y=np.array(points_y)[mask_circle],
    mode="markers",
    marker=dict(
        symbol="circle",
        size=3,
        color="black",
        opacity=1
    ),
    hoverinfo="skip",
    showlegend=False
))


# 2) SQUARES (70–200 cm)


# 2.1 Black outer frame
fig.add_trace(go.Scatter(
    x=np.array(points_x)[mask_square],
    y=np.array(points_y)[mask_square],
    mode="markers",
    marker=dict(
        symbol="square",
        size=23,
        color="rgba(0,0,0,0)",   # transparent fill
        line=dict(color="black", width=2)
    ),
    hoverinfo="skip",
    showlegend=False
))



# AWC halo for squares (outer ring + white cutout)


# 2.2a Outer AWC square
fig.add_trace(go.Scatter(
    x=np.array(points_x)[mask_square],
    y=np.array(points_y)[mask_square],
    mode="markers",
    marker=dict(
        symbol="square",
        size=22,
        color=np.array(awc_values)[mask_square],   # AWC → Jet scale
        colorscale="Jet",
        coloraxis="coloraxis",
        opacity=0.90,
        line=dict(width=0)
    ),
    hoverinfo="skip",
    showlegend=False
))

# 2.2b White inner cutout
fig.add_trace(go.Scatter(
    x=np.array(points_x)[mask_square],
    y=np.array(points_y)[mask_square],
    mode="markers",
    marker=dict(
        symbol="square",
        size=11,
        color="white",   # background color
        line=dict(width=0),
        opacity=1
    ),
    hoverinfo="skip",
    showlegend=False
))



# 2.3 Inner core (soil texture color)

fig.add_trace(go.Scatter(
    x=np.array(points_x)[mask_square],
    y=np.array(points_y)[mask_square],
    mode="markers",
    marker=dict(
        symbol="square",
        size=11,
        color=np.array(ring_colors)[mask_square],   # soil texture class color
        line=dict(color="black", width=1),
        opacity=0.8
    ),
    text=np.array(hover_texts)[mask_square],
    hovertemplate="%{text}<extra></extra>",
    showlegend=False
))



#2.4. Black central dot
fig.add_trace(go.Scatter(

    x=np.array(points_x)[mask_square],
    y=np.array(points_y)[mask_square],
    mode="markers",
    marker=dict(
        symbol="square",
        size=3,
        color="black",
        opacity=1
    ),
    hoverinfo="skip",
    showlegend=False
))



# High-OM > 8% – Black outer ring

if "highom_x" in globals() and len(highom_x) > 0:
    fig.add_trace(go.Scatter(
        x=highom_x,
        y=highom_y,
        mode="markers",
        marker=dict(
            symbol="circle",
            size=22,
            color="rgba(0,0,0,0)",
            line=dict(color="black", width=0)
        ),
        hoverinfo="skip",
        showlegend=False
    ))


# High-OM > 8% – White outer ring (second outline)

if "highom_x" in globals() and len(highom_x) > 0:
  fig.add_trace(go.Scatter(
        x=highom_x,
        y=highom_y,
        mode="markers",
        marker=dict(
            symbol="circle",
            size=24,
            color="white",
            line=dict(color="white", width=0),
            opacity=1
        ),
        hoverinfo="skip",
        showlegend=False
    ))


    # High-OM > 8% – Inner core (soil texture class color)

fig.add_trace(go.Scatter(
    x=highom_x,
    y=highom_y,
    mode="markers",
    marker=dict(
        symbol="circle",
        size=14,
        color=[
            apply_alpha(col, 1.0) for col in highom_inner_color
        ],
        line=dict(color="black", width=0),
        opacity=1.0
    ),
    text=highom_hovertexts,
    hovertemplate="%{text}<extra></extra>",
    showlegend=False
))




# Keep only the single global coloraxis for the AWC colorbar.
fig.update_layout(
    coloraxis=dict(
        colorscale="Jet",
        cmin=awc_min,
        cmax=awc_max,
        colorbar=dict(
            title="",
            tickfont=dict(size=18),
            thickness=20,
            len=0.95,
            y=0.5,
            yanchor="middle",
        )
    )
)



# Light cleaning of the 'location' field:
df_parameters["location_clean"] = (
    df_parameters["location"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# Group by location
for loc, group in df_parameters.groupby("location_clean"):
    if group.shape[0] < 2:
        continue

        # Compute y-position for each point (depends on AB rectangle)
    group = group.copy()
    group["y_position_punkt"] = group.apply(
        lambda row: calculate_y_position_exact(
            row["Unnamed: 1"],  # A
            row["Unnamed: 2"],  # B
            row["Unnamed: 3"],  # C
            row["Unnamed: 4"]  # D
        ),
        axis=1
    )
    # Remove rows with missing y-position or missing Humus values
    group = group.dropna(subset=["y_position_punkt", "Unnamed: 3"])

    # Sort points by depth (if available) or by Index to ensure correct order
    if "Unnamed: 15" in group.columns:
        group = group.sort_values(by="Unnamed: 15")
    else:
        group = group.sort_values(by="Index")

    xs = group["Unnamed: 3"].values
    ys = group["y_position_punkt"].values


# Computes the distribution of soil classes based on the number of points)

print("\n--- Soil texture class distribution based on number of points ---")

# Fixed class order used in the diagram and legend
ordered_legende = [
    "Sand",
    "Silty Sand",
    "Sandy Silt",
    "Loamy Sand",
    "Sandy Loam",
    "Silt",
    "Loamy Silt",
    "Silty Loam",
    "Loam",
    "Clayey Sand",
    "Sandy Clay",
    "Clay Loam",
    "Clay",
    "Organo-Mineral Soils",
    "Organic Soils"
]


# Classes that should NOT be included in percentage calculations
exclude_classes = {"Organo-Mineral Soils", "Organic Soils", "Other"}

# Count occurrences per soil class
class_counts = df_parameters["Bodenklasse"].value_counts().to_dict()

# Compute the total number of points included in the percentage calculation
total_points = sum(
    count for name, count in class_counts.items()
    if name not in exclude_classes
)


# Compute percentage distribution for each class
class_distribution = {
    name: ((class_counts.get(name, 0) / total_points) * 100)
    if (name not in exclude_classes and total_points > 0)
    else 0.0
    for name in ordered_legende
}



# build legend text
legende_text = "<span style='font-size:28px; font-weight:bold;'>Soil texture classes</span><br>"

sorted_classes = [name for name in ordered_legende if name not in exclude_classes]

for name in sorted_classes:
    base_color = apply_alpha(
        soilclass_to_color.get(name, "rgba(0,0,0,1)"),
        ALPHA_LEGEND
    )

    percent = class_distribution[name]
    count = class_counts.get(name, 0)

    legende_text += (
        f"<span style='color:{base_color}; font-size:46px;'>■</span> "
        f"<span style='font-size:26px;'>{name}</span>: "
        f"<b>{percent:.1f}%</b> "
        f"<span style='color:gray; font-size:18px; vertical-align:super;'>"
        f"({count} pts)</span><br>"
    )


legende_text += f"<br><b>Total points:</b> {total_points}<br>"


# Adjust layout to center
fig.update_layout(
plot_bgcolor="white",  # Set plot background to white
    paper_bgcolor="white",  # Set whole figure background to white
    xaxis=dict(
        title=dict(
             text="Sum of Sand % (A) and Silt % (B)",
             font=dict(size=28, color="black",family="Arial Black")
        ),
        range=[-10, 4250],  # korrekt
        tickformat=".0f",
        tickfont=dict(size=25, color="black")
    ),
    yaxis=dict(
        title=dict(
            text="Humus (%) /// Difference between height of AB rectangle and the Humus content (%) equals Clay content (%)",
            font=dict(size=18, color="black",family="Arial Black")
        ),
        range=[-0.3, 11.5],  # ✅ neuer Fokusbereich
        tickformat=".0f",
        dtick=2,  # feinere Ticks (alle 2 %)
        color="black",
        linecolor="gray",
        tickfont=dict(size=25, color="black")
    ),

    autosize=False,
    width=2260,  # Set plot width
    height=1210,  # Set plot height
    margin=dict(l=0, r=90, t=20, b=5),  # Center the plot by minimizing margins
    showlegend=False  # Disable the legend
)

# dashed horizontal lines
y_values = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 20, 30, 40, 50, 60, 70, 80, 90 ]

# add horizontal dashed lines
for y in y_values:
    fig.add_shape(
        type="line",
        x0=0,
        x1=max(df_parameters['Unnamed: 3']) + rechtecke[-1][0] + rechtecke[-1][1],
        y0=y,
        y1=y,
        line=dict(
            color="black",
            width=0.5,
            dash="dash"
        )
    )

# X-values for vertical dashed lines
x_values = []  # Mittlere Positionen von AB90, AB50, AB30

# Add vertical dashed lines
for x in x_values:
    fig.add_shape(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=100,
        line=dict(
            color="black",
            width=1,
            dash="dash"
        )
    )

# rotate plot
for trace in fig.data:
    trace.x, trace.y = trace.y, trace.x


# AB boundary lines
fig.add_annotation(
    text="<b>Available Water Capacity (AWC)</b>",
    xref="paper", yref="paper",
    x=1.015,
    y=0.5,
    showarrow=False,
    textangle=270,
    font=dict(size=20, color="black"),
    xanchor="center",
    yanchor="middle"
)

# legend after rotation
fig.add_annotation(
    x=0.01,
    y=1.015,
    xref="paper",
    yref="paper",
    text=legende_text,
    showarrow=False,
    font=dict(size=20, color="black"),
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="black",
    borderwidth=2,
    xanchor="left",
    yanchor="top",
    align="left",
    width=320,
    height=None
)


# Hover labels
fig.update_layout(
    hoverlabel=dict(
        font_size=22,
        font_family="Arial"
    )
)
# Add a manual annotation pointing to a specific datapoint in the plot
fig.add_annotation(
    x=1070,
    y=5.1,

    ax=-100,
    ay=-100,

    xref="x",
    yref="y",
    axref="pixel",
    ayref="pixel",

    text=(
        "<b>"  
        "Location: Almuferweg, Pettenbach, District Kirchdorf, Upper Austria, Austria<br>"
        "Sand (A): 48.0%<br>"
        "Silt (B): 40.0%<br>"
        "Humus (C): 5.0%<br>"
        "Clay (D): 7.0%<br>"
        "Class: Loamy Sand<br>"
        "AWC: 12.57"
        "</b>"
    ),

    showarrow=True,
    arrowhead=0,
    arrowsize=1.3,
    arrowwidth=1.8,
    arrowcolor="black",

    font=dict(size=18, color="black"),
    bgcolor="rgba(255,255,255,0.9)",
    bordercolor="black",
    borderwidth=1,
    align="left"
)


# Show plot
fig.show()



#                EXPORT SECTION (HTML / PNG / TIFF)


import os
from playwright.sync_api import sync_playwright
from PIL import Image

# Determine the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Create export directory (if not exists)
export_dir = os.path.join(base_dir, "exports")
os.makedirs(export_dir, exist_ok=True)

# Define output file paths
png_path  = os.path.join(export_dir, "cantor_export_interaktiv2.png")
tiff_path = os.path.join(export_dir, "cantor_export_interaktiv2.tiff")
html_output = os.path.join(export_dir, "cantor_export_interaktiv2.html")

#  Export HTML version of the figure 
fig.write_html(html_output, include_plotlyjs="cdn", full_html=True)

# Convert file path to a local browser URL
html_path = "file:///" + html_output.replace("\\", "/")

# Create high-resolution PNG using Playwright 
def export_highres_png():
    print("📸 Erstelle hochauflösendes PNG ...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 2260, "height": 1210, "device_scale_factor": 2}
        )
        page.goto(html_path)
        page.wait_for_timeout(800)  # wait for rendering
        page.screenshot(path=png_path, full_page=True)
        browser.close()
    print("✅ PNG gespeichert unter:", png_path)

#  Convert PNG to TIFF with 400 dpi 
def convert_png_to_tiff_with_dpi(png_path, tiff_path, dpi=(400, 400)):
    print("🖼️ Konvertiere PNG → TIFF (400 dpi) ...")
    if os.path.exists(png_path):
        img = Image.open(png_path)
        img.save(tiff_path, dpi=dpi)
        print("✅ TIFF gespeichert unter:", tiff_path)
    else:
        print("❌ PNG nicht gefunden – TIFF konnte nicht erzeugt werden:", png_path)

# Execute export sequence 
export_highres_png()
convert_png_to_tiff_with_dpi(png_path, tiff_path)

print("\n🎉 EXPORT KOMPLETT – Dateien gespeichert in:", export_dir)
