# Cantor diagram of Austrian soil texture classes with humus as 4th component (syntethic data)
# -------------------------------------------------------------------------
# Creates an interactive Cantor plot showing:
#   • AB-rectangles (AB99–AB1)
#   • Austrian soil texture classes via convex hulls
#   • Classification of samples based on sand, silt, clay and humus
#   • AWC (Available Water Capacity) displayed as color halo
#   • Inner points colored by soil texture class
#   • Export to HTML, PNG (Playwright), and TIFF (400 dpi)
#   • Plots 6500 synthetic soil data points

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from playwright.sync_api import sync_playwright
from PIL import Image
import os
show_points = True

# Largest Remainder Method (Hare–Niemeyer) 
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

# color mappings
color_mapping_files = {
    convex_hulls_file_1: "rgba(0, 0, 139, 0.45)",
    convex_hulls_file_2: "rgba(86, 180, 233, 0.45)",
    convex_hulls_file_3: "rgba(255, 255, 80,0.45)",
    convex_hulls_file_4: "rgba(255, 170, 190, 0.45)",
    convex_hulls_file_5: "rgba(178, 34, 34, 0.45)",
    convex_hulls_file_6: "rgba(255, 140, 0, 0.45)",
    convex_hulls_file_7: "rgba(110, 55, 30, 0.45)",
    convex_hulls_file_8: "rgba(50, 205, 50, 0.45)",
    convex_hulls_file_9: "rgba(0, 70, 50, 0.45)",
    convex_hulls_file_10: "rgba(0, 128, 128, 0.45)",
    convex_hulls_file_11: "rgba(25, 25, 25, 0.45)",
    convex_hulls_file_12: "rgba(123, 104, 238, 0.45)",
    convex_hulls_file_15: "rgba(218, 112, 214, 0.45)",

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


def ensure_transparency(color, alpha=0.45):
    if "rgba" in color:
        return color[:color.rfind(",")] + f", {alpha})"
    elif color.startswith("#"):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    else:
        return f"rgba(0, 0, 0, {alpha})"


# Add rectangles with color gradients along the X-axis
def add_rechtecke_mit_farbverlauf(rechtecke, x_offset, spiegeln=False):
    for i, (y_position, hoehe, label) in enumerate(rechtecke):
        breite = i + 1
        gradient_steps = 14  # Number of steps in the color gradient


        grau_start = 180 # Dark gray tone
        grau_ende = 220  # Light gray tone

        for step in range(gradient_steps):
            # Calculate the gray value within an AB-rectangle
            grau_wert = int(grau_start + (grau_ende - grau_start) * (step / (gradient_steps - 1)))

            # Variation of transparency to achieve a smoother effect
            alpha = 0.8 - (0.6 * (step / (gradient_steps - 1)))
            color = f'rgba({grau_wert}, {grau_wert}, {grau_wert}, {alpha})'

            # Determine the coordinates for the gradient along the  X-axis (sum A + B)
            y_start = y_position + (step / gradient_steps) * hoehe
            y_end = y_position + ((step + 1) / gradient_steps) * hoehe

            if spiegeln:
                x_start, x_end = x_offset - breite, x_offset
            else:
                x_start, x_end = x_offset, x_offset + breite


            fig.add_trace(go.Scatter(
                x=[x_start, x_start, x_end, x_end, x_start],
                y=[y_start, y_end, y_end, y_start, y_start],
                fill="toself",
                mode="lines",
                fillcolor=color,
                line=dict(color="gray", width=0)
            ))
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


# Load data from xlsx file
file_path_gilgen = "data/compendium.xlsx"
df = pd.read_excel(file_path_gilgen, sheet_name='Soil_synth_data')
print(df.columns.tolist())


#  Save original decimal texture values before rounding

df_tex = df[['Unnamed: 1','Unnamed: 2','Unnamed: 3','Unnamed: 4']].copy()


# Working copy for plotting, rounding, and AB calculation

df_parameters = df.copy()


# Keep only rows where (Sand + Silt + Humus + Clay) ≥ 98%

df_parameters = df_parameters[
    df_parameters.apply(
        lambda row: row[['Unnamed: 1','Unnamed: 2','Unnamed: 3','Unnamed: 4']].sum() >= 98,
        axis=1
    )
]

# Apply Largest Remainder Method normalization
df_parameters = df_parameters.apply(normalize_to_100_with_remainders, axis=1)


#  Load origin and index number
df_parameters['Herkunft'] = df.loc[df_parameters.index, 'Unnamed: 5'].values
df_parameters['Index'] = df.loc[df_parameters.index, 'Unnamed: 6'].values
df_parameters['location'] = df.loc[df_parameters.index, 'Unnamed: 7'].values


# Check contents and possible inconsistencies in 'location'

print("\n=== DEBUG: First entries of df_parameters['location'] ===")
print(df_parameters["location"].head(20))
print(f"\n👉 Number of unique locations: {df_parameters['location'].nunique()}")

# Show locations that appear multiple times
duplicates = df_parameters["location"].value_counts()
print("\n=== Locations with multiple entries (Top 10) ===")
print(duplicates[duplicates > 1].head(10))

# Show example of differing spellings / variants
print("\n=== Example of differing location names ===")
print(df_parameters["location"].drop_duplicates().sort_values().head(20))



# Compute exact Y-position inside the Cantor rectangle

def calculate_y_position_exact(a, b, c, d):
    ab_value = a + b
    ab_index = 99 - int(ab_value)

    if 0 <= ab_index < len(rechtecke):
        start_zeile = rechtecke[ab_index][0]
        hoehe = rechtecke[ab_index][1]


        c_int = int(c)
        d_int = int(d)
        c_frac = c - c_int
        d_frac = d - d_int


        field_sum = c + d
        field_int = int(round(field_sum))
        field_frac = field_sum - field_int


        total_frac = c_frac + d_frac
        if total_frac > 0:
            frac_c = c_frac / total_frac
            frac_d = d_frac / total_frac
        else:
            frac_c = frac_d = 0.5


        offset_ratio = field_frac * frac_c  # Anteil nach oben im Feld


        y_position = start_zeile + b + offset_ratio * hoehe
        return y_position

    return None

# Classification of the sample

def classify_soil(a, b, c, d):
    m = (100 - c) / 100.0
    if c > 35:  return "Organic Soils"
    if c > 15:  return "Organo-Mineral Soils"

    if (80*m <= a <= 100*m) and (0 <= b <= 20*m) and (0 <= d <= 10*m): return "Sand"
    if (65*m <= a <= 80*m) and (15*m <= b <= 30*m) and (0 <= d <= 5*m):  return "Sand"
    if (40*m <= a <= 70*m) and (30*m <= b <= 55*m) and (0 <= d <= 5*m):  return "Silty Sand"
    if (30*m <= a <= 80*m) and (10*m <= b <= 55*m) and (5*m <= d <= 15*m): return "Loamy Sand"
    if (10*m <= a <= 45*m) and (55*m <= b <= 75*m) and (0 <= d <= 15*m):  return "Sandy Silt"
    if (0   <= a <= 25*m) and (75*m <= b <= 100*m) and (0 <= d <= 25*m):  return "Silt"
    if (65*m <= a <= 90*m) and (0   <= b <= 10*m)  and (10*m <= d <= 25*m):return "Clayey Sand"
    if (20*m <= a <= 75*m) and (10*m <= b <= 55*m) and (15*m <= d <= 25*m):return "Sandy Loam"
    if (0   <= a <= 30*m) and (55*m <= b <= 75*m) and (15*m <= d <= 25*m):return "Loamy Silt"
    if (50*m <= a <= 75*m) and (0   <= b <= 10*m)  and (25*m <= d <= 40*m):return "Sandy Clay"
    if (5*m  <= a <= 65*m) and (10*m <= b <= 55*m) and (25*m <= d <= 40*m):return "Loam"
    if (0   <= a <= 20*m) and (55*m <= b <= 75*m) and (25*m <= d <= 45*m):return "Silty Loam"
    if (0   <= a <= 60*m) and (0   <= b <= 55*m) and (40*m <= d <= 50*m): return "Clay Loam"
    if (0   <= a <= 50*m) and (0   <= b <= 50*m) and (50*m <= d <= 100*m):return "Clay"
    return "Other"

farbe_to_subklasse = {
    "rgba(123, 104, 238, 0.45)": "Sand",
    "rgba(0, 70, 50, 0.45)": "Silty Sand",
    "rgba(50, 205, 50, 0.45 )": "Loamy Sand",
    "rgba(255, 140, 0, 0.45)": "Sandy Silt",
    "rgba(178, 34, 34, 0.45)": "Silt",
    "rgba(86, 180, 233, 0.45)": "Clayey Sand",
    "rgba(255, 255, 80, 0.45)": "Sandy Loam",
    "rgba(255, 170, 190, 0.45)": "Loamy Silt",
    "rgba(0, 0, 139, 0.45)": "Sandy Clay",
    "rgba(110, 55, 30, 0.45)": "Loam",
    "rgba(0, 128, 128, 0.45)": "Clay Loam",
    "rgba(25, 25, 25, 0.45)": "Clay",
    "rgba(218, 112, 214, 0.45)": "Silty Loam",
    "rgba(70, 45, 30, 0.15)": "Organo-Mineral Soils",
    "rgba(80, 45, 15, 0.22)": "Organic Soils",
}


# Legend text as HTML with fixed display order
legende_text = (
    "<span style='font-size:26px; font-weight:bold;'>Soil texture classes</span><br>"
    "<span style='line-height:25px;'>&nbsp;</span><br>"
)


#List to group points by origin and rectangle (AB)

grouped_points = {}

df_hulls_combined = pd.concat([
    pd.read_excel(path).assign(file_source=path)
    for path in color_mapping_files
])

# Group the hull data based on origin and AB_Value
grouped_hulls_combined = df_hulls_combined.groupby(["Soil texture class", "AB_Value"])



# Group the hull data based on origin and AB_Value
grouped_hulls_combined = df_hulls_combined.groupby(["Soil texture class", "AB_Value"])


# Function to plot convex hulls with different colors based on source file
def plot_imported_hulls_with_file_colors(grouped_hulls, file_color_mapping):
    for (soil_class, ab_value), group in grouped_hulls:

        hull_x = group["X"].values
        hull_y = group["Y"].values

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
            else ensure_transparency(color, alpha=0.40),
            name=f"Class: {soil_class}, AB: {ab_value}"
        ))

# Plot the imported convex hulls using file-specific colors
plot_imported_hulls_with_file_colors(grouped_hulls_combined, color_mapping_files)


# AWC calculation using the original decimal texture values (Saxton & Rawls 2006 with OM)
# Reference: Saxton & Rawls (2006), Soil Sci. Soc. Am. J. 70:1569–1578

tex = df.loc[df_parameters.index, ['Unnamed: 1', 'Unnamed: 3', 'Unnamed: 4']].copy()

# Sand and clay as fractions (0–1)
sand = tex['Unnamed: 1'] / 100.0
clay = tex['Unnamed: 4'] / 100.0


#  OM correction (max. 8%)

om_pct_raw = tex['Unnamed: 3']  # Humus value from file
om_pct = np.clip(om_pct_raw, 0, 8)    # Saxton & Rawls valid up to 8%
# ------------------------------------------

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

awc_min = df_parameters["AWC"].min()
awc_max = df_parameters["AWC"].max()
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
    "Sand": "rgba(123, 104, 238, 0.95)",
    "Silty Sand": "rgba(0, 70, 50, 0.95)",
    "Loamy Sand": "rgba(50, 205, 50, 0.95)",
    "Sandy Silt": "rgba(255, 140, 0, 0.95)",
    "Silt": "rgba(178, 34, 34, 0.95)",
    "Clayey Sand": "rgba(86, 180, 233, 0.95)",
    "Sandy Loam": "rgba(255, 255, 80, 0.95)",
    "Loamy Silt": "rgba(255, 170, 190, 0.95)",
    "Silty Loam": "rgba(218, 112, 214, 0.95)",
    "Sandy Clay": "rgba(0, 30, 80, 0.95)",
    "Loam": "rgba(110, 55, 30, 0.95)",
    "Clay Loam": "rgba(0, 128, 128, 0.95)",
    "Clay": "rgba(25, 25, 25, 0.95)",
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


    # Classify the soil sample based on a, b, c, d
    bodenklasse = classify_soil(a, b, c, d)
    df_parameters.loc[idx, "Bodenklasse"] = bodenklasse
    color_count[bodenklasse] = color_count.get(bodenklasse, 0) + 1

    y_position_punkt = calculate_y_position_exact(a, b, c, d)
    if y_position_punkt is None:
        continue

    # Apply a small jitter to avoid point overlap
    jitter_y = np.random.uniform(-0.0, 0.0)
    jitter_x = np.random.uniform(-0.05, 0.05)
    x_val = c + jitter_x
    y_val = y_position_punkt + jitter_y

    #  AWC point
    points_x.append(x_val)
    points_y.append(y_val)
    awc_values.append(row["AWC"])
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

    # Ring
    ring_x.append(x_val)
    ring_y.append(y_val)
    ring_colors.append(outline_colors.get(bodenklasse, "rgba(0,0,0,0.8)"))

# Ring (halo around the point)
fig.add_trace(go.Scatter(
    x=ring_x, y=ring_y, mode='markers',
    marker=dict(
        size=12.8,
        color='rgba(0,0,0,0)',
        line=dict(color='black', width=2.0),
    ),
    hoverinfo='skip',
    showlegend=False
))

# Lookup outline color based on soil class
fig.add_trace(go.Scatter(
    x=ring_x, y=ring_y, mode='markers',
    marker=dict(
        size=12.8,                       # etwas größer als der Innenpunkt
        color=awc_values,              # Farbe nach AWC
        colorscale="jet",
        cmin=awc_min,
        cmax=awc_max,
        opacity=0.95,                   # 🔹 halbtransparent!
        line=dict(width=0),
        coloraxis="coloraxis"
    ),
    hoverinfo='skip',
    showlegend=False,
))


# Inner point represents soil texture class -> set to minium in this example
fig.add_trace(go.Scatter(
    x=points_x, y=points_y, mode='markers',
    marker=dict(
        size=0.1,
        color=ring_colors,
        line=dict(color="black", width=0.5),
        opacity=1
    ),
    text=hover_texts,
    hovertemplate="%{text}<extra></extra>",
    showlegend=False
))


# All AWC colors are controlled by one shared coloraxis configured below
fig.update_layout(
    coloraxis=dict(
        colorscale="jet",
        cmin=awc_min,
        cmax=awc_max,
        colorbar=dict(
            title="",  # leer lassen, wir ersetzen es durch Annotation
            tickfont=dict(size=18),
            thickness=20,
            len=0.95,
            y=0.5,
            yanchor="middle",
            x=1.02
        )
    )
)

# annotation color bar

# fig.add_annotation(
 #   text="Available Water Capacity (AWC)",
  #  xref="paper", yref="paper",
  #  x=1.09,
  #  y=0.5,
  #  showarrow=False,
  #  textangle=270,
  #  font=dict(size=20, color="black"),
  #  xanchor="center",
  #  yanchor="middle"
#)


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

# mapping soil class
klasse_zu_farbe = {v: k for k, v in farbe_to_subklasse.items()}

# build legend text
legende_text = "<span style='font-size:42px; font-weight:bold;'>Soil texture classes</span><br>"

# Display classes in fixed order
sorted_classes = [name for name in ordered_legende if name not in exclude_classes]


for name in sorted_classes:
    farbe = klasse_zu_farbe.get(name, "rgba(0,0,0,1)")
    legende_text += (
        f"<span style='color:{farbe}; font-size:46px;'>■</span> "
        f"<span style='font-size:36px;'>{name}</span><br>"
    )

# Add total point count at the bottom
legende_text += f"<br><b>Total points:</b> {total_points}<br>"


# Adjust layout to center
fig.update_layout(
plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(
        title=dict(
             text="Sum of Sand % (A) and Silt % (B)",
             font=dict(size=24, color="black", family="Arial Black")
        ),
        range=[-10, 1170],
        tickformat=".0f",
        tickfont=dict(size=25, color="black")
    ),
    yaxis=dict(
        title=dict(
            text="Humus (%) /// Difference between height of AB rectangle and the Humus content (%) equals Clay content (%)",
            font=dict(size=19, color="black", family="Arial Black")
        ),
        range=[-0.3, 8.2],
        tickformat=".0f",
        dtick=2,
        color="black",
        linecolor="gray",
        tickfont=dict(size=25, color="black")
    ),

    autosize=False,
    width=2260,  # Set plot width
    height=1210,  # Set plot height
    margin=dict(l=0, r=5, t=20, b=5),  # Center the plot by minimizing margins
    showlegend=False  # Disable the legend
)

# dashed horizontal lines
y_values = [2,4,6,8, 10,12,14, 20, 30, 40, 50, 60, 70, 80, 90 ]

 # add horizontal dashed lines
for y in y_values:
    fig.add_shape(
      type="line",
        x0=0,
        x1=max(df_parameters['Unnamed: 3']) + rechtecke[-1][0] + rechtecke[-1][1],  # End point on the X-axis (right edge)
        y0=y,
        y1=y,
        line=dict(
            color="black",
            width=0.5,
            dash="dash"
        )
    )


# rotate plot


for trace in fig.data:
    trace.x, trace.y = trace.y, trace.x


# AB boundary lines


for i, (start, hoehe, label) in enumerate(rechtecke):
    if i == 0:
        continue


    x_pos = start + 0.


    fig.add_trace(go.Scatter(
        x=[x_pos, x_pos],
        y=[0, i],
        mode="lines",
        line=dict(color="black", width=3.0, dash="dash"),
        showlegend=False
    ))


    fig.add_trace(go.Scatter(
        x=[x_pos, x_pos],
        y=[-0.24, -0.14],
        mode="lines",
        line=dict(color="grey", width=3),
        showlegend=False
    ))

    # arrows below the diagram
    fig.add_trace(go.Scatter(
        x=[x_pos],
        y=[-0.14],
        mode="markers",
        marker=dict(
            symbol="triangle-up",
            size=16,
            color="black",
            line=dict(color="black", width=0.4)
        ),
        showlegend=False
    ))


# legend after rotation

fig.add_annotation(
    x=0.00, y=0.995,
    xref="paper", yref="paper",
    text=legende_text,
    showarrow=False,
    font=dict(size=25, color="black"),
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="black",
    borderwidth=2,
    xanchor="left",
    yanchor="top",
    align="left",
    width=470
)

# Hover labels

fig.update_layout(
    hoverlabel=dict(
        font_size=22,
        font_family="Arial"
    )
)

fig.show()


# Average AWC per soil texture class


awc_means = (
    df_parameters
    .groupby("Bodenklasse")["AWC"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
)

print("\n=== Average AWC per Soil Texture Class ===")
print(awc_means.to_string())

# Show plot
fig.show()


# Save HTML file

html_write_path = "exports/cantor_export_interaktiv2.html"
fig.write_html(html_write_path)
html_path = "file:///" + html_write_path.replace("\\", "/")
print("📄 HTML file written:", html_write_path)


# PNG and TIFF output paths

png_path = "exports/cantor_soil_zoom_synth.png"
tiff_path = "exports/cantor_soil_zoom_synth_400dpi.tiff"



# Create screenshot (high-resolution PNG)

def export_highres_png():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 2260, "height": 1210, "device_scale_factor": 2})
        page.goto(html_path)
        page.screenshot(path=png_path, full_page=True)
        browser.close()
        print("✅ Screenshot saved:", png_path)


# Convert PNG to TIFF with 400 dpi

def convert_png_to_tiff_with_dpi(png_path, tiff_path, dpi=(400, 400)):
    if os.path.exists(png_path):
        img = Image.open(png_path)
        img.save(tiff_path, dpi=dpi)
        print("✅ TIFF saved with 400 dpi:", tiff_path)
    else:
        print("❌ PNG not found:", png_path)


# Execution sequence

export_highres_png()
convert_png_to_tiff_with_dpi(png_path, tiff_path)
