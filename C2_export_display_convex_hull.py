# This file exports convex hulls based on tuples read from an Excel file. 
# In the present example, the scattered subfields generated by this script 
# refer specifically to the soil texture subcategory "Loamy Sand"
#...all valid tuples of a specific subgroup and were previously generated by tuple_generator_for_subfields.py.

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull
import numpy as np


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


# Add rectangles with color gradients along the new x-axis (after rotation)
def add_rechtecke_mit_farbverlauf(rechtecke, x_offset, spiegeln=False):
    for i, (y_position, hoehe, label) in enumerate(rechtecke):
        breite = i + 1
        gradient_steps = 8  # Anzahl der Schritte im Farbverlauf

        # Der Farbabstufungsverlauf startet intensiv und wird matter
        grau_start = 220  # Dunkler Grauton
        grau_ende = 270  # Heller Grauton

        for step in range(gradient_steps):
            grau_wert = int(grau_start + (grau_ende - grau_start) * (step / gradient_steps))

            # Variiere die Transparenz leicht, um einen smootheren Effekt zu erzielen
            alpha = 0.8 - (0.6 * (step / gradient_steps))  # Reduziert Alpha von 0.8 auf 0.2
            color = f'rgba({grau_wert}, {grau_wert}, {grau_wert}, {alpha})'

            # Bestimme die Koordinaten für den Farbverlauf entlang der neuen x-Achse (Summe A + B)
            y_start = y_position + (step / gradient_steps) * hoehe
            y_end = y_position + ((step + 1) / gradient_steps) * hoehe

            if spiegeln:
                x_start, x_end = x_offset - breite, x_offset
            else:
                x_start, x_end = x_offset, x_offset + breite

            # Rechteck für den aktuellen Schritt hinzufügen
            fig.add_trace(go.Scatter(
                x=[x_start, x_start, x_end, x_end, x_start],
                y=[y_start, y_end, y_end, y_start, y_start],
                fill="toself",
                mode="lines",
                fillcolor=color,
                line=dict(color="rgba(0, 0, 0, 0)", width=0)  # Keine sichtbare Umrandung
            ))
        for x_pos in range(1, breite):
            x_val = x_offset - x_pos if spiegeln else x_offset + x_pos
            fig.add_trace(go.Scatter(
                x=[x_val, x_val],
                y=[y_position, y_position + hoehe],
                mode="lines",
                line=dict(color="gray", width=1),
                showlegend=False
            ))


    # Create  axis labels
    x_labels = {50: "AB99",440: "AB95",  915: "AB90", 1350: "AB85", 1760: "AB80", 2158: "AB75", 2540: "AB70",
                 2870: "AB65", 3195: "AB60", 3480: "AB55", 3755: "AB50", 3995: "AB45", 4209: "AB40", 4405: "AB35", 4570: "AB30", 4830: "AB20", 4990: "AB10" }

    # Update X-axis with labels
    fig.update_layout(
        xaxis=dict(
            title="Summe A + B (%)",
            tickvals=list(x_labels.keys()),  # Positionen der Beschriftungen
            ticktext=list(x_labels.values()),  # Text der Beschriftungen
            tickangle=0,  # Optional: keine Rotation der Beschriftungen

            ))



# Add rectangles
add_rechtecke_mit_farbverlauf(rechtecke, 0)



# Load data from  Excel file
file_path_gilgen = "data/Komp_Pub.xlsx"
df = pd.read_excel(file_path_gilgen, sheet_name='Tuples_for_subfields')

# Remove rows with NaN in columns "Unnamed: 1" to "Unnamed: 4"; filter only rows where the sum is >= 98
df_parameters = df[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].dropna()
df_parameters = df_parameters[df_parameters.apply(lambda row: row[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].sum() >= 98, axis=1)]
df_parameters = df_parameters.astype(float).round()

# Load origin and index number (with the filtered rows without NaN)"
df_parameters['Herkunft'] = df.loc[df_parameters.index, 'Unnamed: 5'].values
df_parameters['Index'] = df.loc[df_parameters.index, 'Unnamed: 6'].values



#  Function to adjust so that the sum equals 100"
def adjust_sum_to_100(row):
    total = row['Unnamed: 1'] + row['Unnamed: 2'] + row['Unnamed: 3'] + row['Unnamed: 4']
    difference = 100 - total
    if difference != 0:
        row['Unnamed: 4'] += difference
    return row


# Apply the adjustment only to rows with a sum between 98 and 100"
df_parameters = df_parameters.apply(lambda row: adjust_sum_to_100(row) if 98 <= row[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].sum() <= 100 else row, axis=1)

# Calculate AB (A + B) for the y-position
df_parameters['AB'] = df_parameters['Unnamed: 1'] + df_parameters['Unnamed: 2']

# Calculate the y-position based on AB
def calculate_y_position(ab_value, b_value):
    ab_index = 99 - int(ab_value)
    if ab_index >= 0 and ab_index < len(rechtecke):
        start_zeile = rechtecke[ab_index][0]
        hoehe = rechtecke[ab_index][1]
        y_position = start_zeile + b_value + 0.5
        return y_position
    return None


# Color palette for origin
herkunfts_list = df_parameters['Herkunft'].unique()
color_palette = px.colors.qualitative.Plotly
color_mapping = {herkunft: color_palette[i % len(color_palette)] for i, herkunft in enumerate(herkunfts_list)}

# List to group points by origin and rectangle (AB
grouped_points = {}

# Plot points
for idx, row in df_parameters.iterrows():
    a = row['Unnamed: 1']
    b = row['Unnamed: 2']
    c = row['Unnamed: 3']
    d = row['Unnamed: 4']
    herkunft = row['Herkunft']
    index = row['Index']
    ab_value = row['AB']
    color = color_mapping.get(herkunft, "black")

    y_position_punkt = calculate_y_position(ab_value, b)
    if y_position_punkt is not None:
        fig.add_trace(go.Scatter(
            x=[c],
            y=[y_position_punkt],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=6,  # Größere Punkte
                color=color,
                opacity=1  # Maximale Deckkraft
            ),
            hovertemplate=(
                f"<b>Index:</b> {index}<br>"
                f"<b>Herkunft:</b> {herkunft}<br>"
                f"<b>Parameter:</b><br>"
                f"Schluff A: {a}%<br>"
                f"Ton B: {b}%<br>"
                f"Humus C: {c}%<br>"
                f"Sand D: {d}%<extra></extra>"
            )
        ))

        # Save points for convex hull by origin and AB value
        grouped_key = (herkunft, ab_value)
        if grouped_key not in grouped_points:
            grouped_points[grouped_key] = []
        grouped_points[grouped_key].append((c, y_position_punkt))


    # Function to display and calculate convex hulls
    def plot_convex_hull(points, color):
        points = np.array(points)

        if len(points) < 3:
            # if not enough points, skip.
            return

        try:
            # calculate convex hull
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_x = hull_points[:, 0]
            hull_y = hull_points[:, 1]
            hull_x = np.append(hull_x, hull_x[0])  # Schließe den Polygon ab
            hull_y = np.append(hull_y, hull_y[0])

            # plot convex hull with alpha 0.7
            fig.add_trace(go.Scatter(
                x=hull_x,
                y=hull_y,
                mode="lines",
                line=dict(color=color, width=1),
                fill="toself",
                fillcolor=ensure_transparency(color, alpha=0.7)
            ))
        except Exception as e:
            # if an error occurs -> warning
            print(f"Fehler beim Berechnen der Convex Hull: {e}")


    # function to ensure the transparency of an RGBA color.
    def ensure_transparency(color, alpha=0.7):
        if "rgba" in color:
            # Ersetze Alpha-Wert
            return color[:color.rfind(",")] + f", {alpha})"
        elif color.startswith("#"):
            # Konvertiere Hex in RGBA
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return f"rgba({r}, {g}, {b}, {alpha})"
        else:
            return f"rgba(0, 0, 0, {alpha})"

# Compute and plot convex hulls for each group
for (herkunft, ab_value), points in grouped_points.items():
    color = color_mapping.get(herkunft, "rgba(0,0,0,0.5)")  # Standardfarbe, falls nicht definiert
    plot_convex_hull(points, color)

# list for all convex hull data
convex_hull_data = []

# Compute convex hulls and store the points in the list
for (herkunft, ab_value), points in grouped_points.items():
    points = np.array(points)

    if len(points) < 3:
        # if not enough points, skip.
        continue

    try:
        # Caclculate convex hull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # Save points of the hull in the list
        for point in hull_points:
            convex_hull_data.append({
                "Herkunft": herkunft,
                "AB_Value": ab_value,
                "X": point[0],  # x coordinate
                "Y": point[1]   # y coordinate
            })
    except Exception as e:
        print(f"Fehler beim Berechnen der Convex Hull für Herkunft {herkunft} und AB {ab_value}: {e}")

# create a DataFrame from the convex hull data.
df_hulls = pd.DataFrame(convex_hull_data)

# Path to the Excel file.
output_file_path = "data/convex_hull_loamy_sand_test.xlsx"


# save the DataFrame to the Excel file
df_hulls.to_excel(output_file_path, index=False)

print(f"Die Convex Hull-Daten wurden erfolgreich in {output_file_path} gespeichert.")


# Adjust layout to center the plot and rotate it by 90 degrees.
fig.update_layout(
    xaxis=dict(
        title=dict(
            text="Sum of Sand and Silt (%)",
            font=dict(
                color="black",       # Farbe des Achsentitels
                size=14              # Schriftgröße
            )
        ),
        range=[0, max(df_parameters['Unnamed: 2']) + 2],  # Achsenbereich
        tickformat=".0f",            # Ganze Zahlen auf der Achse
        color="black",               # Allgemeine Achsenfarbe
        linecolor="black",           # Farbe der Achsenlinie
        tickfont=dict(color="black") # Farbe der Tick-Beschriftungen
    ),
    yaxis=dict(
        title=dict(
            text="Humus (%) /// Difference between height of AB rectangle and the Humus content (%) equals Clay (%)",
            font=dict(
                color="black",       # Farbe des Achsentitels
                size=14              # Schriftgröße
            )
        ),
        range=[0, max(df_parameters['Unnamed: 3']) + 2],  # Achsenbereich
        tickformat=".0f",            # Ganze Zahlen auf der Achse
        color="black",               # Achsenfarbe
        linecolor="black",           # Achsenlinie
        tickfont=dict(color="black") # Tick-Beschriftungen
    ),
    autosize=False,                  # Automatische Größenanpassung deaktivieren
    width=2100,                      # Breite des Plots
    height=1200,                     # Höhe des Plots
    margin=dict(l=0, r=10, t=30, b=10),  # Ränder
    showlegend=False                 # Legende deaktivieren
)

# Eine schwarze Linie entlang der Y-Achse hinzufügen (0 bis 100)
fig.add_shape(
    type="line",
    x0=0, x1=0,
    y0=0, y1=100,
    line=dict(color="black", width=2)
)

# Plot um 90 Grad drehen
for trace in fig.data:
    trace.x, trace.y = trace.y, trace.x

# Plot anzeigen
fig.show()
