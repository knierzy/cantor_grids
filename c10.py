# This script constructs a Cantor diagram for the garnet endmember system
# almandine–spessartine–pyrope–grossular. Scattered provenance fields
# (±1 standard deviation ranges), derived primarily from Suggate and Hall (2014),
# are displayed. The script allows plotting garnet compositions
# onto the diagram and assigns each data point to the most likely provenance
# field. Provenance assignment is computed statistically via Mahalanobis distance using subfield means and standard
# deviations.

import pandas as pd
import plotly.graph_objects as go
import numpy as np

convex_hulls_file_4 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_amphibolites_general.xlsx"
convex_hulls_file_5 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_greenschists_.xlsx"
convex_hulls_file_6 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_granites_.xlsx"
convex_hulls_file_7 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_blueschists_.xlsx"
convex_hulls_file_8 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_calc_silicate_rocks_.xlsx"
convex_hulls_file_9 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_granulites_general.xlsx"
convex_hulls_file_11 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_eclogites_.xlsx"
convex_hulls_file_12 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_ultramafic_.xlsx"

# color mappings
color_mapping_files = {
    convex_hulls_file_5: "rgba(144, 238, 144, 0.9)",  # Greenschists - light green
    convex_hulls_file_4: "rgba(75, 50, 35, 0.9)",  # Amphibolites - black
    convex_hulls_file_11: "rgba(0, 100, 0, 0.9)",  # Eclogites - dark green
    convex_hulls_file_7: "rgba(0, 0, 255, 0.9)",  # Blueschists - blue
    convex_hulls_file_8: "rgba(64, 224, 208, 0.9)",  # Calc-Silicate Rocks - turquoise
    convex_hulls_file_6: "rgba(255, 215, 0, 0.9)",  # Granites - yellow
    convex_hulls_file_9: "rgba(255, 140, 0, 0.9)",  # Granulites General - orange
    convex_hulls_file_12: "rgba(205, 92, 92, 0.9)"  # Ultramafic - brick red
}

# legend mapping
legend_mapping = {
    convex_hulls_file_6: "Granites & Pegmatites",
    convex_hulls_file_5: "Greenschists",
    convex_hulls_file_4: "Amphibolites",
    convex_hulls_file_7: "Blueschists",
    convex_hulls_file_8: "Calc-silicate rocks",
    convex_hulls_file_9: "Granulites",
    convex_hulls_file_11: "Eclogites",
    convex_hulls_file_12: "Ultramafic rocks"
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


# Add rectangles with color gradients along the new x-axis
def add_rechtecke_mit_farbverlauf(rechtecke, x_offset, spiegeln=False):
    for i, (y_position, hoehe, label) in enumerate(rechtecke):
        breite = i + 1
        gradient_steps = 10  # Number of steps in the color gradient


        grau_start = 200  # darker gray at the bottom
        grau_ende = 230 # brighter gray at the top (must stay below 255)

        for step in range(gradient_steps):
            # Calculate the gray value within an AB-rectangle
            grau_wert = int(grau_start + (grau_ende - grau_start) * (step / (gradient_steps - 1)))

            # transparency variation for smoother shading
            alpha = 0.8 - (0.6 * (step / (gradient_steps - 1)))
            color = f'rgba({grau_wert}, {grau_wert}, {grau_wert}, {alpha})'

            # determine coordinates for the gradient along the new x-axis (sum A + B)
            y_start = y_position + (step / gradient_steps) * hoehe
            y_end = y_position + ((step + 1) / gradient_steps) * hoehe

            if spiegeln:
                x_start, x_end = x_offset - breite, x_offset
            else:
                x_start, x_end = x_offset, x_offset + breite

            # add rectangle polygon for this gradient step
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
            title="Summe A and B (%)",
            tickvals=list(x_labels.keys()),
            ticktext=list(x_labels.values()),
            tickangle=0,
            ))


# Add rectangles
add_rechtecke_mit_farbverlauf(rechtecke, 0)

