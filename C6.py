#### Cantor diagram for a soil texture classification system, with humus considered as a fourth component, 
#### including organo-mineral and organic soils

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# data 

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
    convex_hulls_file_1: "rgba(160, 82, 45, 0.75)",       
    convex_hulls_file_2: "rgba(57, 255, 20, 0.85)",    
    convex_hulls_file_3: "rgba(178, 34, 34, 0.75)",       
    convex_hulls_file_4: "rgba(253, 192, 134, 0.75)",     
    convex_hulls_file_5: "rgba(70, 70, 70, 0.75)",      
    convex_hulls_file_6: "rgba(204, 121, 167, 0.75)",       
    convex_hulls_file_7: "rgba(110, 165, 160, 0.75)",     
    convex_hulls_file_8: "rgba(225, 195, 65, 0.75)",      
    convex_hulls_file_9: "rgba(0, 158, 115, 0.75)",       
    convex_hulls_file_10: "rgba(0, 60, 140, 0.75)",      
    convex_hulls_file_11: "rgba(17, 17, 17, 0.85)",       
    convex_hulls_file_12: "rgba(86, 180, 233, 0.75)",     
    convex_hulls_file_14: "rgba(100, 95, 90, 0.2)",  
    convex_hulls_file_15: "rgba(94, 60, 153, 0.75)",  
    convex_hulls_file_16: "rgba(100, 95, 90, 0.2)",
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


def ensure_transparency(color, alpha=0.6):
    if "rgba" in color:
        return color[:color.rfind(",")] + f", {alpha})"
    elif color.startswith("#"):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    else:
        return f"rgba(0, 0, 0, {alpha})"

# color legend
def legend_color(color, alpha=0.45):
    if "rgba" in color:
        return color[:color.rfind(",")] + f", {alpha})"
    return color


def legend_rgba(color, alpha_factor=0.65):
    """
    Adjust alpha for HTML legend so it visually matches Plotly filled polygons.
    """
    if "rgba" in color:
        r, g, b, a = color.replace("rgba(", "").replace(")", "").split(",")
        new_alpha = float(a) * alpha_factor
        return f"rgba({r},{g},{b},{new_alpha})"
    return color


# Add rectangles with color gradients along the X-axis
def add_rechtecke_mit_farbverlauf(rechtecke, x_offset, spiegeln=False):
    for i, (y_position, hoehe, label) in enumerate(rechtecke):
        breite = i + 1
        gradient_steps = 10  # Number of steps in the color gradient


        grau_start = 180  # Dark gray tone
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

            # add rectangle polygon for this gradient step
            fig.add_trace(go.Scatter(
                x=[x_start, x_start, x_end, x_end, x_start],
                y=[y_start, y_end, y_end, y_start, y_start],
                fill="toself",
                mode="lines",
                fillcolor=color,
                line=dict(color="gray", width=0)
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
            title="Summe A + B (%)",
            tickvals=list(x_labels.keys()),
            ticktext=list(x_labels.values()),
            tickangle=0,

        ))


# Add rectangles
add_rechtecke_mit_farbverlauf(rechtecke, 0)




# Mapping of RGBA colors to sub-classes
farbe_to_subklasse = {
    "rgba(86, 180, 233, 0.75)": "Sand",
    "rgba(0, 158, 115, 0.75)": "Silty Sand",
    "rgba(225, 195, 65, 0.75)": "Loamy Sand",
    "rgba(204, 121, 167, 0.75)": "Sandy Silt",
    "rgba(70, 70, 70, 0.75)": "Silt",
    "rgba(57, 255, 20, 0.85)": "Clayey Sand",
    "rgba(178, 34, 34, 0.75)": "Sandy Loam",
    "rgba(253, 192, 134, 0.75)": "Loamy Silt",
    "rgba(160, 82, 45, 0.75)": "Sandy Clay",
    "rgba(123, 204, 196, 0.75)": "Loam",
    "rgba(0, 90, 160, 0.75)": "Clay Loam",
    "rgba(17, 17, 17, 0.85)": "Clay",
    "rgba(94, 60, 153, 0.75)": "Silty Loam",

}







# Legend text as HTML
legende_text = (
    "<span style='font-size:24px; font-weight:bold;'>Soil texture classes</span><br>"

)


# Display order of the legend
ordered_legende = [
    "Clay", "Clay Loam", "Sandy Clay", "Clayey Sand", "Loam",
    "Loamy Silt",  "Silty Loam", "Sandy Loam", "Loamy Sand","Silt",
    "Sandy Silt", "Silty Sand", "Sand", "Organic & Organo-Mineral Soils"
]

for name in ordered_legende:
    if name == "Organic & Organo-Mineral Soils":
        # einheitliche Mischfarbe (mittleres Braun)
        farbe_combined = "rgba(80, 45, 15, 0.14)"  # etwas kräftiger als die Originalfarben

        legende_text += (
            f'<span style="color:{farbe_combined}; font-size:40px;">■</span> '
            f'Organo-Mineral & Organic Soils<br>'
        )

    else:
        farbe = next((k for k, v in farbe_to_subklasse.items() if v == name), None)
        if farbe:
            farbe_legende = legend_rgba(farbe, alpha_factor=0.6)
            legende_text += f'<span style="color:{farbe}; font-size:40px;">■</span> {name}<br>'


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


        hull_x = np.append(hull_x, hull_x[0])
        hull_y = np.append(hull_y, hull_y[0])


        file_source = group["file_source"].iloc[0]
        color = file_color_mapping.get(file_source, "rgba(0, 0, 0, 0.5)")

        # Plot  convex hull
        fig.add_trace(go.Scatter(
            x=hull_x,
            y=hull_y,
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="toself",
            fillcolor=color if file_source in [convex_hulls_file_14, convex_hulls_file_16]
            else ensure_transparency(color, alpha=0.45),
            name=f"Class: {soil_class}, AB: {ab_value}"
        ))

# Plot the imported convex hulls using file-specific colors
plot_imported_hulls_with_file_colors(grouped_hulls_combined, color_mapping_files)

# boundary between Organo-Mineral Soils and Organic Soils
fig.add_shape(
    type="line",
    x0=2905,
    x1=5050,
    y0=35,
    y1=35,
    line=dict(
        color="rgba(50, 25, 10, 1)",  # dunkles Braun
        width=3.8,
        dash="dash"
    ),
    layer="above"
)


# Adjust layout to center
fig.update_layout(
plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(
        title=dict(
            text="Sum of Sand % (A) and Silt % (B)",
            font=dict(size=24, color="black")
        ),
        range=[0, rechtecke[-1][0] + rechtecke[-1][1]],  # korrekt
        tickformat=".0f",
        tickfont=dict(size=25, color="black")
    ),
    yaxis=dict(
        title=dict(
            text="Humus (%) /// Difference between height of AB rectangle and the Humus content (%) equals Clay content (%)",
            font=dict(size=21, color="black")
        ),
        range=[0, 100],
        tickformat=".0f",
        dtick=5,
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

# **Add legend as an annotation
fig.update_layout(
    annotations=[
        dict(
            x=415,
            y=92,
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
# Additional text annotations in the diagram
extra_annotations = [
    dict(
        x=2800,
        y=25,
        text="Organo-Mineral Soils",
        showarrow=False,
        font=dict(size=26, color="rgb(80,45,15)", family="Arial"),
        align="center",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="rgba(80,45,15,0.8)",
        borderwidth=1.5,
        borderpad=4,
        xanchor="center"
    ),
    dict(
        x=4200,
        y=50,
        text="Organic Soils",
        showarrow=False,
        font=dict(size=26, color="rgb(60,35,10)", family="Arial"),
        align="center",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="rgba(60,35,10,0.8)",
        borderwidth=1.5,
        borderpad=4,
        xanchor="center"
    )
]


fig.update_layout(annotations=fig.layout.annotations + tuple(extra_annotations))


#  horizontal lines
y_values = [ 10, 20, 30, 40, 50, 60, 70, 80, 90 ]

# Add horizontal dashed lines
for y in y_values:
    fig.add_shape(
        type="line",
        x0=0,
        x1 = rechtecke[-1][0] + rechtecke[-1][1], # End point on the X-axis (right edge)
        y0=y,
        y1=y,
        line=dict(
            color="black",
            width=0.5,
            dash="dash"
        )
    )

# X-values for vertical dashed lines
x_values = [909.5, 3749.5, 4569.5, 1769.5, 2529.5, 3189.5, 4209.5, 4850.5, 4995.5, 442, 1352, 2162, 2872, 3482, 3992, 4402, 4712, 4922, 5037.5]  
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


#            EXPORT SECTION (HTML / PNG / TIFF) — SCRIPT 2


import os
from playwright.sync_api import sync_playwright
from PIL import Image

# Determine the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Create export directory (if not exists)
export_dir = os.path.join(base_dir, "exports")
os.makedirs(export_dir, exist_ok=True)

#  UNIQUE FILENAMES FOR THIS SCRIPT 
png_path  = os.path.join(export_dir, "cantor_export_organic_soils.png")
tiff_path = os.path.join(export_dir, "cantor_export_organic_soils.tiff")
html_output = os.path.join(export_dir, "cantor_export_organic_soils.html")

# Export HTML version of the figure 
fig.write_html(html_output, include_plotlyjs="cdn", full_html=True)

# Convert file path to a local browser URL
html_path = "file:///" + html_output.replace("\\", "/")

#  Create high-resolution PNG using Playwright 
def export_highres_png():
    print("📸 Erstelle hochauflösendes PNG ...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 2260, "height": 1210, "device_scale_factor": 2}
        )
        page.goto(html_path)
        page.wait_for_timeout(800)
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

