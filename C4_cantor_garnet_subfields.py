#### Cantor diagram of the garnet end-member system Almandine–Spessartine–Pyrope–Grossular. 
#### The subfields in the Cantor diagram allow for an assignment to source rocks. 

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# file paths

convex_hulls_file_4 = "data/convex_hull_amphibolites_general.xlsx"
convex_hulls_file_5 = "data/convex_hull_greenschists_.xlsx"
convex_hulls_file_6 = "data/convex_hull_granites_.xlsx"
convex_hulls_file_7 = "data/convex_hull_blueschists_.xlsx"
convex_hulls_file_8 = "data/convex_hull_calc_silicate_rocks_.xlsx"
convex_hulls_file_9 = "data/convex_hull_granulites_general.xlsx"
convex_hulls_file_11 = "data/convex_hull_eclogites_.xlsx"
convex_hulls_file_12 = "data/convex_hull_ultramafic_.xlsx"



# color mappings 

color_mapping_files = {
    convex_hulls_file_5: "rgba(144, 238, 144, 0.9)",  # Greenschists - Hellgrün
    convex_hulls_file_4: "rgba(75, 50, 35, 0.9)",        # Amphibolites - Schwarz
    convex_hulls_file_11: "rgba(0, 100, 0, 0.9)",     # Eclogites - Dunkelgrün
    convex_hulls_file_7: "rgba(0, 0, 255, 0.9)",      # Blueschists - Blau
    convex_hulls_file_8: "rgba(64, 224, 208, 0.9)",   # Calc-Silicate Rocks - Türkis
    convex_hulls_file_6: "rgba(255, 215, 0, 0.9)",    # Granites - Gelb
    convex_hulls_file_9: "rgba(255, 140, 0, 0.9)",    # Granulites General - Orange
    convex_hulls_file_12: "rgba(205, 92, 92, 0.9)"    # Ultramafic - Kaminrot
}


# legend mapping 
legend_mapping = {
    convex_hulls_file_6: "Granites and Pegmatites",
    convex_hulls_file_5: "Greenschists",
    convex_hulls_file_4: "Amphibolites",
    convex_hulls_file_7: "Blueschists",
    convex_hulls_file_8: "Calc-Silicate Rocks",
    convex_hulls_file_9: "Granulites",
    convex_hulls_file_11: "Eclogites",
    convex_hulls_file_12: "Ultramafic Rocks"
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


# Add rectangles with color gradients along the new x-axis (after rotation)
def add_rechtecke_mit_farbverlauf(rechtecke, x_offset, spiegeln=False):
    for i, (y_position, hoehe, label) in enumerate(rechtecke):
        breite = i + 1
        gradient_steps = 10  # Number of steps in the color gradient

  
        grau_start = 230   # Dark gray tone (RGB value 80)
        grau_ende = 270  # Light gray tone (RGB value 200)

        for step in range(gradient_steps):
          # Calculate the gray value within an AB-rectanglce
            grau_wert = int(grau_start + (grau_ende - grau_start) * (step / (gradient_steps - 1)))

           # Variation of transparency to achieve a smoother effect
            alpha = 0.8 - (0.6 * (step / (gradient_steps - 1)))  # Reduziert Alpha von 0.8 auf 0.2
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
            title="Summe A and B (%)",
            tickvals=list(x_labels.keys()),  
            ticktext=list(x_labels.values()),  
            tickangle=0,  
            ))

# Add rectangles
add_rechtecke_mit_farbverlauf(rechtecke, 0)


# Legend for Diagram
legende_text = "<b>Garnet Provenance Groups:</b><br>"



#  Manually ordered list of convex hulls in the legend
ordered_hulls = [
    convex_hulls_file_5,
    convex_hulls_file_4,  # Amphibolites
    convex_hulls_file_7,
    convex_hulls_file_9,  # Granulites General
    convex_hulls_file_11,
    convex_hulls_file_6,  # Granites zuerst
    convex_hulls_file_12,  # Ultramafic
    convex_hulls_file_8,  # Calc-Silicate Rocks
]

# Prepare legend tex
legende_text += ""
for file_path in ordered_hulls:
    color = color_mapping_files[file_path]  
    hull_name = legend_mapping.get(file_path, file_path.split("\\")[-1].split(".")[0])  
    legende_text += f'<span style="color:{color};">■</span> {hull_name}<br>'


# List to group points by origin and rectangle (AB
grouped_points = {}


df_hulls_4 = pd.read_excel(convex_hulls_file_4)
df_hulls_5 = pd.read_excel(convex_hulls_file_5)
df_hulls_6 = pd.read_excel(convex_hulls_file_6)
df_hulls_7 = pd.read_excel(convex_hulls_file_7)
df_hulls_8 = pd.read_excel(convex_hulls_file_8)
df_hulls_9 = pd.read_excel(convex_hulls_file_9)
df_hulls_11 = pd.read_excel(convex_hulls_file_11)
df_hulls_12 = pd.read_excel(convex_hulls_file_12)

df_hulls_combined = pd.concat([df_hulls_4, df_hulls_5,df_hulls_6,df_hulls_7,df_hulls_8, df_hulls_9, df_hulls_11, df_hulls_12])


# Groups of the hull data based on origin and AB_Value
grouped_hulls_combined = df_hulls_combined.groupby(["Herkunft", "AB_Value"])


# Function to display the convex hulls with different colors
def plot_imported_hulls_with_file_colors(grouped_hulls, file_color_mapping):
    for (herkunft, ab_value), group in grouped_hulls:
      # Extract X and Y coordinates of the hull points
        hull_x = group["X"].values
        hull_y = group["Y"].values

        # close the hull by appending the first point to the end
        hull_x = np.append(hull_x, hull_x[0])
        hull_y = np.append(hull_y, hull_y[0])

        # Determine the color based on the file
        file_source = group["file_source"].iloc[0]  # 
        color = file_color_mapping.get(file_source, "rgba(0, 0, 0, 0.5)")  

        # plot convex hull 
        fig.add_trace(go.Scatter(
            x=hull_x,
            y=hull_y,
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="toself",
            fillcolor=ensure_transparency(color, alpha=0.6),  
            name=f"Herkunft: {herkunft}, AB: {ab_value}"  
        ))

# Add a column to the DataFrames to indicate the file origin


df_hulls_4["file_source"] = convex_hulls_file_4
df_hulls_5["file_source"] = convex_hulls_file_5
df_hulls_6["file_source"] = convex_hulls_file_6
df_hulls_7["file_source"] = convex_hulls_file_7
df_hulls_8["file_source"] = convex_hulls_file_8
df_hulls_9["file_source"] = convex_hulls_file_9
df_hulls_11["file_source"] = convex_hulls_file_11
df_hulls_12["file_source"] = convex_hulls_file_12

# Combine the DataFrames
df_hulls_combined = pd.concat([df_hulls_4, df_hulls_5,df_hulls_6,df_hulls_7,df_hulls_8,df_hulls_9, df_hulls_11, df_hulls_12])

# Group the combined data
grouped_hulls_combined = df_hulls_combined.groupby(["Herkunft", "AB_Value"])

# Plot the imported convex hulls with file-specific colors
plot_imported_hulls_with_file_colors(grouped_hulls_combined, color_mapping_files)


# Compute and plot convex hulls for each group
for (herkunft, ab_value), points in grouped_points.items():
    color = color_mapping.get(herkunft, "rgba(0,0,0,0.5)")  # Standardfarbe, falls nicht definiert
    plot_convex_hull(points, color)

#  Adjust layout to center the plot and rotate it by 90 degrees
fig.update_layout(
    xaxis=dict(
        title=dict(
            text="Sum of Almandine (%) + Spessartine (%)",
            font=dict(size=35, color="black", family="Arial Black")
        ),
        range=[0, rechtecke[-1][0] + rechtecke[-1][1]],
        tickformat=".0f",
        tickfont=dict(size=24, color="black")
    ),

yaxis=dict(
    title=dict(
        text="Pyrope (%) /// Difference between height of AB rectangle and Pyrope content (%) equals Grossular content (%)",
        font=dict(size=19, color="black", family="Arial Black")
    ),
    range=[0, 100],  # Ensure it does not exceed 100
    constrain="domain",
    tickformat=".0f",
    dtick=10,
    color="black",  # Set axis label and axis line color to black
    linecolor="gray",
    tickfont=dict(size=24, color="black")
),


    autosize=False,  # Disable automatic resizing
    width=2100,  
    height=1200,  
    margin=dict(l=0, r=5, t=20, b=5),  # Centering the plot
    showlegend=False  # Deactivate legend
)

# Values at which dashed lines are shown
y_values = [ 10, 20, 30, 40, 50, 60, 70, 80, 90 ]

# Insert horizontal dashed line
for y in y_values:
    fig.add_shape(
        type="line",
        x0=0,  # Starting point of the line on the X-axis (left edge of the plot)
        x1=rechtecke[-1][0] + rechtecke[-1][1],  # End point of the line on the X-axis (right edge)
        y0=y,  #Y-value at which the line is drawn
        y1=y,  # Y-value remains constant (horizontal)
        line=dict(
            color="black",  
            width=0.5,       
            dash="dash"     
        )
    )

# Positions for the vertical dashed lines
x_values = [909.5, 3749.5, 4569.5, 1769.5, 2529.5, 3189.5, 4209.5, 4850.5, 4995.5, 442, 1352, 2162, 2872, 3482, 3992, 4402, 4712, 4922, 5037.5]  

# Insert vertical dashed lines
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

# Rotate the plot by 90 degrees (swap X and Y data)
for trace in fig.data:
    trace.x, trace.y = trace.y, trace.x


# Add the legend to the plot as an annotation (as in the first script)
fig.update_layout(
    annotations=[
        dict(
            x=650,  
            y=80,    
            text=legende_text, 
            showarrow=False,
            font=dict(size=35, color="black"),
            bgcolor="rgba(249, 249, 249,1)",  # White background with slight transparency
            bordercolor="black",
            borderwidth=3,
            xanchor="left",  # Align to the left
            yanchor="top",   # Align to the top
            align="left"     # Left-aligned text
        )
    ]
)

# Show plot
fig.show()
