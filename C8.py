# Cantor diagram of soil texture classes with humus as 4th component
# Generates an interactive Cantor diagram showing AB rectangles,
# soil texture subfields, and a Saxton–Rawls (2006) AWC heatmap,
# with export to HTML, PNG, and high-resolution TIFF.
# generation time with the current parameter settings is approximately 1 minute.


import plotly.graph_objects as go
import numpy as np
import plotly.io as pio


# Rectangle data with the classification system up to AB1”
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

]

# Set up diagram
fig = go.Figure()


# Add rectangles with color gradients along the new x-axis (after rotation)
def add_rechtecke_horizontal(rechtecke):
    for i, (x_start, breite, label) in enumerate(rechtecke):
        hoehe = i + 1  # entspricht AB-Breite

        gradient_steps = 14
        grau_start, grau_ende = 150, 200

        for step in range(gradient_steps):
            grau = int(grau_start + (grau_ende - grau_start) * step / (gradient_steps - 1))
            alpha = 0.8 - 0.6 * step / (gradient_steps - 1)
            color = f"rgba({grau},{grau},{grau},{alpha})"

            x0 = x_start + step / gradient_steps * breite
            x1 = x_start + (step + 1) / gradient_steps * breite

            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[0, 0, hoehe, hoehe, 0],
                fill="toself",
                mode="lines",
                fillcolor=color,
                line=dict(width=0),
                showlegend=False
            ))

        # feine vertikale Linien
        for y in range(1, hoehe):
            fig.add_trace(go.Scatter(
                x=[x_start, x_start + breite],
                y=[y, y],
                mode="lines",
                line=dict(color="gray", width=1),
                showlegend=False
            ))

    # --- AB labels at the END of each AB rectangle ---

    ab_tick_vals = []
    ab_tick_text = []

    for x_start, breite, label in rechtecke:
        ab_value = int(label.replace("AB", ""))  # z.B. "AB95" → 95

        if ab_value % 5 == 0 or ab_value == 99:
            ab_tick_vals.append(x_start + breite)  # rechte Kante des AB-Rechtecks
            ab_tick_text.append(label)

    fig.update_layout(
        xaxis=dict(
            tickvals=ab_tick_vals,
            ticktext=ab_tick_text,
            tickangle=0,
            tickfont=dict(size=18)
        )
    )


add_rechtecke_horizontal(rechtecke)


# Implementation     Saxton & Rawls (2006) Available Water Capacity
#     sand_pct, clay_pct, humus_pct in %
#     returns AWC in %

def saxton_awc(sand_pct, clay_pct, humus_pct):

    sand = sand_pct / 100.0
    clay = clay_pct / 100.0
    om = np.clip(humus_pct, 0, 8)  # gültig bis 8 %

    theta33_t = (
        -0.251 * sand
        + 0.195 * clay
        + 0.011 * om
        + 0.006 * sand * om
        - 0.027 * clay * om
        + 0.452 * sand * clay
        + 0.299
    )
    theta33 = theta33_t + (1.283 * theta33_t**2 - 0.374 * theta33_t - 0.015)

    theta1500_t = (
        -0.024 * sand
        + 0.487 * clay
        + 0.006 * om
        + 0.005 * sand * om
        - 0.013 * clay * om
        + 0.068 * sand * clay
        + 0.031
    )
    theta1500 = theta1500_t + (0.14 * theta1500_t - 0.02)

    return max(0, (theta33 - theta1500) * 100)


def add_saxton_polygons(fig, rechtecke, saxton_awc,
                        n_x=40, zmin=3, zmax=35,
                        colorscale="Turbo"):

    from plotly.colors import sample_colorscale

    AB_MIN = 41  # 🔴 Heatmap nur bis AB41

    for ab_index, (x_start, breite, label) in enumerate(rechtecke):

        AB = 99 - ab_index
        if AB < AB_MIN:
            break

        max_c = 100 - AB
        if max_c <= 0:
            continue

        hoehe = ab_index + 1

        B_vals = np.linspace(0, AB, n_x + 1)
        C_MAX_HEAT = 8  # 🔴 absolute Obergrenze für Heatmap
        C_max_plot = min(max_c, C_MAX_HEAT)
        C_vals = np.arange(0, C_max_plot + 1)

        for i in range(len(B_vals) - 1):
            for j in range(len(C_vals) - 1):

                B0, B1 = B_vals[i], B_vals[i + 1]
                C0, C1 = C_vals[j], C_vals[j + 1]

                A = AB - (B0 + B1) / 2
                D = max_c - (C0 + C1) / 2

                z = saxton_awc(A, D, (C0 + C1) / 2)
                t = np.clip((z - zmin) / (zmax - zmin), 0, 1)
                color = sample_colorscale(colorscale, t)[0]

                y0 = (C0 / max_c) * hoehe
                y1 = (C1 / max_c) * hoehe

                x0 = x_start + B0
                x1 = x_start + B1

                fig.add_trace(go.Scatter(
                    x=[x0, x1, x1, x0, x0],
                    y=[y0, y0, y1, y1, y0],
                    fill="toself",
                    mode="lines",
                    fillcolor=color,
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,

                ))


add_saxton_polygons(fig, rechtecke, saxton_awc)

fig.add_trace(go.Heatmap(
    z=[[3, 35]],                 # min / max
    colorscale="Turbo",
    showscale=True,
    colorbar=dict(
        title=dict(
            text="Available Water Capacity (Vol.%)",
            side="right",        # vertikaler Titel
            font=dict(size=22, family="Arial Black")
        ),
        tickfont=dict(size=20),
        len=0.55,
        thickness=24,
        x=0.945,
        xanchor="left",
        y=0.55,
        yanchor="middle",
        outlinewidth=2
    ),
    opacity=0,                   # unsichtbar
    hoverinfo="skip"
))

# Global layout and axis styling

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",

    xaxis=dict(

        title=dict(
            text="Sum of Sand % (A) + Silt % (B)",
            font=dict(size=28, family="Arial Black", color="black")
        ),
        range=[0, 4500],
        tickfont=dict(size=25, color="black"),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
        showgrid=False
    ),

    yaxis=dict(
        title=dict(
            text="Humus (%) /// Difference between height of AB rectangle and the Humus content (%) equals Clay content (%)",
            font=dict(size=18, family="Arial Black", color="black")
        ),
        range=[-0.35, 8.5],     # oder 15
        dtick=1,
        tickfont=dict(size=25, color="black"),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
        showgrid=False
    ),

    autosize=False,
    width=2260,
    height=1210,
    margin=dict(l=0, r=5, t=20, b=5),
    showlegend=False,

    hoverlabel=dict(
        font_size=22,
        font_family="Arial"
    )
)

# Draw AB class boundaries with dashed lines and arrows below the x-axis
def add_ab_boundaries_with_arrows(fig, rechtecke,
                                  y_top,
                                  arrow_y0=-0.24,
                                  arrow_y1=-0.14):

    for i, (x_start, breite, label) in enumerate(rechtecke):
        if i == 0:
            continue

        x_pos = x_start # AB boundary position on x-axis

        # Vertical dashed AB boundary line
        fig.add_trace(go.Scatter(
            x=[x_pos, x_pos],
            y=[0, y_top],
            mode="lines",
            line=dict(color="black", width=2.5, dash="dot"),
            showlegend=False
        ))

        # Short tick below x-axis
        fig.add_trace(go.Scatter(
            x=[x_pos, x_pos],
            y=[arrow_y0, arrow_y1],
            mode="lines",
            line=dict(color="grey", width=1.5),
            showlegend=False
        ))

        # Arrow marker below x-axis
        fig.add_trace(go.Scatter(
            x=[x_pos],
            y=[arrow_y1],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=16,
                color="black",
                line=dict(color="black", width=0.4)
            ),
            showlegend=False
        ))

# Add AB boundaries (y_top must match y-axis range)
add_ab_boundaries_with_arrows(
    fig,
    rechtecke,
    y_top=9    # muss zu yaxis.range passen!
)

# Add dashed horizontal reference lines
y_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]

x_max = rechtecke[-1][0] + rechtecke[-1][1]  # 🔴 rechnerisch korrektes Diagrammende

for y in y_values:
    fig.add_shape(
        type="line",
        x0=0,
        x1=x_max,
        y0=y,
        y1=y,
        line=dict(
            color="black",
            width=9,
            dash="dot"
        )
    )

# Add horizontal dashed lines for each Cantor level
y_max = 9

for y in range(1, y_max + 1):
    fig.add_shape(
        type="line",
        x0=0,
        x1=rechtecke[-1][0] + rechtecke[-1][1],
        y0=y,
        y1=y,
        line=dict(
            color="black",
            width=2.5,
            dash="dot"
        )
    )


# Update hover label styling
fig.update_layout(
    hoverlabel=dict(
        font_size=22,
        font_family="Arial"
    )
)




# White background for plot and page
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white"
)

# Export interactive HTML (no external dependencies)
pio.write_html(
    fig,
    file="C:/Users/wolfgang.knierzinger/Desktop/cantor_export_interaktiv10.html",
    full_html=True,
    include_plotlyjs='cdn'
)


import os
from playwright.sync_api import sync_playwright
from PIL import Image

# Ensure export directory exists
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Paths (relative, GitHub-safe)
html_path = os.path.join(EXPORT_DIR, "cantor_export_interaktiv10.html")
png_path  = os.path.join(EXPORT_DIR, "cantor_switch2.png")
tiff_path = os.path.join(EXPORT_DIR, "cantor_switch2_400dpi.tiff")

# Write interactive HTML
fig.write_html(html_path, full_html=True, include_plotlyjs="cdn")
print("📄 HTML written:", html_path)

# Create screenshot (high-resolution PNG)
def export_highres_png():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 2260, "height": 1210, "device_scale_factor": 2}
        )
        page.goto(f"file://{os.path.abspath(html_path)}", timeout=120000)
        page.screenshot(path=png_path, full_page=True)
        browser.close()
        print("✅ PNG saved:", png_path)

# Convert PNG to TIFF (400 dpi)
def convert_png_to_tiff_with_dpi(png_path, tiff_path, dpi=(400, 400)):
    if os.path.exists(png_path):
        img = Image.open(png_path)
        img.save(tiff_path, dpi=dpi)
        print("TIFF saved (400 dpi):", tiff_path)
    else:
        print(" PNG not found:", png_path)

# Run export pipeline
export_highres_png()
convert_png_to_tiff_with_dpi(png_path, tiff_path)
# show plot
fig.show()
