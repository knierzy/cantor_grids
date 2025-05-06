import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull
import numpy as np

# Hier kommen die Dateipfade und Farbzuordnungen

convex_hulls_file_4 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_amphibolites_general.xlsx"
convex_hulls_file_5 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_greenschists_.xlsx"
convex_hulls_file_6 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_granites_.xlsx"
convex_hulls_file_7 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_blueschists_.xlsx"
convex_hulls_file_8 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_calc_silicate_rocks_.xlsx"
convex_hulls_file_9 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_granulites_general.xlsx"
convex_hulls_file_11 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_eclogites_.xlsx"
convex_hulls_file_12 = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\pub\export\convex_hull_ultramafic_.xlsx"

# Farbmappings für jede Convex Hull-Datei

# Farbmappings für jede Convex Hull-Datei
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


# Mapping der kurzen Namen für die Legende
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


# Rechteckdaten mit der erweiterten Systematik bis AB1
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

# Diagramm initialisieren
fig = go.Figure()

# Helper-Funktion, um die Transparenz einer RGBA-Farbe sicherzustellen
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


# Rechtecke hinzufügen mit Farbgradienten entlang der neuen x-Achse (nach Drehung)
def add_rechtecke_mit_farbverlauf(rechtecke, x_offset, spiegeln=False):
    for i, (y_position, hoehe, label) in enumerate(rechtecke):
        breite = i + 1
        gradient_steps = 10  # Anzahl der Schritte im Farbverlauf (mehr Schritte für weicheren Übergang)

        # Der Farbabstufungsverlauf startet intensiv und wird matter
        grau_start = 230  # Dunkler Grauton (z. B. RGB-Wert 80)
        grau_ende = 270  # Heller Grauton (z. B. RGB-Wert 200)

        for step in range(gradient_steps):
            # Berechne den Grauwert für diesen Schritt
            grau_wert = int(grau_start + (grau_ende - grau_start) * (step / (gradient_steps - 1)))

            # Variiere die Transparenz leicht, um einen smootheren Effekt zu erzielen
            alpha = 0.8 - (0.6 * (step / (gradient_steps - 1)))  # Reduziert Alpha von 0.8 auf 0.2
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


    # Benutzerdefinierte Achsenbeschriftung erstellen
    x_labels = {50: "AB99",440: "AB95",  915: "AB90", 1350: "AB85", 1760: "AB80", 2158: "AB75", 2540: "AB70",
                 2870: "AB65", 3195: "AB60", 3480: "AB55", 3755: "AB50", 3995: "AB45", 4209: "AB40", 4405: "AB35", 4570: "AB30", 4830: "AB20", 4990: "AB10" }


    # Aktualisierung der X-Achse mit den benutzerdefinierten Labels
    fig.update_layout(
        xaxis=dict(
            title="Summe A and B (%)",
            tickvals=list(x_labels.keys()),  # Positionen der Beschriftungen
            ticktext=list(x_labels.values()),  # Text der Beschriftungen
            tickangle=0,  # Optional: keine Rotation der Beschriftungen
            ))


# Rechtecke hinzufügen (linke Seite)
add_rechtecke_mit_farbverlauf(rechtecke, 0)


# Daten aus der Excel-Datei laden
file_path_gilgen = r"C:\Users\wolfgang.knierzinger\Desktop\cantor_anwend\Rohdaten_für_eq_berch\Kompendium_Ö.xlsx"
df = pd.read_excel(file_path_gilgen, sheet_name='Garn_ex')

# Entfernt Zeilen mit NaN in den Spalten "Unnamed: 1", "Unnamed: 2", "Unnamed: 3", "Unnamed: 4" und filtert nur Zeilen, deren Summe >= 98 beträgt
df_parameters = df[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].dropna()
df_parameters = df_parameters[df_parameters.apply(lambda row: row[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].sum() >= 98, axis=1)]
df_parameters = df_parameters.astype(float).round()

# Herkunft und Indexnummer laden (mit den gefilterten Zeilen ohne NaN)
df_parameters['Herkunft'] = df.loc[df_parameters.index, 'Unnamed: 5'].values
df_parameters['Index'] = df.loc[df_parameters.index, 'Unnamed: 6'].values


# Funktion zur Anpassung, damit die Summe 100 ergibt
def adjust_sum_to_100(row):
    total = row['Unnamed: 1'] + row['Unnamed: 2'] + row['Unnamed: 3'] + row['Unnamed: 4']
    difference = 100 - total
    if difference != 0:
        row['Unnamed: 4'] += difference  # Passe den letzten Parameter an, um die Summe auf 100 zu bringen
    return row

# Wende die Anpassung nur auf Zeilen an, die eine Summe von 98 bis 100 haben
df_parameters = df_parameters.apply(lambda row: adjust_sum_to_100(row) if 98 <= row[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].sum() <= 100 else row, axis=1)

# Berechnung von AB (A + B) für die y-Position
df_parameters['AB'] = df_parameters['Unnamed: 1'] + df_parameters['Unnamed: 2']

# Berechnung der y-Position basierend auf AB
def calculate_y_position(ab_value, b_value):
    ab_index = 99 - int(ab_value)
    if ab_index >= 0 and ab_index < len(rechtecke):
        start_zeile = rechtecke[ab_index][0]
        hoehe = rechtecke[ab_index][1]
        y_position = start_zeile + b_value + 0.5
        return y_position
    return None

# Farbpallette für die Herkunft
herkunfts_list = df_parameters['Herkunft'].unique()
color_palette = px.colors.qualitative.Plotly  # Plotly-Farben
color_mapping = {herkunft: color_palette[i % len(color_palette)] for i, herkunft in enumerate(herkunfts_list)}

# Neue Legende für das Diagramm
legende_text = "<b>Garnet Provenance Groups:</b><br>"

# Punktelegende für die Herkunftsgruppen
#for herkunft, color in color_mapping.items():
 #   legende_text += f'<span style="color:{color};">●</span> {herkunft}<br>'


# Manuelle Reihenfolge der Convex Hulls in der Legende
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

# Legendentext vorbereiten
legende_text += ""
for file_path in ordered_hulls:
    color = color_mapping_files[file_path]  # Farbe aus Mapping holen
    hull_name = legend_mapping.get(file_path, file_path.split("\\")[-1].split(".")[0])  # Namen abrufen
    legende_text += f'<span style="color:{color};">■</span> {hull_name}<br>'


# Liste, um Punkte nach Herkunft und Rechteck (AB) zu gruppieren
grouped_points = {}


df_hulls_4 = pd.read_excel(convex_hulls_file_4)
df_hulls_5 = pd.read_excel(convex_hulls_file_5)
df_hulls_6 = pd.read_excel(convex_hulls_file_6)
df_hulls_7 = pd.read_excel(convex_hulls_file_7)
df_hulls_8 = pd.read_excel(convex_hulls_file_8)
df_hulls_9 = pd.read_excel(convex_hulls_file_9)
df_hulls_11 = pd.read_excel(convex_hulls_file_11)
df_hulls_12 = pd.read_excel(convex_hulls_file_12)
# Lade die Convex Hull-Daten
#df_hulls = pd.read_excel(convex_hulls_file)

 #Kombiniere die beiden Dateien
df_hulls_combined = pd.concat([df_hulls_4, df_hulls_5,df_hulls_6,df_hulls_7,df_hulls_8, df_hulls_9, df_hulls_11, df_hulls_12])


# Gruppen der Hull-Daten basierend auf Herkunft und AB_Value
grouped_hulls_combined = df_hulls_combined.groupby(["Herkunft", "AB_Value"])


# Funktion zur Darstellung der Convex Hulls mit unterschiedlichen Farben
def plot_imported_hulls_with_file_colors(grouped_hulls, file_color_mapping):
    for (herkunft, ab_value), group in grouped_hulls:
        # Extrahiere X- und Y-Koordinaten der Hull-Punkte
        hull_x = group["X"].values
        hull_y = group["Y"].values

        # Schließe die Hull ab, indem du den ersten Punkt ans Ende setzt
        hull_x = np.append(hull_x, hull_x[0])
        hull_y = np.append(hull_y, hull_y[0])

        # Bestimme die Farbe basierend auf der Datei
        file_source = group["file_source"].iloc[0]  # Dateiherkunft
        color = file_color_mapping.get(file_source, "rgba(0, 0, 0, 0.5)")  # Standardfarbe Schwarz

        # Plotte die Convex Hull
        fig.add_trace(go.Scatter(
            x=hull_x,
            y=hull_y,
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="toself",
            fillcolor=ensure_transparency(color, alpha=0.6),  # Transparenter Farbton mit Alpha = 0.4
            name=f"Herkunft: {herkunft}, AB: {ab_value}"  # Legendenname
        ))

# Ergänze eine Spalte in den DataFrames, um die Dateiherkunft zu markieren


df_hulls_4["file_source"] = convex_hulls_file_4
df_hulls_5["file_source"] = convex_hulls_file_5
df_hulls_6["file_source"] = convex_hulls_file_6
df_hulls_7["file_source"] = convex_hulls_file_7
df_hulls_8["file_source"] = convex_hulls_file_8
df_hulls_9["file_source"] = convex_hulls_file_9
df_hulls_11["file_source"] = convex_hulls_file_11
df_hulls_12["file_source"] = convex_hulls_file_12
# Kombiniere die DataFrames
df_hulls_combined = pd.concat([df_hulls_4, df_hulls_5,df_hulls_6,df_hulls_7,df_hulls_8,df_hulls_9, df_hulls_11, df_hulls_12])

# Gruppiere die kombinierten Daten
grouped_hulls_combined = df_hulls_combined.groupby(["Herkunft", "AB_Value"])

# Plotte die importierten Convex Hulls mit Datei-spezifischen Farben
plot_imported_hulls_with_file_colors(grouped_hulls_combined, color_mapping_files)


# Punkte plotten
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



# Berechne und plotte Convex Hulls für jede Gruppe
for (herkunft, ab_value), points in grouped_points.items():
    color = color_mapping.get(herkunft, "rgba(0,0,0,0.5)")  # Standardfarbe, falls nicht definiert
    plot_convex_hull(points, color)

# Layout anpassen, um den Plot zu zentrieren und um 90 Grad zu drehen
fig.update_layout(
    plot_bgcolor="white",  # Hintergrund des Plots auf Weiß setzen
    paper_bgcolor="white",  # Hintergrund des gesamten Diagramms auf Weiß setzen
    xaxis=dict(
        title="Sum of Almandine (%) + Spessartine (%)",  # Neue X-Achsenbeschriftung
        range=[0, rechtecke[-1][0] + rechtecke[-1][1]],  # Bereich der neuen X-Achse
        tickformat=".0f",
        tickfont=dict(size=24, color="black"),
        titlefont=dict(size=25, color="black")
    ),
    yaxis=dict(
        title="Pyrope (%) /// Difference between height of AB rectangle and Pyrope content (%) equals Grossular content (%)",
        range=[0, 100],  # Stelle sicher, dass es nicht über 100 hinausgeht
        constrain="domain",
        tickformat=".0f",
        dtick=10,
        color="black",  # Farbe der Achsenbeschriftung und Achsenlinien auf Schwarz setzen
        linecolor="gray",
        tickfont=dict(size=24, color="black"),
        titlefont=dict(size=20, color="black")
    ),
    autosize=False,  # Deaktiviere automatische Größenanpassung
    width=2100,  # Setze die Breite des Plots größer
    height=1200,  # Setze die Höhe des Plots größer
    margin=dict(l=0, r=5, t=20, b=5),  # Zentriere den Plot, indem du die Ränder minimierst
    showlegend=False  # Deaktiviere die Legende
)

# Werte, bei denen die gestrichelten Linien eingefügt werden sollen
y_values = [ 10, 20, 30, 40, 50, 60, 70, 80, 90 ]

# Waagrechte, gestrichelte Linien einfügen
for y in y_values:
    fig.add_shape(
        type="line",
        x0=0,  # Startpunkt der Linie auf der X-Achse (linker Rand des Diagramms)
        x1=max(df_parameters['Unnamed: 3']) + rechtecke[-1][0] + rechtecke[-1][1],  # Endpunkt der Linie auf der X-Achse (rechter Rand)
        y0=y,  # Y-Wert, bei dem die Linie gezeichnet wird
        y1=y,  # Y-Wert bleibt konstant (waagrecht)
        line=dict(
            color="black",  # Farbe der Linie
            width=0.5,        # Breite der Linie
            dash="dash"     # Stil der Linie: gestrichelt
        )
    )

# Positionen für die senkrechten gestrichelten Linien
x_values = [909.5, 3749.5, 4569.5, 1769.5, 2529.5, 3189.5, 4209.5, 4850.5, 4995.5, 442, 1352, 2162, 2872, 3482, 3992, 4402, 4712, 4922, 5037.5]  # Mittlere Positionen von AB90, AB50, AB30

# Senkrechte gestrichelte Linien einfügen
for x in x_values:
    fig.add_shape(
        type="line",
        x0=x,  # X-Wert, bei dem die Linie gezeichnet wird
        x1=x,  # X-Wert bleibt konstant (senkrecht)
        y0=0,  # Startpunkt der Linie auf der Y-Achse
        y1=100,  # Endpunkt der Linie auf der Y-Achse (höchster Wert)
        line=dict(
            color="black",  # Farbe der Linie
            width=1,        # Breite der Linie
            dash="dash"     # Stil der Linie: gestrichelt
        )
    )

# Drehe den Plot um 90 Grad (tausche X- und Y-Daten)
for trace in fig.data:
    trace.x, trace.y = trace.y, trace.x


# Füge die Legende zum Plot als Annotation hinzu (wie im ersten Skript)
fig.update_layout(
    annotations=[
        dict(
            x=650,  # X-Position der Legende (anpassen falls nötig)
            y=80,    # Y-Position
            text=legende_text,  # Die generierte Legende
            showarrow=False,
            font=dict(size=35, color="black"),
            bgcolor="rgba(249, 249, 249,1)",  # Weißer Hintergrund mit leichter Transparenz
            bordercolor="black",
            borderwidth=3,
            xanchor="left",  # Links ausrichten
            yanchor="top",  # Oben ausrichten
            align="left"  # Links ausgerichteter Text
        )
    ]
)

# Plot anzeigen
fig.show()
