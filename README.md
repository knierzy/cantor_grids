# cantor_grids
**Cantor Grids: A New Domain-Separating 2D Visualization Method for Four-Component Systems, with Exemplary Applications in Garnet Chemistry and Soil Classification**

## Overview

This repository provides a framework for the graphical representation and analysis of four-component systems using customized Cantor grids. The approach is designed to minimize spatial overlap between data points and maximize the resolution of subgroups or classifications within complex, compositional datasets.


The project consists of modular Python scripts labeled `C1` to `C7`:

| File | Description |
|------|-------------|
| `C1_tuple_generator_for_subfields.py` | Reads raw Excel files and prepares the data (e.g. elemental compositions or soil fractions). |
| `C2_export_display_convex_hull.py` | Computes transformed x/y coordinates for plotting in modified ternary-style diagrams. |
| `C3_calculate_overlap.py` | Generates tuples (valid data points) for specific subfields such as *Loamy Sand*, based on threshold definitions. |
| `C4_cantor_garnet_subfields.py` | Defines color schemes for grouping by origin, class, or other metadata. |
| `C5_cantor_garnet_colorbar.py` | Creates the complete Cantor Grid with triangle geometry, axis labels, labeled subfields, and data points. |
| `C6_cantor_soil_texture_classes_organic_soils.py` | (Archived/optional) Previously handled organic soil variants – currently not in use. |
| `C7_cantor_soil_texture_classes_zoom.py` | Computes and visualizes convex hulls for data clusters (e.g. Loamy Sand) based on tuples generated in `C3`. |
