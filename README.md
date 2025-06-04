# cantor_grids
**Cantor Grids: A New Domain-Separating 2D Visualization Method for Four-Component Systems, with Exemplary Applications in Garnet Chemistry and Soil Classification**

## Overview

This repository provides a framework for the graphical representation and analysis of four-component systems using customized Cantor grids. The approach is designed to minimize spatial overlap between data points and maximize the resolution of subgroups or classifications within complex, compositional datasets.


The project consists of modular Python scripts labeled `C1` to `C7`:

| File | Description |
|------|-------------|
| `C1_tuple_generator_for_subfields.py` | Generates valid 4-component integer compositions that sum to 100%.|
| `C2_export_display_convex_hull.py` |  Exports convex hulls based on tuples read from an Excel file. |
| `C3_calculate_overlap.py` | Compares two Excel datasets based on the first four columns to identify identical entries. |
| `C4_cantor_garnet_subfields.py` | Visualizes scattered subfields of garnets based on their end-member composition in a Cantor diagram |
| `C5_cantor_garnet_colorbar.py` | Visualizes scattered subfields of garnets plus colorbar to distinguish. |
| `C6_cantor_soil_texture_classes_organic_soils.py` |  Cantor diagram for a soil texture classification system including organo-mineral and organic soils |
| `C7_cantor_soil_texture_classes_zoom.py` | Computes and visualizes convex hulls for data clusters (e.g. Loamy Sand) based on tuples generated in `C3`. |
