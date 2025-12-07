# cantor_grids
**Cantor Grids: A New 2D Visualization Method for Four-Parameter Systems, with Exemplary Applications in Garnet Chemistry and Soil Classification**

## Overview

This repository provides a framework for the graphical representation and analysis of four-component systems using customized Cantor grids. The approach is designed to minimize spatial overlap between data points and maximize the resolution of subgroups or classifications within complex, compositional datasets.


The project comprises modular Python scripts labeled `C1` to `C8`:

| File | Description |
|------|-------------|
| `C1.py` | Generates valid 4-component integer compositions that sum to 100%.|
| `C2.py` | Exports convex hulls based on tuples read from an Excel file. |
| `C3.py` | Compares two Excel datasets based on the first four columns to identify identical entries. |
| `C4.py` | Visualizes scattered subfields of garnets in a Cantor diagram based on their end-member composition. |
| `C5.py` | Visualizes scattered subfields of garnets in a Cantor Diagram plus colorbar to interpret samples from Pernegg. |
| `C6.py` | Cantor diagram for a soil texture classification system including organo-mineral and organic soils. |
| `C7.py` | Cantor diagram with synthetic data with displayed Available Water Content (colorbar)
| `C8.py` | Cantor diagram with soil samples of district Kirchbach with displayed Available Water Content as halo (colorbar). |
