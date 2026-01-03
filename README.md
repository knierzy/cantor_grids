# cantor_grids
**Cantor Grids: A new 2D Visualization Framework for Four-Parameter Compositional Datasets in Geochemistry and Soil Science**

## Overview

This repository provides a framework for the graphical representation and analysis of four-component systems using customized Cantor grids. The approach preserves the explicit contribution of all four parameters, avoids geometric distortions inherent to extended ternary diagrams, and enables high-resolution discrimination of subfields.


The project comprises modular Python scripts labeled `C1` to `C9`:

| File | Description |
|------|-------------|
| `C1.py` | Generates the constrained Cartesian product for four compositional parameters.|
| `C2.py` | Constructs and exports convex hulls derived from Cartesian products.|
| `C3.py` | Compares two spreadsheet datasets using the first four columns to identify identical records.|
| `C4.py` | Visualizes scattered garnet subfields in a Cantor diagram based on their end-member compositions.|
| `C5.py` | Visualizes scattered garnet subfields in a Cantor Diagram with an additional colorbar for interpreting samples from Pernegg; the classification is based on Mahalanobis distance|
| `C6.py` | Cantor diagram for the Austrian soil texture classification including humus as fourth component. |
| `C7.py` | Cantor diagram for the Austrian soil texture classification including humus, snthetic data, and AWC.|
| `C8.py` | AWC displayed in a Cantor diagram using scalar field visualization.|
| `C9.py` | Cantor diagram with soil samples from the district of Kirchdorf, combined with AWC displayed via a continuous color bar.|
