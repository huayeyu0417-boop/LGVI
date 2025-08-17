# LGVI
TL;DR: Efficiently calculate Greenview / Buildview / Blueview proportions by casting radial lines from observer points, intersecting with urban features, and projecting results onto a 1D coverage array.

Highlights

Radial lines + 1D coverage mapping for fast estimation of visible greenery, buildings, and water
Compatible with vector polygons/lines; (Multi)Polygon features are automatically converted to boundary lines for intersection
Fully configurable: radius, number of angles, observer height, and class mapping (class_new)
Lightweight: single-file script with CLI support for batch integration
Works with metric projections (e.g., UTM), automatic to_crs handling

Method Overview

Project points and target layers to a metric CRS (e.g., EPSG:32650)
For each observation point, generate num_angles radial lines with length buffer_distance Intersect with target features (boundaries), map intersections to a 1D array (covering assigned class values)
Aggregate coverage proportions across all rays to calculate Greenview / Buildview / Blueview
Default class mapping (class_new):
9 = Trees, 2 = Grass → Green
7 = Buildings → Build
6 = Water → Blue
Custom mappings can be specified via parameters.
