# LGVI

**TL;DR**  
Efficiently calculate **Greenview / Buildview / Blueview** proportions by casting radial lines from observer points, intersecting with urban features, and projecting results onto a 1D coverage array.

---

## 🌟 Highlights

- **Radial lines + 1D coverage mapping** for fast estimation of visible greenery, buildings, and water  
- **Compatible** with vector polygons/lines; (Multi)Polygon features are automatically converted to boundary lines for intersection  
- **Fully configurable**: radius, number of angles, observer height, and class mapping (`class_new`)  
- **Lightweight**: single-file script with CLI support for batch integration  
- **Projection-ready**: works with metric CRS (e.g., UTM), with automatic `to_crs` handling  

---

## 🛠️ Method Overview

1. **Projection**  
   Project points and target layers to a metric CRS (e.g., `EPSG:32650`).  

2. **Ray Casting**  
   For each observation point, generate `num_angles` radial lines with length `buffer_distance`.  

3. **Intersection & Mapping**  
   - Intersect radial lines with target features (boundaries).  
   - Map intersections to a **1D array** (covering assigned class values).  

4. **Aggregation**  
   Aggregate coverage proportions across all rays to calculate:  
   - **Greenview**  
   - **Buildview**  
   - **Blueview**  

---

## 📊 Default Class Mapping (`class_new`)

| Code | Class      | Category |
|------|-----------|----------|
| 9    | Trees     | Green    |
| 2    | Grass     | Green    |
| 7    | Buildings | Build    |
| 6    | Water     | Blue     |

👉 Custom mappings can be specified via parameters.

---

## 📌 Example Usage

```bash
# Run LGVI calculation
python lgvi.py --points observer_points.shp --features city_layers.shp --num_angles 360 --radius 100
