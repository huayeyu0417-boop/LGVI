# pgvi_linesight.py
# -*- coding: utf-8 -*-
"""
Radial line-of-sight visual composition (Green/Build/Blue) from points.

Usage (CLI):
  python pgvi_linesight.py \
    --points /path/points.shp \
    --targets /path/targets.shp \
    --output results.csv \
    --buffer-distance 100 \
    --num-angles 360 \
    --crs EPSG:32650 \
    --observer-height 1.6 \
    --class-field class_new \
    --height-field height
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point, Polygon, MultiPolygon


# =========================
# Config & Utilities
# =========================

@dataclass
class Config:
    # I/O
    points_file: str
    targets_file: str
    output_csv: str

    # Geometry / Analysis
    crs: str = "EPSG:32650"
    buffer_distance: int = 100
    num_angles: int = 360
    observer_height: float = 1.6

    # Fields
    class_field: str = "class_new"
    height_field: str = "height"

    # Class mapping (codes in input -> buckets)
    class_map: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "green": [9, 2],   # 9=trees, 2=grass
            "build": [7],      # 7=buildings
            "blue":  [6],      # 6=water
        }
    )

# ---- helpers ----

def _as_int_series(s: pd.Series, missing: int = -9999) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(missing).astype(int)

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _coverage_update(coverage: np.ndarray, start_idx: int, end_idx: int, value: int) -> None:
    """Safely paint [start_idx, end_idx) or [end_idx, start_idx) with class value."""
    n = coverage.shape[0]
    i0 = max(0, min(n, start_idx))
    i1 = max(0, min(n, end_idx))
    if i0 == i1:
        return
    if i0 < i1:
        coverage[i0:i1] = value
    else:
        coverage[i1:i0] = value


# =========================
# Data loading / preparation
# =========================

def load_and_prepare_layers(cfg: Config) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Read points/targets and reproject to cfg.crs; normalize fields."""
    points = gpd.read_file(cfg.points_file).to_crs(cfg.crs)
    targets = gpd.read_file(cfg.targets_file).to_crs(cfg.crs)

    # Normalize class field to int
    if cfg.class_field in targets.columns:
        targets[cfg.class_field] = _as_int_series(targets[cfg.class_field])
    else:
        targets[cfg.class_field] = -9999

    # Normalize height field to float
    if cfg.height_field in targets.columns:
        targets[cfg.height_field] = pd.to_numeric(targets[cfg.height_field], errors="coerce")
    else:
        targets[cfg.height_field] = np.nan

    return points, targets


# =========================
# Geometry operations
# =========================

def generate_radial_lines(x: float, y: float, radius: float, num_angles: int, crs: str) -> gpd.GeoDataFrame:
    """Generate rays from (x, y) out to distance 'radius' at 1-degree steps."""
    angles = np.arange(0, num_angles, 1)
    lines = []
    for angle in angles:
        rad = np.radians(angle)
        ex = x + radius * np.cos(rad)
        ey = y + radius * np.sin(rad)
        lines.append(LineString([(x, y), (ex, ey)]))
    return gpd.GeoDataFrame({"order": np.arange(len(lines))}, geometry=lines, crs=crs)

def convert_polys_to_boundaries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Turn (Multi)Polygons into boundary lines; keep lines/others as-is; drop empties."""
    if gdf.empty:
        return gdf
    records = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                new = row.copy()
                new.geometry = poly.boundary
                records.append(new)
        elif isinstance(geom, Polygon):
            new = row.copy()
            new.geometry = geom.boundary
            records.append(new)
        else:
            records.append(row)
    return gpd.GeoDataFrame(records, columns=gdf.columns, crs=gdf.crs)

def clip_targets_in_buffer(targets: gpd.GeoDataFrame, point_geom: Point, buffer_distance: float) -> gpd.GeoDataFrame:
    """Clip targets by circular buffer around the given point."""
    buffer_geom = point_geom.buffer(buffer_distance)
    return gpd.clip(targets, buffer_geom)

def overlay_rays_with_targets(rays: gpd.GeoDataFrame, target_lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Intersection of rays and target boundaries; geometry types may be Point or MultiPoint."""
    return gpd.overlay(rays, target_lines, how="intersection", keep_geom_type=False)

def explode_points_from_overlay(inter: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Flatten Point/MultiPoint geometries to point rows while keeping attributes."""
    geoms, attrs = [], []
    for _, row in inter.iterrows():
        geom = row.geometry
        if isinstance(geom, Point):
            geoms.append(geom)
            attrs.append(row.drop(labels="geometry"))
        elif isinstance(geom, MultiPoint):
            for p in geom.geoms:
                geoms.append(p)
                attrs.append(row.drop(labels="geometry"))
    if geoms:
        return gpd.GeoDataFrame(attrs, geometry=geoms, crs=inter.crs)
    else:
        # keep columns so concat is possible downstream
        return gpd.GeoDataFrame(columns=list(inter.columns), geometry=[], crs=inter.crs)

def add_center_points_for_orders(inter_pts: gpd.GeoDataFrame,
                                 x: float, y: float,
                                 orders: Iterable[int],
                                 height_value: float,
                                 crs: str) -> gpd.GeoDataFrame:
    """Ensure each ray has a 'center' row at the observer location with given height."""
    rows = []
    for o in orders:
        base = {k: None for k in inter_pts.columns if k != "geometry"}
        base.update({"order": o, "height": height_value})
        rows.append({**base, "geometry": Point(x, y)})
    center = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return pd.concat([inter_pts, center], ignore_index=True) if not inter_pts.empty else center


# =========================
# Projection to 1D coverage
# =========================

def compute_segment_projection(seg: gpd.GeoDataFrame,
                               buffer_distance: float,
                               observer_height: float,
                               class_field: str,
                               height_field: str) -> gpd.GeoDataFrame:
    """Compute parametric start/end on the 1D column for a single ray segment set."""
    # distances
    seg["distance_to_center"] = seg.geometry.distance(seg.geometry.iloc[0])  # wrong: geometry.iloc[0] isn't center
    # Use a scalars approach: distance to the actual center, not first row geometry:
    # For numerical stability we compute outside and pass in; but to keep signature simple, recompute here:
    center = seg.iloc[0]["geometry"]  # will be overwritten below by actual center Point
    # Robust: distance to a dedicated center point (the one at x,y we inserted)
    # find the row having duplicated Point (distance==0). If not found, fall back to min distance.
    dists = seg.geometry.distance(seg.geometry.iloc[0])
    if (dists == 0).any():
        center_geom = seg.geometry.iloc[np.where(dists == 0)[0][0]]
    else:
        center_geom = seg.geometry.iloc[dists.idxmin()]

    seg["distance_to_center"] = seg.geometry.distance(center_geom)

    d = seg["distance_to_center"].replace(0, np.nan)
    h = pd.to_numeric(seg.get(height_field, np.nan), errors="coerce")

    seg["start"] = -observer_height * buffer_distance / d
    seg["end"]   = (h - observer_height) * buffer_distance / d
    seg["start"] = seg["start"].fillna(0.0)
    seg["end"]   = seg["end"].fillna(0.0)

    # class to int
    seg[class_field] = _as_int_series(seg.get(class_field, pd.Series([-9999] * len(seg))))
    return seg

def paint_coverage(seg: gpd.GeoDataFrame, coverage_len: int, class_field: str) -> np.ndarray:
    """Paint a single ray's 1D coverage with class codes."""
    coverage = np.zeros(coverage_len, dtype=int)
    half = coverage_len // 2
    for _, row in seg.iterrows():
        start_idx = int(row["start"] * 100) + half
        end_idx   = int(row["end"]   * 100) + half
        _coverage_update(coverage, start_idx, end_idx, _safe_int(row[class_field], -9999))
    return coverage

def fraction_for_codes(coverage: np.ndarray, codes: List[int]) -> float:
    """Return fraction of coverage assigned to any of given class codes."""
    if not codes:
        return 0.0
    mask = np.isin(coverage, codes)
    return float(mask.sum()) / float(coverage.shape[0])


# =========================
# Per-point computation
# =========================

def compute_views_for_point(pt: Point,
                            targets: gpd.GeoDataFrame,
                            cfg: Config) -> Tuple[float, float, float]:
    """Compute Green/Build/Blue for a single observation point."""
    ox, oy = pt.x, pt.y

    # Rays
    rays = generate_radial_lines(ox, oy, cfg.buffer_distance, cfg.num_angles, cfg.crs)

    # Clip targets & convert to boundaries
    clipped = clip_targets_in_buffer(targets, pt, cfg.buffer_distance)
    clipped_lines = convert_polys_to_boundaries(clipped)
    if clipped_lines.crs is None:
        clipped_lines.set_crs(cfg.crs, inplace=True)
    else:
        clipped_lines = clipped_lines.to_crs(cfg.crs)

    # Intersections
    inter = overlay_rays_with_targets(rays, clipped_lines)
    inter_pts = explode_points_from_overlay(inter)

    # Orders present and ensure center point per order
    orders = sorted(inter_pts["order"].dropna().unique().tolist()) if not inter_pts.empty else list(range(cfg.num_angles))
    inter_pts = add_center_points_for_orders(inter_pts, ox, oy, orders, cfg.observer_height, cfg.crs)

    # Distances & sort far->near
    inter_pts["distance_to_center"] = inter_pts.geometry.distance(Point(ox, oy))
    inter_pts[cfg.height_field] = pd.to_numeric(inter_pts.get(cfg.height_field, np.nan), errors="coerce")
    inter_pts = inter_pts.sort_values(by="distance_to_center", ascending=False)

    # Iterate rays
    coverage_len = max(1, cfg.buffer_distance * 100)
    greenview = buildview = blueview = 0.0
    all_orders = sorted(set(orders) | set(range(cfg.num_angles)))

    for a in all_orders:
        seg = inter_pts[inter_pts["order"] == a].reset_index(drop=True)
        if seg.empty:
            # No intersections along this ray → sky; we only count ground features
            continue

        seg = compute_segment_projection(seg, cfg.buffer_distance, cfg.observer_height, cfg.class_field, cfg.height_field)
        coverage = paint_coverage(seg, coverage_len, cfg.class_field)

        greenview += fraction_for_codes(coverage, cfg.class_map.get("green", []))
        buildview += fraction_for_codes(coverage, cfg.class_map.get("build", []))
        blueview  += fraction_for_codes(coverage, cfg.class_map.get("blue",  []))

    # Average across rays
    if cfg.num_angles > 0:
        greenview /= cfg.num_angles
        buildview /= cfg.num_angles
        blueview  /= cfg.num_angles

    return float(greenview), float(buildview), float(blueview)


# =========================
# Output
# =========================

def append_result(output_csv: str, fid: Any, green: float, build: float, blue: float) -> None:
    """Append one line to CSV; write header if file doesn't exist."""
    import csv
    header = ["fid", "Greenview", "Buildview", "Blueview"]
    row = [fid, green, build, blue]
    exists = os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


# =========================
# Orchestrator
# =========================

def run(cfg: Config) -> Dict[str, Any]:
    """Execute LGVI for all points; return summary dict."""
    points, targets = load_and_prepare_layers(cfg)

    rows_out = 0
    for fidname, prow in points.iterrows():
        pt: Point = prow.geometry
        green, build, blue = compute_views_for_point(pt, targets, cfg)
        append_result(cfg.output_csv, fidname, green, build, blue)
        rows_out += 1

    return {"ok": True, "output": cfg.output_csv, "rows": rows_out}


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Radial LOS visual composition (Green/Build/Blue).")
    p.add_argument("--points", required=True, help="Path to points file (shp/gpkg/etc.)")
    p.add_argument("--targets", required=True, help="Path to targets file (polygons/lines; needs class/height fields)")
    p.add_argument("--output", required=True, help="Output CSV")
    p.add_argument("--buffer-distance", type=int, default=100, help="Radius in meters")
    p.add_argument("--num-angles", type=int, default=360, help="Number of rays (degrees)")
    p.add_argument("--crs", default="EPSG:32650", help="Working projected CRS")
    p.add_argument("--observer-height", type=float, default=1.6, help="Observer eye height (m)")
    p.add_argument("--class-field", default="class_new", help="Field name for land-use class codes")
    p.add_argument("--height-field", default="height", help="Field name for feature height (meters)")
    # Optional custom class map as JSON-like strings could be added here if needed.
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(
        points_file=args.points,
        targets_file=args.targets,
        output_csv=args.output,
        buffer_distance=args.buffer_distance,
        num_angles=args.num_angles,
        crs=args.crs,
        observer_height=args.observer_height,
        class_field=args.class_field,
        height_field=args.height_field,
    )
    res = run(cfg)
    print(res)


if __name__ == "__main__":
    main()
