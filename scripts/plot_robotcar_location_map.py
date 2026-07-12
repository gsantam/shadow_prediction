#!/usr/bin/env python3
"""Create an interactive RobotCar location map from the Kaggle archive splits."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from shadow_prediction.dataset_robotcar_sun import (  # noqa: E402
    ARCHIVE_ROOT,
    DEFAULT_ARCHIVE,
    _load_robotcar_tags,
)
from shadow_prediction.robotcar_sun import utm_to_latlon  # noqa: E402


def timestamp_to_hour(timestamp_us: int, tz: ZoneInfo) -> float:
    dt = datetime.fromtimestamp(timestamp_us / 1_000_000, tz=timezone.utc).astimezone(tz)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def valid_timestamp(value: str | None) -> bool:
    if value is None:
        return False
    value = value.strip()
    return bool(value) and value.lower() not in {"nan", "none"}


def load_split_points(
    archive_path: Path,
    split: str,
    camera: str,
    tz: ZoneInfo,
    sun_runs_only: bool,
) -> list[dict[str, object]]:
    tag_map = _load_robotcar_tags() if sun_runs_only else {}
    points: list[dict[str, object]] = []

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(f"{ARCHIVE_ROOT}/{split}.csv") as f:
            wrapper = io.TextIOWrapper(f, encoding="utf-8", newline="")
            reader = csv.DictReader(wrapper)
            for row in reader:
                track = row["track"]
                if sun_runs_only and "sun" not in tag_map.get(track, set()):
                    continue
                timestamp_value = row.get(camera)
                if not valid_timestamp(timestamp_value):
                    continue

                easting = float(row["easting"])
                northing = float(row["northing"])
                lat, lon = utm_to_latlon(easting=easting, northing=northing)
                timestamp_us = int(timestamp_value)
                points.append(
                    {
                        "lat": round(lat, 7),
                        "lon": round(lon, 7),
                        "easting": easting,
                        "northing": northing,
                        "hour": round(timestamp_to_hour(timestamp_us, tz), 2),
                        "track": track,
                    }
                )

    return points


def point_cells(points: list[dict[str, object]], cell_m: float) -> set[tuple[int, int]]:
    return {
        (
            int(round(float(point["easting"]) / cell_m)),
            int(round(float(point["northing"]) / cell_m)),
        )
        for point in points
    }


def print_overlap_summary(groups: dict[str, list[dict[str, object]]], cell_sizes: list[float]) -> None:
    for prefix in ["all", "sun"]:
        train = groups[f"{prefix}_train"]
        val = groups[f"{prefix}_val"]
        print(prefix)
        print(
            f"  train: n={len(train):,}, tracks={len({point['track'] for point in train}):,}"
        )
        print(f"  val:   n={len(val):,}, tracks={len({point['track'] for point in val}):,}")
        for cell_m in cell_sizes:
            train_cells = point_cells(train, cell_m)
            val_cells = point_cells(val, cell_m)
            overlap = train_cells & val_cells
            val_overlap = 100.0 * len(overlap) / max(len(val_cells), 1)
            train_overlap = 100.0 * len(overlap) / max(len(train_cells), 1)
            print(
                f"  {cell_m:.0f}m cells: train_cells={len(train_cells):,}, "
                f"val_cells={len(val_cells):,}, shared={len(overlap):,}, "
                f"shared/val={val_overlap:.1f}%, shared/train={train_overlap:.1f}%"
            )


def html_document(groups: dict[str, list[dict[str, object]]], camera: str) -> str:
    visible_points = groups["all_train"] + groups["all_val"]
    center_lat = float(np.mean([point["lat"] for point in visible_points]))
    center_lon = float(np.mean([point["lon"] for point in visible_points]))
    payload = json.dumps(groups, separators=(",", ":"))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Oxford RobotCar Location Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map {{
      height: 100%;
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .summary {{
      background: rgba(255, 255, 255, 0.92);
      padding: 10px 12px;
      border-radius: 6px;
      box-shadow: 0 1px 8px rgba(0, 0, 0, 0.2);
      color: #111827;
      font-size: 13px;
      line-height: 1.35;
    }}
    .summary strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const groups = {payload};
    const colors = {{
      all_train: "#2563eb",
      all_val: "#dc2626",
      sun_train: "#0891b2",
      sun_val: "#f97316"
    }};
    const labels = {{
      all_train: "All train",
      all_val: "All val",
      sun_train: "Sun train",
      sun_val: "Sun val"
    }};

    const map = L.map("map", {{ preferCanvas: true }}).setView([{center_lat:.7f}, {center_lon:.7f}], 13);
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    const renderer = L.canvas({{ padding: 0.5 }});
    const overlays = {{}};
    const bounds = [];

    function addLayer(key) {{
      const layer = L.layerGroup();
      for (const point of groups[key]) {{
        const marker = L.circleMarker([point.lat, point.lon], {{
          renderer,
          radius: 3,
          color: colors[key],
          fillColor: colors[key],
          fillOpacity: 0.38,
          opacity: 0.75,
          weight: 1
        }});
        marker.bindTooltip(`${{labels[key]}}<br>${{point.track}}<br>${{point.hour}}h`);
        marker.addTo(layer);
        bounds.push([point.lat, point.lon]);
      }}
      overlays[labels[key] + ` (${{groups[key].length.toLocaleString()}})`] = layer;
      return layer;
    }}

    const allTrain = addLayer("all_train");
    const allVal = addLayer("all_val");
    addLayer("sun_train");
    addLayer("sun_val");
    allTrain.addTo(map);
    allVal.addTo(map);

    L.control.layers(null, overlays, {{ collapsed: false }}).addTo(map);
    if (bounds.length > 0) {{
      map.fitBounds(bounds, {{ padding: [20, 20] }});
    }}

    const summary = L.control({{ position: "topright" }});
    summary.onAdd = function () {{
      const div = L.DomUtil.create("div", "summary");
      div.innerHTML = `
        <strong>Oxford RobotCar {camera}</strong>
        Default: all train/val layers<br>
        Toggle sun-tagged layers in the control.<br>
        Points are split CSV locations.
      `;
      return div;
    }};
    summary.addTo(map);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--camera", default="stereo_centre")
    parser.add_argument("--timezone", default="Europe/London")
    parser.add_argument("--output", type=Path, default=Path("outputs/robotcar_location_map.html"))
    args = parser.parse_args()

    tz = ZoneInfo(args.timezone)
    groups = {
        "all_train": load_split_points(args.archive, "train", args.camera, tz, False),
        "all_val": load_split_points(args.archive, "val", args.camera, tz, False),
        "sun_train": load_split_points(args.archive, "train", args.camera, tz, True),
        "sun_val": load_split_points(args.archive, "val", args.camera, tz, True),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_document(groups, args.camera), encoding="utf-8")
    print(f"Saved map to {args.output}")
    print_overlap_summary(groups, cell_sizes=[20.0, 50.0, 100.0])


if __name__ == "__main__":
    main()
