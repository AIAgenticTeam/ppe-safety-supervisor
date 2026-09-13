"""
Camera zones: which part of the floor a person is standing in, and what PPE that
part of the floor requires.

This is the layer that makes "no gloves in the walkway" and "no gloves at the
grinder" different events. It is deliberately not machine learning -- a polygon
per zone and a point-in-polygon test, so it is inspectable, adjustable by a
safety officer, and never wrong in a way you cannot see.

A person is located by their *foot point* (bottom-centre of the person box), not
the box centre. A worker leaning over a machine has a box that overlaps three
zones; their feet are in one.

Usage:
    from zones import ZoneMap
    zmap = ZoneMap.load("zones.json")
    zone = zmap.locate("cam_3", person_bbox)      # -> Zone or None
    zone.required_ppe                              # -> ("helmet", "gloves", "goggles")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

Point = tuple[float, float]


@dataclass(frozen=True)
class Zone:
    """One named region of one camera's field of view."""

    name: str                            # machine id, e.g. "grinding_station"
    label: str                           # human label for reports, e.g. "Bay 3 - grinding station"
    polygon: tuple[Point, ...]           # image coordinates, clockwise or anti-clockwise
    required_ppe: tuple[str, ...]        # PPE that must be present in this zone
    severity_multiplier: float = 1.0     # scales the agent's severity score
    notes: str = ""                      # free text shown to the supervisor

    def contains(self, point: Point) -> bool:
        """Ray-casting point-in-polygon. No dependencies, handles concave shapes."""
        x, y = point
        inside = False
        n = len(self.polygon)
        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]
            # does the horizontal ray at y cross this edge?
            if (y1 > y) != (y2 > y):
                x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_cross:
                    inside = not inside
        return inside

    def area(self) -> float:
        """Shoelace formula -- useful for picking the tightest zone on overlap."""
        n = len(self.polygon)
        total = 0.0
        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]
            total += x1 * y2 - x2 * y1
        return abs(total) / 2.0


@dataclass(frozen=True)
class Camera:
    camera_id: str
    label: str
    zones: tuple[Zone, ...]
    default_ppe: tuple[str, ...] = ("helmet",)   # applies outside every drawn zone
    frame_size: tuple[int, int] | None = None    # (w, h) -- for validating polygons


@dataclass
class ZoneMap:
    cameras: dict[str, Camera] = field(default_factory=dict)

    @staticmethod
    def foot_point(bbox: tuple[float, float, float, float]) -> Point:
        """Bottom-centre of a person box: where they are standing."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y2)

    def locate(self, camera_id: str, bbox: tuple[float, float, float, float]) -> Zone | None:
        """Which zone is this person standing in? None means no drawn zone matched."""
        cam = self.cameras.get(camera_id)
        if cam is None:
            raise KeyError(f"unknown camera {camera_id!r}; known: {sorted(self.cameras)}")

        point = self.foot_point(bbox)
        hits = [z for z in cam.zones if z.contains(point)]
        if not hits:
            return None
        # Overlapping zones: the smallest one wins, so a machine station drawn
        # inside a general work area takes precedence over it.
        return min(hits, key=lambda z: z.area())

    def required_ppe(self, camera_id: str, bbox) -> tuple[str, ...]:
        """PPE required where this person is standing, falling back to camera default."""
        zone = self.locate(camera_id, bbox)
        if zone is not None:
            return zone.required_ppe
        return self.cameras[camera_id].default_ppe

    def zone_or_default(self, camera_id: str, bbox) -> Zone:
        """Always returns a Zone -- synthesises one for the camera default area."""
        zone = self.locate(camera_id, bbox)
        if zone is not None:
            return zone
        cam = self.cameras[camera_id]
        return Zone(
            name="unzoned",
            label=f"{cam.label} - general area",
            polygon=(),
            required_ppe=cam.default_ppe,
            severity_multiplier=1.0,
            notes="no zone polygon covers this position; camera default PPE applies",
        )

    # ---- loading -------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "ZoneMap":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, raw: dict) -> "ZoneMap":
        cameras: dict[str, Camera] = {}
        for cam_id, cam in raw.get("cameras", {}).items():
            zones = []
            for z in cam.get("zones", []):
                poly = tuple((float(p[0]), float(p[1])) for p in z["polygon"])
                if len(poly) < 3:
                    raise ValueError(f"{cam_id}/{z['name']}: polygon needs >= 3 points")
                zones.append(Zone(
                    name=z["name"],
                    label=z.get("label", z["name"]),
                    polygon=poly,
                    required_ppe=tuple(z.get("required_ppe", ())),
                    severity_multiplier=float(z.get("severity_multiplier", 1.0)),
                    notes=z.get("notes", ""),
                ))
            size = cam.get("frame_size")
            cameras[cam_id] = Camera(
                camera_id=cam_id,
                label=cam.get("label", cam_id),
                zones=tuple(zones),
                default_ppe=tuple(cam.get("default_ppe", ("helmet",))),
                frame_size=tuple(size) if size else None,
            )
        return cls(cameras=cameras)

    def validate(self) -> list[str]:
        """Returns a list of problems. Empty list means the config is sane."""
        problems: list[str] = []
        for cam in self.cameras.values():
            names = [z.name for z in cam.zones]
            for dup in {n for n in names if names.count(n) > 1}:
                problems.append(f"{cam.camera_id}: duplicate zone name {dup!r}")
            if cam.frame_size:
                w, h = cam.frame_size
                for z in cam.zones:
                    for (x, y) in z.polygon:
                        if not (0 <= x <= w and 0 <= y <= h):
                            problems.append(
                                f"{cam.camera_id}/{z.name}: point ({x:.0f},{y:.0f}) "
                                f"outside frame {w}x{h}")
                            break
            for z in cam.zones:
                if not z.required_ppe:
                    problems.append(f"{cam.camera_id}/{z.name}: required_ppe is empty")
                if z.area() < 100:
                    problems.append(f"{cam.camera_id}/{z.name}: area {z.area():.0f}px is tiny")
        return problems


def draw_overlay(image_path: str | Path, zmap: ZoneMap, camera_id: str,
                 out_path: str | Path) -> str:
    """Render zone polygons over a real frame so a safety officer can check them.

    Do this before trusting any zone config -- a polygon that looks right in JSON
    is frequently ten metres off on the actual floor.
    """
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    palette = [(200, 16, 46), (227, 82, 5), (0, 132, 61), (0, 94, 184), (176, 132, 0)]
    for i, z in enumerate(zmap.cameras[camera_id].zones):
        colour = palette[i % len(palette)]
        draw.polygon(list(z.polygon), fill=colour + (55,), outline=colour + (255,), width=3)
        cx = sum(p[0] for p in z.polygon) / len(z.polygon)
        cy = sum(p[1] for p in z.polygon) / len(z.polygon)
        draw.text((cx - 40, cy), f"{z.label}\n{'+'.join(z.required_ppe)}",
                  fill=(255, 255, 255, 255))

    Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB").save(out_path)
    return str(out_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Validate a zone config, or draw it on a frame")
    ap.add_argument("config", help="zones.json")
    ap.add_argument("--overlay", nargs=3, metavar=("FRAME", "CAMERA_ID", "OUT"),
                    help="render zones over a real frame")
    args = ap.parse_args()

    zmap = ZoneMap.load(args.config)
    problems = zmap.validate()
    for cam in zmap.cameras.values():
        print(f"{cam.camera_id}  ({cam.label})  default={'+'.join(cam.default_ppe)}")
        for z in cam.zones:
            print(f"    {z.name:<20} {z.area():>9,.0f}px  x{z.severity_multiplier:<4} "
                  f"{'+'.join(z.required_ppe)}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("\nconfig OK")

    if args.overlay:
        frame, cam_id, out = args.overlay
        print("wrote", draw_overlay(frame, zmap, cam_id, out))
