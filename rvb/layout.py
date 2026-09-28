"""Store layout (floor plan in metres), zones, entrance line and pinhole cameras.

The same layout file drives the simulator (to project people into each camera) and the
zone engine (to turn tracks into footfall / dwell / stop-vs-pass events).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath


@dataclass
class Zone:
    id: str
    polygon: np.ndarray            # (N,2) world metres (or pixels for image-plane zones)
    camera: str | None = None      # camera that covers this zone
    stop_seconds: float = 5.0      # dwell >= this counts as "stopped / engaged"
    label: str = ""

    def contains(self, xy: np.ndarray) -> np.ndarray:
        return MplPath(self.polygon).contains_points(xy)

    def sample_point(self, rng: np.random.Generator, margin: float = 0.15) -> np.ndarray:
        lo, hi = self.polygon.min(0) + margin, self.polygon.max(0) - margin
        for _ in range(200):
            p = rng.uniform(lo, hi)
            if self.contains(p[None])[0]:
                return p
        return self.polygon.mean(0)

    @property
    def centroid(self) -> np.ndarray:
        return self.polygon.mean(0)


@dataclass
class Entrance:
    a: np.ndarray                  # line segment end points
    b: np.ndarray
    inside_normal: np.ndarray      # unit vector pointing into the store
    camera: str | None = None
    outside_spawn: np.ndarray | None = None

    def side(self, xy: np.ndarray) -> np.ndarray:
        """+1 inside, -1 outside (relative to the infinite line)."""
        d = (xy - self.a) @ self.inside_normal
        return np.sign(d)

    def within_span(self, xy: np.ndarray, margin: float = 0.3) -> np.ndarray:
        ab = self.b - self.a
        s = ((xy - self.a) @ ab) / (ab @ ab)
        pad = margin / np.linalg.norm(ab)
        return (s >= -pad) & (s <= 1 + pad)


@dataclass
class Camera:
    id: str
    pos: np.ndarray                # (3,) metres; z = mounting height
    look_at: np.ndarray            # (3,)
    f: float                       # focal length in pixels
    width: int = 640
    height: int = 360
    fps: float = 15.0
    R: np.ndarray = field(init=False)
    K: np.ndarray = field(init=False)

    def __post_init__(self):
        fwd = self.look_at - self.pos
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd])          # world -> camera rotation (rows)
        self.K = np.array([[self.f, 0, self.width / 2], [0, self.f, self.height / 2], [0, 0, 1.0]])

    def project(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N,3) -> pixels (N,2) and depth (N,)."""
        Pc = (P - self.pos) @ self.R.T
        z = Pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([self.f * Pc[:, 0] / z + self.width / 2, self.f * Pc[:, 1] / z + self.height / 2], 1)
        return uv, z

    @property
    def H_ground(self) -> np.ndarray:
        """Homography mapping floor (X,Y,1) -> pixel (u,v,1)."""
        t = -self.R @ self.pos
        return self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], t])

    def pixel_to_ground(self, uv: np.ndarray) -> np.ndarray:
        Hinv = np.linalg.inv(self.H_ground)
        p = np.column_stack([uv, np.ones(len(uv))]) @ Hinv.T
        return p[:, :2] / p[:, 2:3]

    def ground_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        p = np.column_stack([xy, np.ones(len(xy))]) @ self.H_ground.T
        return p[:, :2] / p[:, 2:3]


@dataclass
class Layout:
    name: str
    floor: tuple[float, float]
    zones: list[Zone]
    entrance: Entrance
    cameras: list[Camera]
    fps: float = 15.0
    staff_home: np.ndarray | None = None
    raw: dict = field(default_factory=dict)

    def zone(self, zid: str) -> Zone:
        return next(z for z in self.zones if z.id == zid)

    def camera(self, cid: str) -> Camera:
        return next(c for c in self.cameras if c.id == cid)

    def zones_for_camera(self, cid: str) -> list[Zone]:
        return [z for z in self.zones if z.camera == cid]


def load_layout(path: str | Path) -> Layout:
    d = json.loads(Path(path).read_text())
    fps = d.get("fps", 15.0)
    zones = [Zone(z["id"], np.asarray(z["polygon"], float), z.get("camera"), z.get("stop_seconds", d.get("stop_seconds", 5.0)),
                  z.get("label", z["id"])) for z in d["zones"]]
    e = d["entrance"]
    n = np.asarray(e["inside_normal"], float)
    ent = Entrance(np.asarray(e["line"][0], float), np.asarray(e["line"][1], float), n / np.linalg.norm(n),
                   e.get("camera"), np.asarray(e.get("outside_spawn", [0, 0]), float))
    cams = [Camera(c["id"], np.asarray(c["pos"], float), np.asarray(c["look_at"], float), c["f"],
                   c.get("width", 640), c.get("height", 360), c.get("fps", fps)) for c in d["cameras"]]
    return Layout(d["name"], tuple(d["floor"]), zones, ent, cams, fps,
                  np.asarray(d["staff_home"], float) if "staff_home" in d else None, d)
