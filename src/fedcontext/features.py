"""Causal receiver-side features shared by offline preparation and streaming inference."""
from collections import deque
from dataclasses import dataclass, field
import math
import numpy as np
from .utils import read_json, stable_hash

NAMES = {
    "movement": ["dx", "dy", "speed", "acceleration", "heading_sin", "heading_cos"],
    "timing": ["receive_gap", "send_gap", "delay", "sender_rate_1s", "receive_jitter", "repeat_id_fraction", "repeat_payload_fraction"],
    "context": ["neighbor_count", "neighbor_median_speed", "speed_difference", "nearest_distance", "heading_disagreement", "receiver_distance", "road_distance", "road_heading_difference", "has_neighbors", "has_road_map"],
}
# Fixed engineering scales shared by all RSUs. No pooled private scaler or test fit.
SCALES = {"movement": [10, 10, 40, 10, 1, 1], "timing": [1, 1, .1, 10, .2, 1, 1],
          "context": [20, 40, 40, 300, 180, 300, 10, 180, 1, 1]}


def angle_difference(a, b):
    return abs((a-b+180) % 360-180)


class RoadMap:
    """Optional directed lane/road centreline adapter. Input coordinates MUST be local metres.

    JSON: {"coordinate_system":"local_meters", "roads":[{"points":[[x,y],...],"bidirectional":true}]}
    No use of simulator ground-truth road-distance fields.
    """
    def __init__(self, path=None):
        self.segments = []
        if path:
            obj = read_json(path)
            if obj.get("coordinate_system") != "local_meters":
                raise ValueError("Road map must declare coordinate_system=local_meters; match dataset projection")
            for road in obj["roads"]:
                points = np.asarray(road["points"], dtype=float)
                if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                    raise ValueError("Invalid road coordinates")
                for a, b in zip(points[:-1], points[1:]):
                    if np.linalg.norm(b-a) > 0:
                        self.segments.append((a, b, bool(road.get("bidirectional", False))))
            if not self.segments:
                raise ValueError("No nonzero road segments")

    def match(self, x, y, heading):
        if not self.segments:
            return 0., 0., 0.
        point, best = np.array([x, y]), (float("inf"), 0.)
        for a, b, both in self.segments:
            d = b-a
            u = np.clip(np.dot(point-a, d)/np.dot(d, d), 0, 1)
            distance = np.linalg.norm(point-(a+u*d))
            # SUMO/VeReMi heading: 0=north, clockwise.
            bearing = math.degrees(math.atan2(d[0], d[1])) % 360
            diff = angle_difference(heading, bearing)
            if both:
                diff = min(diff, angle_difference(heading, bearing+180))
            if distance < best[0]:
                best = (float(distance), diff)
        return *best, 1.


@dataclass
class Track:
    last: dict | None = None
    recent: deque = field(default_factory=deque)
    gaps: deque = field(default_factory=lambda: deque(maxlen=10))
    movement: deque = field(default_factory=deque)
    timing: deque = field(default_factory=deque)
    labels: deque = field(default_factory=deque)


class ReceiverFeatures:
    """One instance per receiver session. update consumes receive-time ordered messages.

    Labels are carried ONLY to filter training windows / compute evaluation labels;
    no model input, score, neighbor selection or reset condition uses them.
    """
    def __init__(self, cfg, road=None):
        self.cfg, self.road = cfg, road or RoadMap()
        self.tracks, self.neighbors = {}, {}
        self.last_time = -float("inf")

    def update(self, row):
        t, alias = float(row["rx_time"]), str(row["pseudonym"])
        if t < self.last_time:
            raise ValueError("Receive times must be nondecreasing")
        self.last_time = t
        # Bound inactive sender state; no future information consulted.
        stale = [a for a, tr in self.tracks.items() if tr.last and t-tr.last["rx_time"] > self.cfg["max_gap_s"]]
        for a in stale:
            del self.tracks[a]
        self.neighbors = {a: r for a, r in self.neighbors.items() if t-r["rx_time"] <= self.cfg["neighbor_ttl_s"]}
        tr = self.tracks.setdefault(alias, Track())
        prev = tr.last
        gap = 0. if prev is None else t-prev["rx_time"]
        send_gap = 0. if prev is None else row["tx_time"]-prev["tx_time"]
        if prev is not None:
            tr.gaps.append(gap)
        dx = 0 if prev is None else row["x"]-prev["x"]
        dy = 0 if prev is None else row["y"]-prev["y"]
        h = math.radians(row["heading"])
        m = np.array([dx, dy, row["speed"], row["acceleration"], math.sin(h), math.cos(h)])
        while tr.recent and t-tr.recent[0][0] > 1.:
            tr.recent.popleft()
        # Repeated payload is an observable hint, NOT a definitive replay label.
        payload = tuple(row[k] for k in ("x", "y", "speed", "acceleration", "heading"))
        tr.recent.append((t, str(row["message_id"]), payload))
        n = len(tr.recent)
        repeat_id = (n-len({e[1] for e in tr.recent}))/n
        repeat_payload = (n-len({e[2] for e in tr.recent}))/n
        timing = np.array([gap, send_gap, t-row["tx_time"], n, np.std(tr.gaps) if tr.gaps else 0, repeat_id, repeat_payload])
        near = []
        for a, r in self.neighbors.items():
            if a != alias:
                dist = math.hypot(row["x"]-r["x"], row["y"]-r["y"])
                if dist <= self.cfg["neighbor_radius_m"]:
                    near.append((r, dist))
        if near:
            median_speed = float(np.median([r["speed"] for r, _ in near]))
            speed_diff = row["speed"]-median_speed
            distance = min(d for _, d in near)
            hd = float(np.median([angle_difference(row["heading"], r["heading"]) for r, _ in near]))
        else:
            median_speed = speed_diff = distance = hd = 0.
        rd, rh, has_road = self.road.match(row["x"], row["y"], row["heading"])
        context = np.array([len(near), median_speed, speed_diff, distance, hd,
                            math.hypot(row["x"]-row["rx_x"], row["y"]-row["rx_y"]),
                            rd, rh, bool(near), has_road])
        m = (m/np.array(SCALES["movement"])).astype(np.float32)
        timing = (timing/np.array(SCALES["timing"])).astype(np.float32)
        context = (context/np.array(SCALES["context"])).astype(np.float32)
        length = self.cfg["history"]
        result = None
        if len(tr.movement) == length:
            result = dict(movement=np.array(tr.movement, dtype=np.float32), target=m,
                          timing=np.array([*tr.timing, timing][-length:], dtype=np.float32),
                          context=context,
                          clean_window=not any([*tr.labels, int(row.get("label", 0))]))
        tr.movement.append(m)
        tr.timing.append(timing)
        tr.labels.append(int(row.get("label", 0)))
        for q in (tr.movement, tr.timing, tr.labels):
            while len(q) > length:
                q.popleft()
        tr.last = dict(row)
        self.neighbors[alias] = dict(row)
        return result


def client_id(row, cfg):
    if str(row.get("rsu", "")) not in ("", "nan"):
        key = "manifest:"+str(row["rsu"])
    elif cfg["partition"] == "spatial":
        # Partition by receiver region, NOT attacker-claimed location.
        key = f"{row['scenario']}:{math.floor(row['rx_x']/cfg['region_width_m'])}:{math.floor(row['rx_y']/cfg['region_width_m'])}"
    else:
        key = f"{row['scenario']}:{row['receiver']}"
    return int(stable_hash(key) % cfg["clients"])
