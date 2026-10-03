"""Turning YOLO output into person/dog detections, pairs and pair events."""
import math

import cv2

# COCO class IDs of the two classes we detect
CLASS_PERSON = 0
CLASS_DOG = 16

PERSON_COLOR = (0, 255, 0)  # BGR green
DOG_COLOR = (0, 0, 255)     # BGR red


def class_thresholds(cfg):
    """Return the per-class confidence thresholds {class_id: threshold} from the config."""
    d = cfg["detection"]
    return {CLASS_PERSON: d["conf_person"], CLASS_DOG: d["conf_dog"]}


def draw_bounding_box(frame, x1, y1, x2, y2, cls, conf):
    """Draw a bounding box and class label on the frame."""
    label = "person" if cls == CLASS_PERSON else "dog"
    color = PERSON_COLOR if cls == CLASS_PERSON else DOG_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_detections(frame, persons, dogs):
    """Draw boxes for all persons and dogs on the frame (in place)."""
    for p in persons:
        draw_bounding_box(frame, *p["coords"], CLASS_PERSON, p["conf"])
    for d in dogs:
        draw_bounding_box(frame, *d["coords"], CLASS_DOG, d["conf"])


def collect_detections(boxes, conf_thresholds, dog_inside_person_max_conf):
    """Split YOLO boxes into persons and dogs, applying per-class thresholds and the false-positive filter.

    A dog with confidence below dog_inside_person_max_conf whose centre lies inside a person box is
    dropped (the common "hood as dog" mistake).

    Returns two lists of {"id": int | None, "coords": (x1, y1, x2, y2), "center": (cx, cy), "conf": float}.
    "id" is the tracker ID and is None when the tracker has not assigned one yet.
    """
    persons = []
    dogs = []
    if boxes is None or len(boxes) == 0:
        return persons, dogs

    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if cls not in conf_thresholds or conf < conf_thresholds[cls]:
            continue
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        track_id = int(box.id) if getattr(box, "id", None) is not None else None
        det = {
            "id": track_id,
            "coords": (x1, y1, x2, y2),
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "conf": conf,
        }
        (persons if cls == CLASS_PERSON else dogs).append(det)

    def inside_person(dog):
        cx, cy = dog["center"]
        return any(
            px1 <= cx <= px2 and py1 <= cy <= py2
            for px1, py1, px2, py2 in (p["coords"] for p in persons)
        )

    dogs = [d for d in dogs if not (d["conf"] < dog_inside_person_max_conf and inside_person(d))]
    return persons, dogs


def find_pairs(persons, dogs, distance_threshold):
    """Return {(person_key, dog_key): distance} for every person/dog pair closer than the threshold.

    Keys are tracker IDs ("p3", "d7"); a detection without an ID falls back to its list index ("p#0").
    """
    pairs = {}
    for i, p in enumerate(persons):
        pkey = f"p{p['id']}" if p["id"] is not None else f"p#{i}"
        for j, d in enumerate(dogs):
            dkey = f"d{d['id']}" if d["id"] is not None else f"d#{j}"
            distance = math.dist(p["center"], d["center"])
            if distance < distance_threshold:
                pairs[(pkey, dkey)] = distance
    return pairs


class PairTracker:
    """Turn per-frame person/dog pairs into events: a pair starts, is still going, or ends.

    A pair is considered ended when it has not been seen for lost_timeout seconds, so a few
    missed detections do not split one encounter into several. While a pair persists, an
    "ongoing" event is emitted every snapshot_interval seconds (e.g. to save another screenshot).
    All times are in seconds on whatever clock the caller uses (wall clock or video time).
    """

    def __init__(self, lost_timeout, snapshot_interval):
        self.lost_timeout = lost_timeout
        self.snapshot_interval = snapshot_interval
        self.active = {}

    @classmethod
    def from_config(cls, cfg):
        """Create a tracker from the "pairs" section of the config."""
        p = cfg["pairs"]
        return cls(p["lost_timeout_sec"], p["snapshot_interval_sec"])

    def update(self, pairs, now):
        """Return a list of (event, key, distance, duration) tuples; event is "start", "ongoing" or "end"."""
        events = []
        for key, distance in pairs.items():
            state = self.active.get(key)
            if state is None:
                self.active[key] = {"start": now, "last_seen": now, "last_snap": now}
                events.append(("start", key, distance, 0.0))
                continue
            state["last_seen"] = now
            if now - state["last_snap"] >= self.snapshot_interval:
                state["last_snap"] = now
                events.append(("ongoing", key, distance, now - state["start"]))

        for key in [k for k, s in self.active.items() if k not in pairs and now - s["last_seen"] >= self.lost_timeout]:
            state = self.active.pop(key)
            events.append(("end", key, None, state["last_seen"] - state["start"]))
        return events

    def finish(self):
        """End all active pairs (call when the input is over)."""
        events = [("end", key, None, s["last_seen"] - s["start"]) for key, s in self.active.items()]
        self.active.clear()
        return events
