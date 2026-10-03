import pytest

from pdd.detection import PairTracker, class_thresholds, collect_detections, find_pairs

THRESHOLDS = {0: 0.2, 16: 0.02}
HOOD_MAX_CONF = 0.15


class _Coords:
    def __init__(self, values):
        self.values = values

    def int(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeBox:
    """Minimal stand-in for an ultralytics box."""

    def __init__(self, cls, conf, xyxy, track_id=None):
        self.cls = cls
        self.conf = conf
        self.id = track_id
        self.xyxy = [_Coords(xyxy)]


def collect(boxes):
    return collect_detections(boxes, THRESHOLDS, HOOD_MAX_CONF)


# --- class_thresholds ---

def test_class_thresholds_from_config():
    cfg = {"detection": {"conf_person": 0.3, "conf_dog": 0.05}}
    assert class_thresholds(cfg) == {0: 0.3, 16: 0.05}


# --- collect_detections ---

def test_collect_splits_persons_and_dogs_with_ids():
    boxes = [
        FakeBox(0, 0.9, (0, 0, 100, 200), track_id=3),
        FakeBox(16, 0.5, (300, 100, 350, 150), track_id=7),
    ]
    persons, dogs = collect(boxes)
    assert [p["id"] for p in persons] == [3]
    assert [d["id"] for d in dogs] == [7]
    assert dogs[0]["center"] == (325.0, 125.0)


def test_collect_missing_track_id_is_none():
    persons, _ = collect([FakeBox(0, 0.9, (0, 0, 10, 10))])
    assert persons[0]["id"] is None


def test_collect_applies_per_class_thresholds():
    boxes = [
        FakeBox(0, 0.1, (0, 0, 10, 10)),          # person below 0.2
        FakeBox(16, 0.01, (100, 100, 120, 120)),  # dog below 0.02
        FakeBox(16, 0.03, (200, 200, 220, 220)),  # dog above 0.02
    ]
    persons, dogs = collect(boxes)
    assert persons == []
    assert len(dogs) == 1


def test_collect_ignores_other_classes():
    assert collect([FakeBox(2, 0.99, (0, 0, 10, 10))]) == ([], [])


@pytest.mark.parametrize("dog_conf, dog_box, expected_dogs", [
    (0.05, (40, 40, 60, 60), 0),        # low confidence, inside person -> "hood" false positive
    (0.5, (40, 40, 60, 60), 1),         # confident, inside person -> kept
    (0.05, (300, 100, 350, 150), 1),    # low confidence, outside person -> kept
])
def test_collect_hood_filter(dog_conf, dog_box, expected_dogs):
    boxes = [FakeBox(0, 0.9, (0, 0, 100, 200)), FakeBox(16, dog_conf, dog_box)]
    _, dogs = collect(boxes)
    assert len(dogs) == expected_dogs


def test_collect_hood_filter_limit_comes_from_argument():
    boxes = [FakeBox(0, 0.9, (0, 0, 100, 200)), FakeBox(16, 0.05, (40, 40, 60, 60))]
    _, dogs = collect_detections(boxes, THRESHOLDS, 0.03)
    assert len(dogs) == 1


@pytest.mark.parametrize("boxes", [None, []])
def test_collect_empty_boxes(boxes):
    assert collect(boxes) == ([], [])


# --- find_pairs ---

def det(track_id, cx, cy, height=100):
    """Detection centred at (cx, cy) with a box of the given height."""
    h = height / 2
    return {"id": track_id, "center": (cx, cy), "coords": (cx - 10, cy - h, cx + 10, cy + h)}


def test_find_pairs_uses_track_ids_and_reports_distance_in_person_heights():
    # centres 50 px apart, person is 100 px tall -> 0.5 heights
    pairs = find_pairs([det(3, 0, 0)], [det(7, 30, 40, height=20)], 1.0)
    assert list(pairs) == [("p3", "d7")]
    assert pairs[("p3", "d7")] == pytest.approx(0.5)


def test_find_pairs_distance_equal_to_threshold_is_not_a_pair():
    assert find_pairs([det(1, 0, 0)], [det(2, 50, 0)], 0.5) == {}


def test_find_pairs_far_apart():
    assert find_pairs([det(1, 0, 0)], [det(2, 500, 0)], 0.5) == {}


@pytest.mark.parametrize("scale", [0.25, 1, 4])
def test_find_pairs_is_scale_invariant(scale):
    # The same layout scaled up or down must give the same verdict
    near = find_pairs([det(1, 0, 0, height=200 * scale)], [det(2, 80 * scale, 0)], 0.5)
    far = find_pairs([det(1, 0, 0, height=200 * scale)], [det(2, 120 * scale, 0)], 0.5)
    assert list(near) == [("p1", "d2")]
    assert far == {}


def test_find_pairs_uses_height_of_the_paired_person():
    persons = [det(1, 0, 0, height=400), det(2, 1000, 0, height=100)]
    dogs = [det(5, 150, 0), det(6, 1150, 0)]
    # 150 px is 0.375 heights for the tall person but 1.5 heights for the short one
    assert set(find_pairs(persons, dogs, 0.5)) == {("p1", "d5")}


def test_find_pairs_zero_height_box_does_not_crash():
    person = {"id": 1, "center": (0, 0), "coords": (0, 5, 10, 5)}
    assert find_pairs([person], [det(2, 0, 0)], 0.5) == {("p1", "d2"): 0.0}


def test_find_pairs_falls_back_to_index_without_ids():
    pairs = find_pairs([det(None, 0, 0)], [det(None, 10, 0)], 0.5)
    assert list(pairs) == [("p#0", "d#0")]


def test_find_pairs_multiple():
    persons = [det(1, 0, 0), det(2, 1000, 0)]
    dogs = [det(5, 10, 0), det(6, 1010, 0)]
    assert set(find_pairs(persons, dogs, 0.5)) == {("p1", "d5"), ("p2", "d6")}


# --- PairTracker ---

KEY = ("p1", "d2")


def tracker(lost_timeout=2.0, snapshot_interval=5.0):
    return PairTracker(lost_timeout, snapshot_interval)


def run(t, timeline):
    """timeline: list of (time, pair_present). Returns [(time, event)]."""
    out = []
    for now, present in timeline:
        pairs = {KEY: 50.0} if present else {}
        out += [(now, e[0]) for e in t.update(pairs, now)]
    return out


def test_tracker_starts_once_while_together():
    assert run(tracker(), [(0, True), (1, True), (2, True)]) == [(0, "start")]


def test_tracker_ongoing_every_snapshot_interval():
    result = run(tracker(), [(0, True), (4, True), (5, True), (9, True), (10, True)])
    assert result == [(0, "start"), (5, "ongoing"), (10, "ongoing")]


def test_tracker_ends_after_lost_timeout():
    result = run(tracker(), [(0, True), (1, True), (2, False), (2.9, False), (3.0, False)])
    assert result == [(0, "start"), (3.0, "end")]


def test_tracker_short_gap_does_not_split_pair():
    assert run(tracker(), [(0, True), (1, False), (1.5, False), (2, True)]) == [(0, "start")]


def test_tracker_end_reports_visible_duration():
    t = tracker()
    t.update({KEY: 10.0}, 0)
    t.update({KEY: 10.0}, 4)
    assert t.update({}, 6.5) == [("end", KEY, None, 4)]


def test_tracker_new_start_after_end():
    assert run(tracker(lost_timeout=1.0), [(0, True), (1, False), (2, True)]) == [
        (0, "start"), (1, "end"), (2, "start")]


def test_tracker_finish_ends_active_pairs():
    t = tracker()
    t.update({KEY: 10.0}, 0)
    t.update({KEY: 10.0}, 3)
    assert t.finish() == [("end", KEY, None, 3)]
    assert t.finish() == []


def test_tracker_from_config():
    t = PairTracker.from_config({"pairs": {"lost_timeout_sec": 7, "snapshot_interval_sec": 9}})
    assert (t.lost_timeout, t.snapshot_interval) == (7, 9)
