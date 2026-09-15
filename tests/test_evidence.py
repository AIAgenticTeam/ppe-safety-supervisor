"""
Evidence store tests.

A finding a supervisor cannot independently check is not auditable, so these verify the
store actually produces a reviewable pair -- and that pruning removes photographs of
workers once the event justifying them is gone.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

cv2 = pytest.importorskip("cv2", reason="needs opencv; install into .venv")
import numpy as np  # noqa: E402

from evidence import EvidenceStore  # noqa: E402


@pytest.fixture
def frame():
    """A 720p frame with a bright patch where the 'person' is."""
    img = np.full((720, 1280, 3), 40, dtype=np.uint8)
    img[300:700, 500:700] = (180, 160, 140)
    return img


@pytest.fixture
def store(tmp_path):
    return EvidenceStore(tmp_path / "evidence")


BBOX = (500, 300, 700, 700)


def test_save_writes_both_frame_and_crop(store, frame):
    ev = store.save("evt_test_1", frame, BBOX, label="missing helmet")
    assert Path(ev.frame_path).exists(), "reviewer needs the scene for context"
    assert Path(ev.crop_path).exists(), "reviewer needs the person for detail"


def test_crop_is_tighter_than_the_frame(store, frame):
    ev = store.save("evt_test_2", frame, BBOX)
    crop = cv2.imread(ev.crop_path)
    full = cv2.imread(ev.frame_path)
    assert crop.shape[0] < full.shape[0] and crop.shape[1] < full.shape[1]


def test_crop_includes_padding_around_the_person(store, frame):
    """Context matters: a crop cut exactly to the box hides what is around them."""
    ev = store.save("evt_test_3", frame, BBOX)
    crop = cv2.imread(ev.crop_path)
    box_h, box_w = BBOX[3] - BBOX[1], BBOX[2] - BBOX[0]
    assert crop.shape[0] > box_h and crop.shape[1] > box_w


def test_large_frames_are_downscaled(store):
    """A 4K JPEG per violation fills a disk; 1600px is plenty to judge from."""
    big = np.full((2160, 3840, 3), 50, dtype=np.uint8)
    ev = store.save("evt_big", big, (1000, 500, 1400, 1500))
    assert cv2.imread(ev.frame_path).shape[1] == store.max_frame_width


def test_small_frames_are_not_upscaled(store, frame):
    ev = store.save("evt_small", frame, BBOX)
    assert cv2.imread(ev.frame_path).shape[1] == 1280


def test_annotation_marks_the_frame(store, frame):
    """The box must actually be drawn -- an unannotated crop asks for blind trust."""
    plain = store.save("evt_plain", frame, BBOX)
    a = cv2.imread(plain.frame_path)
    assert not np.array_equal(a, cv2.resize(frame, (a.shape[1], a.shape[0])))


def test_converts_to_the_event_contract(store, frame):
    ev = store.save("evt_contract", frame, BBOX)
    block = ev.as_event_evidence()
    assert block.crop_path == ev.crop_path
    assert block.frame_path == ev.frame_path
    assert block.crop_url is None and block.frame_url is None   # local, no cloud


def test_bbox_at_frame_edge_does_not_crash(store, frame):
    ev = store.save("evt_edge", frame, (0, 0, 120, 200))
    assert cv2.imread(ev.crop_path) is not None


def test_paths_for_finds_saved_evidence(store, frame):
    store.save("evt_lookup", frame, BBOX)
    assert store.paths_for("evt_lookup") is not None
    assert store.paths_for("evt_missing") is None


def test_usage_counts_pairs(store, frame):
    for i in range(3):
        store.save(f"evt_{i}", frame, BBOX)
    u = store.usage()
    assert u["events"] == 3 and u["files"] == 6


def test_prune_removes_evidence_for_deleted_events(store, frame):
    """Evidence outlives nothing. No event, no reason to hold a photo of a worker."""
    for i in range(3):
        store.save(f"evt_{i}", frame, BBOX)
    removed = store.prune(keep_event_ids={"evt_1"})
    assert removed == 4                       # two files each for evt_0 and evt_2
    assert store.paths_for("evt_1") is not None
    assert store.paths_for("evt_0") is None


def test_clear_empties_the_store(store, frame):
    store.save("evt_x", frame, BBOX)
    store.clear()
    assert store.usage()["files"] == 0
