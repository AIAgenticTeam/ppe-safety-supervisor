"""
The stills path of pipeline.py, with the detector stubbed out.

`process_image` had no test, which is how it kept a second evidence format for as long
as it did. The model is faked so this runs without weights or a GPU -- what is being
checked is the wiring either side of the detector, not the detector.

Two properties matter here, and both are about what the console will receive:

  * evidence is written through EvidenceStore, the same way the video path writes it,
    so every event carries an annotated frame AND a crop;
  * the event's id is the id the evidence files were named with, rather than a second
    one derived separately and trusted to match.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

cv2 = pytest.importorskip("cv2", reason="needs opencv; install into .venv")
import numpy as np  # noqa: E402

import pipeline  # noqa: E402
from zones import Zone  # noqa: E402


class FakeAssessment:
    """Stands in for ppe_compliance.PersonAssessment."""

    def __init__(self, missing=("helmet",), person_id=1):
        self.bbox = (500, 300, 700, 700)
        self.person_id = person_id
        self.missing = list(missing)
        self.present = {}
        self.indeterminate = []
        self.status = "violation" if missing else "compliant"
        self.person_confidence = 0.91
        self.reasons = []
        self.zone = Zone(name="grinding_station", label="Grinding station",
                         polygon=[(0, 0), (1280, 0), (1280, 720), (0, 720)],
                         required_ppe=["helmet"])


class FakeResult:
    orig_shape = (720, 1280)
    boxes = []


class FakeModel:
    names = {0: "person", 1: "helmet"}

    def predict(self, _source, **_kwargs):
        return [FakeResult()]


@pytest.fixture
def still(tmp_path):
    img = np.full((720, 1280, 3), 40, dtype=np.uint8)
    img[300:700, 500:700] = (180, 160, 140)
    path = tmp_path / "frame.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def run(monkeypatch, tmp_path, still):
    """Run process_image with the detector replaced by a known assessment."""
    def _run(missing=("helmet",)):
        monkeypatch.setattr(pipeline, "assess_detections",
                            lambda *a, **k: ([FakeAssessment(missing)], []))
        from ppe_compliance import Policy
        from zones import ZoneMap
        out = tmp_path / "out"
        events = pipeline.process_image(
            FakeModel(), still, ZoneMap.load(ROOT / "zones.json"), "cam_3",
            Policy(), out)
        return events, out
    return _run


def test_a_still_produces_an_annotated_frame_and_a_crop(run):
    """The bug: this path used to write a bare crop with no frame and no annotation,
    so a reviewer saw a photograph of a worker with nothing marking the allegation."""
    events, _ = run()
    evidence = events[0].evidence
    assert evidence.frame_path, "no frame -- the reviewer cannot see the context"
    assert evidence.crop_path
    assert Path(evidence.frame_path).exists()
    assert Path(evidence.crop_path).exists()


def test_the_event_points_at_the_files_that_were_actually_written(run):
    """The id used to be minted twice and reconciled by an overwrite afterwards. If the
    two ever disagreed the event referenced filenames nobody wrote."""
    events, _ = run()
    event = events[0]
    assert Path(event.evidence.frame_path).name == f"frame_{event.event_id}.jpg"
    assert Path(event.evidence.crop_path).name == f"crop_{event.event_id}.jpg"


def test_stills_evidence_lands_beside_the_video_path_evidence(run):
    """One directory, one naming scheme, so the console needs no special case."""
    events, out = run()
    assert Path(events[0].evidence.frame_path).parent == out / "evidence"


def test_a_still_is_never_actionable(run):
    """Unchanged by the evidence fix, and worth pinning: at helmet recall 0.796 a
    single frame cannot accuse anyone."""
    events, _ = run()
    assert not events[0].confirmation.confirmed
    assert not events[0].is_actionable
    assert events[0].confirmation.rule == "single_frame"


def test_a_compliant_still_still_gets_evidence_without_an_allegation_label(run):
    events, _ = run(missing=())
    assert events[0].status == "compliant"
    assert Path(events[0].evidence.crop_path).exists()


# ----------------------------------------- ids must survive a folder of stills

def test_frames_sharing_a_timestamp_do_not_collapse_into_one_event(
        monkeypatch, tmp_path):
    """Found by running the real model over a folder of extracted frames.

    Stills take their timestamp from the file's mtime, and person ids restart at 0 on
    every frame. Four frames written in the same second therefore produced four
    identical event ids -- and since events and evidence are both saved under the id,
    three of them were overwritten. The pipeline cheerfully reported "3 events" while
    one file sat on disk.

    Nothing raised. That is the whole problem with it.
    """
    import os

    img = np.full((720, 1280, 3), 40, dtype=np.uint8)
    img[300:700, 500:700] = (180, 160, 140)

    frames = []
    for i in range(4):
        path = tmp_path / f"frame_{i:02d}.jpg"
        cv2.imwrite(str(path), img)
        os.utime(path, (1_760_000_000, 1_760_000_000))   # identical mtime
        frames.append(path)

    monkeypatch.setattr(pipeline, "assess_detections",
                        lambda *a, **k: ([FakeAssessment(("helmet",))], []))
    from ppe_compliance import Policy
    from zones import ZoneMap

    out = tmp_path / "out"
    ids, evidence_paths = [], []
    for path in frames:
        for event in pipeline.process_image(
                FakeModel(), path, ZoneMap.load(ROOT / "zones.json"), "cam_3",
                Policy(), out):
            ids.append(event.event_id)
            evidence_paths.append(event.evidence.crop_path)

    assert len(set(ids)) == len(ids), f"ids collided: {ids}"
    assert len(set(evidence_paths)) == len(evidence_paths), "evidence overwrote itself"
    for path in evidence_paths:
        assert Path(path).exists()


def test_the_source_stem_appears_in_a_stills_event_id(monkeypatch, tmp_path):
    """Traceability: a reviewer holding an event should be able to find the frame."""
    img = np.full((720, 1280, 3), 40, dtype=np.uint8)
    path = tmp_path / "cam3_0930.jpg"
    cv2.imwrite(str(path), img)

    monkeypatch.setattr(pipeline, "assess_detections",
                        lambda *a, **k: ([FakeAssessment(("helmet",))], []))
    from ppe_compliance import Policy
    from zones import ZoneMap

    events = pipeline.process_image(FakeModel(), path,
                                    ZoneMap.load(ROOT / "zones.json"), "cam_3",
                                    Policy(), tmp_path / "out")
    assert "cam3_0930" in events[0].event_id


def test_make_event_id_without_a_source_is_unchanged():
    """The bare contract, which make_fixtures.py still relies on."""
    from datetime import datetime

    from events import RIYADH, make_event_id
    when = datetime(2026, 9, 16, 10, 15, tzinfo=RIYADH)
    assert make_event_id("cam_3", 7, when) == "evt_20260916T101500_cam_3_t7"


def test_two_confirmations_from_one_track_in_one_second_are_distinct():
    """This test previously asserted the opposite, and was wrong.

    It read: "ByteTrack ids are unique within a clip, so video needs no source
    component." Track ids are indeed unique per track — but one track confirms more
    than once, as different items cross the 8-of-10 threshold at different moments,
    and two confirmations a fraction of a second apart round to the same timestamp.

    Real footage of a concrete pour produced exactly that: boots+vest confirmed at
    frame 111 and helmet at frame 123, both track 8, both reported at "1.0s". The
    pipeline printed "2 events" and left one file on disk, because the second id
    collided with the first and overwrote it — in the event store and in the evidence
    directory both. Nothing raised.
    """
    from datetime import datetime

    from events import RIYADH, make_event_id
    when = datetime(2026, 9, 19, 0, 11, 21, tzinfo=RIYADH)

    first = make_event_id("pour_site", 8, when, source="f111")
    second = make_event_id("pour_site", 8, when, source="f123")

    assert first != second, "one track's two confirmations must not share an id"
    assert "f111" in first and "f123" in second, "the confirming frame is traceable"
