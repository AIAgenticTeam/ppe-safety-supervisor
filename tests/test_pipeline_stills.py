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
