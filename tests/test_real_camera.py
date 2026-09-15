"""
Regression tests for the one camera derived from real footage (d_view02).

The worker in scene_02 wears a helmet and a hi-vis vest, so under the real policy he is
compliant and nothing fires. That is correct, and it is also why the confirmed-violation
path needs zones.test.json: requiring an item he plainly is not wearing forces the
assessment to produce a violation, so the path stays covered.

Both configs are exercised here with stub detections -- no model, no video, no GPU.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from ppe_compliance import Policy, assess_detections  # noqa: E402
from zones import ZoneMap  # noqa: E402

NAMES = {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles", 5: "Person"}
W, H = 586, 480                       # the CCTV segment's real frame size

# Roughly where the walking worker sits in scene_02: feet at (380, 300) -> walkway.
PERSON = (340, 15, 420, 300)


class StubBox:
    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf, self.xyxy = cls, conf, [xyxy]


def worker_with_helmet_and_vest():
    """What the detector actually found on him: helmet 0.72, vest 0.58. No goggles."""
    x1, y1, x2, y2 = PERSON
    cx, h = (x1 + x2) / 2, y2 - y1
    return [
        StubBox(5, 0.73, list(PERSON)),
        StubBox(0, 0.72, [cx - 22, y1 + 0.02 * h, cx + 22, y1 + 0.13 * h]),   # helmet
        StubBox(2, 0.58, [x1 + 5, y1 + 0.25 * h, x2 - 5, y1 + 0.60 * h]),     # vest
    ]


@pytest.fixture(scope="module")
def real():
    return ZoneMap.load(ROOT / "zones.json")


@pytest.fixture(scope="module")
def test_cfg():
    return ZoneMap.load(ROOT / "zones.test.json")


# ------------------------------------------------------- the production config

def test_production_config_has_no_test_cameras(real):
    """A deliberately wrong policy must never sit in the config the pipeline loads."""
    for cid, cam in real.cameras.items():
        assert "TEST" not in cam.label.upper(), f"{cid} is a test fixture in zones.json"


def test_real_camera_zones(real):
    walkway = real.zone_or_default("d_view02", PERSON)
    assert walkway.name == "walkway"
    assert set(walkway.required_ppe) == {"helmet", "vest"}

    beyond_the_fence = real.zone_or_default("d_view02", (120, 100, 200, 330))
    assert beyond_the_fence.name == "work_area"
    assert "gloves" in beyond_the_fence.required_ppe


def test_compliant_worker_is_left_alone(real):
    """The result that matters most: he is wearing his PPE, so nothing is alleged."""
    people, _ = assess_detections(worker_with_helmet_and_vest(), NAMES, W, H,
                                  Policy(), real, "d_view02")
    assert len(people) == 1
    p = people[0]
    assert p.status == "compliant"
    assert p.missing == [] and p.indeterminate == []


def test_same_worker_would_need_gloves_beyond_the_fence(real):
    """Zone decides the outcome, not the detections. This is the slide-5 thesis."""
    in_walkway, _ = assess_detections(worker_with_helmet_and_vest(), NAMES, W, H,
                                      Policy(), real, "d_view02")
    assert in_walkway[0].status == "compliant"
    assert "gloves" not in in_walkway[0].required_ppe

    work_area = real.zone_or_default("d_view02", (120, 100, 200, 330))
    assert "gloves" in work_area.required_ppe      # same person, four metres left


def test_worker_clears_the_size_gate(real):
    """d_view02 was chosen over the other dwells because people are big enough here."""
    height = PERSON[3] - PERSON[1]
    assert height >= Policy().min_person_height_px, (
        f"{height}px is under the assessability threshold; this camera would only "
        f"ever produce indeterminate findings")


# ------------------------------------------------------------ the test config

def test_test_config_forces_a_violation(test_cfg):
    """Keeps the confirm/emit path covered on footage where nobody is in breach."""
    people, _ = assess_detections(worker_with_helmet_and_vest(), NAMES, W, H,
                                  Policy(), test_cfg, "d_view02_test")
    p = people[0]
    assert p.status == "violation"
    assert p.missing == ["goggles"]
    assert "helmet" in p.present and "vest" in p.present   # he is credited with both


def test_the_two_configs_share_geometry(real, test_cfg):
    """If the polygons drift apart the test stops proving anything about the real one."""
    a = {z.name: z.polygon for z in real.cameras["d_view02"].zones}
    b = {z.name: z.polygon for z in test_cfg.cameras["d_view02_test"].zones}
    assert a == b


def test_test_config_differs_only_by_goggles(real, test_cfg):
    a = {z.name: set(z.required_ppe) for z in real.cameras["d_view02"].zones}
    b = {z.name: set(z.required_ppe) for z in test_cfg.cameras["d_view02_test"].zones}
    for name in a:
        assert b[name] - a[name] == {"goggles"}, "the test config drifted beyond goggles"
