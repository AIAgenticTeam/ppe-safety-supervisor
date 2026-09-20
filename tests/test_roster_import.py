"""
Bulk roster import from CSV.

The files come from Excel, so most of these are about what Excel does to a CSV: a `;`
delimiter, a byte-order mark, Windows-1256 instead of UTF-8, headers named by whoever built
the HR export. The rest are about the two things an import must never do: write during a
preview, or half-apply a file that has a bad row in it.
"""

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi", reason="needs fastapi; install into .venv")

from fastapi.testclient import TestClient  # noqa: E402

from app import roster_import as ri  # noqa: E402
from app.api import create_app  # noqa: E402
from app.db import EventStore, Worker  # noqa: E402


def csv_bytes(text: str, encoding="utf-8") -> bytes:
    return text.encode(encoding)


BASIC = "worker_id,name,email,role\nW-1,Alpha,alpha@x.com,Welder\nW-2,Beta,,Rigger\n"


# ------------------------------------------------------------- reading the file

def test_a_plain_csv_is_read():
    p = ri.parse(csv_bytes(BASIC))
    assert [(w.worker_id, w.name, w.email, w.role) for _, w in p.rows] == [
        ("W-1", "Alpha", "alpha@x.com", "Welder"), ("W-2", "Beta", "", "Rigger")]
    assert p.issues == [] and p.delimiter == "," and p.encoding == "utf-8"


def test_excel_semicolon_delimiter_and_byte_order_mark():
    """Excel on Arabic or European Windows writes `;`, and "CSV UTF-8" starts with a BOM."""
    text = "﻿worker_id;name;email;role\nW-1;Alpha;;Welder\n"
    p = ri.parse(csv_bytes(text))
    assert p.delimiter == ";" and p.rows[0][1].worker_id == "W-1"     # BOM is not part of the id


def test_utf8_arabic_names_survive():
    p = ri.parse(csv_bytes("worker_id,name\nW-1,محمد الحربي\n"))
    assert p.rows[0][1].name == "محمد الحربي"


def test_windows_1256_is_the_fallback_and_is_reported():
    """Plain "CSV" on Arabic Windows is Windows-1256. Guessing is fine, silently is not."""
    p = ri.parse(csv_bytes("worker_id,name\nW-1,محمد الحربي\n", "cp1256"))
    assert p.encoding == "windows-1256" and p.rows[0][1].name == "محمد الحربي"


def test_excel_unicode_text_is_utf16_and_tab_delimited():
    p = ri.parse("worker_id\tname\nW-1\tAlpha\n".encode("utf-16"))
    assert p.encoding == "utf-16" and p.delimiter == "\t" and p.rows[0][1].name == "Alpha"


@pytest.mark.parametrize("header", [
    "Employee ID,Full Name,E-mail,Job Title",
    "ID,Name,Email Address,Position",
    "employee_no,employee_name,mail,trade",
    "رقم الموظف,الاسم,البريد الإلكتروني,الوظيفة",
])
def test_headers_are_matched_by_name_not_position(header):
    p = ri.parse(csv_bytes(header + "\nW-1,Alpha,a@x.com,Welder\n"))
    w = p.rows[0][1]
    assert (w.worker_id, w.name, w.email, w.role) == ("W-1", "Alpha", "a@x.com", "Welder")


def test_column_order_does_not_matter():
    p = ri.parse(csv_bytes("role,name,worker_id\nWelder,Alpha,W-1\n"))
    assert p.rows[0][1] == Worker("W-1", "Alpha", "", "Welder")


def test_other_columns_are_ignored_and_listed_as_ignored():
    """A salary or national id in the file must be visibly *not taken*."""
    p = ri.parse(csv_bytes("worker_id,name,Salary,National ID\nW-1,Alpha,9000,1234567890\n"))
    assert p.ignored_columns == ["Salary", "National ID"]
    assert "9000" not in repr(p.rows) and "1234567890" not in repr(p.rows)


def test_quoted_fields_with_commas_and_newlines():
    p = ri.parse(csv_bytes('worker_id,name,role\nW-1,"Al-Otaibi, Khalid","Lead\nwelder"\n'))
    w = p.rows[0][1]
    assert w.name == "Al-Otaibi, Khalid" and w.role == "Lead\nwelder"


def test_short_rows_are_read_with_blanks():
    p = ri.parse(csv_bytes("worker_id,name,email,role\nW-1,Alpha\n"))
    assert p.rows[0][1] == Worker("W-1", "Alpha", "", "")


# ------------------------------------------------------- the file cannot be used

@pytest.mark.parametrize("data", [b"", b"   \n \n", "﻿".encode()])
def test_an_empty_file_is_refused(data):
    with pytest.raises(ri.RosterFileError):
        ri.parse(data)


def test_a_missing_required_column_says_what_was_found():
    with pytest.raises(ri.RosterFileError) as exc:
        ri.parse(csv_bytes("Employee,Department\nAlpha,Site\n"))
    msg = str(exc.value)
    assert "worker_id" in msg and "Employee" in msg and "Department" in msg


def test_a_file_with_no_header_is_refused_rather_than_misread():
    """Row one being data (W-1,Alpha) must not be silently taken as the header."""
    with pytest.raises(ri.RosterFileError):
        ri.parse(csv_bytes("W-1,Alpha,a@x.com\nW-2,Beta,b@x.com\n"))


def test_too_many_rows_and_too_large_a_file_are_refused():
    many = "worker_id,name\n" + "\n".join(f"W-{i},P{i}" for i in range(ri.MAX_ROWS + 1))
    with pytest.raises(ri.RosterFileError, match="more than"):
        ri.parse(csv_bytes(many))
    with pytest.raises(ri.RosterFileError, match="larger"):
        ri.parse(b"x" * (ri.MAX_BYTES + 1))


# ------------------------------------------------------------- bad rows

def test_bad_rows_are_issues_with_spreadsheet_row_numbers_and_the_rest_still_read():
    text = ("worker_id,name,email\n"
            "W-1,Alpha,a@x.com\n"          # row 2
            "\n"                           # row 3 is blank, and still a row in Excel
            ",Nameless,\n"                 # row 4: no id
            "W-4,,\n"                      # row 5: no name
            "W-5,Bad Mail,not-an-email\n"  # row 6
            "W-1,Again,a@x.com\n"          # row 7: duplicate of row 2
            "W-8,Fine,\n")                 # row 8
    p = ri.parse(csv_bytes(text))
    assert {i.row for i in p.issues} == {4, 5, 6, 7}
    by_row = {i.row: i.problem for i in p.issues}
    assert "no id" in by_row[4] and "no name" in by_row[5]
    assert "not an email" in by_row[6] and "row 2" in by_row[7]
    assert [w.worker_id for _, w in p.rows] == ["W-1", "W-8"]


@pytest.mark.parametrize("email,ok", [
    ("a@x.com", True), ("first.last@site.co.uk", True), ("a@x", False), ("a@.com", False),
    ("a@x.", False), ("@x.com", False), ("a b@x.com", False), ("a@@x.com", False),
    ("ahmed.ali", False), ("ahmed@", False)])
def test_the_email_check_is_loose_but_catches_the_usual_mistakes(email, ok):
    assert ri._looks_like_email(email) is ok


def test_a_hostile_cell_cannot_make_validation_slow():
    started = time.perf_counter()
    p = ri.parse(csv_bytes("worker_id,name,email\nW-1,A,a@" + "." * 100_000 + "\n"))
    assert time.perf_counter() - started < 2.0
    assert p.issues and not p.rows


def test_an_absurdly_large_cell_is_a_clear_refusal_not_a_crash(api):
    """Python's csv module raises on a cell over 128 KB. That must reach the person as a
    message, not as a 500."""
    client, _ = api
    data = csv_bytes("worker_id,name\nW-1," + "x" * 400_000 + "\n")
    with pytest.raises(ri.RosterFileError, match="CSV"):
        ri.parse(data)
    assert post(client, data).status_code == 422


# ------------------------------------------------------------- planning

def test_the_plan_sorts_rows_into_new_updated_and_unchanged():
    existing = [Worker("W-1", "Alpha", "a@x.com", "Welder"), Worker("W-2", "Beta", "", "Rigger")]
    text = "worker_id,name,email,role\nW-1,Alpha,a@x.com,Welder\nW-2,Beta Renamed,,Rigger\nW-3,Gamma,,\n"
    plan = ri.plan(ri.parse(csv_bytes(text)), existing)
    assert [w.worker_id for w in plan.unchanged] == ["W-1"]
    assert [w.worker_id for w in plan.updated] == ["W-2"]
    assert [w.worker_id for w in plan.new] == ["W-3"]
    assert [w.worker_id for w in plan.to_write] == ["W-3", "W-2"]      # unchanged is not rewritten


def test_a_column_the_file_lacks_leaves_that_field_alone():
    """A file of just ids and names must not blank every existing email and role."""
    existing = [Worker("W-1", "Alpha", "a@x.com", "Welder")]
    plan = ri.plan(ri.parse(csv_bytes("worker_id,name\nW-1,Alpha Renamed\n")), existing)
    assert plan.updated == [Worker("W-1", "Alpha Renamed", "a@x.com", "Welder")]


def test_a_column_the_file_has_is_the_truth_blanks_included():
    existing = [Worker("W-1", "Alpha", "a@x.com", "Welder")]
    plan = ri.plan(ri.parse(csv_bytes("worker_id,name,email\nW-1,Alpha,\n")), existing)
    assert plan.updated == [Worker("W-1", "Alpha", "", "Welder")]      # email cleared, role kept


# ------------------------------------------------------------- the endpoints

@pytest.fixture
def api(tmp_path):
    app = create_app(tmp_path / "r.db")
    return TestClient(app), app


def post(client, data, **params):
    return client.post("/roster/import", content=data, params=params,
                       headers={"content-type": "text/csv"})


def roster_ids(client):
    return sorted(w["worker_id"] for w in client.get("/roster").json()["workers"])


def test_a_preview_reports_what_would_change_and_writes_nothing(api):
    client, _ = api
    r = post(client, csv_bytes(BASIC), dry_run="true")
    body = r.json()
    assert r.status_code == 200 and body["dry_run"] is True and body["applied"] == 0
    assert (body["new"], body["updated"], body["unchanged"], body["issue_count"]) == (2, 0, 0, 0)
    assert roster_ids(client) == []


def test_the_default_is_a_preview_so_a_forgotten_flag_cannot_write(api):
    client, _ = api
    assert post(client, csv_bytes(BASIC)).json()["dry_run"] is True
    assert roster_ids(client) == []


def test_confirming_writes_the_roster(api):
    client, _ = api
    r = post(client, csv_bytes(BASIC), dry_run="false")
    assert r.status_code == 200 and r.json()["applied"] == 2
    assert roster_ids(client) == ["W-1", "W-2"]


def test_the_same_file_twice_changes_nothing_the_second_time(api):
    client, _ = api
    post(client, csv_bytes(BASIC), dry_run="false")
    again = post(client, csv_bytes(BASIC), dry_run="false").json()
    assert (again["new"], again["updated"], again["unchanged"], again["applied"]) == (0, 0, 2, 0)


def test_updating_someone_who_already_has_findings_keeps_them_attached(api, tmp_path):
    """The reason a re-upload is safe: a rename must not detach a person's history."""
    client, app = api
    post(client, csv_bytes(BASIC), dry_run="false")
    import json
    event = json.loads(next((ROOT / "fixtures" / "events").glob("*.json")).read_text(encoding="utf-8"))
    app.state.db.record_event(event)
    assert app.state.db.attach_identity(event["event_id"], "W-1", "supervisor:t")

    changed = "worker_id,name,email,role\nW-1,Alpha Renamed,alpha@x.com,Foreman\n"
    assert post(client, csv_bytes(changed), dry_run="false").json()["updated"] == 1

    row = next(r for r in client.get("/events").json()["events"] if r["event_id"] == event["event_id"])
    assert row["worker_id"] == "W-1"
    assert any(w["name"] == "Alpha Renamed" for w in client.get("/roster").json()["workers"])


def test_one_bad_row_blocks_the_import_and_nothing_is_half_written(api):
    client, _ = api
    text = "worker_id,name\nW-1,Alpha\n,Nameless\nW-3,Gamma\n"
    preview = post(client, csv_bytes(text), dry_run="true").json()
    assert preview["issue_count"] == 1 and preview["issues"][0]["row"] == 3
    r = post(client, csv_bytes(text), dry_run="false")
    assert r.status_code == 422 and "nothing was imported" in r.json()["detail"]
    assert roster_ids(client) == []


def test_skip_invalid_imports_only_the_valid_rows(api):
    client, _ = api
    text = "worker_id,name\nW-1,Alpha\n,Nameless\nW-3,Gamma\n"
    r = post(client, csv_bytes(text), dry_run="false", skip_invalid="true")
    assert r.status_code == 200 and r.json()["applied"] == 2
    assert roster_ids(client) == ["W-1", "W-3"]


def test_a_file_with_only_bad_rows_imports_nothing_even_when_skipping(api):
    client, _ = api
    r = post(client, csv_bytes("worker_id,name\n,A\n,B\n"), dry_run="false", skip_invalid="true")
    assert r.status_code == 422


def test_an_unusable_file_is_a_422_that_says_why(api):
    client, _ = api
    r = post(client, csv_bytes("Employee,Department\nAlpha,Site\n"))
    assert r.status_code == 422 and "Columns found" in r.json()["detail"]
    assert post(client, b"").status_code == 422


def test_an_oversized_upload_is_a_413(api):
    client, _ = api
    assert post(client, b"x" * (ri.MAX_BYTES + 10)).status_code == 413


def test_the_preview_says_what_it_detected_and_what_it_ignored(api):
    client, _ = api
    text = "﻿Employee ID;Full Name;Salary\nW-1;Alpha;9000\n"
    body = post(client, csv_bytes(text)).json()
    assert body["delimiter"] == "semicolon" and body["encoding"] == "utf-8"
    assert body["ignored_columns"] == ["Salary"]
    assert body["columns"] == {"worker_id": "Employee ID", "name": "Full Name"}


def test_the_preview_shows_a_sample_with_each_rows_status(api):
    client, _ = api
    post(client, csv_bytes("worker_id,name\nW-1,Alpha\n"), dry_run="false")
    text = "worker_id,name\nW-1,Alpha\nW-2,Beta\nW-3,Gamma Renamed\n"
    sample = post(client, csv_bytes(text)).json()["preview"]
    assert {r["worker_id"]: r["status"] for r in sample} == {
        "W-1": "unchanged", "W-2": "new", "W-3": "new"}


def test_the_template_is_served_as_a_download_and_is_itself_importable(api):
    """Round trip: the file we tell people to start from must be one we accept."""
    client, _ = api
    r = client.get("/roster/template.csv")
    assert r.status_code == 200 and "text/csv" in r.headers["content-type"]
    assert "attachment" in r.headers["content-disposition"]
    body = post(client, r.content).json()
    assert body["new"] == 1 and body["issue_count"] == 0


def test_the_store_write_is_all_or_nothing(tmp_path):
    db = EventStore(tmp_path / "s.db")
    with pytest.raises(Exception):
        db.add_workers([Worker("W-1", "Alpha"), Worker("W-2", None)])     # NOT NULL name
    assert db.roster() == []
