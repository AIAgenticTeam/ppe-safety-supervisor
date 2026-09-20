"""
Bulk roster import from a CSV file, so nobody types a few hundred employees in one by one.

Pure functions, no web and no database: bytes in, a plan out. The endpoint asks for a plan
first (a preview -- nothing is written) and applies it only when a person confirms, because
a roster import is easy to get subtly wrong and hard to notice afterwards.

The files this has to cope with come from Excel, not from us:

  * the delimiter is `;` on Arabic and European Windows, and a tab in "Unicode text";
  * "CSV UTF-8" starts with a byte-order mark, and plain "CSV" on Arabic Windows is
    Windows-1256, not UTF-8;
  * the headers are whatever the HR export calls them ("Employee ID", "Full Name").

So the delimiter and encoding are detected, headers are matched by name rather than
position, and what was detected is reported back so it is never silent.

Only four things are kept: id, name, email, role. Any other column (a salary, a national id)
is ignored and *listed as ignored*, so somebody uploading a sensitive file can see it was not
taken. Identity stays a supervisor's call: this file says who exists, never who was seen.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

from app.db import Worker

MAX_BYTES = 1_000_000
MAX_ROWS = 5_000

# canonical field -> the header spellings accepted for it, compared after normalising
ALIASES: dict[str, set[str]] = {
    "worker_id": {"worker_id", "workerid", "id", "employee_id", "employeeid", "employee_no",
                  "employee_number", "emp_id", "emp_no", "staff_id", "badge", "badge_id",
                  "رقم_الموظف", "الرقم_الوظيفي", "الرقم"},
    "name": {"name", "full_name", "fullname", "employee_name", "worker_name", "employee",
             "الاسم", "اسم_الموظف", "اسم"},
    "email": {"email", "e_mail", "email_address", "mail",
              "البريد_الإلكتروني", "البريد_الالكتروني", "البريد"},
    "role": {"role", "job_title", "jobtitle", "title", "position", "trade", "job",
             "الوظيفة", "المسمى_الوظيفي", "المهنة"},
}
REQUIRED = ("worker_id", "name")


def _looks_like_email(value: str) -> bool:
    """Deliberately loose: it catches `ahmed.ali` and `ahmed@`, not every invalid address.
    Written without a regex so a hostile 1 MB cell cannot make it slow."""
    if len(value) > 254 or any(c.isspace() for c in value) or value.count("@") != 1:
        return False
    local, _, domain = value.partition("@")
    return bool(local) and "." in domain and not domain.startswith(".") and not domain.endswith(".")


class RosterFileError(ValueError):
    """The file as a whole cannot be used. The message says what to do about it."""


@dataclass
class Issue:
    row: int          # the row number as a spreadsheet shows it: the header is row 1
    problem: str


@dataclass
class Parsed:
    encoding: str
    delimiter: str
    columns: dict[str, str]                       # canonical field -> the header it came from
    ignored_columns: list[str]
    rows: list[tuple[int, Worker]] = field(default_factory=list)     # valid rows, with row no.
    issues: list[Issue] = field(default_factory=list)


# ---------------------------------------------------------------- reading the file

def decode(data: bytes) -> tuple[str, str]:
    """(text, encoding). UTF-8 first, since that is what a modern export is; Windows-1256
    only as the fallback, because that is what Excel writes for plain "CSV" on Arabic
    Windows -- and the encoding used is reported, so a wrong guess is visible."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16"
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("cp1256", errors="replace"), "windows-1256"


def _detect_delimiter(text: str) -> str:
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    counts = {d: first.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def _norm(header: str) -> str:
    return re.sub(r"[\s\-./]+", "_", header.strip().lower()).strip("_")


def _map_columns(headers: list[str]) -> tuple[dict[str, str], list[str]]:
    columns: dict[str, str] = {}
    ignored: list[str] = []
    for raw in headers:
        key = _norm(raw)
        hit = next((f for f, names in ALIASES.items() if key in names and f not in columns), None)
        if hit:
            columns[hit] = raw
        elif raw.strip():
            ignored.append(raw.strip())
    return columns, ignored


# ---------------------------------------------------------------- parsing

def _numbered_records(text: str, delimiter: str) -> list[tuple[int, list[str]]]:
    """Non-blank records with spreadsheet row numbers: a blank line still occupies a row."""
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    return [(n, rec) for n, rec in enumerate(reader, start=1) if any(c.strip() for c in rec)]


def _row_problem(wid: str, name: str, email: str, first_seen: dict[str, int]) -> str | None:
    if not wid:
        return "no id"
    if not name:
        return f"{wid}: no name"
    if len(wid) > 40:
        return f"{wid[:20]}…: id is longer than 40 characters"
    if len(name) > 200:
        return f"{wid}: name is longer than 200 characters"
    if email and not _looks_like_email(email):
        return f"{wid}: '{email[:60]}' is not an email address"
    if wid in first_seen:
        return f"{wid}: appears again (first on row {first_seen[wid]})"
    return None


def parse(data: bytes) -> Parsed:
    """Read and validate a whole file. Raises RosterFileError only when the file itself is
    unusable; a bad *row* is an Issue, so the person sees every problem in one pass instead
    of fixing them one upload at a time."""
    if not data.strip():
        raise RosterFileError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise RosterFileError(f"The file is larger than {MAX_BYTES // 1_000_000} MB. "
                              "A roster should not need that; split it, or remove extra columns.")

    text, encoding = decode(data)
    delimiter = _detect_delimiter(text)
    try:
        records = _numbered_records(text, delimiter)
    except csv.Error as exc:          # e.g. a single cell over 128 KB: not a roster
        raise RosterFileError(f"This does not read as a CSV roster ({exc}).") from exc
    if not records:
        raise RosterFileError("The file has no rows.")

    headers = records[0][1]
    columns, ignored = _map_columns(headers)
    missing = [f for f in REQUIRED if f not in columns]
    if missing:
        seen = ", ".join(h.strip() for h in headers if h.strip()) or "(none)"
        raise RosterFileError(
            f"The first row must be a header naming at least an id and a name. "
            f"Missing: {', '.join(missing)}. Columns found: {seen}. "
            "Accepted names include: worker_id / employee_id / id, and name / full_name.")

    body = records[1:]
    if len(body) > MAX_ROWS:
        raise RosterFileError(f"{len(body)} rows is more than the {MAX_ROWS} allowed in one import.")

    index = {f: headers.index(h) for f, h in columns.items()}
    out = Parsed(encoding=encoding, delimiter=delimiter, columns=columns, ignored_columns=ignored)
    first_seen: dict[str, int] = {}

    for row_no, rec in body:
        cells = {f: rec[i].strip() if i < len(rec) else "" for f, i in index.items()}
        wid, name = cells["worker_id"], cells["name"]
        email, role = cells.get("email", ""), cells.get("role", "")
        problem = _row_problem(wid, name, email, first_seen)
        if problem:
            out.issues.append(Issue(row_no, problem))
            continue
        first_seen[wid] = row_no
        out.rows.append((row_no, Worker(wid, name, email, role)))
    return out


# ---------------------------------------------------------------- planning

@dataclass
class Plan:
    parsed: Parsed
    new: list[Worker]
    updated: list[Worker]
    unchanged: list[Worker]

    @property
    def to_write(self) -> list[Worker]:
        """Only what actually changes: re-uploading the same file writes nothing."""
        return self.new + self.updated


def plan(parsed: Parsed, existing: list[Worker]) -> Plan:
    """Sort the valid rows into new / updated / unchanged against the current roster.

    A column the file does not have leaves that field alone -- a file of just ids and names
    must not blank every existing email. A column the file *does* have is the truth, blanks
    included: that is what "update the roster from this file" means."""
    have = {w.worker_id: w for w in existing}
    new, updated, unchanged = [], [], []
    for _, w in parsed.rows:
        old = have.get(w.worker_id)
        if old is None:
            new.append(w)
            continue
        merged = Worker(
            w.worker_id,
            w.name,
            w.email if "email" in parsed.columns else old.email,
            w.role if "role" in parsed.columns else old.role)
        (unchanged if merged == old else updated).append(merged)
    return Plan(parsed, new, updated, unchanged)


TEMPLATE = ("worker_id,name,email,role\r\n"
            "W-0412,Khalid Al-Otaibi,khalid@example.com,Welder\r\n")
