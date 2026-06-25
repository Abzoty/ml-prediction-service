from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd

# Must match the training script's mapping.
GRADE_ORDER = {
    "Not_Registered": 0, "F": 1, "Abs": 2, "Con": 3,
    "D": 4, "D+": 5, "C": 6, "C+": 7,
    "P": 8, "B": 9, "B+": 10, "A": 11, "A+": 12,
}

GRADE_TO_POINTS = {
    "Not_Registered": 0.0, "F": 0.0, "Abs": 0.0, "Con": 0.0,
    "D": 2.0, "D+": 2.2, "C": 2.4, "C+": 2.7,
    "P": 0.0, "B": 3.0, "B+": 3.3, "A": 3.7, "A+": 4.0,
}


# ── Custom exception ──────────────────────────────────────────────────────────

class InsufficientCourseDataError(ValueError):
    """
    Raised when the student's course registrations are incomplete for
    model inference.
    """
    def __init__(
        self,
        missing_courses: List[str],
        incomplete_courses: List[str],
    ) -> None:
        self.missing_courses = missing_courses
        self.incomplete_courses = incomplete_courses
        super().__init__(
            f"Insufficient course data for prediction: "
            f"{len(missing_courses)} required course(s) not registered, "
            f"{len(incomplete_courses)} registered course(s) missing a final grade."
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _normalise_grade(raw) -> str:
    if raw is None:
        return "Not_Registered"
    s = str(raw).strip()
    return s if s and s not in ("nan", "None") else "Not_Registered"


# ── Public API ────────────────────────────────────────────────────────────────

def validate_course_data(
    courses: List[Dict],
    course_prefixes: List[str],
) -> None:
    by_code: Dict[str, Dict] = {c["code"]: c for c in courses}

    missing_courses: List[str] = []
    incomplete_courses: List[str] = [] # <-- Changed from List[Dict]

    for code in course_prefixes:
        record = by_code.get(code)

        if record is None:
            missing_courses.append(code)
            continue

        if record.get("grade") is None:
            # <-- SIMPLIFIED: Just append the code string
            incomplete_courses.append(code) 

    if missing_courses or incomplete_courses:
        raise InsufficientCourseDataError(missing_courses, incomplete_courses)


def build_feature_vector(
    courses: List[Dict],
    expected_features: List[str],
    course_prefixes: List[str],
) -> pd.DataFrame:
    """
    Build a single-row DataFrame aligned to ``expected_features``,
    ready to be fed to the model.

    **Precondition**: ``validate_course_data`` has already been called and
    passed without raising, so every prefix in ``course_prefixes`` exists
    in ``courses`` and carries a non-null grade.  No graceful fallbacks for
    absent or partially-graded courses are applied here; a ``KeyError``
    would indicate a bug in the call sequence.
    """
    by_code: Dict[str, Dict] = {c["code"]: c for c in courses}
    row: Dict[str, object] = {}

    for prefix in course_prefixes:
        # Guaranteed to exist after validation — no get() with a default.
        c = by_code[prefix]

        grade_col  = f"{prefix}_grade"
        points_col = f"{prefix}_points"
        term_col   = f"{prefix}_termWork"
        exam_col   = f"{prefix}_examWork"
        result_col = f"{prefix}_result"
        reg_col    = f"{prefix}_registered"

        row[reg_col] = 1

        grade_str        = _normalise_grade(c.get("grade"))
        row[grade_col]   = GRADE_ORDER.get(grade_str, 0)

        p                = c.get("points")
        row[points_col]  = (
            _safe_float(p, 0.0) if p is not None
            else GRADE_TO_POINTS.get(grade_str, 0.0)
        )

        row[term_col]    = _safe_float(c.get("term_work"), 0.0)
        row[exam_col]    = _safe_float(c.get("exam_work"), 0.0)

        r                = c.get("result")
        row[result_col]  = (
            _safe_float(r, 0.0) if r is not None
            else row[term_col] + row[exam_col]
        )

    # Non-course aggregate features
    if "gpa" in expected_features:
        pts = [
            row[f"{p}_points"]
            for p in course_prefixes
            if row.get(f"{p}_registered", 0) == 1
            and row.get(f"{p}_points", 0) > 0
        ]
        row["gpa"] = float(np.mean(pts)) if pts else 0.0

    return pd.DataFrame([{feat: row.get(feat, 0) for feat in expected_features}])