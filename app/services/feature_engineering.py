from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd

# Must match the training script's mapping (cleaned of the trailing-space
# artefacts that appear in the pasted source).
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


def build_feature_vector(
    courses: List[Dict],
    expected_features: List[str],
    course_prefixes: List[str],
) -> pd.DataFrame:
    """
    Build a single-row DataFrame whose columns are exactly `expected_features`,
    ready to be fed to the model.
    """
    by_code: Dict[str, Dict] = {c["code"]: c for c in courses}
    row: Dict[str, object] = {}

    # --- per-course columns -------------------------------------------------
    for prefix in course_prefixes:
        c = by_code.get(prefix)

        grade_col  = f"{prefix}_grade"
        points_col = f"{prefix}_points"
        term_col   = f"{prefix}_termWork"
        exam_col   = f"{prefix}_examWork"
        result_col = f"{prefix}_result"
        reg_col    = f"{prefix}_registered"

        if c is None:
            # Student never took this course.
            row[grade_col]  = GRADE_ORDER["Not_Registered"]
            row[points_col] = 0.0
            row[term_col]   = 0.0
            row[exam_col]   = 0.0
            row[result_col] = 0.0
            row[reg_col]    = 0
            continue

        # "registered" = any original field was non-null (matches training).
        raw_vals = [c.get("grade"), c.get("points"),
                    c.get("term_work"), c.get("exam_work"), c.get("result")]
        row[reg_col] = 1 if any(v is not None for v in raw_vals) else 0

        grade_str = _normalise_grade(c.get("grade"))
        row[grade_col] = GRADE_ORDER.get(grade_str, 0)

        # points: use provided value, else derive from grade.
        p = c.get("points")
        row[points_col] = _safe_float(p, default=np.nan) \
            if p is not None else np.nan
        if pd.isna(row[points_col]):
            row[points_col] = GRADE_TO_POINTS.get(grade_str, 0.0)

        row[term_col] = _safe_float(c.get("term_work"), 0.0)
        row[exam_col] = _safe_float(c.get("exam_work"), 0.0)

        r = c.get("result")
        row[result_col] = _safe_float(r, default=np.nan) \
            if r is not None else np.nan
        if pd.isna(row[result_col]):
            row[result_col] = row[term_col] + row[exam_col]
        if pd.isna(row[result_col]):
            row[result_col] = 0.0

    # --- non-course features (e.g. gpa) -------------------------------------
    if "gpa" in expected_features:
        pts = [row[f"{p}_points"]
                for p in course_prefixes
                if row.get(f"{p}_registered", 0) == 1 and row.get(f"{p}_points", 0) > 0]
        row["gpa"] = float(np.mean(pts)) if pts else 0.0

    # Pin the exact column order the model expects; missing → 0.
    df = pd.DataFrame([{feat: row.get(feat, 0) for feat in expected_features}])
    return df