from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CourseData(BaseModel):
    code: str
    term_work: Optional[float] = None
    exam_work: Optional[float] = None
    result:    Optional[float] = None
    grade:     Optional[str]   = None
    points:    Optional[float] = None


class PredictionRequest(BaseModel):
    courses: Optional[List[CourseData]] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    probabilities: Dict[str, float]
    model_version: str            # serialised as "model_version"


# ── Error schema returned as HTTP 422 ─────────────────────────────────────────

class InsufficientDataErrorBody(BaseModel):
    """
    Body of the HTTP 422 response returned when the student's course data
    does not meet the minimum requirements for model inference.
    """
    error: str = "INSUFFICIENT_COURSE_DATA"
    message: str
    missing_courses: List[str]
    incomplete_courses: List[str]  # <--- FIXED: Changed from List[IncompleteCourseDetail]