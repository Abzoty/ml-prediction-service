from __future__ import annotations
import logging
from typing import Dict, List

from app.core.model_manager import ModelManager
from app.services.feature_engineering import build_feature_vector, validate_course_data

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager

    def predict(self, courses: List[dict], model_version: str) -> Dict:
        if not self.mm.is_loaded:
            raise RuntimeError("Model artifacts are not loaded")

        # Strict pre-flight check: raises InsufficientCourseDataError (→ HTTP 422)
        # if any course required by the model is absent or has a null grade.
        # No fallbacks are applied; the student must complete their course data
        # before the model can run.
        validate_course_data(courses, self.mm.course_prefixes)

        features_df = build_feature_vector(
            courses,
            self.mm.selected_features,
            self.mm.course_prefixes,
        )
        probabilities = self.mm.predict(features_df)

        logger.info(
            "Prediction complete | top=%s",
            max(probabilities, key=probabilities.get) if probabilities else None,
        )

        return {
            "probabilities": probabilities,
            "model_version": model_version,
        }