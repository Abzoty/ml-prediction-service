from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.model_manager import ModelManager
from app.schemas import (
    # <--- FIXED: Removed IncompleteCourseDetail from imports
    InsufficientDataErrorBody,
    PredictionRequest,
    PredictionResponse,
)
from app.services.feature_engineering import InsufficientCourseDataError
from app.services.prediction_service import PredictionService

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

model_manager = ModelManager(settings.artifacts_dir, settings.models_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_manager.load()
    except Exception as e:
        logger.exception("Failed to load model artifacts: %s", e)
        raise
    yield


app = FastAPI(
    title="Department Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
)

service = PredictionService(model_manager)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    courses = [c.model_dump() for c in (request.courses or [])]
    try:
        result = service.predict(courses, settings.model_version)
        return PredictionResponse(**result)

    except InsufficientCourseDataError as e:
        # Business validation error — the student's course data is incomplete.
        logger.warning(
            "Prediction rejected — missing=%s  incomplete=%s",
            e.missing_courses,
            e.incomplete_courses, # <--- FIXED: No longer needs list comprehension
        )
        body = InsufficientDataErrorBody(
            message=str(e),
            missing_courses=e.missing_courses,
            incomplete_courses=e.incomplete_courses, # <--- FIXED: Pass list of strings directly
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_manager.is_loaded,
        "model_version": settings.model_version,
    }


@app.get("/metadata")
def metadata():
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_version":   settings.model_version,
        "features_count":  len(model_manager.selected_features),
        "course_prefixes": len(model_manager.course_prefixes),
        "classes":         list(model_manager.label_encoder.classes_),
        "best_model_name": model_manager.model_info.get("best_model_name"),
    }