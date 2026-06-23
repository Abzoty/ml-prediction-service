from __future__ import annotations
from ast import Dict
import json, logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_COURSE_SUFFIXES = ("_grade", "_points", "_termWork", "_examWork", "_result", "_registered")

class ModelManager:
    def __init__(self, artifacts_dir: Path, models_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.models_dir    = models_dir
        self.label_encoder     = None
        self.selected_features: List[str] = []
        self.model             = None
        self.model_info: dict  = {}
        self.course_prefixes: List[str] = []
        self.is_loaded = False

    # ---- lifecycle --------------------------------------------------------
    def load(self) -> None:
        le_path    = self.artifacts_dir / "label_encoder.pkl"
        sf_path    = self.artifacts_dir / "selected_features.pkl"
        model_path = self.models_dir    / "Best_Calibrated_Model.pkl"
        info_path  = self.artifacts_dir / "best_model_info.json"

        for p in (le_path, sf_path, model_path, info_path):
            if not p.exists():
                raise FileNotFoundError(f"Required artifact missing: {p}")

        self.label_encoder     = joblib.load(le_path)
        self.selected_features = joblib.load(sf_path)
        self.model             = joblib.load(model_path)
        with info_path.open() as f:
            self.model_info = json.load(f)

        self.course_prefixes = self._extract_course_prefixes()
        self.is_loaded = True

        logger.info(
            "Model loaded | features=%d  prefixes=%d  classes=%s",
            len(self.selected_features),
            len(self.course_prefixes),
            list(self.label_encoder.classes_),
        )

    def _extract_course_prefixes(self) -> List[str]:
        prefixes = set()
        for feat in self.selected_features:
            for suf in _COURSE_SUFFIXES:
                if feat.endswith(suf):
                    prefixes.add(feat[: -len(suf)])
                    break
        return sorted(prefixes)

    # ---- inference --------------------------------------------------------
    def predict(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Return {department_name: probability}."""
        # Align columns exactly as the model was trained.
        features_df = features_df.reindex(columns=self.selected_features, fill_value=0)

        probas       = self.model.predict_proba(features_df)[0]
        int_classes  = self.model.classes_
        dept_names   = self.label_encoder.inverse_transform(np.asarray(int_classes, dtype=int))

        return {str(name): round(float(p), 4) for name, p in zip(dept_names, probas)}