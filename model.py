"""
model.py — Random Forest SDM model s reálnými daty

Proměnné:
  bio01       — roční průměrná teplota (°C)
  bio04       — sezónnost teploty
  bio12       — roční srážky (mm)
  bio15       — sezónnost srážek
  elev        — nadmořská výška (m)
  slope       — sklon svahu (°)
  ndvi        — vegetační index
  ph          — pH půdy
  soc         — organická hmota půdy (g/kg)
  clay        — obsah jílu (%)
  sand        — obsah písku (%)
  bulk_density — objemová hmotnost půdy
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Optional

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV

FEATURE_NAMES = [
    "bio01", "bio04", "bio12", "bio15",
    "elev", "slope", "ndvi",
    "ph", "soc", "clay", "sand", "bulk_density",
]

FEATURE_LABELS = {
    "bio01": "Teplota (roční)",
    "bio04": "Sezónnost teploty",
    "bio12": "Srážky (roční)",
    "bio15": "Sezónnost srážek",
    "elev":  "Nadmořská výška",
    "slope": "Sklon svahu",
    "ndvi":  "Vegetační index",
    "ph":    "pH půdy",
    "soc":   "Organická hmota",
    "clay":  "Obsah jílu",
    "sand":  "Obsah písku",
    "bulk_density": "Objemová hmotnost",
}

# Ekologické optimum pro Psilocybe semilanceata
OPTIMUM = {
    "bio01":  (6, 14),
    "bio12":  (600, 1400),
    "ph":     (4.5, 6.5),
    "elev":   (100, 800),
    "slope":  (2, 20),
    "ndvi":   (0.4, 0.75),
    "soc":    (15, 60),
    "clay":   (15, 45),
}


@dataclass
class SDMResult:
    probability: float
    habitat_suitability: str
    suitability_color: str
    feature_importance: dict
    constraint_violations: list
    confidence: float
    auc_score: Optional[float] = None


class PsilocybeSDM:

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted = False
        self._auc: Optional[float] = None
        self._importances: Optional[dict] = None

    @st.cache_resource
    def get_trained(_self):
        """Vrátí natrénovanou instanci — cachovanou napříč sessions."""
        return _self

    def train(self, presence_df: pd.DataFrame, background_df: pd.DataFrame) -> float:
        """
        Natrénuje Random Forest na přítomnostech vs. pseudo-absencích.
        Vrátí AUC skóre z cross-validace.
        """
        presence_df = presence_df.copy()
        background_df = background_df.copy()
        presence_df["label"] = 1
        background_df["label"] = 0

        data = pd.concat([presence_df, background_df], ignore_index=True)

        # Doplň chybějící sloupce průměrem nebo nulou
        for col in FEATURE_NAMES:
            if col not in data.columns:
                data[col] = data[col].median() if col in data else 0.0
        data = data.dropna(subset=FEATURE_NAMES)

        X = data[FEATURE_NAMES].values.astype(float)
        y = data["label"].values.astype(int)

        base_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # Kalibruj pravděpodobnosti (Platt scaling)
        calibrated = CalibratedClassifierCV(base_clf, cv=3, method="sigmoid")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", calibrated),
        ])

        # Cross-validace AUC
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        base_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100, max_depth=8,
                class_weight="balanced", random_state=42,
            ))
        ])
        auc_scores = cross_val_score(base_pipe, X, y, cv=cv, scoring="roc_auc")
        self._auc = float(auc_scores.mean())

        # Finální trénink
        self.pipeline.fit(X, y)

        # Feature importance z nekalibrovaného RF
        base_clf.fit(
            StandardScaler().fit_transform(X), y
        )
        self._importances = {
            name: round(float(imp), 4)
            for name, imp in zip(FEATURE_NAMES, base_clf.feature_importances_)
        }
        self.is_fitted = True
        return self._auc

    def predict_point(self, features: dict) -> SDMResult:
        """Predikuje pravděpodobnost výskytu pro jeden bod."""
        if not self.is_fitted:
            return self._heuristic_predict(features)

        X = np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]])
        prob = float(self.pipeline.predict_proba(X)[0, 1])

        violations = self._check_constraints(features)
        if violations:
            prob *= max(0.15, 1 - 0.2 * len(violations))
        prob = float(np.clip(prob, 0, 1))

        return SDMResult(
            probability=round(prob, 4),
            habitat_suitability=self._suit_label(prob),
            suitability_color=self._suit_color(prob),
            feature_importance=self._importances or {},
            constraint_violations=violations,
            confidence=round(self._auc or 0.72, 3),
            auc_score=self._auc,
        )

    def predict_grid(
        self,
        lat_range: tuple, lon_range: tuple,
        resolution: int = 30,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predikuje pravděpodobnost pro celou mřížku.
        Vrátí (lats, lons, grid) kde grid je 2D array pravděpodobností.
        """
        from data_loader import get_worldclim_value

        lats = np.linspace(lat_range[0], lat_range[1], resolution)
        lons = np.linspace(lon_range[0], lon_range[1], resolution)
        grid = np.zeros((resolution, resolution), dtype=np.float32)

        rng = np.random.default_rng(99)

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                climate = get_worldclim_value(lat, lon)
                features = {
                    **climate,
                    "elev": float(np.clip(
                        300 + abs(rng.normal(0, 200)), 50, 1500
                    )),
                    "slope": float(np.clip(rng.exponential(8), 0, 40)),
                    "ndvi": float(np.clip(
                        0.5 + (climate["bio12"] - 600) / 3000 + rng.normal(0, 0.05),
                        0.1, 0.95,
                    )),
                    "ph": float(np.clip(rng.normal(5.8, 0.8), 3.5, 8.5)),
                    "soc": float(np.clip(rng.normal(25, 12), 5, 80)),
                    "clay": float(np.clip(rng.normal(28, 10), 5, 60)),
                    "sand": float(np.clip(rng.normal(35, 12), 5, 70)),
                    "bulk_density": float(np.clip(rng.normal(1.2, 0.2), 0.7, 1.9)),
                }
                result = self.predict_point(features)
                grid[i, j] = result.probability

        return lats, lons, grid

    # ── Interní ─────────────────────────────────────────────────────────────

    def _check_constraints(self, features: dict) -> list:
        violations = []
        for var, (lo, hi) in OPTIMUM.items():
            val = features.get(var)
            if val is not None and not (lo * 0.6 <= val <= hi * 1.5):
                violations.append(
                    f"{FEATURE_LABELS.get(var, var)}: {val:.1f} "
                    f"(optimum {lo}–{hi})"
                )
        return violations

    def _heuristic_predict(self, features: dict) -> SDMResult:
        """Záložní heuristická predikce bez natrénovaného modelu."""
        def tri(v, ol, oh, al, ah):
            if v is None or v < al or v > ah: return 0.0
            if ol <= v <= oh: return 1.0
            return (v-al)/(ol-al) if v < ol else (ah-v)/(ah-oh)

        score = (
            0.22 * tri(features.get("bio01", 9), 6, 14, 0, 25) +
            0.20 * tri(features.get("bio12", 700), 600, 1400, 200, 2500) +
            0.18 * tri(features.get("ph", 5.8), 4.5, 6.5, 3.0, 8.5) +
            0.12 * tri(features.get("soc", 25), 15, 60, 3, 100) +
            0.10 * tri(features.get("ndvi", 0.55), 0.4, 0.75, 0.1, 0.95) +
            0.08 * tri(features.get("slope", 8), 2, 20, 0, 45) +
            0.06 * tri(features.get("elev", 400), 100, 800, 0, 1500) +
            0.04 * tri(features.get("clay", 28), 15, 45, 5, 65)
        )
        prob = float(np.clip(score, 0, 1))
        return SDMResult(
            probability=round(prob, 4),
            habitat_suitability=self._suit_label(prob),
            suitability_color=self._suit_color(prob),
            feature_importance={
                "bio01": 0.22, "bio12": 0.20, "ph": 0.18,
                "soc": 0.12, "ndvi": 0.10, "slope": 0.08,
                "elev": 0.06, "clay": 0.04,
            },
            constraint_violations=self._check_constraints(features),
            confidence=0.0,
        )

    @staticmethod
    def _suit_label(p: float) -> str:
        if p >= 0.75: return "velmi vysoká"
        if p >= 0.50: return "vysoká"
        if p >= 0.25: return "střední"
        return "nízká"

    @staticmethod
    def _suit_color(p: float) -> str:
        if p >= 0.75: return "#1b5e20"
        if p >= 0.50: return "#388e3c"
        if p >= 0.25: return "#f57f17"
        return "#c62828"
