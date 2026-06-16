"""
SDM model — samostatný modul pro Streamlit verzi.
Nevyžaduje FastAPI ani databázi.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


FEATURE_NAMES = ["bio01", "bio04", "bio12", "bio15", "ph", "elev", "ndvi"]

SPECIES_CONSTRAINTS = {
    "bio01": (2.0,  25.0),
    "bio12": (300,  2200),
    "ph":    (4.0,  7.8),
    "elev":  (0,    1500),
    "ndvi":  (0.15, 0.90),
}


@dataclass
class SDMFeatures:
    bio01: float
    bio04: float
    bio12: float
    bio15: float
    ph:    float
    elev:  float
    ndvi:  float

    def to_dict(self):
        return {k: getattr(self, k) for k in FEATURE_NAMES}


@dataclass
class SDMResult:
    probability: float
    habitat_suitability: str
    feature_importance: dict
    constraint_violations: list
    confidence: float


class PsilocybeSDM:
    """
    SDM model v demo módu — bez externího tréninku.
    Používá ekologicky kalibrované pravidlové skóre
    (trojúhelníkové funkce pro každou proměnnou).
    """

    IMPORTANCE = {
        "bio01": 0.28, "bio04": 0.08, "bio12": 0.27,
        "bio15": 0.07, "ph":    0.18, "elev":  0.07, "ndvi": 0.05,
    }

    def predict_point(self, features: SDMFeatures) -> SDMResult:
        fd = features.to_dict()
        violations = [
            f"{var}={fd[var]:.1f} (mimo {lo}–{hi})"
            for var, (lo, hi) in SPECIES_CONSTRAINTS.items()
            if not (lo <= fd[var] <= hi)
        ]

        ts = self._tri(features.bio01,  8,  18,  2,  25)
        rs = self._tri(features.bio12, 600, 1400, 300, 2200)
        ps = self._tri(features.ph,    5.0,  6.8, 4.0,  7.8)
        es = self._tri(features.elev,  200,  900,   0, 1500)
        ns = self._tri(features.ndvi,  0.4,  0.75, 0.15, 0.90)

        score = (
            0.28 * ts + 0.08 * 0.5 +   # bio04 fixní
            0.27 * rs + 0.07 * 0.5 +   # bio15 fixní
            0.18 * ps + 0.07 * es +
            0.05 * ns
        )
        score *= max(0.1, 1 - 0.25 * len(violations))
        score = float(np.clip(score, 0, 1))

        suit = (
            "velmi vysoká" if score >= 0.75 else
            "vysoká"       if score >= 0.50 else
            "střední"      if score >= 0.25 else
            "nízká"
        )
        return SDMResult(
            probability=round(score, 4),
            habitat_suitability=suit,
            feature_importance=self.IMPORTANCE,
            constraint_violations=violations,
            confidence=0.72,
        )

    def predict_grid(
        self,
        lat_range: tuple,
        lon_range: tuple,
        resolution: int = 20,
    ) -> np.ndarray:
        lats = np.linspace(lat_range[0], lat_range[1], resolution)
        lons = np.linspace(lon_range[0], lon_range[1], resolution)
        grid = np.zeros((resolution, resolution), dtype=np.float32)
        rng = np.random.default_rng(42)

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # Syntetické prostředí aproximující ČR/SR
                temp = 12.0 - (lat - 48.5) * 1.5 + rng.normal(0, 1.2)
                rain = 700 + (51.2 - lat) * 80 + rng.normal(0, 80)
                ph_v = 5.6 + rng.normal(0, 0.6)
                elev_v = 300 + abs(rng.normal(0, 250))
                ndvi_v = float(np.clip(0.55 + rng.normal(0, 0.12), 0, 1))

                f = SDMFeatures(
                    bio01=float(temp), bio04=750.0,
                    bio12=float(rain), bio15=25.0,
                    ph=float(np.clip(ph_v, 3, 9)),
                    elev=float(np.clip(elev_v, 0, 1500)),
                    ndvi=ndvi_v,
                )
                grid[i, j] = self.predict_point(f).probability
        return grid

    @staticmethod
    def _tri(v, opt_lo, opt_hi, abs_lo, abs_hi):
        if v < abs_lo or v > abs_hi: return 0.0
        if opt_lo <= v <= opt_hi:    return 1.0
        if v < opt_lo: return (v - abs_lo) / (opt_lo - abs_lo)
        return (abs_hi - v) / (abs_hi - opt_hi)
