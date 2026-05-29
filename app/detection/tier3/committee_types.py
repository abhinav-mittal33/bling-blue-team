from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CommitteeMode(Enum):
    SHADOW = "shadow"
    LIVE = "live"


SCORER_IDS = ("A", "B", "C", "D", "F")

# Fallback aggregate weights when meta-learner not yet trained
# A=GBM, B=Graph, C=Prototype, D=Sequence, F=Remark
FALLBACK_WEIGHTS: dict[str, float] = {
    "A": 0.40,
    "B": 0.20,
    "C": 0.20,
    "D": 0.10,
    "F": 0.10,
}


@dataclass
class ScorerOutput:
    score: float          # [0.0, 1.0]
    confidence: float     # [0.0, 1.0] — how certain the scorer is
    missing_flag: bool    # True = scorer could not produce a reliable score
    scorer_id: str        # one of SCORER_IDS

    def __post_init__(self) -> None:
        self.score = float(max(0.0, min(1.0, self.score)))
        self.confidence = float(max(0.0, min(1.0, self.confidence)))

    @classmethod
    def unavailable(cls, scorer_id: str) -> "ScorerOutput":
        """Degraded output when scorer fails or model is absent."""
        return cls(score=0.5, confidence=0.0, missing_flag=True, scorer_id=scorer_id)


@dataclass
class CommitteeResult:
    scorer_outputs: list[ScorerOutput]
    final_score: float
    specialist_override: bool = False
    meta_score: Optional[float] = None
    mapie_lower: Optional[float] = None
    mapie_upper: Optional[float] = None

    def as_breakdown_dict(self) -> dict:
        """Serializable dict for alert committee_breakdown field."""
        return {
            "scorers": {
                o.scorer_id: {
                    "score": round(o.score, 4),
                    "confidence": round(o.confidence, 4),
                    "missing": o.missing_flag,
                }
                for o in self.scorer_outputs
            },
            "meta_score": round(self.meta_score, 4) if self.meta_score is not None else None,
            "specialist_override": self.specialist_override,
            "final_score": round(self.final_score, 4),
            "mapie_lower": round(self.mapie_lower, 4) if self.mapie_lower is not None else None,
            "mapie_upper": round(self.mapie_upper, 4) if self.mapie_upper is not None else None,
        }
