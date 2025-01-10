from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class EvaluationType(Enum):
    TRANSLATION = "translation"
    SEARCH = "search"
    WRITING = "writing"
    CODE = "code"
    CUSTOM = "custom"

@dataclass
class EvaluationCriteria:
    """Criteria for evaluation"""
    name: str
    description: str
    weight: float
    min_score: float = 0.0
    max_score: float = 1.0

@dataclass
class EvaluationResult:
    """Result of an evaluation"""
    scores: Dict[str, float]
    feedback: Dict[str, str]
    overall_score: float
    suggestions: List[str]
    iteration: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scores": self.scores,
            "feedback": self.feedback,
            "overall_score": self.overall_score,
            "suggestions": self.suggestions,
            "iteration": self.iteration
        }