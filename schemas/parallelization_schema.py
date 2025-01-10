from typing import List
from dataclasses import dataclass
from enum import Enum
class ParallelizationType(Enum):
    SECTIONING = "sectioning"
    VOTING = "voting"

@dataclass
class Section:
    """Configuration for a section in sectioning parallelization"""
    name: str
    system_prompt: str
    task_prompt: str
    weight: float = 1.0

@dataclass
class VotingConfig:
    """Configuration for voting parallelization"""
    prompt: str
    variations: List[str]
    threshold: float
    aggregation_method: str = "majority"  # majority, unanimous, weighted