from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    CODE = "code"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"

@dataclass
class SubTask:
    """Representation of a subtask created by the orchestrator"""
    id: str
    task_type: TaskType
    description: str
    context: Dict[str, Any]
    dependencies: List[str] = None
    priority: int = 0

@dataclass
class TaskResult:
    """Result of a completed subtask"""
    task_id: str
    status: str
    result: Any
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result,
            "metadata": self.metadata
        }