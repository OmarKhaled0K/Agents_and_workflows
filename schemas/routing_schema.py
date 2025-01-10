from dataclasses import dataclass
from enum import Enum
@dataclass
class RouteConfig:
    """Configuration for a specific route including prompts"""
    name: str
    description: str
    system_prompt: str
    response_template: str
    confidence_threshold: float = 0.5
    priority: int = 0

class RouteType(Enum):
    """Enum for different types of routing strategies"""
    SINGLE = "single"
    MULTI = "multi"
    PRIORITY = "priority"
