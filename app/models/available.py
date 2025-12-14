from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AvailabilityResult:
    """Type-safe result for availability checks."""
    status: str 
    message: str
    available_slots: List[str]
    checked_time: Optional[str] = None
    checked_date: Optional[str] = None