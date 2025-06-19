from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class ConceptMastery(BaseModel):
    concept_id: str
    mastery_level: int  # 0-5
    last_updated: datetime

class SectionProgress(BaseModel):
    section_id: str
    completed: bool
    time_spent: int  # minutes
    completed_at: Optional[datetime] = None

class UserProgress(BaseModel):
    user_id: str
    current_chapter: int
    completed_sections: List[str]
    concept_mastery: Dict[str, int]
    total_time_spent: int  # minutes
    last_active: datetime
    achievements: List[str] = []
    
class LearningPath(BaseModel):
    user_id: str
    recommended_next: List[str]
    estimated_time: int  # minutes
    mastery_gaps: List[str]