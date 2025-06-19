from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
from datetime import datetime
from app.schemas.progress import UserProgress, SectionProgress, ConceptMastery

router = APIRouter()

# In-memory storage for demo
user_progress: Dict[str, UserProgress] = {}

@router.get("/{user_id}", response_model=UserProgress)
async def get_user_progress(user_id: str):
    """Get user's learning progress"""
    if user_id not in user_progress:
        # Create new progress
        user_progress[user_id] = UserProgress(
            user_id=user_id,
            current_chapter=1,
            completed_sections=[],
            concept_mastery={},
            total_time_spent=0,
            last_active=datetime.utcnow()
        )
    
    return user_progress[user_id]

@router.post("/{user_id}/complete-section")
async def complete_section(user_id: str, section_id: str):
    """Mark a section as completed"""
    if user_id not in user_progress:
        user_progress[user_id] = UserProgress(
            user_id=user_id,
            current_chapter=1,
            completed_sections=[],
            concept_mastery={},
            total_time_spent=0,
            last_active=datetime.utcnow()
        )
    
    progress = user_progress[user_id]
    if section_id not in progress.completed_sections:
        progress.completed_sections.append(section_id)
        progress.last_active = datetime.utcnow()
    
    return {"message": "Section completed", "section_id": section_id}

@router.post("/{user_id}/update-mastery")
async def update_concept_mastery(
    user_id: str,
    concept_id: str,
    mastery_level: int
):
    """Update mastery level for a concept"""
    if mastery_level < 0 or mastery_level > 5:
        raise HTTPException(status_code=400, detail="Mastery level must be between 0 and 5")
    
    if user_id not in user_progress:
        user_progress[user_id] = UserProgress(
            user_id=user_id,
            current_chapter=1,
            completed_sections=[],
            concept_mastery={},
            total_time_spent=0,
            last_active=datetime.utcnow()
        )
    
    progress = user_progress[user_id]
    progress.concept_mastery[concept_id] = mastery_level
    progress.last_active = datetime.utcnow()
    
    return {
        "message": "Mastery updated",
        "concept_id": concept_id,
        "mastery_level": mastery_level
    }

@router.get("/{user_id}/achievements")
async def get_achievements(user_id: str):
    """Get user's achievements"""
    # Simulated achievements based on progress
    if user_id not in user_progress:
        return {"achievements": []}
    
    progress = user_progress[user_id]
    achievements = []
    
    if len(progress.completed_sections) >= 1:
        achievements.append({
            "id": "first_step",
            "name": "First Step",
            "description": "Complete your first section",
            "unlocked_at": datetime.utcnow()
        })
    
    if len(progress.completed_sections) >= 5:
        achievements.append({
            "id": "dedicated_learner",
            "name": "Dedicated Learner",
            "description": "Complete 5 sections",
            "unlocked_at": datetime.utcnow()
        })
    
    return {"achievements": achievements}