from fastapi import APIRouter
from app.api.endpoints import training, progress, code_execution, assessment

router = APIRouter()

router.include_router(training.router, prefix="/training", tags=["training"])
router.include_router(progress.router, prefix="/progress", tags=["progress"])
router.include_router(code_execution.router, prefix="/code", tags=["code"])
router.include_router(assessment.router, prefix="/assessment", tags=["assessment"])