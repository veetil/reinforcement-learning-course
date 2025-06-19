from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import socketio

from app.api.routes import router as api_router
from app.core.config import settings
from app.api.websocket.training_ws import sio_app

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting PPO Course API...")
    yield
    # Shutdown
    print("Shutting down PPO Course API...")

app = FastAPI(
    title="PPO Interactive Course API",
    description="Backend API for the PPO learning platform",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Mount Socket.IO app
app.mount("/ws", sio_app)

@app.get("/")
async def root():
    return {
        "message": "PPO Interactive Course API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}