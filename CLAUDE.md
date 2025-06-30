# PPO Interactive Learning Platform - Development Guide

## Project Overview
An interactive web application for learning Proximal Policy Optimization (PPO) through hands-on visualizations, practical coding exercises, and real-world applications.

## Tech Stack
- **Frontend**: Next.js 14 with TypeScript, React Flow, Framer Motion, Three.js
- **Backend**: FastAPI with WebSockets, PyTorch, Stable Baselines3
- **Database**: PostgreSQL, Redis
- **Infrastructure**: Docker, Docker Compose

## Quick Start

### Prerequisites
- Node.js 20+
- Python 3.9+
- Docker & Docker Compose (optional)

### Running the Application

#### Option 1: Using Docker Compose (Recommended)
```bash
docker-compose up
```

#### Option 2: Running Services Individually

1. **Install Dependencies**
```bash
# Backend dependencies
cd backend
pip install -r requirements.txt
pip install python-socketio  # Additional required dependency

# Frontend dependencies
cd ../ppo-course
npm install
```

2. **Fix Known Issues**
- Edit `backend/app/schemas/code.py` to import `Any` from typing:
  ```python
  from typing import List, Optional, Dict, Any
  ```
  And update lines 35-36 to use `Any` instead of `any`

3. **Start Services**
```bash
# Backend (from /backend directory)
python run.py

# Frontend (from /ppo-course directory)
npm run dev
```

### Access URLs
- Frontend: http://localhost:3001
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Project Structure
```
course/
├── backend/                # FastAPI backend
│   ├── app/
│   │   ├── api/           # API endpoints
│   │   ├── core/          # Core configuration
│   │   ├── schemas/       # Pydantic models
│   │   └── services/      # Business logic
│   └── requirements.txt
├── ppo-course/            # Next.js frontend
│   ├── src/
│   │   ├── app/          # App router pages
│   │   ├── components/   # React components
│   │   └── lib/          # Utilities and stores
│   └── package.json
└── docker-compose.yml     # Development environment
```

## Key Features
1. **Neural Network Visualizer**: Interactive visualization with real-time updates
2. **PPO Algorithm Stepper**: Step-by-step algorithm walkthrough
3. **Code Playground**: Secure sandboxed environment for testing
4. **Progress Tracking**: Comprehensive learning progress tracking
5. **Assessment System**: Automated grading for quizzes and projects

## Common Issues & Solutions

### Pydantic Version Conflicts
If you encounter pydantic-related errors, install compatible versions:
```bash
pip install pydantic==2.5.0 pydantic-settings==2.0.3
```

### Docker Build Failures
If Docker build fails due to network issues, run services individually instead.

### Port Conflicts
- Frontend runs on port 3001 (can be changed in package.json)
- Backend runs on port 8000 (can be changed in run.py)

## Development Commands

### Backend
- Run tests: `pytest`
- Format code: `black .`
- Lint: `flake8`

### Frontend
- Run tests: `npm test`
- Build: `npm run build`
- Lint: `npm run lint`
- Type check: `npm run type-check`

## Course Content Structure
- **Phase 1**: Foundation (Chapters 1-4)
- **Phase 2**: Deep Dive (Chapters 5-9)
- **Phase 3**: Advanced (Chapters 10-14)

## Contributing
See contributing guidelines for details on code style, testing requirements, and submission process.