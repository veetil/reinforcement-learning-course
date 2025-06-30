# PPO Interactive Learning Platform

An interactive web application for learning Proximal Policy Optimization (PPO) through hands-on visualizations, practical coding exercises, and real-world applications.

## Features

- 🎯 **Interactive Visualizations**: See neural networks come to life with real-time animations
- 💻 **Hands-on Coding**: Build PPO implementations with instant feedback
- 🧠 **Adaptive Learning**: AI-powered confusion detection and personalized paths
- 🏆 **Gamification**: Achievements, progress tracking, and certifications
- 🤝 **Community Learning**: Collaborate with peers and share insights

## Tech Stack

### Frontend
- Next.js 14 with TypeScript
- React Flow for neural network visualizations
- Framer Motion for animations
- Three.js for 3D visualizations
- Monaco Editor for code editing
- Tailwind CSS for styling

### Backend
- FastAPI for REST API
- Socket.IO for real-time updates
- PyTorch & Stable Baselines3 for RL
- Docker for code sandboxing
- PostgreSQL for data persistence
- Redis for caching

## Getting Started

### Prerequisites
- Node.js 20+
- Python 3.9+
- Docker & Docker Compose
- PostgreSQL (or use Docker)
- Redis (or use Docker)

### Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd course
```

2. Install frontend dependencies:
```bash
cd ppo-course
npm install
```

3. Install backend dependencies:
```bash
cd ../backend
pip install -r requirements.txt
```

4. Start services with Docker Compose:
```bash
docker-compose up
```

Or run services individually:

**Frontend:**
```bash
cd ppo-course
npm run dev
```

**Backend:**
```bash
cd backend
python run.py
```

5. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Project Structure

```
course/
├── ppo-course/          # Next.js frontend
│   ├── src/
│   │   ├── app/        # App router pages
│   │   ├── components/ # React components
│   │   └── lib/        # Utilities and stores
│   └── public/         # Static assets
├── backend/            # FastAPI backend
│   ├── app/
│   │   ├── api/       # API endpoints
│   │   ├── core/      # Core configuration
│   │   ├── schemas/   # Pydantic models
│   │   └── services/  # Business logic
│   └── tests/         # Backend tests
└── docker-compose.yml  # Development environment
```

## Key Components

### 1. Neural Network Visualizer
Interactive visualization of neural networks with real-time weight and activation updates.

### 2. PPO Algorithm Stepper
Step-by-step walkthrough of PPO algorithm execution with visualizations.

### 3. Code Playground
Secure sandboxed environment for writing and testing PPO implementations.

### 4. Progress Tracking
Comprehensive tracking of learning progress, concept mastery, and achievements.

### 5. Assessment System
Quizzes, coding assignments, and practical projects with automated grading.

## Course Content

### Phase 1: Foundation (Chapters 1-4)
- Introduction to Reinforcement Learning
- Value Functions and Critics
- Actor-Critic Architecture
- Introduction to PPO

### Phase 2: Deep Dive (Chapters 5-9)
- PPO Objective Function
- Advantage Estimation
- Implementation Architecture
- Mini-batch Training
- PPO for Language Models

### Phase 3: Advanced (Chapters 10-14)
- Scaling PPO Systems
- Advanced Reward Modeling
- Complex Domains
- Production Deployment
- PPO Variants

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for PPO algorithm
- Stable Baselines3 team
- React Flow contributors
- All open-source contributors