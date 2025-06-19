#!/bin/bash

echo "🚀 Starting PPO Interactive Course Demo..."
echo ""
echo "This will start the frontend on port 3001"
echo ""

cd ppo-course

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install --legacy-peer-deps
fi

# Check if next is available
if [ ! -f "node_modules/.bin/next" ]; then
    echo "⚠️  Next.js not found, installing..."
    npm install next@15.3.3 --legacy-peer-deps
fi

echo ""
echo "🎮 Starting the application on port 3001..."
echo "📍 Visit http://localhost:3001 in your browser"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the dev server
./node_modules/.bin/next dev -p 3001