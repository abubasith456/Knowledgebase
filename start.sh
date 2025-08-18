#!/bin/bash

# Knowledge Base System Startup Script
# This script starts both the backend and frontend services

echo "🚀 Starting Knowledge Base System..."

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "❌ Port $1 is already in use. Please stop the service using port $1 first."
        return 1
    fi
    return 0
}

# Function to start backend
start_backend() {
    echo "🔧 Starting Backend API..."
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies if requirements.txt is newer than venv
    if [ requirements.txt -nt venv/lib/python*/site-packages/.installed ] 2>/dev/null; then
        echo "📥 Installing Python dependencies..."
        pip install -r requirements.txt
        touch venv/lib/python*/site-packages/.installed
    fi
    
    # Start backend
    echo "🌟 Backend starting on http://localhost:8000"
    echo "📖 API Documentation: http://localhost:8000/docs"
    python main.py &
    BACKEND_PID=$!
    cd ..
}

# Function to start frontend
start_frontend() {
    echo "🎨 Starting Frontend..."
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "📦 Installing Node.js dependencies..."
        npm install
    fi
    
    # Start frontend
    echo "🌟 Frontend starting on http://localhost:3000"
    npm start &
    FRONTEND_PID=$!
    cd ..
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "✅ Backend stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "✅ Frontend stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check ports
echo "🔍 Checking ports..."
if ! check_port 8000; then
    exit 1
fi
if ! check_port 3000; then
    exit 1
fi

# Start services
start_backend
sleep 3  # Give backend time to start

start_frontend

echo ""
echo "🎉 Knowledge Base System is starting up!"
echo ""
echo "📱 Frontend: http://localhost:3000"
echo "🔗 Backend API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for user to stop
wait