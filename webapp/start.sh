#!/bin/bash
# Start script for the web application

echo "Starting GMM Fitting Web Application..."
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Virtual environment not activated."
    echo "Please activate it with: source .venv/bin/activate"
    echo ""
fi

# Start FastAPI backend
echo "Starting FastAPI backend on http://localhost:8000"
python -m webapp.api &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start React frontend
echo "Starting React frontend on http://localhost:3000"
cd webapp/frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait

