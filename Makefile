.PHONY: test test-verbose test-cov test-frontend test-all clean install help webapp-install webapp-start webapp-stop webapp-backend webapp-frontend webapp-status webapp-logs webapp-clean

# Default target
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python
VENV := .venv
VENV_BIN := $(VENV)/bin
PYTEST := $(VENV_BIN)/pytest

# Test configuration
TEST_DIR := tests
COV_MODULE := gmm_fitting
COV_REPORT := term-missing

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

test: ## Run all tests (Python + Frontend)
	@echo "Running Python tests..."
	$(PYTEST) $(TEST_DIR)
	@echo ""
	@echo "Running frontend tests..."
	@cd webapp/frontend && npm test -- --watchAll=false || (echo "Frontend tests failed. Make sure dependencies are installed: cd webapp/frontend && npm install" && exit 1)
	@echo ""
	@echo "All tests passed!"

test-python: ## Run Python tests only
	@echo "Running Python tests..."
	$(PYTEST) $(TEST_DIR)

test-frontend: ## Run frontend tests only
	@echo "Running frontend tests..."
	@cd webapp/frontend && npm test -- --watchAll=false

test-all: test ## Alias for test (runs all tests)

test-verbose: ## Run tests with verbose output
	@echo "Running tests with verbose output..."
	$(PYTEST) $(TEST_DIR) -v

test-cov: ## Run Python tests with coverage report
	@echo "Running Python tests with coverage report..."
	$(PYTEST) $(TEST_DIR) --cov=$(COV_MODULE) --cov-report=$(COV_REPORT)

test-cov-all: ## Run all tests with coverage reports
	@echo "Running Python tests with coverage report..."
	$(PYTEST) $(TEST_DIR) --cov=$(COV_MODULE) --cov-report=$(COV_REPORT)
	@echo ""
	@echo "Running frontend tests with coverage report..."
	@cd webapp/frontend && npm run test:coverage || (echo "Frontend tests failed. Make sure dependencies are installed: cd webapp/frontend && npm install" && exit 1)

test-cov-html: ## Run tests and generate HTML coverage report
	@echo "Running tests and generating HTML coverage report..."
	$(PYTEST) $(TEST_DIR) --cov=$(COV_MODULE) --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

test-fast: ## Run Python tests quickly (no coverage)
	@echo "Running Python tests (fast mode)..."
	$(PYTEST) $(TEST_DIR) -q

test-specific: ## Run specific test file (usage: make test-specific FILE=tests/test_pdf_calculation.py)
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE variable is required. Usage: make test-specific FILE=tests/test_pdf_calculation.py"; \
		exit 1; \
	fi
	$(PYTEST) $(FILE) -v

install: ## Install dependencies
	@echo "Installing dependencies..."
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment with uv..."; \
		uv venv; \
	fi
	uv pip install -r requirements.txt

clean: ## Clean generated files
	@echo "Cleaning generated files..."
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf src/**/__pycache__
	rm -rf webapp/__pycache__
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete."

# Web application targets
WEBAPP_PID_DIR := .webapp_pids
WEBAPP_BACKEND_PID := $(WEBAPP_PID_DIR)/backend.pid
WEBAPP_FRONTEND_PID := $(WEBAPP_PID_DIR)/frontend.pid

webapp-install: ## Install web application dependencies
	@echo "Installing web application dependencies..."
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment with uv..."; \
		uv venv; \
	fi
	uv pip install -r webapp/requirements.txt
	@echo "Installing frontend dependencies..."
	@cd webapp/frontend && npm install
	@echo "Web application dependencies installed."

webapp-start: ## Start web application (backend + frontend)
	@echo "Starting web application..."
	@mkdir -p $(WEBAPP_PID_DIR)
	@if [ -f $(WEBAPP_BACKEND_PID) ] || [ -f $(WEBAPP_FRONTEND_PID) ]; then \
		echo "Warning: Web application may already be running. Use 'make webapp-stop' first."; \
		exit 1; \
	fi
	@echo "Starting FastAPI backend on http://localhost:8000..."
	@uv run uvicorn webapp.api:app --host 0.0.0.0 --port 8000 > $(WEBAPP_PID_DIR)/backend.log 2>&1 & \
		echo $$! > $(WEBAPP_BACKEND_PID)
	@sleep 2
	@echo "Starting React frontend on http://localhost:3000..."
	@cd webapp/frontend && npm run dev > $(CURDIR)/$(WEBAPP_PID_DIR)/frontend.log 2>&1 & \
		echo $$! > $(CURDIR)/$(WEBAPP_FRONTEND_PID)
	@sleep 3
	@echo "Web application started!"
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"
	@echo "API docs: http://localhost:8000/docs"
	@echo "Use 'make webapp-stop' to stop the servers."

webapp-stop: ## Stop web application
	@echo "Stopping web application..."
	@if [ -f $(WEBAPP_BACKEND_PID) ]; then \
		BACKEND_PID=$$(cat $(WEBAPP_BACKEND_PID)); \
		if ps -p $$BACKEND_PID > /dev/null 2>&1; then \
			echo "Stopping backend (PID: $$BACKEND_PID)..."; \
			kill $$BACKEND_PID 2>/dev/null || true; \
		fi; \
		rm -f $(WEBAPP_BACKEND_PID); \
	fi
	@if [ -f $(WEBAPP_FRONTEND_PID) ] || [ -f $(CURDIR)/$(WEBAPP_FRONTEND_PID) ]; then \
		if [ -f $(WEBAPP_FRONTEND_PID) ]; then \
			FRONTEND_PID_FILE=$(WEBAPP_FRONTEND_PID); \
		else \
			FRONTEND_PID_FILE=$(CURDIR)/$(WEBAPP_FRONTEND_PID); \
		fi; \
		FRONTEND_PID=$$(cat $$FRONTEND_PID_FILE); \
		if ps -p $$FRONTEND_PID > /dev/null 2>&1; then \
			echo "Stopping frontend (PID: $$FRONTEND_PID)..."; \
			kill $$FRONTEND_PID 2>/dev/null || true; \
		fi; \
		rm -f $$FRONTEND_PID_FILE; \
	fi
	@# Also kill any remaining uvicorn/vite processes
	@pkill -f "uvicorn webapp.api" 2>/dev/null || true
	@pkill -f "uv run uvicorn" 2>/dev/null || true
	@pkill -f "vite" 2>/dev/null || true
	@echo "Web application stopped."

webapp-backend: ## Start backend only
	@echo "Starting FastAPI backend..."
	@mkdir -p $(WEBAPP_PID_DIR)
	@if [ -f $(WEBAPP_BACKEND_PID) ]; then \
		echo "Backend may already be running. Use 'make webapp-stop' first."; \
		exit 1; \
	fi
	@uv run uvicorn webapp.api:app --host 0.0.0.0 --port 8000 --reload > $(WEBAPP_PID_DIR)/backend.log 2>&1 & \
		echo $$! > $(WEBAPP_BACKEND_PID)
	@sleep 2
	@echo "Backend started on http://localhost:8000"
	@echo "API docs: http://localhost:8000/docs"
	@echo "Use 'make webapp-stop' to stop."

webapp-frontend: ## Start frontend only
	@echo "Starting React frontend..."
	@mkdir -p $(WEBAPP_PID_DIR)
	@if [ -f $(WEBAPP_FRONTEND_PID) ]; then \
		echo "Frontend may already be running. Use 'make webapp-stop' first."; \
		exit 1; \
	fi
	@cd webapp/frontend && npm run dev > $(CURDIR)/$(WEBAPP_PID_DIR)/frontend.log 2>&1 & \
		echo $$! > $(CURDIR)/$(WEBAPP_FRONTEND_PID)
	@sleep 3
	@echo "Frontend started on http://localhost:3000"
	@echo "Use 'make webapp-stop' to stop."

webapp-status: ## Show web application status
	@echo "Web Application Status:"
	@echo "======================"
	@if [ -f $(WEBAPP_BACKEND_PID) ]; then \
		BACKEND_PID=$$(cat $(WEBAPP_BACKEND_PID)); \
		if ps -p $$BACKEND_PID > /dev/null 2>&1; then \
			echo "Backend: Running (PID: $$BACKEND_PID) - http://localhost:8000"; \
		else \
			echo "Backend: Not running (stale PID file)"; \
		fi; \
	else \
		echo "Backend: Not running"; \
	fi
	@if [ -f $(WEBAPP_FRONTEND_PID) ] || [ -f $(CURDIR)/$(WEBAPP_FRONTEND_PID) ]; then \
		if [ -f $(WEBAPP_FRONTEND_PID) ]; then \
			FRONTEND_PID_FILE=$(WEBAPP_FRONTEND_PID); \
		else \
			FRONTEND_PID_FILE=$(CURDIR)/$(WEBAPP_FRONTEND_PID); \
		fi; \
		FRONTEND_PID=$$(cat $$FRONTEND_PID_FILE); \
		if ps -p $$FRONTEND_PID > /dev/null 2>&1; then \
			echo "Frontend: Running (PID: $$FRONTEND_PID) - http://localhost:3000"; \
		else \
			echo "Frontend: Not running (stale PID file)"; \
		fi; \
	else \
		echo "Frontend: Not running"; \
	fi

webapp-logs: ## Show web application logs
	@echo "Backend logs:"
	@echo "============="
	@if [ -f $(WEBAPP_PID_DIR)/backend.log ]; then \
		tail -n 50 $(WEBAPP_PID_DIR)/backend.log; \
	else \
		echo "No backend logs found."; \
	fi
	@echo ""
	@echo "Frontend logs:"
	@echo "=============="
	@if [ -f $(WEBAPP_PID_DIR)/frontend.log ]; then \
		tail -n 50 $(WEBAPP_PID_DIR)/frontend.log; \
	else \
		echo "No frontend logs found."; \
	fi

webapp-clean: ## Clean web application files (logs, PID files)
	@echo "Cleaning web application files..."
	@make webapp-stop 2>/dev/null || true
	@rm -rf $(WEBAPP_PID_DIR)
	@rm -f webapp/temp_plot.png
	@echo "Web application files cleaned."

