# =========================
# Predictive Maintenance API
# =========================

IMAGE_NAME ?= pm-api
CONTAINER_NAME ?= pm-api
PORT ?= 8000

API_URL ?= http://127.0.0.1:$(PORT)


help:
	@echo "Available commands:"
	@echo "  make build                Build Docker image ($(IMAGE_NAME))"
	@echo "  make run                  Run container on port $(PORT)"
	@echo "  make run-d                Run container in background (detached)"
	@echo "  make stop                 Stop & remove container ($(CONTAINER_NAME))"
	@echo "  make rebuild              Stop + build + run"
	@echo "  make clean                Remove image + prune build cache"
	@echo "  make health               Call GET /health"
	@echo "  make predict              Call POST /predict (single record)"
	@echo "  make explain              Call POST /explain (top contributors)"
	@echo "  make explain-plot          Call POST /explain/plot -> saves shap.png"
	@echo "  make test                 Run all tests (local)"
	@echo "  make test-cov             Run tests with coverage"
	@echo "  make test-unit            Run unit tests only"
	@echo "  make test-integration     Run integration tests only"
	@echo "  make test-docker          Run tests inside Docker"
	@echo ""
	@echo "Vars:"
	@echo "  IMAGE_NAME=pm-api PORT=8000 CONTAINER_NAME=pm-api"

# Build Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run container (foreground)
run: stop
	docker run --rm \
		--name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
		$(IMAGE_NAME)

# Run container (detached)
run-d: stop
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
		$(IMAGE_NAME)
	@echo "Running: $(CONTAINER_NAME) on $(API_URL)"



# Stop & remove container by name (safe if not running)
stop:
	@docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1 || true
	@echo "Container $(CONTAINER_NAME) stopped/removed (if existed)."

# Rebuild (stop + build + run-d)
rebuild: stop build run-d

# Healthcheck
health:
	curl -s $(API_URL)/health | python3 -m json.tool

# Test /predict
predict:
	curl -s -X POST $(API_URL)/predict \
		-H "Content-Type: application/json" \
		-d '{"Air temperature [K]":300,"Process temperature [K]":310,"Rotational speed [rpm]":1500,"Torque [Nm]":40,"Tool wear [min]":100,"Type":"M"}' \
	| python3 -m json.tool

# Test /explain
explain:
	curl -s -X POST "$(API_URL)/explain?top_k=8" \
		-H "Content-Type: application/json" \
		-d '{"Air temperature [K]":300,"Process temperature [K]":310,"Rotational speed [rpm]":1500,"Torque [Nm]":40,"Tool wear [min]":100,"Type":"M"}' \
	| python3 -m json.tool

# Test /explain/plot (saves PNG)
explain-plot:
	curl -s -X POST $(API_URL)/explain/plot \
		-H "Content-Type: application/json" \
		-d '{"Air temperature [K]":300,"Process temperature [K]":310,"Rotational speed [rpm]":1500,"Torque [Nm]":40,"Tool wear [min]":100,"Type":"M"}' \
	-o shap.png
	@echo "Saved SHAP plot to ./shap.png"


# Clean image + build cache 
clean: stop
	@docker image rm -f $(IMAGE_NAME) >/dev/null 2>&1 || true
	@docker builder prune -f >/dev/null 2>&1 || true
	@echo "🧹 Cleanup done."

# Run tests inside Docker container
test-docker: build
	docker run --rm \
		$(IMAGE_NAME) \
		pytest -v

test-docker-cov: build
	docker run --rm \
		$(IMAGE_NAME) \
		pytest -v --cov=src --cov=api --cov-report=term-missing