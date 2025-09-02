.PHONY: help install train_smoke train_quick train_full clean check install-dev

# Default target
help:
	@echo "RLHF Systems Kit - Available targets:"
	@echo ""
	@echo "Setup:"
	@echo "  install        - Install dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo ""
	@echo "Training:"
	@echo "  train_smoke    - Quick smoke test (<2 min)"
	@echo "  train_quick    - Quick training run (5 min)"
	@echo "  train_full     - Full training run (15 min)"
	@echo ""
	@echo "Profiling:"
	@echo "  profile        - Run profiling job (<2 min)"
	@echo ""
	@echo "Visualization:"
	@echo "  plots          - Generate PNG plots from latest training run"
	@echo "  plots-run      - Generate plots from specific run (use RUN=path)"
	@echo "  anomaly-check  - Generate anomaly detection report from latest run"
	@echo "  anomaly-check-run - Generate anomaly report from specific run (use RUN=path)"
	@echo ""
	@echo "Verification:"
	@echo "  verify_m1      - Verify M1: Determinism"
	@echo "  verify_m2      - Verify M2: Profiler artifacts"
	@echo "  verify_m3      - Verify M3: Dashboard metrics"
	@echo ""
	@echo "Utilities:"
	@echo "  check          - Check installation and dependencies"
	@echo "  clean          - Clean generated files"
	@echo "  tensorboard    - Launch TensorBoard for latest run"

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev: install
	pip install pytest black flake8 mypy

# Quick smoke test (<2 minutes)
train_smoke:
	@echo "Running smoke test (target: <2 minutes)..."
	timeout 130s python train.py --epochs 2 --steps_per_epoch 6 --batch_size 2 --max_new_tokens 10
	@echo "Smoke test completed!"

# Quick training run (~5 minutes)
train_quick:
	@echo "Running quick training (target: ~5 minutes)..."
	python train.py --epochs 5 --steps_per_epoch 10 --batch_size 4 --max_new_tokens 15

# Full training run (~15 minutes)
train_full:
	@echo "Running full training (target: ~15 minutes)..."
	python train.py --epochs 10 --steps_per_epoch 15 --batch_size 8 --max_new_tokens 20

# Check installation
check:
	@echo "Checking installation..."
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	@python -c "from rlhf_core.policy import PolicyModel; print('Policy model: OK')"
	@python -c "from rlhf_core.reward import ToyRewardModel; print('Reward model: OK')"
	@python -c "from rlhf_core.ppo import PPOTrainer; print('PPO trainer: OK')"
	@echo "All checks passed!"

# Clean generated files
clean:
	rm -rf runs/
	rm -rf profiles/
	rm -rf __pycache__/
	rm -rf rlhf_core/__pycache__/
	rm -rf profiler/__pycache__/
	rm -rf tools/__pycache__/
	rm -rf *.pyc
	@echo "Cleaned generated files"

# Launch TensorBoard
tensorboard:
	@echo "Launching TensorBoard for latest run..."
	@if [ -L runs/latest ]; then \
		tensorboard --logdir=runs/latest/tb --port=6006; \
	else \
		echo "No latest run found. Run training first."; \
	fi

# Run profiling job (<2 minutes)
profile:
	@echo "Running profiling job (target: <2 minutes)..."
	timeout 130s python3 tools/run_profile.py --steps 1 --batch_size 2 --seq_len 10
	@echo "Profiling completed!"

# Generate plots from latest training run
plots:
	@echo "Generating plots from latest training run..."
	python3 tools/generate_plots_simple.py --run runs/latest
	@echo "Plot generation completed!"

# Generate plots from specific run
plots-run:
	@echo "Usage: make plots-run RUN=runs/run_20250829_024038"
	@if [ -z "$(RUN)" ]; then \
		echo "Error: Please specify RUN variable, e.g., make plots-run RUN=runs/run_20250829_024038"; \
		exit 1; \
	fi
	@echo "Generating plots from run: $(RUN)"
	python3 tools/generate_plots_simple.py --run $(RUN)

# Generate anomaly detection report from latest run
anomaly-check:
	@echo "Generating anomaly detection report from latest training run..."
	python3 tools/generate_anomaly_plots.py --run runs/latest
	@echo "Anomaly check completed!"

# Generate anomaly detection report from specific run
anomaly-check-run:
	@echo "Usage: make anomaly-check-run RUN=runs/run_20250829_024038"
	@if [ -z "$(RUN)" ]; then \
		echo "Error: Please specify RUN variable, e.g., make anomaly-check-run RUN=runs/run_20250829_024038"; \
		exit 1; \
	fi
	@echo "Generating anomaly report from run: $(RUN)"
	python3 tools/generate_anomaly_plots.py --run $(RUN)

# M1: Verify determinism
verify_m1:
	@echo "Verifying M1: Determinism..."
	SEED=123 python train.py --seed 123 --output_dir runs/m1a --epochs 1 --steps_per_epoch 2 --batch_size 2 --max_new_tokens 5
	SEED=123 python train.py --seed 123 --output_dir runs/m1b --epochs 1 --steps_per_epoch 2 --batch_size 2 --max_new_tokens 5
	@echo "Comparing log files..."
	head -n 50 runs/m1a/run_*/logs/train.jsonl > /tmp/a.jsonl
	head -n 50 runs/m1b/run_*/logs/train.jsonl > /tmp/b.jsonl
	diff /tmp/a.jsonl /tmp/b.jsonl || (echo "❌ M1 FAILED: Logs differ between runs"; exit 1)
	@echo "✅ M1 PASSED: Determinism verified"

# M2: Verify profiler artifacts and sanity checks
verify_m2:
	@echo "Verifying M2: Profiler artifacts and sanity checks..."
	make profile
	python tools/check_profile.py
	test -s profiles/summary.csv
	test -s profiles/chrome_trace.json
	test -s profiles/op_stats.csv
	@echo "✅ M2 PASSED: Profiler artifacts verified"

# M3: Verify dashboard metrics and live server
verify_m3:
	@echo "Verifying M3: Dashboard metrics and live server..."
	@echo "Starting monitoring server..."
	uvicorn monitor.app:app --port 8765 --reload &
	@sleep 3
	@echo "Testing server endpoints..."
	curl -sf "http://localhost:8765/health" || (echo "❌ Health check failed"; exit 1)
	curl -sf "http://localhost:8765/alerts?test_alerts=1" | grep -i warning || (echo "❌ Test alerts failed"; exit 1)
	python tools/check_dashboard_metrics.py
	@echo "✅ M3 PASSED: Dashboard metrics and server verified"
	@pkill -f "uvicorn monitor.app:app" || true

# Run tests (if pytest is installed)
test:
	pytest tests/ -v

# Format code (if black is installed)
format:
	black rlhf_core/ train.py

# Lint code (if flake8 is installed)
lint:
	flake8 rlhf_core/ train.py

# Type check (if mypy is installed)
typecheck:
	mypy rlhf_core/ train.py

# All checks
all-checks: format lint typecheck test

# Default training target
train: train_smoke