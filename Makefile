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
	@echo "Monitoring:"
	@echo "  dashboard      - Launch live stability dashboard"
	@echo "  tensorboard    - Launch TensorBoard for latest run"
	@echo ""
	@echo "Utilities:"
	@echo "  check          - Check installation and dependencies"
	@echo "  clean          - Clean generated files"

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev: install
	pip install pytest black flake8 mypy

# Quick smoke test (<2 minutes)
train_smoke:
	@echo "Running smoke test (target: <2 minutes)..."
	timeout 130s python3 train.py --epochs 2 --steps_per_epoch 6 --batch_size 2 --max_new_tokens 10
	@echo "Smoke test completed!"

# Quick training run (~5 minutes)
train_quick:
	@echo "Running quick training (target: ~5 minutes)..."
	python3 train.py --epochs 5 --steps_per_epoch 10 --batch_size 4 --max_new_tokens 15

# Full training run (~15 minutes)
train_full:
	@echo "Running full training (target: ~15 minutes)..."
	python3 train.py --epochs 10 --steps_per_epoch 15 --batch_size 8 --max_new_tokens 20

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

# Launch Stability Dashboard
dashboard:
	@echo "Launching RLHF Stability Dashboard..."
	@python3 scripts/serve_dashboard.py

# Run profiling job (<2 minutes)
profile:
	@echo "Running profiling job (target: <2 minutes)..."
	timeout 130s python3 tools/run_profile.py --steps 1 --batch_size 2 --seq_len 10
	@echo "Profiling completed!"

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