#!/usr/bin/env python3
"""Test import of evaluation suite and replay system."""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    print("Testing evaluation suite import...")
    from rlhf_core.evaluation_suite import EvaluationSuite, EvaluationConfig
    print("✅ Evaluation suite imported successfully")
except Exception as e:
    print(f"❌ Evaluation suite import failed: {e}")

try:
    print("Testing replay system import...")
    from rlhf_core.replay_system import ReplaySystem, ReplayConfig
    print("✅ Replay system imported successfully")
except Exception as e:
    print(f"❌ Replay system import failed: {e}")

try:
    print("Testing integrated system import...")
    from rlhf_core.integrated_evaluation_replay import IntegratedEvaluationReplay
    print("✅ Integrated system imported successfully")
except Exception as e:
    print(f"❌ Integrated system import failed: {e}")

print("All imports completed")