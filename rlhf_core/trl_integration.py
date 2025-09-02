"""
TRL Integration Module

This module provides seamless integration with TRL (Transformers Reinforcement Learning)
for advanced RLHF training with real-time monitoring, PPO-specific debugging, and
checkpoint analysis capabilities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# TRL imports
try:
    from trl import PPOTrainer as TRLPPOTrainer, PPOConfig
    from trl.trainer import PPOTrainer as BasePPOTrainer
    from trl.core import LengthSampler
    from trl.trainer.utils import disable_dropout_in_model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    print("Warning: TRL not available. Install with: pip install trl>=0.7.0")

# Local imports
from .logging import JSONLLogger
from .profiler import ProfilerManager, stage_timer


@dataclass
class TRLIntegrationConfig:
    """Configuration for TRL integration."""
    
    # Training configuration
    model_name: str = "gpt2"
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    max_grad_norm: float = 1.0
    
    # PPO specific
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    kl_penalty: str = "kl"
    target: float = 6.0
    horizon: float = 10000.0
    init_kl_coef: float = 0.2
    adap_kl_ctrl: bool = True
    
    # Monitoring and logging
    log_with: str = "tensorboard"
    logging_dir: str = "./logs"
    save_freq: int = 100
    eval_freq: int = 50
    project_name: str = "rlhf-trl-integration"
    
    # Advanced monitoring
    enable_profiling: bool = True
    enable_checkpoint_analysis: bool = True
    enable_reward_monitoring: bool = True
    anomaly_detection_threshold: float = 3.0
    
    # Device and optimization
    device: str = "auto"
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False


class TrainingCallback(ABC):
    """Abstract base class for training callbacks."""
    
    @abstractmethod
    def on_step_begin(self, step: int, logs: Dict[str, Any]) -> None:
        """Called at the beginning of each training step."""
        pass
    
    @abstractmethod
    def on_step_end(self, step: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each training step."""
        pass
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        pass


class PPOMonitoringCallback(TrainingCallback):
    """PPO-specific monitoring callback for real-time debugging."""
    
    def __init__(self, 
                 anomaly_threshold: float = 3.0,
                 log_dir: str = "./logs",
                 enable_detailed_logging: bool = True):
        self.anomaly_threshold = anomaly_threshold
        self.log_dir = log_dir
        self.enable_detailed_logging = enable_detailed_logging
        self.logger = JSONLLogger(os.path.join(log_dir, "ppo_monitoring.jsonl"))
        
        # PPO-specific metrics tracking
        self.kl_divergences = []
        self.policy_losses = []
        self.value_losses = []
        self.reward_scores = []
        self.clip_ratios = []
        self.entropy_scores = []
        
        # Anomaly detection
        self.metric_history = {
            'kl_div': [],
            'policy_loss': [],
            'value_loss': [],
            'reward': [],
            'clip_ratio': []
        }
        
        os.makedirs(log_dir, exist_ok=True)
    
    def on_step_begin(self, step: int, logs: Dict[str, Any]) -> None:
        """Monitor step beginning."""
        if self.enable_detailed_logging:
            self.logger.log({
                "event": "step_begin",
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "memory_usage": self._get_memory_usage()
            })
    
    def on_step_end(self, step: int, logs: Dict[str, Any]) -> None:
        """Monitor step end with PPO-specific analysis."""
        # Extract PPO metrics
        ppo_metrics = self._extract_ppo_metrics(logs)
        
        # Update history
        for metric, value in ppo_metrics.items():
            if metric in self.metric_history:
                self.metric_history[metric].append(value)
                # Keep only last 100 values for anomaly detection
                if len(self.metric_history[metric]) > 100:
                    self.metric_history[metric] = self.metric_history[metric][-100:]
        
        # Anomaly detection
        anomalies = self._detect_anomalies(step, ppo_metrics)
        
        # Log comprehensive metrics
        log_entry = {
            "event": "step_end",
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "ppo_metrics": ppo_metrics,
            "anomalies": anomalies,
            "memory_usage": self._get_memory_usage(),
            "gpu_utilization": self._get_gpu_utilization()
        }
        
        self.logger.log(log_entry)
        
        # Alert on critical anomalies
        if anomalies:
            self._handle_anomalies(step, anomalies)
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Monitor epoch beginning."""
        self.logger.log({
            "event": "epoch_begin",
            "epoch": epoch,
            "timestamp": datetime.now().isoformat()
        })
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Monitor epoch end with comprehensive analysis."""
        # Calculate epoch statistics
        epoch_stats = self._calculate_epoch_statistics()
        
        self.logger.log({
            "event": "epoch_end",
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "epoch_statistics": epoch_stats
        })
        
        # Save epoch summary
        self._save_epoch_summary(epoch, epoch_stats)
    
    def _extract_ppo_metrics(self, logs: Dict[str, Any]) -> Dict[str, float]:
        """Extract PPO-specific metrics from training logs."""
        metrics = {}
        
        # Standard PPO metrics
        for key in ['kl_div', 'policy_loss', 'value_loss', 'reward', 'clip_ratio', 'entropy']:
            if key in logs:
                metrics[key] = float(logs[key])
        
        # Additional computed metrics
        if 'kl_div' in logs and 'policy_loss' in logs:
            metrics['kl_policy_ratio'] = logs['kl_div'] / (logs['policy_loss'] + 1e-8)
        
        if 'value_loss' in logs and 'policy_loss' in logs:
            metrics['value_policy_ratio'] = logs['value_loss'] / (logs['policy_loss'] + 1e-8)
        
        return metrics
    
    def _detect_anomalies(self, step: int, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect anomalies in PPO training metrics."""
        anomalies = []
        
        for metric, value in metrics.items():
            if metric not in self.metric_history or len(self.metric_history[metric]) < 10:
                continue
            
            history = self.metric_history[metric]
            mean_val = np.mean(history)
            std_val = np.std(history)
            
            # Z-score based anomaly detection
            if std_val > 0:
                z_score = abs(value - mean_val) / std_val
                if z_score > self.anomaly_threshold:
                    anomalies.append({
                        "metric": metric,
                        "value": value,
                        "z_score": z_score,
                        "mean": mean_val,
                        "std": std_val,
                        "severity": "high" if z_score > 5.0 else "medium"
                    })
        
        return anomalies
    
    def _handle_anomalies(self, step: int, anomalies: List[Dict[str, Any]]) -> None:
        """Handle detected anomalies."""
        for anomaly in anomalies:
            if anomaly["severity"] == "high":
                logging.warning(f"CRITICAL ANOMALY at step {step}: {anomaly}")
                # Could trigger automatic checkpoint saving, learning rate adjustment, etc.
            else:
                logging.info(f"Anomaly detected at step {step}: {anomaly}")
    
    def _calculate_epoch_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive epoch statistics."""
        stats = {}
        
        for metric, history in self.metric_history.items():
            if history:
                stats[metric] = {
                    "mean": np.mean(history),
                    "std": np.std(history),
                    "min": np.min(history),
                    "max": np.max(history),
                    "trend": self._calculate_trend(history)
                }
        
        return stats
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a metric."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        result = {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024
        }
        
        if torch.cuda.is_available():
            result["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            result["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return result
    
    def _get_gpu_utilization(self) -> Dict[str, float]:
        """Get GPU utilization metrics."""
        if not torch.cuda.is_available():
            return {}
        
        return {
            "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0,
            "memory_utilization": torch.cuda.memory_utilization() if hasattr(torch.cuda, 'memory_utilization') else 0.0
        }
    
    def _save_epoch_summary(self, epoch: int, stats: Dict[str, Any]) -> None:
        """Save epoch summary to file."""
        summary_path = os.path.join(self.log_dir, f"epoch_{epoch}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "statistics": stats
            }, f, indent=2)


class CheckpointAnalyzer:
    """Analyzes model checkpoints for health monitoring."""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        self.logger = JSONLLogger(os.path.join(log_dir, "checkpoint_analysis.jsonl"))
        os.makedirs(log_dir, exist_ok=True)
    
    def analyze_checkpoint(self, 
                          model_path: str, 
                          step: int,
                          reference_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a model checkpoint for health indicators."""
        
        analysis = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "file_size_mb": self._get_file_size(model_path),
            "health_score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Analyze model weights
            weight_analysis = self._analyze_weights(checkpoint)
            analysis.update(weight_analysis)
            
            # Analyze training state
            if 'trainer_state' in checkpoint:
                training_analysis = self._analyze_training_state(checkpoint['trainer_state'])
                analysis.update(training_analysis)
            
            # Compare with reference if provided
            if reference_checkpoint:
                comparison = self._compare_checkpoints(model_path, reference_checkpoint)
                analysis['comparison'] = comparison
            
            # Calculate overall health score
            analysis['health_score'] = self._calculate_health_score(analysis)
            
        except Exception as e:
            analysis['issues'].append(f"Failed to load checkpoint: {str(e)}")
            analysis['health_score'] = 0.0
        
        # Log analysis
        self.logger.log(analysis)
        
        return analysis
    
    def _get_file_size(self, file_path: str) -> float:
        """Get file size in MB."""
        try:
            return os.path.getsize(file_path) / 1024 / 1024
        except:
            return 0.0
    
    def _analyze_weights(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model weights for health indicators."""
        analysis = {
            "weight_stats": {},
            "gradient_stats": {},
            "issues": [],
            "recommendations": []
        }
        
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            
            # Analyze weight statistics
            for name, param in model_state.items():
                if isinstance(param, torch.Tensor):
                    weight_stats = {
                        "mean": float(param.mean()),
                        "std": float(param.std()),
                        "min": float(param.min()),
                        "max": float(param.max()),
                        "norm": float(param.norm())
                    }
                    analysis["weight_stats"][name] = weight_stats
                    
                    # Check for potential issues
                    if abs(weight_stats["mean"]) > 10.0:
                        analysis["issues"].append(f"Large mean weight in {name}: {weight_stats['mean']}")
                    
                    if weight_stats["std"] > 5.0:
                        analysis["issues"].append(f"High weight variance in {name}: {weight_stats['std']}")
                    
                    if weight_stats["norm"] > 100.0:
                        analysis["issues"].append(f"Large weight norm in {name}: {weight_stats['norm']}")
        
        return analysis
    
    def _analyze_training_state(self, trainer_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training state for health indicators."""
        analysis = {
            "training_metrics": {},
            "optimizer_state": {},
            "issues": [],
            "recommendations": []
        }
        
        # Analyze training metrics
        for key, value in trainer_state.items():
            if isinstance(value, (int, float)):
                analysis["training_metrics"][key] = value
        
        # Check for training issues
        if 'global_step' in trainer_state:
            step = trainer_state['global_step']
            if step > 0:
                # Check learning rate
                if 'learning_rate' in trainer_state:
                    lr = trainer_state['learning_rate']
                    if lr < 1e-8:
                        analysis["issues"].append(f"Very low learning rate: {lr}")
                    elif lr > 1e-2:
                        analysis["issues"].append(f"Very high learning rate: {lr}")
        
        return analysis
    
    def _compare_checkpoints(self, current_path: str, reference_path: str) -> Dict[str, Any]:
        """Compare current checkpoint with reference."""
        comparison = {
            "weight_differences": {},
            "overall_drift": 0.0,
            "issues": []
        }
        
        try:
            current = torch.load(current_path, map_location='cpu')
            reference = torch.load(reference_path, map_location='cpu')
            
            if 'model' in current and 'model' in reference:
                current_model = current['model']
                reference_model = reference['model']
                
                total_drift = 0.0
                param_count = 0
                
                for name in current_model:
                    if name in reference_model:
                        current_param = current_model[name]
                        reference_param = reference_model[name]
                        
                        if isinstance(current_param, torch.Tensor) and isinstance(reference_param, torch.Tensor):
                            diff = (current_param - reference_param).norm().item()
                            comparison["weight_differences"][name] = diff
                            total_drift += diff
                            param_count += 1
                
                if param_count > 0:
                    comparison["overall_drift"] = total_drift / param_count
                    
                    if comparison["overall_drift"] > 10.0:
                        comparison["issues"].append("High model drift detected")
        
        except Exception as e:
            comparison["issues"].append(f"Failed to compare checkpoints: {str(e)}")
        
        return comparison
    
    def _calculate_health_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall health score (0-1)."""
        score = 1.0
        
        # Deduct points for issues
        issue_penalty = len(analysis.get("issues", [])) * 0.1
        score -= issue_penalty
        
        # Check weight statistics
        weight_stats = analysis.get("weight_stats", {})
        for name, stats in weight_stats.items():
            if abs(stats["mean"]) > 5.0:
                score -= 0.05
            if stats["std"] > 3.0:
                score -= 0.05
            if stats["norm"] > 50.0:
                score -= 0.05
        
        return max(0.0, min(1.0, score))


class RewardModelIntegrator:
    """Integrates reward model monitoring with RLDK capabilities."""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        self.logger = JSONLLogger(os.path.join(log_dir, "reward_model_integration.jsonl"))
        os.makedirs(log_dir, exist_ok=True)
        
        # Reward model reliability tracking
        self.reward_history = []
        self.reward_variance_history = []
        self.reward_correlation_history = []
    
    def monitor_reward_model(self, 
                           reward_scores: List[float],
                           step: int,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Monitor reward model performance and reliability."""
        
        analysis = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "reward_stats": self._calculate_reward_stats(reward_scores),
            "reliability_metrics": {},
            "anomalies": [],
            "recommendations": []
        }
        
        # Update history
        self.reward_history.extend(reward_scores)
        if len(self.reward_history) > 1000:  # Keep last 1000 rewards
            self.reward_history = self.reward_history[-1000:]
        
        # Calculate reliability metrics
        analysis["reliability_metrics"] = self._calculate_reliability_metrics()
        
        # Detect anomalies
        analysis["anomalies"] = self._detect_reward_anomalies(reward_scores)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        # Log analysis
        self.logger.log(analysis)
        
        return analysis
    
    def _calculate_reward_stats(self, reward_scores: List[float]) -> Dict[str, float]:
        """Calculate basic reward statistics."""
        if not reward_scores:
            return {}
        
        return {
            "mean": float(np.mean(reward_scores)),
            "std": float(np.std(reward_scores)),
            "min": float(np.min(reward_scores)),
            "max": float(np.max(reward_scores)),
            "median": float(np.median(reward_scores)),
            "q25": float(np.percentile(reward_scores, 25)),
            "q75": float(np.percentile(reward_scores, 75))
        }
    
    def _calculate_reliability_metrics(self) -> Dict[str, float]:
        """Calculate reward model reliability metrics."""
        if len(self.reward_history) < 10:
            return {"insufficient_data": True}
        
        # Calculate variance over time
        recent_rewards = self.reward_history[-100:] if len(self.reward_history) >= 100 else self.reward_history
        variance = np.var(recent_rewards)
        
        # Calculate stability (inverse of coefficient of variation)
        mean_reward = np.mean(recent_rewards)
        stability = 1.0 / (1.0 + np.std(recent_rewards) / (abs(mean_reward) + 1e-8))
        
        # Calculate consistency (how often rewards are within expected range)
        expected_range = (mean_reward - 2 * np.std(recent_rewards), 
                         mean_reward + 2 * np.std(recent_rewards))
        within_range = sum(1 for r in recent_rewards 
                          if expected_range[0] <= r <= expected_range[1])
        consistency = within_range / len(recent_rewards)
        
        return {
            "variance": float(variance),
            "stability": float(stability),
            "consistency": float(consistency),
            "mean_reward": float(mean_reward),
            "sample_size": len(recent_rewards)
        }
    
    def _detect_reward_anomalies(self, reward_scores: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in reward scores."""
        anomalies = []
        
        if len(self.reward_history) < 20:
            return anomalies
        
        # Use historical data for anomaly detection
        historical_mean = np.mean(self.reward_history[-100:])
        historical_std = np.std(self.reward_history[-100:])
        
        for i, score in enumerate(reward_scores):
            z_score = abs(score - historical_mean) / (historical_std + 1e-8)
            
            if z_score > 3.0:
                anomalies.append({
                    "index": i,
                    "score": score,
                    "z_score": z_score,
                    "severity": "high" if z_score > 5.0 else "medium",
                    "expected_range": (historical_mean - 2*historical_std, 
                                     historical_mean + 2*historical_std)
                })
        
        return anomalies
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        reliability = analysis.get("reliability_metrics", {})
        
        if reliability.get("stability", 1.0) < 0.5:
            recommendations.append("Consider reward model calibration - low stability detected")
        
        if reliability.get("consistency", 1.0) < 0.7:
            recommendations.append("Reward model shows inconsistent behavior - review training data")
        
        if len(analysis.get("anomalies", [])) > len(analysis.get("reward_stats", {}).get("mean", 0)) * 0.1:
            recommendations.append("High anomaly rate - investigate reward model training")
        
        return recommendations


class TRLIntegrationManager:
    """Main manager for TRL integration with all monitoring capabilities."""
    
    def __init__(self, config: TRLIntegrationConfig):
        self.config = config
        self.log_dir = config.logging_dir
        
        # Initialize components
        self.ppo_callback = PPOMonitoringCallback(
            anomaly_threshold=config.anomaly_detection_threshold,
            log_dir=os.path.join(self.log_dir, "ppo_monitoring")
        )
        
        self.checkpoint_analyzer = CheckpointAnalyzer(
            log_dir=os.path.join(self.log_dir, "checkpoint_analysis")
        )
        
        self.reward_integrator = RewardModelIntegrator(
            log_dir=os.path.join(self.log_dir, "reward_integration")
        )
        
        # Profiler
        if config.enable_profiling:
            self.profiler = ProfilerManager(log_dir=os.path.join(self.log_dir, "profiling"))
        else:
            self.profiler = None
        
        # TRL trainer (will be initialized when needed)
        self.trl_trainer = None
        
        os.makedirs(self.log_dir, exist_ok=True)
    
    def setup_trl_trainer(self, 
                         model_name: str,
                         dataset_name: str,
                         reward_model_path: Optional[str] = None) -> 'TRLPPOTrainer':
        """Setup TRL PPO trainer with integrated monitoring."""
        
        if not TRL_AVAILABLE:
            raise ImportError("TRL is not available. Install with: pip install trl>=0.7.0")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map=self.config.device
        )
        
        # PPO Configuration
        ppo_config = PPOConfig(
            model_name=model_name,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            ppo_epochs=self.config.ppo_epochs,
            max_grad_norm=self.config.max_grad_norm,
            cliprange=self.config.cliprange,
            cliprange_value=self.config.cliprange_value,
            vf_coef=self.config.vf_coef,
            gamma=self.config.gamma,
            lam=self.config.lam,
            kl_penalty=self.config.kl_penalty,
            target=self.config.target,
            horizon=self.config.horizon,
            init_kl_coef=self.config.init_kl_coef,
            adap_kl_ctrl=self.config.adap_kl_ctrl,
            log_with=self.config.log_with,
            logging_dir=self.config.logging_dir,
            save_freq=self.config.save_freq,
            eval_freq=self.config.eval_freq,
            project_name=self.config.project_name
        )
        
        # Initialize TRL trainer
        self.trl_trainer = TRLPPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=None,  # Will be set automatically
            tokenizer=tokenizer,
            dataset=dataset_name,
            data_collator=None,  # Will use default
            num_shared_layers=None  # Will be determined automatically
        )
        
        return self.trl_trainer
    
    def train_with_monitoring(self, 
                            num_steps: int,
                            save_checkpoints: bool = True) -> Dict[str, Any]:
        """Train with comprehensive monitoring."""
        
        if self.trl_trainer is None:
            raise ValueError("TRL trainer not initialized. Call setup_trl_trainer first.")
        
        training_results = {
            "total_steps": num_steps,
            "start_time": datetime.now().isoformat(),
            "checkpoints_saved": [],
            "anomalies_detected": [],
            "final_metrics": {}
        }
        
        try:
            # Training loop with monitoring
            for step in range(num_steps):
                # Begin step monitoring
                self.ppo_callback.on_step_begin(step, {})
                
                # Training step
                if self.profiler:
                    with self.profiler.profile_step(f"training_step_{step}"):
                        logs = self.trl_trainer.step()
                else:
                    logs = self.trl_trainer.step()
                
                # End step monitoring
                self.ppo_callback.on_step_end(step, logs)
                
                # Checkpoint analysis
                if save_checkpoints and step % self.config.save_freq == 0:
                    checkpoint_path = os.path.join(self.log_dir, f"checkpoint_step_{step}")
                    self.trl_trainer.save_model(checkpoint_path)
                    
                    analysis = self.checkpoint_analyzer.analyze_checkpoint(
                        checkpoint_path, step
                    )
                    
                    if analysis["health_score"] < 0.7:
                        training_results["anomalies_detected"].append({
                            "step": step,
                            "issue": "Low health score",
                            "score": analysis["health_score"]
                        })
                    
                    training_results["checkpoints_saved"].append({
                        "step": step,
                        "path": checkpoint_path,
                        "health_score": analysis["health_score"]
                    })
                
                # Reward model monitoring
                if "rewards" in logs:
                    reward_analysis = self.reward_integrator.monitor_reward_model(
                        logs["rewards"], step
                    )
                    
                    if reward_analysis["anomalies"]:
                        training_results["anomalies_detected"].extend([
                            {"step": step, "issue": "reward_anomaly", "details": anomaly}
                            for anomaly in reward_analysis["anomalies"]
                        ])
        
        except Exception as e:
            training_results["error"] = str(e)
            logging.error(f"Training failed at step {step}: {e}")
        
        finally:
            training_results["end_time"] = datetime.now().isoformat()
            training_results["final_metrics"] = self.ppo_callback._calculate_epoch_statistics()
        
        return training_results
    
    def generate_training_report(self, training_results: Dict[str, Any]) -> str:
        """Generate comprehensive training report."""
        
        report_path = os.path.join(self.log_dir, "training_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# TRL Integration Training Report\n\n")
            f.write(f"**Training Period**: {training_results['start_time']} - {training_results['end_time']}\n")
            f.write(f"**Total Steps**: {training_results['total_steps']}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- Checkpoints Saved: {len(training_results['checkpoints_saved'])}\n")
            f.write(f"- Anomalies Detected: {len(training_results['anomalies_detected'])}\n")
            
            if training_results.get('error'):
                f.write(f"- **Training Error**: {training_results['error']}\n")
            
            f.write("\n")
            
            # Final Metrics
            if training_results.get('final_metrics'):
                f.write("## Final Training Metrics\n\n")
                for metric, stats in training_results['final_metrics'].items():
                    f.write(f"### {metric}\n")
                    f.write(f"- Mean: {stats.get('mean', 'N/A')}\n")
                    f.write(f"- Std: {stats.get('std', 'N/A')}\n")
                    f.write(f"- Trend: {stats.get('trend', 'N/A')}\n\n")
            
            # Anomalies
            if training_results['anomalies_detected']:
                f.write("## Detected Anomalies\n\n")
                for anomaly in training_results['anomalies_detected']:
                    f.write(f"- Step {anomaly['step']}: {anomaly['issue']}\n")
                    if 'details' in anomaly:
                        f.write(f"  - Details: {anomaly['details']}\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. Review anomaly logs for potential issues\n")
            f.write("2. Check checkpoint health scores\n")
            f.write("3. Monitor reward model reliability metrics\n")
            f.write("4. Consider adjusting hyperparameters based on trends\n")
        
        return report_path


# Example usage and integration functions
def create_trl_integration_example():
    """Create an example of TRL integration usage."""
    
    # Configuration
    config = TRLIntegrationConfig(
        model_name="gpt2",
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=1,
        enable_profiling=True,
        enable_checkpoint_analysis=True,
        enable_reward_monitoring=True,
        logging_dir="./trl_integration_logs"
    )
    
    # Initialize integration manager
    integration_manager = TRLIntegrationManager(config)
    
    # Setup TRL trainer
    trainer = integration_manager.setup_trl_trainer(
        model_name="gpt2",
        dataset_name="imdb"  # Example dataset
    )
    
    # Train with monitoring
    results = integration_manager.train_with_monitoring(
        num_steps=100,
        save_checkpoints=True
    )
    
    # Generate report
    report_path = integration_manager.generate_training_report(results)
    print(f"Training report saved to: {report_path}")
    
    return integration_manager, results


if __name__ == "__main__":
    # Run example
    integration_manager, results = create_trl_integration_example()