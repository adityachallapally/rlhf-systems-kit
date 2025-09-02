"""
Report generation utilities for RLHF profiling.

Provides utilities to aggregate profiling results into CSV files
and generate simple visualizations.
"""

import os
import csv
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np


class ProfilerReport:
    """Generate profiling reports from collected data."""
    
    def __init__(self, output_dir: str = "profiles"):
        self.output_dir = output_dir
        self.summary_dir = os.path.join(output_dir, "summary")
        self.figures_dir = os.path.join(output_dir, "figures")
        
        # Ensure output directories exist
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def generate_summary_csv(self, results: List[Dict[str, Any]], filename: str = "summary.csv"):
        """Generate summary CSV with stage breakdown.
        
        Args:
            results: List of profiling results from hooks
            filename: Output CSV filename
        """
        if not results:
            print("Warning: No profiling results to generate summary")
            return
        
        # Group results by stage
        stage_results = {}
        for result in results:
            stage = result.get('stage', 'unknown')
            if stage not in stage_results:
                stage_results[stage] = []
            stage_results[stage].append(result)
        
        # Aggregate results by stage
        summary_data = []
        for stage, stage_data in stage_results.items():
            # Calculate averages for numeric fields
            wall_times = [r.get('wall_time_ms', 0) for r in stage_data]
            cpu_mems = [r.get('cpu_mem_mb', 0) for r in stage_data]
            cuda_peaks = [r.get('cuda_mem_mb_peak', 0) for r in stage_data]
            
            # Get metadata from first result (should be consistent within stage)
            first_result = stage_data[0]
            
            # Get the maximum global_step for this stage to show the latest step
            max_global_step = max(r.get('global_step', 0) for r in stage_data)
            
            summary_row = {
                'stage': stage,
                'wall_time_ms': np.mean(wall_times),
                'cpu_mem_mb': np.mean(cpu_mems),
                'cuda_mem_mb_peak': np.mean(cuda_peaks),
                'tokens': first_result.get('tokens_processed'),
                'batch_size': first_result.get('batch_size'),
                'seq_len': first_result.get('seq_len'),
                'step': max_global_step,
                'count': len(stage_data)
            }
            summary_data.append(summary_row)
        
        # Sort by wall time (descending)
        summary_data.sort(key=lambda x: x['wall_time_ms'], reverse=True)
        
        # Write to CSV
        csv_path = os.path.join(self.summary_dir, filename)
        with open(csv_path, 'w', newline='') as f:
            if summary_data:
                fieldnames = summary_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)
        
        print(f"Generated summary CSV: {csv_path}")
        return csv_path
    
    def generate_stage_breakdown_figure(self, results: List[Dict[str, Any]], 
                                      filename: str = "stage_breakdown.png"):
        """Generate a horizontal bar chart showing stage breakdown.
        
        Args:
            results: List of profiling results from hooks
            filename: Output figure filename
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available, skipping figure generation")
            return None
        
        if not results:
            print("Warning: No profiling results to generate figure")
            return None
        
        # Group results by stage
        stage_results = {}
        for result in results:
            stage = result.get('stage', 'unknown')
            if stage not in stage_results:
                stage_results[stage] = []
            stage_results[stage].append(result)
        
        # Calculate total time per stage
        stage_times = {}
        for stage, stage_data in stage_results.items():
            total_time = sum(r.get('wall_time_ms', 0) for r in stage_data)
            stage_times[stage] = total_time
        
        # Sort stages by time (descending)
        sorted_stages = sorted(stage_times.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_stages:
            print("Warning: No valid stage data for figure generation")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stages = [s[0] for s in sorted_stages]
        times = [s[1] for s in sorted_stages]
        
        # Create horizontal bar chart
        bars = ax.barh(stages, times, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, time_val) in enumerate(zip(bars, times)):
            ax.text(bar.get_width() + max(times) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{time_val:.1f}ms', va='center', ha='left', fontsize=10)
        
        # Customize chart
        ax.set_xlabel('Wall Time (ms)')
        ax.set_title('RLHF Training Stage Breakdown')
        ax.grid(axis='x', alpha=0.3)
        
        # Add total time annotation
        total_time = sum(times)
        ax.text(0.02, 0.98, f'Total: {total_time:.1f}ms', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.figures_dir, filename)
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Generated stage breakdown figure: {fig_path}")
        return fig_path
    
    def print_console_summary(self, results: List[Dict[str, Any]]):
        """Print a formatted console summary of profiling results.
        
        Args:
            results: List of profiling results from hooks
        """
        if not results:
            print("No profiling results to display")
            return
        
        # Group results by stage
        stage_results = {}
        for result in results:
            stage = result.get('stage', 'unknown')
            if stage not in stage_results:
                stage_results[stage] = []
            stage_results[stage].append(result)
        
        # Calculate totals
        total_wall_time = 0
        total_cpu_mem = 0
        max_cuda_peak = 0
        
        # Prepare stage data
        stage_data = []
        for stage, stage_results_list in stage_results.items():
            wall_times = [r.get('wall_time_ms', 0) for r in stage_results_list]
            cpu_mems = [r.get('cpu_mem_mb', 0) for r in stage_results_list]
            cuda_peaks = [r.get('cuda_mem_mb_peak', 0) for r in stage_results_list]
            
            avg_wall_time = np.mean(wall_times)
            avg_cpu_mem = np.mean(cpu_mems)
            avg_cuda_peak = np.mean(cuda_peaks)
            
            stage_data.append({
                'stage': stage,
                'wall_time_ms': avg_wall_time,
                'cpu_mb': avg_cpu_mem,
                'cuda_peak_mb': avg_cuda_peak
            })
            
            total_wall_time += avg_wall_time
            total_cpu_mem += avg_cpu_mem
            max_cuda_peak = max(max_cuda_peak, avg_cuda_peak)
        
        # Sort by wall time
        stage_data.sort(key=lambda x: x['wall_time_ms'], reverse=True)
        
        # Print header
        print("\n" + "="*60)
        print("RLHF Training Profiling Summary")
        print("="*60)
        print(f"{'stage':<20} {'wall_ms':<10} {'cpu_mb':<10} {'cuda_peak_mb':<15}")
        print("-"*60)
        
        # Print stage data
        for stage in stage_data:
            print(f"{stage['stage']:<20} {stage['wall_time_ms']:<10.1f} "
                  f"{stage['cpu_mb']:<10.1f} {stage['cuda_peak_mb']:<15.1f}")
        
        # Print totals
        print("-"*60)
        print(f"{'total':<20} {total_wall_time:<10.1f} "
              f"{total_cpu_mem:<10.1f} {max_cuda_peak:<15.1f}")
        print("="*60)
        
        # Print additional info
        print(f"\nTotal stages profiled: {len(stage_data)}")
        print(f"Total wall time: {total_wall_time:.1f}ms ({total_wall_time/1000:.3f}s)")
        if max_cuda_peak > 0:
            print(f"Peak CUDA memory: {max_cuda_peak:.1f}MB")
        print()
    
    def generate_full_report(self, results: List[Dict[str, Any]], 
                           summary_filename: str = "summary.csv",
                           figure_filename: str = "stage_breakdown.png",
                           json_filename: str = "profiler_summary.json"):
        """Generate a complete profiling report.
        
        Args:
            results: List of profiling results from hooks
            summary_filename: Output CSV filename
            figure_filename: Output figure filename
            json_filename: Output JSON summary filename
        
        Returns:
            Dict containing paths to generated files
        """
        report_files = {}
        
        # Generate summary CSV
        summary_path = self.generate_summary_csv(results, summary_filename)
        if summary_path:
            report_files['summary_csv'] = summary_path
        
        # Generate profiler summary JSON
        json_path = self.generate_profiler_summary_json(results, json_filename)
        if json_path:
            report_files['profiler_summary_json'] = json_path
        
        # Generate figure
        figure_path = self.generate_stage_breakdown_figure(results, figure_filename)
        if figure_path:
            report_files['figure'] = figure_path
        
        # Print console summary
        self.print_console_summary(results)
        
        return report_files
    
    def generate_profiler_summary_json(self, results: List[Dict[str, Any]], 
                                     filename: str = "profiler_summary.json"):
        """Generate profiler summary JSON with total_steps derived from recorded data.
        
        Args:
            results: List of profiling results from hooks
            filename: Output JSON filename
        
        Returns:
            Path to generated JSON file
        """
        if not results:
            print("Warning: No profiling results to generate summary JSON")
            return None
        
        # Calculate total_steps from the recorded data
        # Get unique global_step values from results
        global_steps = set()
        for result in results:
            global_step = result.get('global_step', 0)
            if global_step is not None:
                global_steps.add(global_step)
        
        # total_steps should be the maximum global_step value
        total_steps = max(global_steps) if global_steps else 0
        
        # Count total number of profiler records
        total_records = len(results)
        
        # Group results by stage
        stage_results = {}
        for result in results:
            stage = result.get('stage', 'unknown')
            if stage not in stage_results:
                stage_results[stage] = []
            stage_results[stage].append(result)
        
        # Calculate stage statistics
        stage_stats = {}
        for stage, stage_data in stage_results.items():
            wall_times = [r.get('wall_time_ms', 0) for r in stage_data]
            cpu_mems = [r.get('cpu_mem_mb', 0) for r in stage_data]
            cuda_peaks = [r.get('cuda_mem_mb_peak', 0) for r in stage_data]
            
            stage_stats[stage] = {
                'count': len(stage_data),
                'avg_wall_time_ms': np.mean(wall_times) if wall_times else 0,
                'total_wall_time_ms': np.sum(wall_times) if wall_times else 0,
                'avg_cpu_mem_mb': np.mean(cpu_mems) if cpu_mems else 0,
                'avg_cuda_mem_mb_peak': np.mean(cuda_peaks) if cuda_peaks else 0
            }
        
        # Create summary data
        summary_data = {
            'total_steps': total_steps,
            'step_count': total_steps,  # Alias for compatibility
            'total_records': total_records,
            'stages': stage_stats,
            'generated_at': datetime.now().isoformat(),
            'metadata': {
                'description': 'RLHF training profiler summary',
                'total_steps_derived_from': 'recorded profiler data global_step values'
            }
        }
        
        # Write to JSON
        json_path = os.path.join(self.summary_dir, filename)
        with open(json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Generated profiler summary JSON: {json_path}")
        print(f"  Total steps: {total_steps}")
        print(f"  Total records: {total_records}")
        print(f"  Stages: {list(stage_stats.keys())}")
        
        return json_path


def create_profiler_report(output_dir: str = "profiles") -> ProfilerReport:
    """Create a profiler report instance.
    
    Args:
        output_dir: Directory to save reports
    
    Returns:
        ProfilerReport instance
    """
    return ProfilerReport(output_dir)