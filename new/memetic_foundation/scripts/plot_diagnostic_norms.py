import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# Patch for numpy 2.0+ incompatibility in tensorboard
if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tb_scalars(event_file, tag):
    """Extract scalar values and steps for a specific tag from a TensorBoard file."""
    try:
        ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
        ea.Reload()
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            return steps, values
    except Exception as e:
        print(f"Error reading {event_file}: {e}")
    return [], []

def plot_diagnostic_norms(log_dirs, output_dir="plots"):
    """
    Plots the diagnostic memory and message norms required to prove
    functional correctness of the Memetic architecture ablations over training.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Average Memory Norm 
    plt.figure(figsize=(8, 5))
    for variant, log_dir in log_dirs.items():
        event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events' in f]
        if not event_files:
            continue
        event_file = max(event_files, key=os.path.getctime)
        steps, vals = extract_tb_scalars(event_file, "Norms/memory_norm")
        if steps:
            plt.plot(steps, vals, label=variant)
            
    plt.title("Average Memory Norm ($||M||$) Over Time", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Matrix Norm", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagnostic_memory_norm.png"), dpi=300)
    plt.close()

    # 2. Average Message Norm 
    plt.figure(figsize=(8, 5))
    for variant, log_dir in log_dirs.items():
        event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events' in f]
        if not event_files:
            continue
            
        event_file = max(event_files, key=os.path.getctime)
        steps, vals = extract_tb_scalars(event_file, "Norms/message_out_norm")
        if steps:
            plt.plot(steps, vals, label=variant)
            
    plt.title("Average Message Out Norm ($||m_{out}||$) Over Time", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Vector Norm", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagnostic_message_norm.png"), dpi=300)
    plt.close()

    # 3. Average Memory Update Norm 
    plt.figure(figsize=(8, 5))
    for variant, log_dir in log_dirs.items():
        event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events' in f]
        if not event_files:
            continue
            
        event_file = max(event_files, key=os.path.getctime)
        steps, vals = extract_tb_scalars(event_file, "Norms/memory_delta_norm")
        if steps:
            plt.plot(steps, vals, label=variant)
            
    plt.title("Average Memory Update Norm ($||M_{t+1} - M_t||$)", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Norm Delta", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagnostic_memory_delta.png"), dpi=300)
    plt.close()

    print(f"Saved diagnostic plots to {output_dir}/")

if __name__ == "__main__":
    # Point mapping to the last run log directories in 
    checkpoints_dir = "/Users/almondgod/Repositories/memeplex-capstone/checkpoints"
    
    # We attempt to auto-detect the latest directory for each variant
    variants = ["baseline", "memory_only", "comm_only", "full"]
    log_dirs = {}
    
    if os.path.exists(checkpoints_dir):
        for v in variants:
            # Find directories matching this variant
            dirs = [d for d in os.listdir(checkpoints_dir) if d.startswith(f"memfound_{v}")]
            if dirs:
                # Sort by name (which has timestamp) and pick the latest
                latest_dir = sorted(dirs)[-1]
                
                # Convert "memory_only" to "Memory Only" for plot labels
                label = v.replace("_", " ").title()
                log_dirs[label] = os.path.join(checkpoints_dir, latest_dir)
                
    if not log_dirs:
        print(f"No checkpoint directories found in {checkpoints_dir}. Cannot plot.")
    else:
        print(f"Plotting diagnostic norms from: {log_dirs}")
        plot_diagnostic_norms(log_dirs)
