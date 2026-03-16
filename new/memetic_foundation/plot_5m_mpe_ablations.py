import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob

def plot_5m_mpe_ablations(checkpoints_dir="checkpoints"):
    print("Gathering 5M-step MPE simple_tag_v2 ablation results...")
    
    variants = ["baseline", "memory_only", "comm_only", "full"]
    variant_data = {v: [] for v in variants}
    
    for v in variants:
        pattern = os.path.join(checkpoints_dir, f"memfound_{v}_*/*results.json")
        for file in glob.glob(pattern):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    
                # Ensure this is an MPE run with 5M steps
                if data.get("environment", "smacv2") != "mpe":
                    continue
                if data.get("total_steps", 0) < 4_900_000:
                    continue
                    
                steps = data.get("steps_history", [])
                rewards = data.get("rewards_history", [])
                
                if len(steps) > 0 and len(rewards) > 0:
                    variant_data[v].append((steps, rewards))
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
    plt.figure(figsize=(12, 7))
    colors = {"baseline": "gray", "memory_only": "blue", "comm_only": "green", "full": "purple"}
    labels = {"baseline": "Baseline", "memory_only": "Memory Only", "comm_only": "Comm Only", "full": "Full Architecture"}
    
    for v in variants:
        runs = variant_data[v]
        if not runs:
            print(f"No 5M data found for {v} yet.")
            continue
            
        print(f"Plotting {v}: found {len(runs)} 5M-step seeds.")
        steps, rewards = runs[0] # There is only 1 seed for the 5M runs
        
        # Apply a moving average for readability over 5M steps
        window = max(1, len(rewards) // 100)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        smoothed_steps = steps[:len(smoothed)]
        
        plt.plot(smoothed_steps, smoothed, label=f"{labels[v]}", color=colors[v], linewidth=2, alpha=0.9)
        plt.plot(steps, rewards, color=colors[v], linewidth=0.5, alpha=0.1) # raw data underneath
        
    plt.title("Memetic Foundation Ablations: MPE (5 Million Steps)", fontsize=16)
    plt.xlabel("Environment Steps", fontsize=14)
    plt.ylabel("Team Reward", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    
    out_file = "mpe_5m_ablation_results.png"
    plt.savefig(out_file, dpi=300)
    print(f"\nPlot saved successfully to {out_file}")

if __name__ == "__main__":
    plot_5m_mpe_ablations()
