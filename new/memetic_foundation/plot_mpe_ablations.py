import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob

def plot_mpe_ablations(checkpoints_dir="checkpoints"):
    print("Gathering MPE simple_tag_v2 ablation results...")
    
    # We expect 4 variants: baseline, memory_only, comm_only, full
    # 3 seeds each.
    
    variants = ["baseline", "memory_only", "comm_only", "full"]
    variant_data = {v: [] for v in variants}
    
    for v in variants:
        pattern = os.path.join(checkpoints_dir, f"memfound_{v}_*/*results.json")
        for file in glob.glob(pattern):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    
                # Ensure this is an MPE run (not an old SMAC run)
                env_type = data.get("environment", "smacv2")
                if env_type != "mpe":
                    continue
                    
                steps = data.get("steps_history", [])
                rewards = data.get("rewards_history", [])
                
                if len(steps) > 0 and len(rewards) > 0:
                    variant_data[v].append((steps, rewards))
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
    # Interpolate to a common step grid
    common_steps = np.linspace(0, 1_000_000, 100)
    
    plt.figure(figsize=(10, 6))
    colors = {"baseline": "gray", "memory_only": "blue", "comm_only": "green", "full": "purple"}
    labels = {"baseline": "Baseline (None)", "memory_only": "Memory Only", "comm_only": "Comm Only", "full": "Full (Mem+Comm)"}
    
    for v in variants:
        runs = variant_data[v]
        if not runs:
            print(f"No MPE data found for {v}")
            continue
            
        print(f"Plotting {v}: grouped {len(runs)} seeds.")
        interp_rewards = []
        for steps, rewards in runs:
            # Interpolate to the common grid
            interp_r = np.interp(common_steps, steps, rewards)
            interp_rewards.append(interp_r)
            
        interp_rewards = np.array(interp_rewards)
        mean_r = np.mean(interp_rewards, axis=0)
        std_r = np.std(interp_rewards, axis=0) / np.sqrt(len(runs)) # Standard error
        
        plt.plot(common_steps, mean_r, label=f"{labels[v]} (n={len(runs)})", color=colors[v], linewidth=2)
        plt.fill_between(common_steps, mean_r - std_r, mean_r + std_r, color=colors[v], alpha=0.2)
        
    plt.title("Memetic Foundation Ablations: MPE (simple_tag_v2)", fontsize=14)
    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Team Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    out_file = "mpe_ablation_results.png"
    plt.savefig(out_file, dpi=300)
    print(f"\nPlot saved successfully to {out_file}")

if __name__ == "__main__":
    plot_mpe_ablations()
