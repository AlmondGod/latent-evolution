import os
import numpy as np
if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dirs = {
    'Baseline': '/Users/almondgod/Repositories/memeplex-capstone/checkpoints/memfound_baseline_20260315_003002',
    'Memory Only': '/Users/almondgod/Repositories/memeplex-capstone/checkpoints/memfound_memory_only_20260315_003005',
    'Comm Only': '/Users/almondgod/Repositories/memeplex-capstone/checkpoints/memfound_comm_only_20260315_003008',
    'Full': '/Users/almondgod/Repositories/memeplex-capstone/checkpoints/memfound_full_20260315_003011'
}

metrics = ["Norms/memory_norm", "Norms/memory_delta_norm", "Norms/message_out_norm"]

for variant, log_dir in log_dirs.items():
    print(f"\n--- {variant} ---")
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events' in f]
    if not event_files:
        print("No event files found.")
        continue
    event_file = max(event_files, key=os.path.getctime)
    ea = EventAccumulator(event_file)
    ea.Reload()
    available_tags = ea.Tags().get('scalars', [])
    for m in metrics:
        if m in available_tags:
            events = ea.Scalars(m)
            if events:
                last_val = events[-1].value
                print(f"{m}: {last_val:.4f}")
            else:
                print(f"{m}: empty")
        else:
            print(f"{m}: not found")
