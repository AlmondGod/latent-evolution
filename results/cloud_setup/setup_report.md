# Cloud Setup Report — Latent Evolution

**Date:** 2026-04-24  
**Machine:** RunPod GPU instance (Docker container)

---

## System Inventory

| Item | Value |
|------|-------|
| **OS** | Linux 6.8.0-107-generic x86_64 (Ubuntu 22.04 container) |
| **GPU** | NVIDIA GeForce RTX 3090 (24 GB VRAM) |
| **CUDA Driver** | 580.126.20 |
| **CUDA Version** | 13.0 |
| **Python** | 3.11.10 (`/usr/bin/python3`) |
| **PyTorch** | 2.4.1+cu124 (pre-installed) |
| **torch.cuda.is_available()** | ✅ `True` |
| **RAM** | 1.0 TiB |
| **Disk (workspace)** | 834T shared (MFS), 20G overlay |

---

## Package Installation

No venv was used; the RunPod image ships with Python 3.11 and PyTorch 2.4.1+cu124 pre-installed.  
All packages were installed globally as root.

### OS packages (apt)
```
git build-essential cmake swig unzip wget curl
python3-venv python3-dev
libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libffi-dev
```

### Python packages (pip)
```
numpy==1.26.4
gymnasium==1.2.3
pettingzoo[mpe]==1.25.0
matplotlib
tqdm
scikit-learn==1.5.1
tensorboard
pygame
vmas
rware          # (was missing, required by rware_wrapper.py import)
smacv2 @ git+https://github.com/oxwhirl/smacv2.git@main
six (upgraded)
hanabi-learning-environment==0.0.4
scikit-build   # (needed to build hanabi with modern CMake)
```

### Hanabi install note
The default `pip install hanabi-learning-environment` fails because the
package's `CMakeLists.txt` requires `cmake_minimum_required(VERSION 2.8)`,
which is rejected by modern CMake (≥3.30). Fix:
```bash
pip install scikit-build
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install hanabi-learning-environment
```

---

## Code Fix Applied

**File:** `new/memetic_foundation/training/mpe_wrapper.py`  
**Issue:** PettingZoo 1.25.0 deprecated `simple_tag_v2` and `simple_spread_v2`; calling `.parallel_env()` raises `DeprecatedEnv`.  
**Fix:** Updated imports from `simple_tag_v2` / `simple_spread_v2` to `simple_tag_v3` / `simple_spread_v3` (aliased to keep internal names stable). The v3 API is compatible; the wrapper already handles both 4-tuple and 5-tuple step returns.

```diff
-from pettingzoo.mpe import simple_tag_v2
-from pettingzoo.mpe import simple_spread_v2
+import warnings
+warnings.filterwarnings("ignore", message=".*pettingzoo.mpe.*moved to.*mpe2.*")
+from pettingzoo.mpe import simple_tag_v3 as simple_tag_v2
+from pettingzoo.mpe import simple_spread_v3 as simple_spread_v2
```

---

## Smoke Test Results

| Test | Result | Notes |
|------|--------|-------|
| **MPE simple_spread_v2** (CPU, 1 ep) | ✅ PASS | `Smoke test passed!` — reward=-796.91, 400 steps |
| **VMAS discovery** (CPU, 1 ep) | ✅ PASS | `Smoke test passed!` — reward=0.50, 400 steps |
| **VMAS transport** (CPU, 1 ep) | ✅ PASS | `Smoke test passed!` — reward=0.00, 400 steps |
| **GPU training smoke** (2000 steps, MPE) | ✅ PASS | Device: **cuda**, 2000 steps in 6.8s, checkpoint saved |
| **SMACv2** | ⏭️ SKIPPED | Package installed; StarCraft II binary not set up |
| **Hanabi install/import** | ✅ PASS | `hanabi_learning_environment.rl_env` imports, env creates and resets |

### GPU Training Details
```
Device: cuda
Variant: full_commnet
Parameters: 191,878
Steps: 2000 in 6.8s
Checkpoint: results/cloud_setup/gpu_smoke/memfound_full_*/memfound_full_latest.pt (2.3 MB)
Tensorboard events and training plot also saved.
```

---

## Verified Imports
```
torch 2.4.1+cu124  cuda True
numpy 1.26.4
gymnasium 1.2.3
pettingzoo 1.25.0
sklearn 1.5.1
vmas OK
hanabi OK
```

---

## Known Issues / Blockers

1. **PettingZoo MPE deprecation warning:** `pettingzoo.mpe` is moving to the `mpe2` standalone package. Current code works with v3 alias but may need a full migration eventually.
2. **`rware` not in requirements.txt:** The `rware_wrapper.py` imports `rware` but it was not listed in dependencies. Added at install time.
3. **No `sudo` on RunPod:** Container runs as root; use direct `apt-get` / `pip` commands.

---

## Recommended Next Steps

1. **Update `requirements.txt`** to include `rware` and pin working versions.
2. **Run full benchmark training** on MPE (simple_spread_v2) and VMAS (discovery, transport) with the 4 ablation variants.
3. **Build Hanabi wrapper** (`new/memetic_foundation/training/hanabi_wrapper.py`) matching the existing env interface.
4. **Set up StarCraft II** binary if SMACv2 benchmarks are needed (see `old/SETUP.md`).
5. **Consider migrating** from `pettingzoo.mpe` to standalone `mpe2` package before PettingZoo drops MPE entirely.
