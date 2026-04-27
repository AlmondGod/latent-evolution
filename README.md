# Aligning Latent Evolution in Multi-Agent Communication

**Anand Majmudar** — University of Pennsylvania — `majanand@seas.upenn.edu`  
*Advisor: Victor Preciado* — `preciado@seas.upenn.edu`

Research code for memetic (culture-like) adaptation in multi-agent communication: **Reward-Aligned Latent Evolution (RALE)** and **ALEC** (Attention-guided Latent Evolution in Communication), with baselines and experiments on MPE, VMAS, and SMACv2.

---

## Overview

Multi-agent RL usually improves coordination through per-agent weight updates. This project studies **selection over latent communication strategies**—*memetic evolution*—in addition to (or instead of) pure gradient steps.

- **Formalism:** Communicated latents are treated as transmissible strategies whose frequencies evolve under a learned **transmission** mechanism. **RALE** measures how well patterns that *spread* are also the ones that **improve joint return** (alignment between replicative and host “fitness”).
- **ALEC** instantiates the idea with a **frozen** MAPPO backbone, an evolvable **communication adapter** (low-rank Q/K/V/O deltas + **memetic state** for decentralized latent memory), and an outer loop (**OpenAI-ES** or **MAPPO** on the adapter) depending on the experiment.
- **Environments (among others):** MPE2 Predator-Prey, VMAS Discovery/Transport, **StarCraft II (SMACv2)**. Experiments compare ALEC to MAPPO under matched budgets; **RALE** analyses probe alignment between $F$ and $R$ on logged latents (see `results/phase2_rale_alignment/` for paper-ready LaTeX snippets).

*Abstract (shortened):* We formalize communicated latents as transmissible strategies, define the RALE objective, build ALEC with ES/RL over the attention graph and GRU-based latent memory, and show competitiveness with MAPPO. RALE measurements indicate that higher-reward latents can become more probable under both ES- and RL-trained ALEC.

**Code (related capstone repository):** [github.com/AlmondGod/memeplex-capstone](https://github.com/AlmondGod/memeplex-capstone)  
**This repository:** [github.com/AlmondGod/latent-evolution](https://github.com/AlmondGod/latent-evolution)

---

## Installation

**Requirements:** Python 3.10+ recommended, PyTorch, and the dependencies in `requirements.txt`. Some experiments need **StarCraft II** and **SMACv2** for full runs.

```bash
cd latent-evolution
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

**SMACv2 (optional, for StarCraft experiments):** Install the SC2 client (e.g. Linux 4.10+) and set `SC2PATH`. Install `smac` / `pysc2` per the [SMAC](https://github.com/oxwhirl/smac) project. Large maps may require extra disk space. Local helper scripts (e.g. `install_sc2.sh`) may exist in the environment you use; adapt paths to your machine.

**VMAS / MPE:** Pull versions compatible with the pinned `torch`/gym stack in `requirements.txt`; minor version conflicts are resolved by the versions already specified where possible.

---

## Repository layout

| Path | Description |
|------|-------------|
| **`new/memetic_foundation/`** | Main implementation: `modules/memetic_adapter.py` (`MemeticCommAdapter`, `MemeticStateCell`, `LowRankDelta`), frozen backbone `models/frozen_attention_hu_actor.py`, `training/` (PPO, ES, env wrappers, SMAC utilities). |
| **`new/memetic_foundation/scripts/`** | Runners: `run_memetic_selection_phase2.py` (ALEC+ES), `run_memetic_rl_phase2.py` (MAPPO adapter baseline), `phase2_rale_probe.py`, `analyze_rale_alignment.py`, SMAC/phase-1/3-arm launchers, RALE paper export (`export_rale_paper_tables.py`, `make_rale_paper_figure.py`). |
| **`new/memetic_foundation/docs/`** | Design notes (memetic selection, phase-2 budget protocol). |
| **`old/`** | Earlier experiments and baselines (e.g. MPE distillation, legacy layouts); keep for history—prefer `new/` for ALEC/SMAC work. |
| **`results/`** | Generated runs: metrics, RALE LaTeX under `results/phase2_rale_alignment/` (tables and section snippets). Large artifacts (`*.npz`, `*.csv`, `*.png`, most logs) are **gitignored**; regenerate with the scripts or download separately. |
| **`checkpoints/`** | Trained weights (local only; not committed). |
| **`scripts/`** | Small top-level shell helpers. |
| **`read_tb.py`**, **`plots/`** | Misc. plotting / TensorBoard helpers. |

Path for imports in scripts: set `PYTHONPATH` to the repo root (e.g. `export PYTHONPATH=/path/to/latent-evolution`).

---

## Citation

If you use this code or the RALE/ALEC formalism, please cite the thesis (replace year/venue as appropriate when published):

```bibtex
@mastersthesis{majmudar2026aligning,
  author  = {Majmudar, Anand},
  title   = {Aligning Latent Evolution in Multi-Agent Communication},
  school  = {University of Pennsylvania},
  year    = {2026},
  address = {Philadelphia, PA},
  type    = {Undergraduate / Masters Thesis},
  note    = {Advised by Victor Preciado. Code: https://github.com/AlmondGod/memeplex-capstone},
}
```

**Plain text (abstract snippet for reference lists):**  
*Multi-Agent Reinforcement Learning (MARL) typically improves coordination through individual weight updates. Taking inspiration from cultural evolution in human populations, we investigate a different axis of adaptation: communicating multi-agent systems changing through selection over latent communication strategies, known abstractly as “memetic evolution.” We formalize communicated latents as transmissible strategies whose frequencies evolve under a learned transmission operator, and we define the central objective of Reward-Aligned Latent Evolution (RALE): a measure of how much the communication patterns that spread are also those that improve joint return. We instantiate this framework in Attention-guided Latent Evolution in Communication (ALEC), which applies an optimizer (ES or MARL) over the attention-based communication graph and introduces decentralized latent memory via GRUs. The proposed method is competitive with MAPPO in MPE2 Predator-Prey, VMAS Discovery/Transport, and StarCraft MultiAgent Challenge v2. We measure alignment between latent replication success and downstream reward in ALEC using RALE and see that, with both ES- and RL-based ALEC, higher-reward latents become more probable.*

---

## Quick pointers

- **Phase-2 ALEC (ES):** `new/memetic_foundation/scripts/run_memetic_selection_phase2.py`
- **Phase-2 MAPPO adapter:** `new/memetic_foundation/scripts/run_memetic_rl_phase2.py`
- **RALE probe + analysis:** `phase2_rale_probe.py`, `analyze_rale_alignment.py`
- **Paper (LaTeX) snippets:** `results/phase2_rale_alignment/paper_section_rale_streamlined.tex`, `tab_alec_formal_mapping.tex`

---

## Legacy: LA-IPPO (Method I)

The original README in git history described **Latent-Aligned IPPO** (representation distillation between agents) and MPE `simple_spread` experiments. That line of work lives in older scripts at the repo root (`train_method_i.py`, etc.) and in `old/`. The **current** research emphasis for this thesis is **memetic foundation + ALEC + RALE** under `new/memetic_foundation/`.
