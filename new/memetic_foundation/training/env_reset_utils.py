# Tiny helper: reset env, only passing seed if the env's reset signature accepts it.
# SMACv2's StarCraftCapabilityEnvWrapper.reset() takes no kwargs.

from __future__ import annotations
import inspect
from typing import Any, Optional


def reset_env(env: Any, seed: Optional[int] = None) -> Any:
    if seed is None:
        return env.reset()
    sig = inspect.signature(env.reset)
    if "seed" in sig.parameters:
        return env.reset(seed=int(seed))
    return env.reset()
