from __future__ import annotations

import numpy as np


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """Map fitness values to centered ranks in [-0.5, 0.5]."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    if len(x) > 1:
        ranks /= (len(x) - 1)
    return ranks - 0.5


class OpenAIES:
    """Simple OpenAI-ES style optimizer for small communication genotypes.

    This is intended for the phase-2 post-MAPPO setting where the genotype is a
    compact communication adapter rather than the full network.
    """

    def __init__(
        self,
        theta0: np.ndarray,
        sigma: float = 0.05,
        lr: float = 0.02,
        antithetic: bool = True,
    ) -> None:
        self.theta = np.asarray(theta0, dtype=np.float64).copy()
        self.sigma = float(sigma)
        self.lr = float(lr)
        self.antithetic = antithetic
        self.dim = int(self.theta.size)

    def ask(self, population_size: int) -> tuple[np.ndarray, np.ndarray]:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.antithetic:
            half = population_size // 2
            noise_half = np.random.randn(half, self.dim)
            noises = np.concatenate([noise_half, -noise_half], axis=0)
            if population_size % 2 == 1:
                noises = np.concatenate([noises, np.random.randn(1, self.dim)], axis=0)
        else:
            noises = np.random.randn(population_size, self.dim)
        candidates = self.theta[None, :] + self.sigma * noises
        return candidates, noises

    def tell(self, noises: np.ndarray, fitnesses: np.ndarray) -> dict[str, float]:
        noises = np.asarray(noises, dtype=np.float64)
        fitnesses = np.asarray(fitnesses, dtype=np.float64)
        if noises.ndim != 2 or noises.shape[1] != self.dim:
            raise ValueError("noises must be (population, dim)")
        if fitnesses.shape[0] != noises.shape[0]:
            raise ValueError("fitnesses length must match population size")

        shaped = centered_ranks(fitnesses)
        grad = (shaped[:, None] * noises).mean(axis=0) / max(self.sigma, 1e-8)
        self.theta = self.theta + self.lr * grad

        return {
            "fitness_mean": float(np.mean(fitnesses)),
            "fitness_std": float(np.std(fitnesses)),
            "fitness_max": float(np.max(fitnesses)),
            "grad_norm": float(np.linalg.norm(grad)),
            "theta_norm": float(np.linalg.norm(self.theta)),
        }
