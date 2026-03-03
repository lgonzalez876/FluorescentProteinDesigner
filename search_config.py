"""Configuration dataclass for evolutionary protein design search."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SearchConfig:
    """All hyperparameters for the evolutionary search algorithm."""

    # ── Population ────────────────────────────────────────────────────────────
    population_size: int = 50
    num_seeds: int = 8
    mutations_per_parent: int = 20

    # ── Mutation ──────────────────────────────────────────────────────────────
    multi_site_mutation_fraction: float = 0.3
    max_mutations_per_sequence: int = 4
    entropy_threshold_percentile: float = 70.0
    mutation_rate_decay: float = 0.98

    # ── Filters ───────────────────────────────────────────────────────────────
    pll_threshold_percentile: float = 5.0
    emission_tolerance_nm: float = 20.0
    emission_tolerance_decay: float = 0.90
    emission_tolerance_min_nm: float = 5.0

    # ── Selection ─────────────────────────────────────────────────────────────
    num_clusters: int = 8
    min_per_cluster: int = 3
    num_elites: int = 5

    # ── Fitness weights ───────────────────────────────────────────────────────
    w_emission: float = 1.0
    w_plausibility: float = 0.3
    w_novelty: float = 0.1
    w_diversity: float = 0.2
    w_dead_zone: float = 0.5

    # ── Termination ───────────────────────────────────────────────────────────
    max_generations: int = 100
    min_generations: int = 10
    convergence_threshold_nm: float = 3.0
    convergence_min_candidates: int = 5
    stagnation_limit: int = 15

    # ── Paths ─────────────────────────────────────────────────────────────────
    artifact_path: str = "artifacts/cross/ensemble.json"
    output_dir: str = "results/"

    # ── Seeding ────────────────────────────────────────────────────────────────
    use_similarity_seeds: bool = False  # True = old wavelength-proximity seeding

    # ── PLL ────────────────────────────────────────────────────────────────────
    pll_sample_positions: int = 30
    pll_sigmoid_midpoint: float = -65.0

    # ── Target (set via CLI) ──────────────────────────────────────────────────
    target_nm: float = 520.0

    def get_emission_tolerance(self, generation: int) -> float:
        """Compute emission tolerance for a given generation, with decay and floor."""
        tol = self.emission_tolerance_nm * (self.emission_tolerance_decay ** generation)
        return max(tol, self.emission_tolerance_min_nm)

    def get_mutation_rate(self, generation: int) -> float:
        """Compute multi-site mutation fraction decayed by generation."""
        return self.multi_site_mutation_fraction * (self.mutation_rate_decay ** generation)
