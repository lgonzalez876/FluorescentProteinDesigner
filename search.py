#!/usr/bin/env python
"""Evolutionary protein design search.

Given a target emission wavelength, discovers novel fluorescent protein sequences
predicted to emit at that wavelength using ESM-C as a mutation prior and the
trained cross-embedding ensemble as a fitness oracle.

Usage:
    python search.py --target 520
    python search.py --target 620 --population-size 80 --max-generations 50 --output results/red/
"""

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import data
import embeddings
import models
from search_config import SearchConfig

# Standard amino acid alphabet (no ambiguous codes)
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA_ALPHABET)


# ── Candidate ────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """A protein sequence candidate in the evolutionary search."""

    sequence: str
    parent_name: str
    generation: int
    mutations: list[str] = field(default_factory=list)

    # Cached scores (filled lazily)
    predicted_emission: float | None = None
    pll_score: float | None = None
    embedding: np.ndarray | None = None  # ESM-C mean embedding
    esm2_augmented: np.ndarray | None = None  # ESM-2 augmented embedding

    # Fitness components
    emission_score: float = 0.0
    plausibility_score: float = 0.0
    novelty_score: float = 0.0
    diversity_score: float = 0.0
    dead_zone_penalty: float = 0.0
    fitness: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "parent_name": self.parent_name,
            "generation": self.generation,
            "mutations": self.mutations,
            "predicted_emission": self.predicted_emission,
            "pll_score": self.pll_score,
            "fitness": self.fitness,
        }


# ── ESM-C Client ─────────────────────────────────────────────────────────────

class ESMCClient:
    """Shared ESM-C 600M client for mutation logits, PLL, and embeddings."""

    def __init__(self):
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig

        self._ESMProtein = ESMProtein
        self._LogitsConfig = LogitsConfig

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "mps":
            try:
                self.client = ESMC.from_pretrained("esmc_600m").to(device)
            except Exception:
                print("  MPS not supported for ESM-C, falling back to CPU")
                device = "cpu"
                self.client = ESMC.from_pretrained("esmc_600m").to(device)
        else:
            self.client = ESMC.from_pretrained("esmc_600m").to(device)

        self.device = device
        print(f"ESM-C 600M loaded on {device}")

    def get_logits_and_embeddings(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass returning logits and mean-pooled embedding.

        Returns:
            logits: (1, seq_len, vocab_size) raw logits including BOS/EOS positions
            mean_embedding: (1, 1152) mean-pooled hidden states (BOS/EOS excluded)
        """
        protein = self._ESMProtein(sequence=sequence)
        tensor = self.client.encode(protein)
        output = self.client.logits(
            tensor, self._LogitsConfig(sequence=True, return_embeddings=True)
        )

        hidden = output.embeddings          # (1, seq_len, 1152)
        logits = output.logits.sequence     # (1, seq_len, vocab_size)

        # Pool hidden states for embedding
        seq_len = hidden.shape[1]
        mask = torch.ones(1, seq_len, device=hidden.device)
        mask = embeddings._mask_bos_eos(mask)
        mean_emb, _ = embeddings._pool_hidden_states(hidden, mask)

        return logits, mean_emb

    def get_logits(self, sequence: str) -> torch.Tensor:
        """Get per-position logits. Returns (1, seq_len, vocab_size)."""
        logits, _ = self.get_logits_and_embeddings(sequence)
        return logits

    def get_pooled_embedding(self, sequence: str) -> np.ndarray:
        """Get mean-pooled embedding. Returns (1, 1152) numpy array."""
        _, mean_emb = self.get_logits_and_embeddings(sequence)
        return mean_emb.cpu().detach().numpy()

    def compute_pll_approximate(self, sequence: str, num_positions: int = 30,
                                rng: np.random.Generator | None = None) -> float:
        """Approximate pseudo-log-likelihood using unmasked logits.

        Samples `num_positions` random residue positions, computes log P(true_aa)
        at each from the unmasked forward pass logits, and returns the sum.
        This requires only 1 forward pass (vs seq_len masked passes for true PLL).
        """
        if rng is None:
            rng = np.random.default_rng()

        logits = self.get_logits(sequence)  # (1, seq_len, vocab_size)
        logits_2d = logits[0]  # (seq_len, vocab_size)

        # Positions 1..seq_len-2 are residue positions (skip BOS=0 and EOS=last)
        seq_len = logits_2d.shape[0]
        residue_positions = list(range(1, seq_len - 1))
        n_sample = min(num_positions, len(residue_positions))
        sampled = rng.choice(residue_positions, size=n_sample, replace=False)

        # Get token IDs for this sequence
        protein = self._ESMProtein(sequence=sequence)
        tokens = self.client.encode(protein)
        token_ids = tokens.sequence  # (seq_len,) — 1D tensor

        log_probs = torch.log_softmax(logits_2d, dim=-1)
        total_ll = 0.0
        for pos in sampled:
            true_token = token_ids[pos].item()
            total_ll += log_probs[pos, true_token].item()

        return total_ll


# ── Batch Predictor ──────────────────────────────────────────────────────────

class ESM2Embedder:
    """Cached ESM-2 model for embedding sequences without reloading each time."""

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading ESM-2 ({embeddings.ESM2_MODEL_NAME}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(embeddings.ESM2_MODEL_NAME)
        self.model = AutoModel.from_pretrained(embeddings.ESM2_MODEL_NAME).to(self.device)
        self.model.eval()
        print(f"  ESM-2 loaded ({sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M params)")

    def embed(self, sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Embed sequences. Returns (mean, augmented) arrays."""
        all_mean, all_aug = [], []

        for i in range(0, len(sequences), embeddings.BATCH_SIZE):
            batch = sequences[i : i + embeddings.BATCH_SIZE]
            tokens = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=embeddings.MAX_LENGTH + 2,
            ).to(self.device)

            with torch.no_grad():
                hidden = self.model(**tokens).last_hidden_state

            mask = embeddings._mask_bos_eos(tokens["attention_mask"])
            mean_emb, augmented = embeddings._pool_hidden_states(hidden, mask)
            all_mean.append(mean_emb.cpu().numpy())
            all_aug.append(augmented.cpu().numpy())

        return np.concatenate(all_mean, axis=0), np.concatenate(all_aug, axis=0)


class BatchPredictor:
    """Wraps the cross-embedding ensemble for batch prediction.

    The cross ensemble has:
      - Ridge component: ESM-2 augmented embeddings
      - MLP component: ESM-C 600M mean embeddings

    Both embedding models are loaded once and reused across all predictions.
    """

    def __init__(self, artifact_path: str, esmc_client: ESMCClient):
        self.artifact = models.load_artifact(Path(artifact_path))
        self.esmc_client = esmc_client
        self.esm2 = ESM2Embedder()

        if self.artifact["model_type"] != "ensemble":
            raise ValueError(f"Expected ensemble artifact, got {self.artifact['model_type']}")

        self.components = self.artifact["components"]
        self.weights = self.artifact["weights"]
        print(f"Loaded ensemble with {len(self.components)} components, weights={self.weights}")

    def predict_single(self, sequence: str, esmc_mean_emb: np.ndarray | None = None) -> float:
        """Predict emission for a single sequence.

        If esmc_mean_emb is provided (1, 1152), reuses it for the MLP component
        instead of re-computing.
        """
        return float(self.predict_batch([sequence], [esmc_mean_emb] if esmc_mean_emb is not None else None)[0])

    def predict_batch(self, sequences: list[str],
                      esmc_mean_embs: list[np.ndarray | None] | None = None) -> np.ndarray:
        """Predict emission for multiple sequences."""
        preds = np.zeros(len(sequences))

        for comp, w in zip(self.components, self.weights):
            emb_name = comp["embedding"]
            pooling = comp["pooling"]

            if emb_name == "esm2":
                # Ridge component: ESM-2 augmented embeddings (cached model)
                _, aug = self.esm2.embed(sequences)
                X = aug

            elif "esmc" in emb_name:
                # MLP component: ESM-C mean embeddings (shared client)
                all_embs = []
                for i, seq in tqdm(list(enumerate(sequences)), desc="  Predicting emission", leave=False):
                    if esmc_mean_embs is not None and esmc_mean_embs[i] is not None:
                        all_embs.append(esmc_mean_embs[i])
                    else:
                        emb = self.esmc_client.get_pooled_embedding(seq)
                        all_embs.append(emb)
                X = np.concatenate(all_embs, axis=0)
            else:
                all_embs = []
                for seq in sequences:
                    m, a = embeddings.embed_single(seq, emb_name)
                    all_embs.append(m if pooling == "mean" else a)
                X = np.concatenate(all_embs, axis=0)

            preds += models.predict(comp, X) * w

        return preds


# ── Chromophore Protection ───────────────────────────────────────────────────

def find_protected_positions(sequence: str) -> set[int]:
    """Detect chromophore-forming residues via motif matching.

    Searches for [TSAC]YG tripeptide (chromophore core) and catalytic residues:
    - Arg at offset +31 (±3 window) from chromophore start
    - Glu at offset +157 (±3 window) from chromophore start

    Returns set of 0-indexed positions that should never be mutated.
    """
    protected = set()
    pattern = re.compile(r"[TSAC]YG")

    for match in pattern.finditer(sequence):
        start = match.start()
        # Protect the chromophore tripeptide itself
        protected.update([start, start + 1, start + 2])

        # Catalytic Arg at ~+31
        for offset in range(28, 35):
            pos = start + offset
            if 0 <= pos < len(sequence) and sequence[pos] == "R":
                protected.add(pos)

        # Catalytic Glu at ~+157
        for offset in range(154, 161):
            pos = start + offset
            if 0 <= pos < len(sequence) and sequence[pos] == "E":
                protected.add(pos)

    return protected


# ── Seed Selection ───────────────────────────────────────────────────────────

def select_ancestor_seeds() -> list[Candidate]:
    """Seed with all natural ancestor proteins (roots of FPBase lineage trees).

    Returns ~37 Candidate objects, one per ancestor with a known sequence.
    These are the wild-type proteins found in nature before directed evolution.
    """
    ancestors = data.get_ancestors()

    seeds = []
    for _, row in ancestors.iterrows():
        seeds.append(Candidate(
            sequence=row["sequence"],
            parent_name=row["name"],
            generation=0,
        ))

    print(f"Selected {len(seeds)} ancestor seed proteins:")
    for s in seeds:
        print(f"  {s.parent_name} (len={len(s.sequence)})")

    return seeds


def select_similarity_seeds(target_nm: float, num_seeds: int) -> list[Candidate]:
    """Pick diverse starting proteins from FPBase within ±50nm of target.

    Uses spectral binning (10nm bins) to ensure diversity across families.
    """
    df = data.get_dataset()
    nearby = df[(df["em_max"] >= target_nm - 50) & (df["em_max"] <= target_nm + 50)].copy()

    if nearby.empty:
        print(f"WARNING: No FPBase proteins within ±50nm of {target_nm}nm.")
        print("  Falling back to closest proteins.")
        nearby = df.copy()
        nearby["dist"] = (nearby["em_max"] - target_nm).abs()
        nearby = nearby.nsmallest(num_seeds * 3, "dist")

    # Spectral binning for diversity
    nearby = nearby.copy()
    nearby["bin"] = ((nearby["em_max"] - target_nm + 50) // 10).astype(int)

    seeds = []
    bins = sorted(nearby["bin"].unique())

    # Round-robin across bins
    bin_groups = {b: nearby[nearby["bin"] == b].sort_values("em_max", key=lambda x: (x - target_nm).abs()) for b in bins}

    bin_idx = 0
    bin_offsets = {b: 0 for b in bins}
    seen_seqs = set()

    while len(seeds) < num_seeds and any(bin_offsets[b] < len(bin_groups[b]) for b in bins):
        b = bins[bin_idx % len(bins)]
        grp = bin_groups[b]

        if bin_offsets[b] < len(grp):
            row = grp.iloc[bin_offsets[b]]
            bin_offsets[b] += 1

            if row["sequence"] not in seen_seqs:
                seen_seqs.add(row["sequence"])
                seeds.append(Candidate(
                    sequence=row["sequence"],
                    parent_name=row["name"],
                    generation=0,
                ))

        bin_idx += 1

    # If still need more, fill from closest
    if len(seeds) < num_seeds:
        remaining = nearby.sort_values("em_max", key=lambda x: (x - target_nm).abs())
        for _, row in remaining.iterrows():
            if row["sequence"] not in seen_seqs:
                seen_seqs.add(row["sequence"])
                seeds.append(Candidate(
                    sequence=row["sequence"],
                    parent_name=row["name"],
                    generation=0,
                ))
                if len(seeds) >= num_seeds:
                    break

    print(f"Selected {len(seeds)} seed proteins:")
    for s in seeds:
        print(f"  {s.parent_name} (len={len(s.sequence)})")

    return seeds


# ── Mutation ─────────────────────────────────────────────────────────────────

def mutate_sequence(candidate: Candidate, esmc: ESMCClient,
                    protected: set[int], generation: int,
                    config: SearchConfig, rng: np.random.Generator) -> Candidate:
    """Generate a mutant using ESM-C MLM-guided mutation.

    1. Single forward pass to get per-position logits
    2. Compute entropy at each mutable position
    3. Select high-entropy positions (above entropy_threshold_percentile)
    4. 70%/30% single-site vs multi-site (2-4 mutations), decaying with generation
    5. Sample new AA from ESM-C distribution at each selected position
    """
    seq = candidate.sequence
    logits_3d = esmc.get_logits(seq)  # (1, seq_len, vocab_size)
    logits_2d = logits_3d[0]          # (seq_len, vocab_size)

    # Residue positions are 1..len(seq) (skip BOS=0, EOS=last)
    # We need to map to 0-indexed sequence positions
    seq_len = len(seq)

    # Compute entropy at each mutable position
    mutable_positions = []
    entropies = []

    for seq_pos in range(seq_len):
        if seq_pos in protected:
            continue

        token_pos = seq_pos + 1  # +1 for BOS token
        if token_pos >= logits_2d.shape[0] - 1:  # skip EOS
            continue

        probs = torch.softmax(logits_2d[token_pos], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        mutable_positions.append(seq_pos)
        entropies.append(entropy)

    if not mutable_positions:
        return None

    entropies = np.array(entropies)
    threshold = np.percentile(entropies, config.entropy_threshold_percentile)
    high_entropy_mask = entropies >= threshold
    high_entropy_positions = [mutable_positions[i] for i in range(len(mutable_positions)) if high_entropy_mask[i]]

    if not high_entropy_positions:
        high_entropy_positions = mutable_positions  # fallback to all mutable

    # Decide single-site vs multi-site
    multi_rate = config.get_mutation_rate(generation)
    is_multi = rng.random() < multi_rate

    if is_multi:
        n_mutations = rng.integers(2, config.max_mutations_per_sequence + 1)
        n_mutations = min(n_mutations, len(high_entropy_positions))
    else:
        n_mutations = 1

    # Select positions to mutate
    chosen_positions = rng.choice(high_entropy_positions, size=n_mutations, replace=False)

    # Build mutant sequence
    seq_list = list(seq)
    mutation_labels = []

    for seq_pos in chosen_positions:
        token_pos = seq_pos + 1  # +1 for BOS
        probs = torch.softmax(logits_2d[token_pos], dim=-1).cpu().numpy()

        # Build distribution over standard AAs only
        # We need the tokenizer's mapping; use ESMProtein to get token IDs
        old_aa = seq_list[seq_pos]

        # Sample from the model's distribution, excluding the current AA
        # We'll create a simple AA-level distribution from the logits
        aa_probs = _extract_aa_probs(probs, esmc, old_aa)

        if aa_probs is None or sum(aa_probs.values()) == 0:
            continue

        aas = list(aa_probs.keys())
        weights = np.array([aa_probs[a] for a in aas])
        weights /= weights.sum()

        new_aa = rng.choice(aas, p=weights)
        mutation_labels.append(f"{old_aa}{seq_pos + 1}{new_aa}")
        seq_list[seq_pos] = new_aa

    if not mutation_labels:
        return None

    new_seq = "".join(seq_list)
    return Candidate(
        sequence=new_seq,
        parent_name=candidate.parent_name,
        generation=generation,
        mutations=candidate.mutations + mutation_labels,
    )


# Token-to-AA mapping cache (populated on first use)
_TOKEN_TO_AA = {}
_AA_TO_TOKEN = {}


def _build_aa_token_mapping(esmc: ESMCClient):
    """Build mapping between AA characters and ESM-C token IDs."""
    global _TOKEN_TO_AA, _AA_TO_TOKEN
    if _TOKEN_TO_AA:
        return

    # Encode a known sequence to discover the token mapping
    test_seq = AA_ALPHABET  # "ACDEFGHIKLMNPQRSTVWY"
    protein = esmc._ESMProtein(sequence=test_seq)
    tokens = esmc.client.encode(protein)
    token_ids = tokens.sequence  # (seq_len,) — already 1D

    # tokens include BOS + 20 AAs + EOS
    for i, aa in enumerate(test_seq):
        tok_id = token_ids[i + 1].item()  # +1 for BOS
        _TOKEN_TO_AA[tok_id] = aa
        _AA_TO_TOKEN[aa] = tok_id


def _extract_aa_probs(probs: np.ndarray, esmc: ESMCClient,
                      exclude_aa: str) -> dict[str, float] | None:
    """Extract per-AA probabilities from a full vocab probability vector."""
    _build_aa_token_mapping(esmc)

    aa_probs = {}
    for aa in AA_ALPHABET:
        if aa == exclude_aa:
            continue
        tok_id = _AA_TO_TOKEN.get(aa)
        if tok_id is not None and tok_id < len(probs):
            aa_probs[aa] = float(probs[tok_id])

    return aa_probs if aa_probs else None


# ── Filter Stack ─────────────────────────────────────────────────────────────

def check_chromophore_intact(candidate: Candidate, protected: set[int],
                             parent_sequence: str) -> bool:
    """Verify that protected positions are unchanged."""
    for pos in protected:
        if pos >= len(candidate.sequence) or pos >= len(parent_sequence):
            return False
        if candidate.sequence[pos] != parent_sequence[pos]:
            return False
    return True


def filter_by_pll(candidates: list[Candidate], esmc: ESMCClient,
                  config: SearchConfig, rng: np.random.Generator) -> list[Candidate]:
    """Reject bottom percentile by approximate PLL."""
    if not candidates:
        return candidates

    for c in tqdm(candidates, desc="  Computing PLL"):
        if c.pll_score is None:
            c.pll_score = esmc.compute_pll_approximate(
                c.sequence, config.pll_sample_positions, rng
            )

    scores = np.array([c.pll_score for c in candidates])
    threshold = np.percentile(scores, config.pll_threshold_percentile)

    survivors = [c for c in candidates if c.pll_score >= threshold]
    return survivors


def filter_by_emission(candidates: list[Candidate], predictor: BatchPredictor,
                       target_nm: float, tolerance: float) -> list[Candidate]:
    """Reject candidates whose predicted emission is too far from target."""
    if not candidates:
        return candidates

    # Predict in batch for candidates without predictions
    needs_pred = [i for i, c in enumerate(candidates) if c.predicted_emission is None]

    if needs_pred:
        seqs = [candidates[i].sequence for i in needs_pred]
        preds = predictor.predict_batch(seqs)
        for idx, pred in zip(needs_pred, preds):
            candidates[idx].predicted_emission = float(pred)

    survivors = [c for c in candidates if abs(c.predicted_emission - target_nm) <= tolerance]
    return survivors


# ── Fitness Scoring ──────────────────────────────────────────────────────────

def compute_fitness(candidates: list[Candidate], target_nm: float,
                    known_embeddings: np.ndarray, dead_zones: list[tuple[np.ndarray, float]],
                    config: SearchConfig):
    """Compute composite fitness scores in-place on candidates."""
    if not candidates:
        return

    # Ensure all have embeddings and predictions
    emb_matrix = np.array([c.embedding.flatten() for c in candidates])  # (N, 1152)

    # Emission score
    for c in candidates:
        c.emission_score = -abs(c.predicted_emission - target_nm) / 100.0

    # Plausibility score (absolute sigmoid on raw PLL)
    for c in candidates:
        c.plausibility_score = 1.0 / (1.0 + np.exp(-0.1 * (c.pll_score - config.pll_sigmoid_midpoint)))

    # Novelty score (min distance to known FP embeddings)
    for i, c in enumerate(candidates):
        dists = np.linalg.norm(known_embeddings - emb_matrix[i], axis=1)
        c.novelty_score = float(np.min(dists))

    # Normalize novelty
    novelty_vals = np.array([c.novelty_score for c in candidates])
    nov_max = novelty_vals.max() if novelty_vals.max() > 0 else 1.0
    for c in candidates:
        c.novelty_score /= nov_max

    # Diversity score (mean distance to other candidates)
    for i, c in enumerate(candidates):
        if len(candidates) > 1:
            others = np.delete(emb_matrix, i, axis=0)
            c.diversity_score = float(np.mean(np.linalg.norm(others - emb_matrix[i], axis=1)))
        else:
            c.diversity_score = 0.0

    # Normalize diversity
    div_vals = np.array([c.diversity_score for c in candidates])
    div_max = div_vals.max() if div_vals.max() > 0 else 1.0
    for c in candidates:
        c.diversity_score /= div_max

    # Dead zone penalty
    for c in candidates:
        c.dead_zone_penalty = 0.0
        for centroid, radius in dead_zones:
            dist = np.linalg.norm(emb_matrix[candidates.index(c)] - centroid)
            penalty = max(0.0, 1.0 - dist / radius) if radius > 0 else 0.0
            c.dead_zone_penalty = max(c.dead_zone_penalty, penalty)

    # Composite fitness
    for c in candidates:
        c.fitness = (
            config.w_emission * c.emission_score
            + config.w_plausibility * c.plausibility_score
            + config.w_novelty * c.novelty_score
            + config.w_diversity * c.diversity_score
            - config.w_dead_zone * c.dead_zone_penalty
        )


# ── Selection ────────────────────────────────────────────────────────────────

def select_population(candidates: list[Candidate], config: SearchConfig) -> list[Candidate]:
    """Diversity-preserving selection via elites + clustering + fitness fill."""
    if len(candidates) <= config.population_size:
        return candidates

    # Sort by fitness descending
    ranked = sorted(candidates, key=lambda c: c.fitness, reverse=True)

    # 1. Keep top elites
    selected = []
    selected_seqs = set()
    for c in ranked[:config.num_elites]:
        if c.sequence not in selected_seqs:
            selected.append(c)
            selected_seqs.add(c.sequence)

    remaining = [c for c in ranked if c.sequence not in selected_seqs]

    if not remaining:
        return selected

    # 2. KMeans clustering by ESM-C embedding
    from sklearn.cluster import KMeans

    emb_matrix = np.array([c.embedding.flatten() for c in remaining])
    n_clusters = min(config.num_clusters, len(remaining))

    if n_clusters < 2:
        # Too few candidates for clustering, just take by fitness
        for c in remaining:
            if len(selected) >= config.population_size:
                break
            if c.sequence not in selected_seqs:
                selected.append(c)
                selected_seqs.add(c.sequence)
        return selected

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(emb_matrix)

    # 3. Take top min_per_cluster from each cluster
    for cluster_id in range(n_clusters):
        cluster_members = [remaining[i] for i in range(len(remaining)) if labels[i] == cluster_id]
        cluster_members.sort(key=lambda c: c.fitness, reverse=True)

        for c in cluster_members[:config.min_per_cluster]:
            if c.sequence not in selected_seqs:
                selected.append(c)
                selected_seqs.add(c.sequence)

    # 4. Fill remaining slots from overall fitness ranking
    for c in ranked:
        if len(selected) >= config.population_size:
            break
        if c.sequence not in selected_seqs:
            selected.append(c)
            selected_seqs.add(c.sequence)

    return selected[:config.population_size]


# ── Dead Zones ───────────────────────────────────────────────────────────────

def update_dead_zones(dead_zones: list[tuple[np.ndarray, float]],
                      population: list[Candidate]) -> list[tuple[np.ndarray, float]]:
    """Add a dead zone around the current population centroid."""
    if not population:
        return dead_zones

    emb_matrix = np.array([c.embedding.flatten() for c in population])
    centroid = emb_matrix.mean(axis=0)

    # Radius = max distance from centroid to any member
    dists = np.linalg.norm(emb_matrix - centroid, axis=1)
    radius = float(np.max(dists)) if len(dists) > 0 else 1.0

    dead_zones.append((centroid, radius))
    return dead_zones


# ── Progress Saving ──────────────────────────────────────────────────────────

def _save_top_candidates(population: list[Candidate], output_dir: Path,
                         config: SearchConfig, target_nm: float,
                         total_generations: int, converged: bool = False):
    """Save current top candidates to results.json and top_candidates.fasta."""
    ranked = sorted(population, key=lambda c: c.fitness, reverse=True)
    top_n = min(20, len(ranked))
    top_candidates = ranked[:top_n]

    results = {
        "config": asdict(config),
        "total_generations": total_generations,
        "converged": converged,
        "top_candidates": [c.to_dict() for c in top_candidates],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "top_candidates.fasta", "w") as f:
        for i, c in enumerate(top_candidates):
            header = (f">candidate_{i+1} parent={c.parent_name} gen={c.generation} "
                      f"emission={c.predicted_emission:.1f}nm fitness={c.fitness:.4f}")
            f.write(header + "\n")
            for j in range(0, len(c.sequence), 80):
                f.write(c.sequence[j:j+80] + "\n")


# ── Main Search Loop ─────────────────────────────────────────────────────────

def run_search(config: SearchConfig) -> dict:
    """Run the evolutionary search. Returns results dict."""
    target_nm = config.target_nm
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Evolutionary Protein Design Search")
    print(f"Target emission: {target_nm} nm")
    print(f"Population: {config.population_size}, Seeds: {config.num_seeds}")
    print(f"Max generations: {config.max_generations}")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(42)

    # 1. Load ESM-C 600M (shared client)
    print("Loading ESM-C 600M...")
    esmc = ESMCClient()

    # 2. Load ensemble predictor
    print(f"Loading ensemble from {config.artifact_path}...")
    predictor = BatchPredictor(config.artifact_path, esmc)

    # 3. Load known FP embeddings for novelty scoring
    print("Loading known FP embeddings...")
    known_data = embeddings.get_embeddings("esmc600m")
    known_embeddings = known_data["mean"]  # (N, 1152)
    print(f"  {known_embeddings.shape[0]} known FP embeddings loaded")

    # 4. Select seeds
    if config.use_similarity_seeds:
        print(f"\nSelecting {config.num_seeds} seeds near {target_nm} nm...")
        population = select_similarity_seeds(target_nm, config.num_seeds)
    else:
        print(f"\nSelecting ancestor seeds...")
        population = select_ancestor_seeds()
        # Ensure population_size is at least as large as the ancestor count
        if len(population) > config.population_size:
            config.population_size = len(population)
            print(f"  Adjusted population_size to {config.population_size}")

    # 5. Compute initial embeddings, PLL, and predictions for seeds
    print("Computing initial embeddings for seeds...")
    for c in population:
        emb = esmc.get_pooled_embedding(c.sequence)
        c.embedding = emb
        c.pll_score = esmc.compute_pll_approximate(c.sequence, config.pll_sample_positions, rng)
        c.predicted_emission = predictor.predict_single(c.sequence, esmc_mean_emb=emb)
        print(f"  {c.parent_name}: pred={c.predicted_emission:.1f}nm, PLL={c.pll_score:.1f}")

    # Detect protected positions per seed sequence (keyed by root parent)
    protected_map = {}
    parent_seqs = {}
    for c in population:
        if c.parent_name not in protected_map:
            protected_map[c.parent_name] = find_protected_positions(c.sequence)
            parent_seqs[c.parent_name] = c.sequence
            print(f"  Protected positions for {c.parent_name}: {len(protected_map[c.parent_name])} residues")

    # Initialize tracking
    dead_zones: list[tuple[np.ndarray, float]] = []
    best_fitness_ever = -float("inf")
    best_emission_ever = None
    stagnation_count = 0
    generation_log = []

    print(f"\nStarting evolution...")

    for gen in range(1, config.max_generations + 1):
        gen_start = time.time()
        tolerance = config.get_emission_tolerance(gen)

        print(f"\n── Generation {gen} ──────────────────────────────────────")
        print(f"  Population: {len(population)}, Tolerance: {tolerance:.1f}nm")

        # 6. Generate mutations
        mutants = []
        for parent in tqdm(population, desc="  Mutating"):
            protected = protected_map.get(parent.parent_name, set())
            for _ in range(config.mutations_per_parent):
                mutant = mutate_sequence(parent, esmc, protected, gen, config, rng)
                if mutant is not None:
                    mutants.append(mutant)

        print(f"  Generated {len(mutants)} mutants")

        # 7. Filter: chromophore check → dedup → PLL → emission prediction
        # Chromophore check
        survivors = []
        for m in mutants:
            parent_seq = parent_seqs.get(m.parent_name, population[0].sequence)
            protected = protected_map.get(m.parent_name, set())
            if check_chromophore_intact(m, protected, parent_seq):
                survivors.append(m)
        print(f"  After chromophore check: {len(survivors)}")

        # Dedup (against existing population and within mutants)
        existing_seqs = {c.sequence for c in population}
        seen = set()
        deduped = []
        for m in survivors:
            if m.sequence not in existing_seqs and m.sequence not in seen:
                seen.add(m.sequence)
                deduped.append(m)
        survivors = deduped
        print(f"  After dedup: {len(survivors)}")

        if not survivors:
            print("  No survivors after filtering, carrying population forward")
            stagnation_count += 1
        else:
            # PLL filter
            survivors = filter_by_pll(survivors, esmc, config, rng)
            print(f"  After PLL filter: {len(survivors)}")

            # Emission filter
            survivors = filter_by_emission(survivors, predictor, target_nm, tolerance)
            print(f"  After emission filter: {len(survivors)}")

        # 8. Embed survivors via ESM-C
        for c in tqdm(survivors, desc="  Embedding survivors"):
            if c.embedding is None:
                c.embedding = esmc.get_pooled_embedding(c.sequence)

        # Combine population + survivors
        all_candidates = population + survivors

        # Ensure all have embeddings and predictions
        for c in all_candidates:
            if c.embedding is None:
                c.embedding = esmc.get_pooled_embedding(c.sequence)
            if c.pll_score is None:
                c.pll_score = esmc.compute_pll_approximate(
                    c.sequence, config.pll_sample_positions, rng
                )
            if c.predicted_emission is None:
                c.predicted_emission = predictor.predict_single(c.sequence, c.embedding)

        # 9. Score fitness
        compute_fitness(all_candidates, target_nm, known_embeddings, dead_zones, config)

        # 10. Cluster-based selection → next population
        population = select_population(all_candidates, config)

        # Track best
        best = max(population, key=lambda c: c.fitness)
        gen_elapsed = time.time() - gen_start

        print(f"  Best: fitness={best.fitness:.4f}, emission={best.predicted_emission:.1f}nm, "
              f"PLL={best.pll_score:.1f} ({gen_elapsed:.1f}s)")

        # Log generation
        gen_entry = {
            "generation": gen,
            "population_size": len(population),
            "best_fitness": round(best.fitness, 4),
            "best_emission": round(best.predicted_emission, 1),
            "emission_tolerance": round(tolerance, 1),
            "dead_zones": len(dead_zones),
            "elapsed_seconds": round(gen_elapsed, 1),
        }
        generation_log.append(gen_entry)

        # Save progress incrementally
        with open(output_dir / "generations.json", "w") as f:
            json.dump(generation_log, f, indent=2)
        _save_top_candidates(population, output_dir, config, target_nm, gen)

        # 11. Check convergence (require min generations + multiple good candidates)
        if gen >= config.min_generations:
            n_within = sum(
                1 for c in population
                if abs(c.predicted_emission - target_nm) <= config.convergence_threshold_nm
            )
            if n_within >= config.convergence_min_candidates:
                print(f"\n  CONVERGED at gen {gen}! {n_within} candidates within "
                      f"{config.convergence_threshold_nm}nm of target {target_nm}nm "
                      f"(best={best.predicted_emission:.1f}nm)")
                break

        # 12. Check stagnation
        if best.fitness > best_fitness_ever:
            best_fitness_ever = best.fitness
            best_emission_ever = best.predicted_emission
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= config.stagnation_limit:
            print(f"  Stagnation detected ({stagnation_count} generations). Adding dead zone + new seeds.")
            dead_zones = update_dead_zones(dead_zones, population)

            # Inject new seeds
            new_seeds = select_similarity_seeds(target_nm, max(2, config.num_seeds // 2))
            for s in new_seeds:
                s.embedding = esmc.get_pooled_embedding(s.sequence)
                s.pll_score = esmc.compute_pll_approximate(s.sequence, config.pll_sample_positions, rng)
                s.predicted_emission = predictor.predict_single(s.sequence, s.embedding)
                if s.parent_name not in protected_map:
                    protected_map[s.parent_name] = find_protected_positions(s.sequence)
                    parent_seqs[s.parent_name] = s.sequence

            population.extend(new_seeds)
            stagnation_count = 0

    # ── Final Output ─────────────────────────────────────────────────────────

    population.sort(key=lambda c: c.fitness, reverse=True)
    top_candidates = population[:min(20, len(population))]

    converged = (gen < config.max_generations and
                 abs(top_candidates[0].predicted_emission - target_nm) <= config.convergence_threshold_nm)

    # Final save (overwrites incremental files with converged flag)
    with open(output_dir / "generations.json", "w") as f:
        json.dump(generation_log, f, indent=2)
    _save_top_candidates(population, output_dir, config, target_nm, gen, converged)

    print(f"\nSaved results to {output_dir / 'results.json'}")
    print(f"Saved generation log to {output_dir / 'generations.json'}")
    print(f"Saved top {len(top_candidates)} candidates to {output_dir / 'top_candidates.fasta'}")

    print(f"\n{'='*60}")
    print(f"Search complete. {gen} generations, {'converged' if converged else 'max generations reached'}.")
    print(f"Best: emission={top_candidates[0].predicted_emission:.1f}nm, "
          f"fitness={top_candidates[0].fitness:.4f}")
    print(f"{'='*60}")

    return {
        "config": asdict(config),
        "total_generations": gen,
        "converged": converged,
        "top_candidates": [c.to_dict() for c in top_candidates],
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evolutionary search for fluorescent protein sequences with target emission wavelength"
    )
    parser.add_argument("--target", type=float, required=True,
                        help="Target emission wavelength in nm (e.g., 520)")
    parser.add_argument("--population-size", type=int, default=None,
                        help=f"Population size (default: {SearchConfig.population_size})")
    parser.add_argument("--num-seeds", type=int, default=None,
                        help=f"Number of seed proteins (default: {SearchConfig.num_seeds})")
    parser.add_argument("--max-generations", type=int, default=None,
                        help=f"Maximum generations (default: {SearchConfig.max_generations})")
    parser.add_argument("--mutations-per-parent", type=int, default=None,
                        help=f"Mutations per parent (default: {SearchConfig.mutations_per_parent})")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Output directory (default: {SearchConfig.output_dir})")
    parser.add_argument("--artifact", type=str, default=None,
                        help=f"Path to ensemble artifact (default: {SearchConfig.artifact_path})")
    parser.add_argument("--similarity-seeds", action="store_true",
                        help="Use old wavelength-proximity seeding instead of ancestor seeding")

    args = parser.parse_args()

    config = SearchConfig(target_nm=args.target)

    if args.population_size is not None:
        config.population_size = args.population_size
    if args.num_seeds is not None:
        config.num_seeds = args.num_seeds
    if args.max_generations is not None:
        config.max_generations = args.max_generations
    if args.mutations_per_parent is not None:
        config.mutations_per_parent = args.mutations_per_parent
    if args.output is not None:
        config.output_dir = args.output
    if args.artifact is not None:
        config.artifact_path = args.artifact
    if args.similarity_seeds:
        config.use_similarity_seeds = True

    run_search(config)


if __name__ == "__main__":
    main()
