"""Score evolutionary search candidates for fluorescence likelihood using ESM3.

Evaluates candidates on 4 axes:
  1. Secondary structure composition (ESM3 SS prediction)
  2. Chromophore-critical residue conservation (alignment to avGFP)
  3. Fold confidence (ESM3 pTM/pLDDT)
  4. Homology to known fluorescent proteins

Usage:
  python scripts/score_candidates.py --input results/full_test/
  python scripts/score_candidates.py --input results/full_test/ --self-test
  python scripts/score_candidates.py --input results/full_test/ --top-n 5
  python scripts/score_candidates.py --setup  # check HuggingFace auth + download model
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow importing project modules from repo root
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── avGFP reference (UniProt P42212, 238 aa) ────────────────────────────────
AVGFP_SEQUENCE = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN"
    "RIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHY"
    "QQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

# EGFP for self-test (avGFP + F64L, S65T)
EGFP_SEQUENCE = (
    "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    "VTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN"
    "RIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHY"
    "QQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

# Insulin B chain for negative self-test
INSULIN_B = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"

# Score weights
WEIGHTS = {
    "ss": 0.25,
    "chromophore": 0.35,
    "ptm": 0.25,
    "homology": 0.15,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def gaussian_score(value: float, ideal: float, sigma: float) -> float:
    """Gaussian similarity: 1.0 at ideal, decays with sigma."""
    return math.exp(-0.5 * ((value - ideal) / sigma) ** 2)


def detect_device(requested: str = "auto") -> str:
    """Select compute device: cuda > mps > cpu."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_model():
    """Check HuggingFace auth and ensure ESM3 model is downloadable."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"HuggingFace authenticated as: {user.get('name', user.get('fullname', 'unknown'))}")
        return True
    except Exception:
        print("Not logged in to HuggingFace.")
        print()
        print("To use ESM3, you need to:")
        print("  1. Accept the license at https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1")
        print("  2. Run: huggingface-cli login")
        print()
        return False


def load_esm3(device: str, use_forge: bool = False, forge_token: str | None = None,
              forge_model: str | None = None):
    """Load ESM3 model, with MPS→CPU fallback."""
    if use_forge:
        from esm.sdk.forge import ESM3ForgeInferenceClient
        token = forge_token or os.environ.get("ESM_FORGE_TOKEN") or os.environ.get("EVOLUTIONARY_SCALE_API_KEY", "")
        model_name = forge_model or "esm3-open-2024-03"
        print(f"Connecting to ESM3 Forge ({model_name})...")
        return ESM3ForgeInferenceClient(model=model_name, token=token), device

    from esm.models.esm3 import ESM3
    print(f"Loading ESM3-open on {device}...")
    try:
        model = ESM3.from_pretrained("esm3-open").to(device)
        return model, device
    except Exception as e:
        if device == "mps":
            print(f"  MPS failed ({e}), falling back to CPU")
            model = ESM3.from_pretrained("esm3-open").to("cpu")
            return model, "cpu"
        raise


# ── Score Components ─────────────────────────────────────────────────────────

def score_secondary_structure(model, sequence: str, device: str) -> dict:
    """Score SS composition against GFP ideals (beta-barrel)."""
    from esm.sdk.api import ESMProtein, GenerationConfig

    protein = ESMProtein(sequence=sequence)
    config = GenerationConfig(track="secondary_structure", num_steps=8)
    result = model.generate(protein, config)

    ss_string = result.secondary_structure or ""
    total = len(ss_string) if ss_string else 1

    beta_frac = ss_string.count("E") / total
    helix_frac = ss_string.count("H") / total

    # Count contiguous beta strand segments
    strand_count = 0
    in_strand = False
    for c in ss_string:
        if c == "E" and not in_strand:
            strand_count += 1
            in_strand = True
        elif c != "E":
            in_strand = False

    s_beta = gaussian_score(beta_frac, 0.50, 0.10)
    s_helix = gaussian_score(helix_frac, 0.12, 0.05)
    s_strands = gaussian_score(strand_count, 11, 2)
    score = (s_beta + s_helix + s_strands) / 3.0

    return {
        "score": round(score, 4),
        "beta_fraction": round(beta_frac, 3),
        "helix_fraction": round(helix_frac, 3),
        "strand_count": strand_count,
        "ss_string": ss_string,
    }


def score_chromophore(sequence: str) -> dict:
    """Check chromophore-critical residues via alignment to avGFP."""
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import align_optimal, SubstitutionMatrix

    seq_cand = ProteinSequence(sequence)
    seq_ref = ProteinSequence(AVGFP_SEQUENCE)
    matrix = SubstitutionMatrix.std_protein_matrix()

    alignments = align_optimal(
        seq_ref, seq_cand, matrix,
        gap_penalty=(-10, -1),
        terminal_penalty=False,
    )
    alignment = alignments[0]
    trace = alignment.trace  # (N, 2) array: ref_idx, cand_idx

    # Build ref_pos -> cand_pos mapping (0-indexed)
    ref_to_cand = {}
    for ref_idx, cand_idx in trace:
        if ref_idx >= 0 and cand_idx >= 0:
            ref_to_cand[ref_idx] = cand_idx

    checks = {
        "G67": {"ref_pos": 66, "test": lambda aa: aa == "G", "weight": 0.35},
        "aromatic66": {"ref_pos": 65, "test": lambda aa: aa in "FYWH", "weight": 0.30},
        "R96": {"ref_pos": 95, "test": lambda aa: aa == "R", "weight": 0.20},
        "E222": {"ref_pos": 221, "test": lambda aa: aa == "E", "weight": 0.15},
    }

    total = 0.0
    details = {}
    for name, chk in checks.items():
        cand_pos = ref_to_cand.get(chk["ref_pos"])
        if cand_pos is not None and cand_pos < len(sequence):
            aa = sequence[cand_pos]
            passed = chk["test"](aa)
            details[name] = {"candidate_aa": aa, "passed": passed}
            if passed:
                total += chk["weight"]
        else:
            details[name] = {"candidate_aa": "-", "passed": False}

    return {"score": round(total, 4), "checks": details}


def score_ptm(model, sequence: str, device: str) -> dict:
    """Fold confidence via ESM3 structure prediction (pTM + pLDDT)."""
    from esm.sdk.api import ESMProtein, GenerationConfig

    protein = ESMProtein(sequence=sequence)
    config = GenerationConfig(track="structure", num_steps=1, temperature=0.0)
    result = model.generate(protein, config)

    ptm_raw = getattr(result, "ptm", None)
    if ptm_raw is None:
        ptm = 0.0
    else:
        ptm = float(ptm_raw)
    ptm = max(0.0, min(1.0, ptm))

    plddt = None
    if hasattr(result, "plddt") and result.plddt is not None:
        plddt_vals = result.plddt
        if hasattr(plddt_vals, "float"):
            plddt_vals = plddt_vals.float()
        plddt = float(torch.mean(torch.as_tensor(plddt_vals)))

    return {"score": round(ptm, 4), "ptm": round(ptm, 4), "mean_plddt": round(plddt, 2) if plddt else None}


def score_homology(sequence: str, known_fps: list[str]) -> dict:
    """Max sequence identity to known fluorescent proteins."""
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import align_optimal, SubstitutionMatrix

    matrix = SubstitutionMatrix.std_protein_matrix()
    seq_cand = ProteinSequence(sequence)
    cand_len = len(sequence)

    best_identity = 0.0
    best_name = ""

    for i, fp_seq in enumerate(known_fps):
        # Skip if length differs > 50%
        fp_len = len(fp_seq)
        if abs(cand_len - fp_len) / max(cand_len, fp_len) > 0.50:
            continue

        seq_fp = ProteinSequence(fp_seq)
        alignments = align_optimal(
            seq_cand, seq_fp, matrix,
            gap_penalty=(-10, -1),
            terminal_penalty=False,
        )
        trace = alignments[0].trace
        matches = sum(1 for ri, ci in trace if ri >= 0 and ci >= 0 and sequence[ri] == fp_seq[ci])
        aligned_len = sum(1 for ri, ci in trace if ri >= 0 and ci >= 0)
        identity = matches / aligned_len if aligned_len > 0 else 0.0

        if identity > best_identity:
            best_identity = identity

    # Tiered scoring
    if best_identity >= 0.70:
        score = 1.0
    elif best_identity >= 0.50:
        score = 0.8
    elif best_identity >= 0.30:
        score = 0.5
    else:
        score = 0.2

    return {"score": score, "best_identity": round(best_identity, 4)}


# ── Main Scoring Pipeline ───────────────────────────────────────────────────

def score_candidate(model, sequence: str, device: str, known_fp_seqs: list[str]) -> dict:
    """Compute all 4 score components for a single candidate."""
    scores = {}

    # 1. Secondary structure
    try:
        scores["ss"] = score_secondary_structure(model, sequence, device)
    except Exception as e:
        print(f"    [WARN] SS scoring failed: {e}")
        scores["ss"] = {"score": 0.0, "error": str(e)}

    # 2. Chromophore (no ESM3 needed)
    try:
        scores["chromophore"] = score_chromophore(sequence)
    except Exception as e:
        print(f"    [WARN] Chromophore scoring failed: {e}")
        scores["chromophore"] = {"score": 0.0, "error": str(e)}

    # 3. pTM
    try:
        scores["ptm"] = score_ptm(model, sequence, device)
    except Exception as e:
        print(f"    [WARN] pTM scoring failed: {e}")
        scores["ptm"] = {"score": 0.0, "error": str(e)}

    # 4. Homology (no ESM3 needed)
    try:
        scores["homology"] = score_homology(sequence, known_fp_seqs)
    except Exception as e:
        print(f"    [WARN] Homology scoring failed: {e}")
        scores["homology"] = {"score": 0.0, "error": str(e)}

    # Composite
    fl = sum(WEIGHTS[k] * scores[k]["score"] for k in WEIGHTS)
    scores["fluorescence_likelihood"] = round(fl, 4)

    return scores


def run_self_test(model, device: str, known_fp_seqs: list[str]) -> bool:
    """Score EGFP (should be high) and insulin B (should be low)."""
    print("\n=== Self-Test ===\n")
    all_pass = True

    # Positive: EGFP
    print("Scoring EGFP (positive control)...")
    egfp_scores = score_candidate(model, EGFP_SEQUENCE, device, known_fp_seqs)
    fl_egfp = egfp_scores["fluorescence_likelihood"]
    status = "PASS" if fl_egfp > 0.7 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  EGFP fluorescence_likelihood = {fl_egfp:.4f}  [{status}] (expect > 0.7)")
    for k in WEIGHTS:
        print(f"    {k}: {egfp_scores[k]['score']:.4f}")

    # Negative: insulin B chain
    print("\nScoring insulin B chain (negative control)...")
    ins_scores = score_candidate(model, INSULIN_B, device, known_fp_seqs)
    fl_ins = ins_scores["fluorescence_likelihood"]
    status = "PASS" if fl_ins < 0.3 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  Insulin B fluorescence_likelihood = {fl_ins:.4f}  [{status}] (expect < 0.3)")
    for k in WEIGHTS:
        print(f"    {k}: {ins_scores[k]['score']:.4f}")

    print(f"\nSelf-test: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return all_pass


def print_summary_table(scored_candidates: list[dict]) -> None:
    """Print a formatted summary table to stdout."""
    header = f"{'ID':<28} {'FL':>6} {'SS':>6} {'Chrom':>6} {'pTM':>6} {'Homo':>6} {'Em(nm)':>7} {'Gen':>4}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for c in scored_candidates:
        s = c["scores"]
        print(
            f"{c['id']:<28} "
            f"{s['fluorescence_likelihood']:>6.3f} "
            f"{s['ss']['score']:>6.3f} "
            f"{s['chromophore']['score']:>6.3f} "
            f"{s['ptm']['score']:>6.3f} "
            f"{s['homology']['score']:>6.3f} "
            f"{c.get('predicted_emission', 0):>7.1f} "
            f"{c.get('generation', '?'):>4}"
        )
    print("=" * len(header))


def _explain_candidate(rank: int, c: dict) -> str:
    """Generate a human-readable explanation for a candidate's ranking."""
    s = c["scores"]
    fl = s["fluorescence_likelihood"]
    lines = [
        f"#{rank}  {c['id']}  —  Fluorescence Likelihood: {fl:.4f}",
        f"    Parent: {c['parent_name']}  |  Generation: {c['generation']}  |  Predicted emission: {c.get('predicted_emission', 0):.1f} nm",
        f"    Mutations ({len(c.get('mutations', []))}): {', '.join(c.get('mutations', [])) or 'none'}",
        f"    PLL: {c.get('pll_score', 0):.2f}  |  Search fitness: {c.get('fitness', 0):.4f}",
        "",
    ]

    # SS
    ss = s["ss"]
    if "error" in ss:
        lines.append(f"    SS (w=0.25):         {ss['score']:.3f}  — FAILED: {ss['error']}")
    else:
        lines.append(f"    SS (w=0.25):         {ss['score']:.3f}  — beta={ss['beta_fraction']:.2f} (ideal 0.50), helix={ss['helix_fraction']:.2f} (ideal 0.12), strands={ss['strand_count']} (ideal 11)")

    # Chromophore
    chrom = s["chromophore"]
    if "error" in chrom:
        lines.append(f"    Chromophore (w=0.35): {chrom['score']:.3f}  — FAILED: {chrom['error']}")
    else:
        checks = chrom.get("checks", {})
        passed = [name for name, info in checks.items() if info["passed"]]
        failed = [f"{name}(got {info['candidate_aa']})" for name, info in checks.items() if not info["passed"]]
        lines.append(f"    Chromophore (w=0.35): {chrom['score']:.3f}  — passed: {', '.join(passed) or 'none'}; failed: {', '.join(failed) or 'none'}")

    # pTM
    ptm = s["ptm"]
    if "error" in ptm:
        lines.append(f"    pTM (w=0.25):        {ptm['score']:.3f}  — FAILED: {ptm['error']}")
    else:
        plddt_str = f", mean pLDDT={ptm['mean_plddt']:.1f}" if ptm.get("mean_plddt") else ""
        lines.append(f"    pTM (w=0.25):        {ptm['score']:.3f}  — fold confidence={ptm['ptm']:.3f}{plddt_str}")

    # Homology
    homo = s["homology"]
    if "error" in homo:
        lines.append(f"    Homology (w=0.15):   {homo['score']:.3f}  — FAILED: {homo['error']}")
    else:
        lines.append(f"    Homology (w=0.15):   {homo['score']:.3f}  — best identity to known FP: {homo['best_identity']:.1%}")

    # Overall assessment
    strengths = []
    weaknesses = []
    for component, label in [("chromophore", "Chromophore"), ("ss", "SS"), ("ptm", "Fold"), ("homology", "Homology")]:
        val = s[component]["score"]
        if val >= 0.8:
            strengths.append(label)
        elif val <= 0.3:
            weaknesses.append(label)
    if strengths:
        lines.append(f"    Strengths: {', '.join(strengths)}")
    if weaknesses:
        lines.append(f"    Weaknesses: {', '.join(weaknesses)}")

    lines.append("")
    return "\n".join(lines)


def write_report(scored: list[dict], output_dir: Path, timestamp: str) -> tuple[Path, Path]:
    """Write full ranked report and top-10 report."""
    # Full report
    report_path = output_dir / "scores_report.txt"
    lines = [
        "Fluorescence Likelihood Scoring Report",
        f"Generated: {timestamp}",
        f"Candidates: {len(scored)}",
        f"Weights: SS={WEIGHTS['ss']}, Chromophore={WEIGHTS['chromophore']}, pTM={WEIGHTS['ptm']}, Homology={WEIGHTS['homology']}",
        "",
        "=" * 90,
        "",
    ]
    for rank, c in enumerate(scored, 1):
        lines.append(_explain_candidate(rank, c))

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    # Top-10 JSON
    top10_path = output_dir / "top10.json"
    top10_data = {
        "metadata": {
            "description": "Top 10 candidates by fluorescence likelihood",
            "timestamp": timestamp,
            "weights": WEIGHTS,
        },
        "candidates": scored[:10],
    }
    with open(top10_path, "w") as f:
        json.dump(top10_data, f, indent=2)

    return report_path, top10_path


def main():
    parser = argparse.ArgumentParser(description="Score FP candidates for fluorescence likelihood")
    parser.add_argument("--input", default=None, help="Path to results.json or directory containing it")
    parser.add_argument("--output", default=None, help="Output JSON path (default: <input_dir>/scores.json)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--top-n", type=int, default=None, help="Only show top N candidates")
    parser.add_argument("--self-test", action="store_true", help="Run EGFP/insulin self-test")
    parser.add_argument("--setup", action="store_true", help="Check HuggingFace auth and download model, then exit")
    parser.add_argument("--use-forge", action="store_true", help="Use ESM3 Forge API")
    parser.add_argument("--forge-token", default=None, help="Forge API token")
    parser.add_argument("--forge-model", default=None, help="Forge model name")
    args = parser.parse_args()

    # --setup: just check auth and download model
    if args.setup:
        if not ensure_model():
            sys.exit(1)
        print("Auth OK. Loading model to verify download...")
        device = detect_device(args.device)
        load_esm3(device, args.use_forge, args.forge_token, args.forge_model)
        print("Model ready.")
        sys.exit(0)

    if not args.input:
        parser.error("--input is required (unless using --setup)")

    input_path = Path(args.input)
    # Accept a directory — look for results.json inside
    if input_path.is_dir():
        input_path = input_path / "results.json"
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.parent / "scores.json"

    # Load candidates
    with open(input_path) as f:
        data = json.load(f)
    candidates = data.get("top_candidates", [])
    print(f"Loaded {len(candidates)} candidates from {input_path}")

    # Device & model
    device = detect_device(args.device)
    model, device = load_esm3(device, args.use_forge, args.forge_token, args.forge_model)

    # Load known FP sequences for homology scoring
    print("Loading known fluorescent proteins...")
    import data as fp_data
    fp_df = fp_data.get_dataset()
    known_fp_seqs = fp_df["sequence"].tolist()
    print(f"  {len(known_fp_seqs)} known FPs loaded")

    # Self-test
    if args.self_test:
        passed = run_self_test(model, device, known_fp_seqs)
        if not passed:
            print("\nSelf-test failed. Proceeding anyway...\n")

    # Score candidates
    if not candidates:
        print("No candidates to score.")
        sys.exit(0)

    has_tqdm = False
    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(candidates), total=len(candidates), desc="Scoring candidates")
        has_tqdm = True
    except ImportError:
        iterator = enumerate(candidates)
        print("Scoring candidates (install tqdm for progress bar)...")

    scored = []
    for idx, cand in iterator:
        seq = cand["sequence"]
        parent = cand.get("parent_name", "unknown")
        gen = cand.get("generation", 0)
        cand_id = f"{parent}_g{gen}_{idx:03d}"

        if has_tqdm:
            iterator.set_postfix_str(cand_id)

        scores = score_candidate(model, seq, device, known_fp_seqs)

        scored.append({
            "id": cand_id,
            "sequence": seq,
            "parent_name": parent,
            "generation": gen,
            "mutations": cand.get("mutations", []),
            "predicted_emission": cand.get("predicted_emission"),
            "pll_score": cand.get("pll_score"),
            "fitness": cand.get("fitness"),
            "scores": scores,
        })

    # Sort by fluorescence likelihood (descending)
    scored.sort(key=lambda c: c["scores"]["fluorescence_likelihood"], reverse=True)

    # Apply top-n filter for display
    display = scored[:args.top_n] if args.top_n else scored

    # Print summary
    print_summary_table(display)

    # Save output
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    output_data = {
        "metadata": {
            "input_file": str(input_path),
            "num_candidates": len(scored),
            "device": device,
            "weights": WEIGHTS,
            "timestamp": timestamp,
        },
        "candidates": scored,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nScores saved to {output_path}")

    # Write human-readable report + top-10
    report_path, top10_path = write_report(scored, output_path.parent, timestamp)
    print(f"Full report: {report_path}")
    print(f"Top 10: {top10_path}")


if __name__ == "__main__":
    main()
