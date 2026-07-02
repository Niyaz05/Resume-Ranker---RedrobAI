"""
ranker.py — Hybrid Candidate Ranking Engine (Soft-Penalty Architecture)
========================================================================

DESIGN PHILOSOPHY:
    Instead of SQL-style WHERE clauses that drop candidates (risking 0 results),
    every JD constraint is expressed as a continuous multiplier ∈ (0, 1].
    
    This guarantees:
      • The output ALWAYS has exactly TOP_K candidates
      • Perfect matches surface naturally (all multipliers ≈ 1.0)
      • Weak matches sink to the bottom (multiplied down, never deleted)
      • The "least bad" candidate still occupies Rank 1 if no perfect match exists

SCORING PIPELINE:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  Phase 1: Semantic Retrieval                                        │
    │    • Embed JD text with MiniLM                                      │
    │    • Query Qdrant for top-2000 by cosine similarity                 │
    │    • Only filter: honeypot_flag = false (synthetic garbage)          │
    │                                                                     │
    │  Phase 2: Penalty Multipliers (in Polars)                           │
    │    • Experience penalty      → continuous decay outside [5, 9]       │
    │    • Industry penalty        → pure services crushed to 0.10        │
    │    • Title relevance penalty → non-tech titles crushed to 0.30      │
    │    • Notice period penalty   → smooth sigmoid decay beyond 30d      │
    │    • Location penalty        → non-India + won't relocate           │
    │    • Staleness penalty       → inactive > 180 days                  │
    │    • Skills match bonus      → multiplicative reward for JD skills  │
    │                                                                     │
    │  Phase 3: Behavioral Score (bounded boost)                          │
    │    • GitHub, response rate, freshness, interview completion          │
    │    • Normalized to [0, 1], applied as bounded multiplier            │
    │                                                                     │
    │  Phase 4: Final Score Fusion                                        │
    │    Final = Semantic × Π(penalties) × (1 + Behavioral × 0.25)        │
    │                                                                     │
    │  Phase 5: Output                                                    │
    │    • Sort descending, take top 100                                  │
    │    • Generate reasoning string per candidate                        │
    │    • Write CSV matching submission spec                             │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    python ranker.py --output submission.csv
    python ranker.py --output submission.csv --top-k 100 --retrieve 2000
"""

import argparse
import math
import sys
from datetime import date

import polars as pl
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

from config import (
    BEHAVIORAL_BOOST_CAP,
    BEHAVIORAL_WEIGHTS,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    JD_TEXT,
    NON_TECH_TITLES,
    AI_RELEVANT_TITLES,
    QDRANT_HOST,
    QDRANT_PORT,
    REFERENCE_DATE,
    RETRIEVAL_LIMIT,
    TOP_K_OUTPUT,
)


# =============================================================================
# PHASE 1: SEMANTIC RETRIEVAL
# =============================================================================

def retrieve_candidates(
    client: QdrantClient,
    model: SentenceTransformer,
    limit: int = RETRIEVAL_LIMIT,
) -> list[dict]:
    """
    Embed the JD and retrieve the top-N candidates by cosine similarity.

    We apply only ONE hard filter: honeypot_flag = false.
    Honeypots are synthetic impossible profiles that would poison the ranking.
    Everything else is handled by soft penalties in Phase 2.

    Args:
        client: Qdrant client.
        model:  The same MiniLM model used for ingestion.
        limit:  How many candidates to retrieve (default: 2000).

    Returns:
        List of dicts, each containing:
            - candidate_id: str
            - semantic_score: float (cosine similarity, 0 to 1)
            - All payload fields from ingestion
    """
    print(f"\n[ranker] Phase 1: Semantic Retrieval (top {limit})")
    print(f"[ranker] Embedding JD text ({len(JD_TEXT)} chars)...")

    # Embed the JD
    jd_embedding = model.encode(
        JD_TEXT,
        normalize_embeddings=True,  # Must match ingestion normalisation
    ).tolist()

    # Query Qdrant — ONLY filter out honeypots
    # Everything else enters the scoring pipeline
    print(f"[ranker] Querying Qdrant (limit={limit}, filter=honeypot_flag=false)...")

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=jd_embedding,
        limit=limit,
        with_payload=True,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="honeypot_flag",
                    match=MatchValue(value=False),
                ),
            ]
        ),
    )

    # Flatten into list of dicts
    candidates = []
    for point in results.points:
        record = dict(point.payload)
        record["semantic_score"] = point.score  # Cosine similarity [0, 1]
        candidates.append(record)

    print(f"[ranker] Retrieved {len(candidates)} candidates.")
    print(f"[ranker] Semantic score range: "
          f"{candidates[-1]['semantic_score']:.4f} — "
          f"{candidates[0]['semantic_score']:.4f}")

    return candidates


# =============================================================================
# PHASE 2: PENALTY MULTIPLIERS
# =============================================================================
# Each penalty function maps a candidate field to a multiplier ∈ (0, 1].
#
# MATHEMATICAL DESIGN PRINCIPLES:
#   1. Perfect match → multiplier = 1.0 (no penalty)
#   2. Near-miss → multiplier in [0.5, 0.9] (proportional penalty)
#   3. Clear disqualification → multiplier in [0.05, 0.3] (crushed, but not zero)
#   4. NEVER return exactly 0.0 — that would make the final score 0 regardless
#      of semantic match, which is equivalent to a hard filter.
#
# Why continuous functions instead of step functions?
#   Step functions create cliff edges: a candidate with 4.9 years gets 0.8
#   but 5.0 years gets 1.0.  Continuous functions are smooth and differentiable,
#   which means small changes in input produce small changes in output.
#   This is crucial for ranking stability.

def compute_experience_penalty(years: float) -> float:
    """
    Experience penalty — smooth decay outside the [5, 9] sweet spot.

    The JD says "5-9 years" but also "we'll seriously consider outside the band
    if other signals are strong."  So we use a generous function:

    Mathematical model:
        • Inside [5, 9]:  penalty = 1.0 (perfect fit)
        • Outside [5, 9]: penalty = exp(-((distance from band)^2) / (2 × σ²))
          where σ = 3.0 (controls how fast the penalty decays)

    This is a Gaussian-shaped penalty centered on the band:
        • 4 years → exp(-1/18) ≈ 0.946   (barely penalised)
        • 3 years → exp(-4/18) ≈ 0.800   (noticeable but recoverable)
        • 2 years → exp(-9/18) ≈ 0.607   (significant penalty)
        • 1 year  → exp(-16/18) ≈ 0.411  (heavy penalty)
        • 0 years → exp(-25/18) ≈ 0.249  (near-disqualification)
        • 12 years → exp(-9/18) ≈ 0.800
        • 15 years → exp(-36/18) ≈ 0.135

    Args:
        years: candidate's years_of_experience

    Returns:
        Multiplier ∈ (0, 1]
    """
    BAND_LOW = 5.0
    BAND_HIGH = 9.0
    SIGMA = 3.0  # Standard deviation — controls decay speed

    if BAND_LOW <= years <= BAND_HIGH:
        return 1.0

    # Distance from nearest edge of the band
    distance = min(abs(years - BAND_LOW), abs(years - BAND_HIGH))
    # Gaussian decay
    penalty = math.exp(-(distance ** 2) / (2 * SIGMA ** 2))

    # Floor at 0.05 to avoid zeroing out
    return max(penalty, 0.05)


def compute_industry_penalty(
    is_pure_services: bool,
    has_product_experience: bool,
    services_fraction: float,
) -> float:
    """
    Industry penalty — penalise IT services background, reward product experience.

    The JD explicitly disqualifies:
        "People who have ONLY worked at consulting firms in their ENTIRE career."

    But it also says:
        "If you're currently at one of these companies but have prior
         product-company experience, that's fine."

    Mathematical model:
        • Pure services, zero product experience:  0.10 (crushed, nearly dead)
        • Majority services but SOME product exp:  lerp(0.40, 0.80) based on fraction
        • Minority services:                       lerp(0.80, 1.0)
        • Zero services:                           1.0

    We use linear interpolation (lerp) based on services_fraction ∈ [0, 1]:
        penalty = 1.0 - services_fraction × 0.9   (if no product experience)
        penalty = 1.0 - services_fraction × 0.6   (if has product experience)

    Args:
        is_pure_services:       True if ALL career is at services companies
        has_product_experience: True if ANY role was at a product company
        services_fraction:      Fraction of career months at services companies

    Returns:
        Multiplier ∈ [0.10, 1.0]
    """
    if is_pure_services and not has_product_experience:
        # JD says "we will not move forward" → heavy crush but not zero
        return 0.10

    if not has_product_experience:
        # All services but not in the known-bad list
        return max(0.15, 1.0 - services_fraction * 0.85)

    # Has product experience — scale penalty by services fraction
    # services_fraction=0.0 → 1.0, services_fraction=0.8 → 0.52
    return max(0.40, 1.0 - services_fraction * 0.60)


def compute_title_penalty(title_lower: str) -> float:
    """
    Title relevance penalty — non-technical titles are strongly penalised.

    The JD is for an "AI Engineer" role.  A candidate whose current title is
    "Marketing Manager" or "Accountant" is extremely unlikely to be a fit,
    regardless of what skills they list.

    The JD warns: "A candidate who has all the AI keywords listed as skills
    but whose title is 'Marketing Manager' is not a fit, no matter how
    perfect their skill list looks."

    Mathematical model:
        • AI-relevant title (AI Engineer, ML Engineer, etc.): 1.0
        • Generic tech title (Software Engineer, Backend):    0.85
        • Non-tech title (Accountant, Marketing Manager):     0.30

    Args:
        title_lower: Lowercased current_title

    Returns:
        Multiplier ∈ [0.30, 1.0]
    """
    title_clean = title_lower.strip()

    # Direct AI/ML match — full credit
    if title_clean in AI_RELEVANT_TITLES:
        return 1.0

    # Partial match — check for AI/ML keywords in the title
    ai_keywords = {"ai", "ml", "machine learning", "data scien", "nlp",
                   "deep learning", "research scien", "applied scien"}
    for kw in ai_keywords:
        if kw in title_clean:
            return 0.95

    # Generic tech — some credit
    tech_keywords = {"engineer", "developer", "architect", "programmer",
                     "scientist", "analyst", "technical", "devops", "sre",
                     "platform", "infrastructure"}
    for kw in tech_keywords:
        if kw in title_clean:
            return 0.75

    # Known non-tech — heavy penalty
    if title_clean in NON_TECH_TITLES:
        return 0.30

    # Unknown title — moderate penalty
    return 0.50


def compute_notice_penalty(notice_days: int) -> float:
    """
    Notice period penalty — smooth sigmoid decay beyond 30 days.

    The JD says:
        "Sub-30-day notice preferred. We can buy out up to 30 days.
         30+ day notice candidates are still in scope but the bar gets higher."

    Mathematical model (sigmoid decay):
        penalty = 1 / (1 + exp((notice_days - midpoint) / steepness))

    Where:
        midpoint = 45 days (inflection point, 50% penalty)
        steepness = 15 (controls transition sharpness)

    This produces:
        • 0 days  → 1 / (1 + exp(-3.0)) ≈ 0.953
        • 15 days → 1 / (1 + exp(-2.0)) ≈ 0.881
        • 30 days → 1 / (1 + exp(-1.0)) ≈ 0.731
        • 45 days → 1 / (1 + exp(0.0))  = 0.500  (midpoint)
        • 60 days → 1 / (1 + exp(1.0))  ≈ 0.269
        • 90 days → 1 / (1 + exp(3.0))  ≈ 0.047

    We floor at 0.05 so even 180-day candidates don't get zeroed.

    Args:
        notice_days: candidate's notice_period_days

    Returns:
        Multiplier ∈ [0.05, ~0.95]
    """
    MIDPOINT = 45.0
    STEEPNESS = 15.0

    # Sigmoid decay
    exponent = (notice_days - MIDPOINT) / STEEPNESS
    penalty = 1.0 / (1.0 + math.exp(exponent))

    return max(penalty, 0.05)


def compute_location_penalty(
    country: str,
    willing_to_relocate: bool,
    preferred_work_mode: str,
) -> float:
    """
    Location penalty — India-based or willing to relocate preferred.

    The JD says:
        "Pune/Noida-preferred but flexible."
        "Outside India: case-by-case, but we don't sponsor work visas."

    Mathematical model:
        • India + hybrid/onsite/flexible:              1.0
        • India + remote-only:                         0.90
        • Non-India + willing to relocate + onsite:    0.80
        • Non-India + willing to relocate + remote:    0.70
        • Non-India + not willing to relocate:         0.50

    Args:
        country:              candidate's country
        willing_to_relocate:  boolean
        preferred_work_mode:  "remote" | "hybrid" | "onsite" | "flexible"

    Returns:
        Multiplier ∈ [0.50, 1.0]
    """
    is_india = country.strip().lower() == "india"

    if is_india:
        # In India — slight penalty for remote-only
        if preferred_work_mode == "remote":
            return 0.90
        return 1.0

    # Non-India
    if willing_to_relocate:
        if preferred_work_mode in ("onsite", "hybrid", "flexible"):
            return 0.80
        return 0.70

    # Non-India, not willing to relocate
    return 0.50


def compute_staleness_penalty(days_since_active: int) -> float:
    """
    Staleness penalty — exponential decay for inactive profiles.

    The JD says:
        "A perfect-on-paper candidate who hasn't logged in for 6 months
         and has a 5% recruiter response rate is, for hiring purposes,
         not actually available. Down-weight them appropriately."

    Mathematical model (exponential decay):
        penalty = exp(-days_since_active / τ)

    Where τ = 180 (half-life ≈ 125 days).  This means:
        • 0 days inactive   → exp(0)          = 1.000
        • 30 days inactive  → exp(-0.167)     ≈ 0.846
        • 90 days inactive  → exp(-0.500)     ≈ 0.607
        • 180 days inactive → exp(-1.0)       ≈ 0.368
        • 365 days inactive → exp(-2.028)     ≈ 0.131

    Floor at 0.05.

    Args:
        days_since_active: Days since last_active_date

    Returns:
        Multiplier ∈ [0.05, 1.0]
    """
    TAU = 180.0  # Time constant in days

    penalty = math.exp(-days_since_active / TAU)
    return max(penalty, 0.05)


def compute_skills_bonus(
    must_have_count: int,
    nice_to_have_count: int,
    total_skill_count: int,
) -> float:
    """
    Skills match bonus — reward candidates who have JD-relevant skills.

    Unlike penalties (which crush bad candidates), this is a BONUS multiplier
    that lifts good candidates above the baseline.

    Mathematical model:
        must_have_score = min(must_have_count / 4, 1.0)  → saturates at 4 skills
        nice_to_have_score = min(nice_to_have_count / 6, 1.0) → saturates at 6
        combined = must_have_score × 0.70 + nice_to_have_score × 0.30

        bonus = 1.0 + combined × 0.30  (up to 30% boost)

    So a candidate with 4+ must-have skills and 6+ nice-to-have skills gets
    a 1.30 multiplier, while a candidate with zero relevant skills gets 1.0.

    We also apply a SLIGHT penalty if total_skill_count is suspiciously high
    (> 15 skills) — this catches keyword stuffers.

    Args:
        must_have_count:     JD must-have skills found
        nice_to_have_count:  JD nice-to-have skills found
        total_skill_count:   Total skills listed

    Returns:
        Multiplier ∈ [0.85, 1.30]
    """
    # Normalise skill counts with saturation
    # min(x/cap, 1.0) ensures diminishing returns
    must_score = min(must_have_count / 4.0, 1.0)
    nice_score = min(nice_to_have_count / 6.0, 1.0)

    # Weighted combination (must-have matters more)
    combined = must_score * 0.70 + nice_score * 0.30

    # Convert to bonus multiplier: [1.0, 1.30]
    bonus = 1.0 + combined * 0.30

    # Keyword-stuffer detection: if total skills > 15 and most aren't relevant,
    # the candidate might be padding.  Apply a mild dampening.
    if total_skill_count > 15:
        relevant_ratio = (must_have_count + nice_to_have_count) / total_skill_count
        if relevant_ratio < 0.3:
            # Less than 30% of skills are relevant — mild penalty
            bonus *= 0.90

    return bonus


# =============================================================================
# PHASE 3: BEHAVIORAL SCORE
# =============================================================================

def compute_behavioral_scores(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute normalised behavioral score from Redrob signals using Polars.

    Each signal is normalised to [0, 1] and weighted according to
    BEHAVIORAL_WEIGHTS.  The composite behavioral score is then used
    as a bounded multiplier in the final fusion step.

    NORMALISATION METHODS:
    ┌─────────────────────────┬──────────────────────────────────────────┐
    │ Signal                  │ Method                                   │
    ├─────────────────────────┼──────────────────────────────────────────┤
    │ github_activity_score   │ clamp(score/100, 0, 1); -1 → 0.3        │
    │ recruiter_response_rate │ Already ∈ [0, 1]; use directly           │
    │ freshness (days active) │ exp(-days/180); exponential decay        │
    │ interview_completion    │ Already ∈ [0, 1]; use directly           │
    │ profile_completeness    │ score / 100                              │
    │ verification            │ (email + phone + linkedin) / 3           │
    │ open_to_work            │ 1.0 if true else 0.3                     │
    └─────────────────────────┴──────────────────────────────────────────┘

    Why exponential decay for freshness?
        Linear decay (1 - days/max) has a cliff at max and doesn't capture
        the real-world pattern where recency matters exponentially:
        yesterday vs 2 days ago is less important than 1 month vs 7 months.

    Args:
        df: Polars DataFrame with raw signal columns.

    Returns:
        DataFrame with added columns: each normalized signal + behavioral_score
    """
    w = BEHAVIORAL_WEIGHTS

    df = df.with_columns([
        # ── GitHub activity ─────────────────────────────────────────────
        # -1 means "no GitHub linked" → treat as neutral (0.3), not zero.
        # A candidate without GitHub isn't penalised to death; they just
        # don't get the open-source bonus.
        pl.when(pl.col("github_activity_score") < 0)
          .then(pl.lit(0.3))
          .otherwise(pl.col("github_activity_score").clip(0, 100) / 100.0)
          .alias("github_norm"),

        # ── Recruiter response rate ─────────────────────────────────────
        # Already normalised ∈ [0, 1]. Direct use.
        pl.col("recruiter_response_rate").clip(0.0, 1.0).alias("response_norm"),

        # ── Freshness ───────────────────────────────────────────────────
        # exp(-days/τ) where τ=180.  This gives:
        #   0 days → 1.0, 30d → 0.85, 90d → 0.61, 180d → 0.37, 365d → 0.13
        (-pl.col("days_since_active").cast(pl.Float64) / 180.0)
          .exp()
          .clip(0.0, 1.0)
          .alias("freshness_norm"),

        # ── Interview completion rate ───────────────────────────────────
        # Already ∈ [0, 1]. Direct use.
        pl.col("interview_completion_rate").clip(0.0, 1.0).alias("interview_norm"),

        # ── Profile completeness ────────────────────────────────────────
        # 0-100 scale → normalise to [0, 1]
        (pl.col("profile_completeness_score") / 100.0)
          .clip(0.0, 1.0)
          .alias("completeness_norm"),

        # ── Verification (trust signals) ────────────────────────────────
        # Average of three boolean flags → [0, 0.33, 0.67, 1.0]
        ((pl.col("verified_email").cast(pl.Float64)
          + pl.col("verified_phone").cast(pl.Float64)
          + pl.col("linkedin_connected").cast(pl.Float64)) / 3.0)
          .alias("verification_norm"),

        # ── Open to work ────────────────────────────────────────────────
        # Binary: open = 1.0, not open = 0.3 (not zero — they might still respond)
        pl.when(pl.col("open_to_work_flag") == True)
          .then(pl.lit(1.0))
          .otherwise(pl.lit(0.3))
          .alias("open_to_work_norm"),
    ])

    # ── Weighted composite behavioral score ─────────────────────────────
    # This is the weighted sum of all normalised signals.
    # Result ∈ [0, 1] by construction (each component ∈ [0,1], weights sum to 1).
    df = df.with_columns(
        (
            pl.col("github_norm")       * w["github_activity"]
            + pl.col("response_norm")   * w["recruiter_response"]
            + pl.col("freshness_norm")  * w["freshness"]
            + pl.col("interview_norm")  * w["interview_completion"]
            + pl.col("completeness_norm") * w["profile_completeness"]
            + pl.col("verification_norm") * w["verification"]
            + pl.col("open_to_work_norm") * w["open_to_work"]
        ).alias("behavioral_score")
    )

    return df


# =============================================================================
# PHASE 4: SCORE FUSION
# =============================================================================

def compute_final_scores(candidates: list[dict]) -> pl.DataFrame:
    """
    Fuse semantic scores with penalty multipliers and behavioral boost.

    FORMULA:
        Final Score = Semantic × Π(penalties) × Skills Bonus × (1 + Behavioral × α)

    Where:
        Semantic    = cosine similarity from Qdrant ∈ [0, 1]
        Π(penalties)= product of all penalty multipliers ∈ (0, 1]
        Skills Bonus= multiplicative reward for JD-relevant skills ∈ [0.85, 1.30]
        Behavioral  = weighted behavioral score ∈ [0, 1]
        α           = BEHAVIORAL_BOOST_CAP = 0.25 (max 25% boost)

    WHY MULTIPLICATIVE FUSION?
        Additive fusion (semantic + behavioral) allows a candidate with
        high behavioral score but zero semantic relevance to rank highly.
        Multiplicative fusion ensures that semantic relevance is the
        FOUNDATION — behavioral signals can only amplify or slightly
        attenuate the base score, never replace it.

    WHY BOUNDED MULTIPLIER for behavioral?
        (1 + x × 0.25) maps x ∈ [0,1] to [1.0, 1.25].  This means:
        • A perfect behavioral score gives at most +25% boost
        • A zero behavioral score gives exactly +0% (no penalty either)
        This prevents behavioral signals from overwhelming semantic match.

    Args:
        candidates: List of dicts from Phase 1 retrieval.

    Returns:
        Polars DataFrame sorted by final_score descending, with all
        intermediate scores for transparency.
    """
    print(f"\n[ranker] Phase 2-4: Computing penalties, behavioral scores, and fusion...")

    # ── Apply penalty functions row-by-row ──────────────────────────────
    # These are Python-level functions applied to each candidate dict.
    # For 2000 candidates this is fast (~5ms); Polars vectorisation
    # is used for the behavioral score computation.

    for c in candidates:
        c["experience_penalty"] = compute_experience_penalty(
            c.get("years_of_experience", 0.0)
        )
        c["industry_penalty"] = compute_industry_penalty(
            c.get("is_pure_services", False),
            c.get("has_product_experience", True),
            c.get("services_fraction", 0.0),
        )
        c["title_penalty"] = compute_title_penalty(
            c.get("current_title_lower", "")
        )
        c["notice_penalty"] = compute_notice_penalty(
            c.get("notice_period_days", 90)
        )
        c["location_penalty"] = compute_location_penalty(
            c.get("country", ""),
            c.get("willing_to_relocate", False),
            c.get("preferred_work_mode", "remote"),
        )
        c["staleness_penalty"] = compute_staleness_penalty(
            c.get("days_since_active", 999)
        )
        c["skills_bonus"] = compute_skills_bonus(
            c.get("must_have_skill_count", 0),
            c.get("nice_to_have_count", 0),
            c.get("total_skill_count", 0),
        )

    # ── Convert to Polars DataFrame ─────────────────────────────────────
    df = pl.DataFrame(candidates)

    # ── Compute behavioral scores (vectorised in Polars) ────────────────
    df = compute_behavioral_scores(df)

    # ── Compute combined penalty product ────────────────────────────────
    # Product of all penalty multipliers — each ∈ (0, 1]
    df = df.with_columns(
        (
            pl.col("experience_penalty")
            * pl.col("industry_penalty")
            * pl.col("title_penalty")
            * pl.col("notice_penalty")
            * pl.col("location_penalty")
            * pl.col("staleness_penalty")
        ).alias("penalty_product")
    )

    # ── Final score fusion ──────────────────────────────────────────────
    # Final = Semantic × Penalties × Skills_Bonus × (1 + Behavioral × α)
    df = df.with_columns(
        (
            pl.col("semantic_score")
            * pl.col("penalty_product")
            * pl.col("skills_bonus")
            * (1.0 + pl.col("behavioral_score") * BEHAVIORAL_BOOST_CAP)
        ).alias("final_score")
    )

    # ── Sort descending by final score ──────────────────────────────────
    df = df.sort("final_score", descending=True)

    # ── Print score distribution ────────────────────────────────────────
    print(f"\n[ranker] Score distribution (top 2000 retrieved):")
    print(f"  Semantic  : {df['semantic_score'].mean():.4f} mean, "
          f"{df['semantic_score'].min():.4f}–{df['semantic_score'].max():.4f}")
    print(f"  Behavioral: {df['behavioral_score'].mean():.4f} mean, "
          f"{df['behavioral_score'].min():.4f}–{df['behavioral_score'].max():.4f}")
    print(f"  Penalty   : {df['penalty_product'].mean():.4f} mean, "
          f"{df['penalty_product'].min():.4f}–{df['penalty_product'].max():.4f}")
    print(f"  Final     : {df['final_score'].mean():.4f} mean, "
          f"{df['final_score'].min():.4f}–{df['final_score'].max():.4f}")

    return df


# =============================================================================
# PHASE 5: OUTPUT
# =============================================================================

def generate_reasoning(row: dict) -> str:
    """
    Generate a 1-2 sentence reasoning string for a ranked candidate.

    The submission spec requires a `reasoning` column explaining why each
    candidate was ranked at their position.  This function produces a
    concise, data-driven explanation.

    Args:
        row: Dict with all computed scores and payload fields.

    Returns:
        A 1-2 sentence reasoning string.
    """
    parts = []

    # Title and experience
    title = row.get("current_title", "Unknown")
    yoe = row.get("years_of_experience", 0)
    parts.append(f"{title} with {yoe:.1f} yrs exp")

    # Semantic match quality
    sem = row.get("semantic_score", 0)
    if sem >= 0.6:
        parts.append(f"strong semantic match ({sem:.2f})")
    elif sem >= 0.4:
        parts.append(f"moderate semantic match ({sem:.2f})")
    else:
        parts.append(f"weak semantic match ({sem:.2f})")

    # Key signals
    signals = []
    if row.get("must_have_skill_count", 0) > 0:
        signals.append(f"{row['must_have_skill_count']} core AI skills")
    if row.get("has_product_experience"):
        signals.append("product co. exp")
    if row.get("recruiter_response_rate", 0) >= 0.5:
        signals.append(f"responsive ({row['recruiter_response_rate']:.0%})")
    if row.get("notice_period_days", 999) <= 30:
        signals.append(f"{row['notice_period_days']}d notice")
    if row.get("github_activity_score", -1) >= 50:
        signals.append(f"active GitHub ({row['github_activity_score']:.0f})")

    if signals:
        parts.append("; ".join(signals))

    # Key penalties (if significant)
    penalties = []
    if row.get("industry_penalty", 1) < 0.5:
        penalties.append("pure services background")
    if row.get("title_penalty", 1) < 0.5:
        penalties.append("non-tech title")
    if row.get("staleness_penalty", 1) < 0.5:
        penalties.append("inactive profile")

    if penalties:
        parts.append(f"penalised: {', '.join(penalties)}")

    return ". ".join(parts) + "."


def output_csv(df: pl.DataFrame, output_path: str, top_k: int = TOP_K_OUTPUT) -> None:
    """
    Write the top-K ranked candidates to a submission CSV.

    Format (per submission_spec.md):
        candidate_id,rank,score,reasoning
        CAND_0001234,1,0.9520,"AI Engineer with 6.5 yrs..."
        ...

    Rules enforced:
        • Exactly top_k rows
        • Ranks 1 to top_k (1-indexed)
        • Scores non-increasing (higher rank = higher score)
        • No duplicate candidate_ids

    Args:
        df:          Sorted Polars DataFrame from Phase 4.
        output_path: Path to write the CSV.
        top_k:       Number of candidates to output.
    """
    print(f"\n[ranker] Phase 5: Generating output CSV ({top_k} candidates)...")

    # Take top-K
    top = df.head(top_k)

    # Build output rows
    rows = []
    for rank, row_dict in enumerate(top.iter_rows(named=True), start=1):
        reasoning = generate_reasoning(row_dict)
        score = round(row_dict["final_score"], 4)
        rows.append({
            "candidate_id": row_dict["candidate_id"],
            "rank": rank,
            "score": score,
            "reasoning": reasoning,
        })

    # Write CSV
    output_df = pl.DataFrame(rows)
    output_df.write_csv(output_path)

    print(f"[ranker] ✓ Written {len(rows)} candidates to {output_path}")
    print(f"[ranker]   Rank 1:   {rows[0]['candidate_id']} "
          f"(score={rows[0]['score']:.4f})")
    print(f"[ranker]   Rank {top_k}: {rows[-1]['candidate_id']} "
          f"(score={rows[-1]['score']:.4f})")


# =============================================================================
# DETAILED SCORE REPORT (printed to stdout)
# =============================================================================

def print_top_candidates(df: pl.DataFrame, n: int = 20) -> None:
    """Print a detailed breakdown of the top-N candidates for debugging."""
    print(f"\n{'='*100}")
    print(f"  TOP {n} CANDIDATES — DETAILED SCORE BREAKDOWN")
    print(f"{'='*100}")

    columns_to_show = [
        "candidate_id", "current_title", "years_of_experience",
        "country", "notice_period_days",
        "semantic_score", "experience_penalty", "industry_penalty",
        "title_penalty", "notice_penalty", "location_penalty",
        "staleness_penalty", "skills_bonus", "behavioral_score",
        "penalty_product", "final_score",
    ]

    # Only select columns that exist
    available = [c for c in columns_to_show if c in df.columns]
    top = df.head(n).select(available)

    for rank, row in enumerate(top.iter_rows(named=True), start=1):
        print(f"\n  Rank {rank:>3}: {row['candidate_id']}")
        print(f"    Title:      {row.get('current_title', 'N/A')}")
        print(f"    YOE:        {row.get('years_of_experience', 'N/A'):.1f} years")
        print(f"    Country:    {row.get('country', 'N/A')}")
        print(f"    Notice:     {row.get('notice_period_days', 'N/A')} days")
        print(f"    ─── Scores ───")
        print(f"    Semantic:    {row.get('semantic_score', 0):.4f}")
        print(f"    Exp Pen:     {row.get('experience_penalty', 0):.4f}")
        print(f"    Ind Pen:     {row.get('industry_penalty', 0):.4f}")
        print(f"    Title Pen:   {row.get('title_penalty', 0):.4f}")
        print(f"    Notice Pen:  {row.get('notice_penalty', 0):.4f}")
        print(f"    Loc Pen:     {row.get('location_penalty', 0):.4f}")
        print(f"    Stale Pen:   {row.get('staleness_penalty', 0):.4f}")
        print(f"    Skills Bon:  {row.get('skills_bonus', 0):.4f}")
        print(f"    Behavioral:  {row.get('behavioral_score', 0):.4f}")
        print(f"    Pen Product: {row.get('penalty_product', 0):.4f}")
        print(f"    ─── FINAL:   {row.get('final_score', 0):.4f}")

    print(f"\n{'='*100}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid candidate ranking engine with soft-penalty architecture."
    )
    parser.add_argument(
        "--output", "-o",
        default="submission.csv",
        help="Output CSV path (default: submission.csv).",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=TOP_K_OUTPUT,
        help=f"Number of candidates to output (default: {TOP_K_OUTPUT}).",
    )
    parser.add_argument(
        "--retrieve", "-r",
        type=int,
        default=RETRIEVAL_LIMIT,
        help=f"Qdrant retrieval depth (default: {RETRIEVAL_LIMIT}).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed score breakdown for top candidates.",
    )
    args = parser.parse_args()

    # ── Load model ──────────────────────────────────────────────────────
    print(f"[ranker] Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # ── Connect to Qdrant ───────────────────────────────────────────────
    print(f"[ranker] Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    # Verify collection exists and has points
    try:
        info = client.get_collection(COLLECTION_NAME)
        point_count = info.points_count
        print(f"[ranker] Collection '{COLLECTION_NAME}' has {point_count:,} points.")
        if point_count == 0:
            print("[ranker] ✗ Collection is empty. Run ingest.py first.")
            sys.exit(1)
    except Exception as e:
        print(f"[ranker] ✗ Cannot access collection: {e}")
        sys.exit(1)

    # ── Phase 1: Retrieve ───────────────────────────────────────────────
    candidates = retrieve_candidates(client, model, limit=args.retrieve)

    if not candidates:
        print("[ranker] ✗ No candidates retrieved. Check your data and filters.")
        sys.exit(1)

    # ── Phases 2-4: Score ───────────────────────────────────────────────
    df = compute_final_scores(candidates)

    # ── Detailed output (optional) ──────────────────────────────────────
    print_top_candidates(df, n=20)

    # ── Phase 5: Output CSV ─────────────────────────────────────────────
    output_csv(df, args.output, top_k=args.top_k)

    print(f"\n[ranker] ✓ Done! Submission file: {args.output}")


if __name__ == "__main__":
    main()
