"""
ingest.py — Streaming Candidate Ingestion Pipeline
====================================================

Reads candidate data (JSONL or JSON), computes embeddings, extracts structured
payload fields, detects honeypots, and upserts everything into Qdrant.

Usage:
    # Quick test with 50-candidate sample
    python ingest.py --input data_and_problem_statement/sample_candidates.json --format json

    # Full 100K dataset (streaming, ~15-20 min on M1)
    python ingest.py --input data_and_problem_statement/candidates.jsonl --format jsonl

Architecture:
    ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────┐
    │ JSONL/   │───▶│ Text Builder  │───▶│ MiniLM       │───▶│ Qdrant │
    │ JSON     │    │ + Payload     │    │ Embedder     │    │ Upsert │
    │ Stream   │    │ Extractor     │    │ (batch=64)   │    │        │
    └──────────┘    └──────────────┘    └──────────────┘    └────────┘
                          │
                    ┌─────▼─────┐
                    │ Honeypot  │
                    │ Detector  │
                    └───────────┘

Memory Management:
    • File is streamed line-by-line (never loaded into RAM entirely)
    • Embeddings are computed in batches of 64 (peak ~200 MB for model+batch)
    • Upserted in batches of 100, then the batch is freed
    • Total peak RAM: ~500 MB (model=80MB + batch=200MB + Qdrant client=50MB)
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generator, Optional

import orjson
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    MAX_TEXT_CHARS,
    QDRANT_HOST,
    QDRANT_PORT,
    REFERENCE_DATE,
    UPSERT_BATCH_SIZE,
    JD_HARD_CONSTRAINTS,
)


# =============================================================================
# TEXT BUILDER
# =============================================================================

def build_composite_text(candidate: dict) -> str:
    """
    Build a single text string for embedding from a candidate's profile.

    Strategy:
        We concatenate the profile summary with ALL career history descriptions.
        This captures both the candidate's self-description AND their actual
        work history, which is crucial for detecting "plain-language Tier 5s"
        (candidates who don't use AI buzzwords but have relevant experience).

        We also prepend the headline and current title as they're high-signal,
        short-form identifiers.

    Truncation:
        MiniLM has a 256 word-piece limit (~200 words).  We truncate the
        composite text to MAX_TEXT_CHARS (2000 chars) before feeding it to
        the model, which handles its own tokenisation and truncation.
        This avoids wasting compute on text that will be thrown away.

    Args:
        candidate: A single candidate dict from the JSONL.

    Returns:
        A single string suitable for embedding.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    parts = []

    # ── High-signal short identifiers ───────────────────────────────────
    if profile.get("headline"):
        parts.append(f"Title: {profile['headline']}")
    if profile.get("current_title"):
        parts.append(f"Current Role: {profile['current_title']}")

    # ── Professional summary ────────────────────────────────────────────
    if profile.get("summary"):
        parts.append(f"Summary: {profile['summary']}")

    # ── Career history descriptions (chronological) ─────────────────────
    for job in career:
        role_text = f"{job.get('title', '')} at {job.get('company', '')}"
        if job.get("description"):
            role_text += f": {job['description']}"
        parts.append(role_text)

    # ── Top skills (name + proficiency) ─────────────────────────────────
    # Only include advanced/expert skills to keep text focused
    top_skills = [
        s["name"] for s in skills
        if s.get("proficiency") in ("advanced", "expert")
    ]
    if top_skills:
        parts.append(f"Key Skills: {', '.join(top_skills)}")

    composite = " | ".join(parts)
    return composite[:MAX_TEXT_CHARS]


# =============================================================================
# PAYLOAD EXTRACTOR
# =============================================================================

def extract_payload(candidate: dict) -> dict[str, Any]:
    """
    Extract a flat payload dict from a nested candidate record.

    This flattens the deeply nested JSON structure into top-level keys that
    Qdrant can filter on.  We pre-compute several derived fields:

    Pre-computed Fields:
        is_pure_services:      True if ALL career roles are at IT services companies
        is_pure_academic:      True if ALL career roles are in academia/research
        has_product_experience: True if ANY career role is at a non-services company
        career_industries:     List of unique industries across career history
        total_career_months:   Sum of duration_months across all roles
        skill_names_lower:     List of skill names (lowercased) for matching
        must_have_skill_count: Count of JD must-have skills the candidate has
        nice_to_have_skill_count: Count of JD nice-to-have skills
        honeypot_flag:         True if the profile shows impossible characteristics

    Args:
        candidate: A single candidate dict.

    Returns:
        Flat dict suitable for Qdrant payload.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])

    # ── Basic profile fields ────────────────────────────────────────────
    years_exp = profile.get("years_of_experience", 0.0)
    current_title = profile.get("current_title", "")
    current_industry = profile.get("current_industry", "")
    country = profile.get("country", "")

    # ── Career analysis ─────────────────────────────────────────────────
    services_set = set(JD_HARD_CONSTRAINTS["services_companies"])
    academic_set = set(JD_HARD_CONSTRAINTS["academic_industries"])

    career_industries = []
    career_companies_lower = []
    total_career_months = 0
    services_months = 0
    product_months = 0

    for job in career:
        industry = job.get("industry", "").strip()
        company = job.get("company", "").strip()
        duration = job.get("duration_months", 0)
        career_industries.append(industry)
        career_companies_lower.append(company.lower())
        total_career_months += duration

        # Classify: is this a services-company role?
        if company.lower() in services_set or industry.lower() in (
            "it services", "it consulting", "consulting",
            "information technology & services",
            "information technology",
        ):
            services_months += duration
        else:
            product_months += duration

    unique_industries = list(set(career_industries))

    # Pure services: ALL roles at services companies, zero product months
    is_pure_services = (
        total_career_months > 0
        and product_months == 0
        and services_months == total_career_months
    )

    # Pure academic: ALL roles in academic industries
    is_pure_academic = (
        len(career) > 0
        and all(
            ind.lower() in academic_set
            for ind in career_industries
            if ind
        )
    )

    has_product_experience = product_months > 0

    # Fraction of career in services (for penalty gradient)
    services_fraction = (
        services_months / total_career_months
        if total_career_months > 0 else 0.0
    )

    # ── Skills matching ─────────────────────────────────────────────────
    from config import JD_MUST_HAVE_SKILLS, JD_NICE_TO_HAVE_SKILLS

    skill_names_lower = [s.get("name", "").lower() for s in skills]
    skill_proficiencies = {
        s.get("name", "").lower(): s.get("proficiency", "beginner")
        for s in skills
    }
    skill_durations = {
        s.get("name", "").lower(): s.get("duration_months", 0)
        for s in skills
    }

    # Count how many JD-required skills the candidate has
    must_have_count = sum(
        1 for s in skill_names_lower
        if s in JD_MUST_HAVE_SKILLS
    )
    nice_to_have_count = sum(
        1 for s in skill_names_lower
        if s in JD_NICE_TO_HAVE_SKILLS
    )

    # ── Education tier ──────────────────────────────────────────────────
    best_tier = "unknown"
    tier_rank = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 5}
    for edu in education:
        t = edu.get("tier", "unknown")
        if tier_rank.get(t, 5) < tier_rank.get(best_tier, 5):
            best_tier = t

    # ── Redrob signals (flattened) ──────────────────────────────────────
    notice_period = signals.get("notice_period_days", 180)
    github_score = signals.get("github_activity_score", -1)
    response_rate = signals.get("recruiter_response_rate", 0.0)
    avg_response_hours = signals.get("avg_response_time_hours", 999)
    interview_rate = signals.get("interview_completion_rate", 0.0)
    offer_rate = signals.get("offer_acceptance_rate", -1.0)
    completeness = signals.get("profile_completeness_score", 0.0)
    open_to_work = signals.get("open_to_work_flag", False)
    willing_to_relocate = signals.get("willing_to_relocate", False)
    preferred_work_mode = signals.get("preferred_work_mode", "remote")

    # Date parsing
    last_active_str = signals.get("last_active_date", "2020-01-01")
    signup_str = signals.get("signup_date", "2020-01-01")
    try:
        last_active = date.fromisoformat(last_active_str)
        days_since_active = (REFERENCE_DATE - last_active).days
    except (ValueError, TypeError):
        days_since_active = 999

    try:
        signup_date = date.fromisoformat(signup_str)
        days_since_signup = (REFERENCE_DATE - signup_date).days
    except (ValueError, TypeError):
        days_since_signup = 999

    # ── Honeypot detection ──────────────────────────────────────────────
    honeypot_flag = detect_honeypot(candidate, years_exp, total_career_months, skills)

    # ── Build flat payload ──────────────────────────────────────────────
    return {
        # Profile
        "candidate_id":          candidate.get("candidate_id", ""),
        "anonymized_name":       profile.get("anonymized_name", ""),
        "headline":              profile.get("headline", ""),
        "current_title":         current_title,
        "current_title_lower":   current_title.lower().strip(),
        "current_company":       profile.get("current_company", ""),
        "current_industry":      current_industry,
        "country":               country,
        "location":              profile.get("location", ""),
        "years_of_experience":   years_exp,
        "company_size":          profile.get("current_company_size", ""),

        # Career analysis (pre-computed)
        "career_industries":     unique_industries,
        "total_career_months":   total_career_months,
        "services_fraction":     round(services_fraction, 3),
        "is_pure_services":      is_pure_services,
        "is_pure_academic":      is_pure_academic,
        "has_product_experience": has_product_experience,

        # Skills matching (pre-computed)
        "skill_names_lower":     skill_names_lower,
        "must_have_skill_count": must_have_count,
        "nice_to_have_count":    nice_to_have_count,
        "total_skill_count":     len(skills),

        # Education
        "best_edu_tier":         best_tier,

        # Redrob signals (flattened)
        "notice_period_days":    notice_period,
        "github_activity_score": github_score,
        "recruiter_response_rate": response_rate,
        "avg_response_time_hours": avg_response_hours,
        "interview_completion_rate": interview_rate,
        "offer_acceptance_rate": offer_rate,
        "profile_completeness_score": completeness,
        "open_to_work_flag":     open_to_work,
        "willing_to_relocate":   willing_to_relocate,
        "preferred_work_mode":   preferred_work_mode,
        "days_since_active":     days_since_active,
        "days_since_signup":     days_since_signup,
        "connection_count":      signals.get("connection_count", 0),
        "endorsements_received": signals.get("endorsements_received", 0),
        "profile_views_30d":     signals.get("profile_views_received_30d", 0),
        "search_appearance_30d": signals.get("search_appearance_30d", 0),
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d", 0),
        "applications_submitted_30d": signals.get("applications_submitted_30d", 0),
        "verified_email":        signals.get("verified_email", False),
        "verified_phone":        signals.get("verified_phone", False),
        "linkedin_connected":    signals.get("linkedin_connected", False),
        "salary_min_lpa":        signals.get("expected_salary_range_inr_lpa", {}).get("min", 0),
        "salary_max_lpa":        signals.get("expected_salary_range_inr_lpa", {}).get("max", 0),

        # Honeypot
        "honeypot_flag":         honeypot_flag,
    }


# =============================================================================
# HONEYPOT DETECTOR
# =============================================================================

def detect_honeypot(
    candidate: dict,
    years_exp: float,
    total_career_months: int,
    skills: list[dict],
) -> bool:
    """
    Detect honeypot candidates with subtly impossible profiles.

    The dataset contains ~80 honeypots with characteristics like:
        • 8 years claimed experience at a company founded 3 years ago
        • "Expert" proficiency in 10 skills with 0 months duration each
        • Career history that doesn't add up to claimed experience

    Detection Heuristics:
        1. YOE vs Career Duration mismatch:
           If |years_of_experience - total_career_months/12| > 5 years,
           something is wrong.  Real candidates might have a 1-2 year gap
           (parental leave, education), but a 5+ year discrepancy is synthetic.

        2. Impossible skill proficiency:
           If a candidate lists "expert" proficiency in 5+ skills with
           ≤ 6 months duration each, that's impossible.  You can't become
           an expert in anything in 6 months.

        3. Too many advanced skills with zero endorsements:
           A pattern of advanced/expert skills with 0 endorsements suggests
           synthetic profiles (real experts accumulate some endorsements).

    Args:
        candidate:          Full candidate dict
        years_exp:          profile.years_of_experience
        total_career_months: Sum of career_history[*].duration_months
        skills:             candidate["skills"] list

    Returns:
        True if the candidate is likely a honeypot.
    """
    flags = 0

    # ── Heuristic 1: YOE ↔ career duration mismatch ────────────────────
    # Allow 2 years of slack (education, career breaks, overlapping roles)
    career_years = total_career_months / 12.0
    if abs(years_exp - career_years) > 5.0:
        flags += 1

    # ── Heuristic 2: Expert skills with negligible duration ─────────────
    # Count skills marked as "expert" or "advanced" with ≤ 6 months usage
    suspect_skills = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 0) <= 6
    )
    if suspect_skills >= 5:
        flags += 1

    # ── Heuristic 3: Many advanced skills with zero endorsements ────────
    zero_endorsement_advanced = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("endorsements", 0) == 0
    )
    if zero_endorsement_advanced >= 4:
        flags += 1

    # ── Heuristic 4: Absurd experience claim ────────────────────────────
    # e.g., 15 years experience but career history starts 3 years ago
    career = candidate.get("career_history", [])
    if career:
        try:
            earliest_start = min(
                date.fromisoformat(j["start_date"])
                for j in career
                if j.get("start_date")
            )
            actual_span_years = (REFERENCE_DATE - earliest_start).days / 365.25
            if years_exp > actual_span_years + 3:
                flags += 1
        except (ValueError, TypeError):
            pass

    # A candidate is flagged as a honeypot if 2+ heuristics trigger
    return flags >= 2


# =============================================================================
# DATA STREAMING
# =============================================================================

def stream_candidates(
    filepath: str,
    file_format: str = "jsonl",
) -> Generator[dict, None, None]:
    """
    Memory-efficient candidate streaming.

    For JSONL: reads one line at a time (never loads the full 465 MB file).
    For JSON:  loads the array (fine for small files like sample_candidates.json).

    Args:
        filepath:    Path to the data file.
        file_format: "jsonl" or "json".

    Yields:
        One candidate dict at a time.
    """
    path = Path(filepath)

    if file_format == "json":
        # JSON array — load into memory (only for small sample files)
        with open(path, "rb") as f:
            candidates = orjson.loads(f.read())
        yield from candidates

    elif file_format == "jsonl":
        # JSONL — stream line by line
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield orjson.loads(line)
    else:
        raise ValueError(f"Unsupported format: {file_format}")


# =============================================================================
# INGESTION PIPELINE
# =============================================================================

def ingest(
    filepath: str,
    file_format: str = "jsonl",
    skip_existing: bool = False,
) -> None:
    """
    Main ingestion pipeline.

    Steps:
        1. Load the embedding model (once, ~80 MB)
        2. Stream candidates from file
        3. For each batch of EMBED_BATCH_SIZE candidates:
           a. Build composite text
           b. Extract flat payload
           c. Compute embeddings (batch)
           d. Upsert to Qdrant
        4. Print summary statistics

    Args:
        filepath:      Path to candidate data file.
        file_format:   "jsonl" or "json".
        skip_existing: If True, check if candidate exists before upserting.
    """
    # ── Load model ──────────────────────────────────────────────────────
    print(f"[ingest] Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"[ingest] Model loaded. Max seq length: {model.max_seq_length}")

    # ── Connect to Qdrant ───────────────────────────────────────────────
    print(f"[ingest] Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    # Verify collection exists
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        print(f"[ingest] ✗ Collection '{COLLECTION_NAME}' not found. "
              f"Run setup_db.py first.")
        sys.exit(1)

    # ── Count candidates for progress bar ───────────────────────────────
    # For JSONL, we do a fast line count; for JSON, we don't know ahead of time
    total = None
    if file_format == "jsonl":
        print(f"[ingest] Counting lines in {filepath}...")
        with open(filepath, "rb") as f:
            total = sum(1 for _ in f)
        print(f"[ingest] Found {total:,} candidates.")

    # ── Streaming ingestion ─────────────────────────────────────────────
    batch_texts: list[str] = []
    batch_payloads: list[dict] = []
    batch_ids: list[str] = []

    total_ingested = 0
    total_honeypots = 0
    total_pure_services = 0

    progress = tqdm(
        stream_candidates(filepath, file_format),
        total=total,
        desc="Ingesting",
        unit="cand",
    )

    for candidate in progress:
        # Build text and payload
        text = build_composite_text(candidate)
        payload = extract_payload(candidate)

        batch_texts.append(text)
        batch_payloads.append(payload)
        batch_ids.append(candidate.get("candidate_id", ""))

        # Track stats
        if payload["honeypot_flag"]:
            total_honeypots += 1
        if payload["is_pure_services"]:
            total_pure_services += 1

        # ── Flush batch when full ───────────────────────────────────────
        if len(batch_texts) >= EMBED_BATCH_SIZE:
            _flush_batch(model, client, batch_texts, batch_payloads, batch_ids)
            total_ingested += len(batch_texts)
            progress.set_postfix(ingested=total_ingested, honeypots=total_honeypots)

            # Clear batch (free memory)
            batch_texts.clear()
            batch_payloads.clear()
            batch_ids.clear()

    # ── Flush remaining ─────────────────────────────────────────────────
    if batch_texts:
        _flush_batch(model, client, batch_texts, batch_payloads, batch_ids)
        total_ingested += len(batch_texts)

    # ── Summary ─────────────────────────────────────────────────────────
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"\n[ingest] ✓ Ingestion complete!")
    print(f"  Total candidates ingested:  {total_ingested:,}")
    print(f"  Honeypots detected:         {total_honeypots:,}")
    print(f"  Pure services (flagged):    {total_pure_services:,}")
    print(f"  Collection point count:     {collection_info.points_count:,}")


def _flush_batch(
    model: SentenceTransformer,
    client: QdrantClient,
    texts: list[str],
    payloads: list[dict],
    ids: list[str],
) -> None:
    """
    Embed a batch of texts and upsert to Qdrant.

    The embedding step is the bottleneck.  On an M1 Mac with MiniLM:
        • 64 texts × ~200 words each ≈ 0.5 seconds
        • 100K candidates ÷ 64 per batch = ~1,562 batches ≈ 13 minutes

    Args:
        model:    The SentenceTransformer model.
        client:   The Qdrant client.
        texts:    List of composite texts to embed.
        payloads: Corresponding payload dicts.
        ids:      Corresponding candidate IDs.
    """
    # ── Compute embeddings ──────────────────────────────────────────────
    # show_progress_bar=False because we have our own tqdm wrapper
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,  # Pre-normalise for cosine similarity
    )

    # ── Build Qdrant points ─────────────────────────────────────────────
    # We use a hash of the candidate_id as the numeric point ID.
    # Qdrant requires uint64 IDs; we hash the string ID to get one.
    points = []
    for i, (cid, embedding, payload) in enumerate(zip(ids, embeddings, payloads)):
        # Convert candidate_id "CAND_0000001" → integer 1
        try:
            point_id = int(cid.replace("CAND_", ""))
        except (ValueError, AttributeError):
            point_id = hash(cid) % (2**63)

        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload,
        ))

    # ── Upsert in sub-batches ───────────────────────────────────────────
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        sub_batch = points[i : i + UPSERT_BATCH_SIZE]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=sub_batch,
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ingest candidate data into Qdrant with embeddings."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to candidate data file (JSONL or JSON).",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["jsonl", "json"],
        default="jsonl",
        help="File format: 'jsonl' (line-delimited) or 'json' (array).",
    )
    args = parser.parse_args()

    ingest(args.input, args.format)


if __name__ == "__main__":
    main()
