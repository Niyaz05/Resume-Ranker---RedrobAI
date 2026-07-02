"""
app.py — Streamlit Sandbox / Demo App
=======================================

A lightweight, self-contained Streamlit app that demonstrates the candidate
reranking pipeline end-to-end WITHOUT requiring Qdrant.

This satisfies Section 10.5 of the submission spec:
  • Accepts a small candidate sample (≤100 candidates) via file upload or pre-loaded
  • Runs the ranking system end-to-end and produces a ranked CSV
  • Completes within the compute budget (≤5 min on CPU)
  • Does NOT make any external API calls

How it works:
  1. User uploads a JSON / JSONL file (or uses the pre-loaded sample)
  2. Embeddings are computed in-memory using sentence-transformers
  3. Candidates are scored using the same soft-penalty architecture as ranker.py
  4. Results are displayed and downloadable as CSV

Usage:
    streamlit run app.py
"""

import io
import json
import math
import time
from datetime import date, datetime

import polars as pl
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import (
    BEHAVIORAL_BOOST_CAP,
    BEHAVIORAL_WEIGHTS,
    EMBEDDING_MODEL_NAME,
    JD_HARD_CONSTRAINTS,
    JD_TEXT,
    MAX_TEXT_CHARS,
    NON_TECH_TITLES,
    AI_RELEVANT_TITLES,
    REFERENCE_DATE,
    TOP_K_OUTPUT,
)

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Candidate Reranker — Redrob Hackathon",
    page_icon="🎯",
    layout="wide",
)


# =============================================================================
# CACHED MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model() -> SentenceTransformer:
    """Load and cache the MiniLM embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# =============================================================================
# TEXT BUILDER (duplicated from ingest.py to keep sandbox self-contained)
# =============================================================================

def build_composite_text(candidate: dict) -> str:
    """Build a composite text representation for embedding."""
    parts = []

    # Headline + current title (high signal, short-form)
    if candidate.get("headline"):
        parts.append(candidate["headline"])
    if candidate.get("current_title"):
        parts.append(f"Current role: {candidate['current_title']}")

    # Profile summary
    if candidate.get("profile_summary"):
        parts.append(candidate["profile_summary"])

    # Career history descriptions
    for exp in candidate.get("career_history", []):
        if exp.get("description"):
            parts.append(exp["description"])
        if exp.get("title"):
            parts.append(f"Role: {exp['title']} at {exp.get('company_name', 'N/A')}")

    # Skills as text
    skills = candidate.get("skills", [])
    if skills:
        skill_names = [s.get("name", s) if isinstance(s, dict) else str(s)
                       for s in skills]
        parts.append("Skills: " + ", ".join(skill_names))

    text = " | ".join(parts)
    return text[:MAX_TEXT_CHARS]


# =============================================================================
# PAYLOAD EXTRACTION (from ingest.py, simplified for sandbox)
# =============================================================================

def extract_payload(candidate: dict) -> dict:
    """Extract structured payload fields from a candidate profile."""
    payload = {}

    payload["candidate_id"] = candidate.get("candidate_id", "UNKNOWN")
    payload["current_title"] = candidate.get("current_title", "")
    payload["current_title_lower"] = payload["current_title"].lower().strip()
    payload["headline"] = candidate.get("headline", "")

    # Years of experience
    payload["years_of_experience"] = float(
        candidate.get("years_of_experience", 0) or 0
    )

    # Notice period
    payload["notice_period_days"] = int(
        candidate.get("notice_period_days", 90) or 90
    )

    # Location
    payload["country"] = candidate.get("country", "")
    payload["city"] = candidate.get("city", "")
    payload["willing_to_relocate"] = bool(
        candidate.get("willing_to_relocate", False)
    )
    payload["preferred_work_mode"] = candidate.get(
        "preferred_work_mode", "remote"
    )

    # Industry signals
    services_set = set(JD_HARD_CONSTRAINTS["services_companies"])
    career = candidate.get("career_history", [])
    total_months = 0
    services_months = 0
    has_product = False

    for exp in career:
        company = (exp.get("company_name", "") or "").lower().strip()
        duration = int(exp.get("duration_months", 0) or 0)
        total_months += duration
        if any(s in company for s in services_set):
            services_months += duration
        else:
            if duration > 0:
                has_product = True

    payload["services_fraction"] = (
        services_months / total_months if total_months > 0 else 0.0
    )
    payload["is_pure_services"] = (
        services_months == total_months and total_months > 0
    )
    payload["has_product_experience"] = has_product

    # Academic check
    industry = (candidate.get("current_industry", "") or "").lower()
    academic_set = set(JD_HARD_CONSTRAINTS["academic_industries"])
    payload["is_pure_academic"] = industry in academic_set
    payload["current_industry"] = candidate.get("current_industry", "")

    # Skills
    skills = candidate.get("skills", [])
    skill_names = set()
    for s in skills:
        name = (s.get("name", s) if isinstance(s, dict) else str(s)).lower()
        skill_names.add(name)

    from config import JD_MUST_HAVE_SKILLS, JD_NICE_TO_HAVE_SKILLS

    payload["must_have_skill_count"] = len(skill_names & JD_MUST_HAVE_SKILLS)
    payload["nice_to_have_count"] = len(skill_names & JD_NICE_TO_HAVE_SKILLS)
    payload["total_skill_count"] = len(skills)

    # Behavioral signals
    signals = candidate.get("signals", {})
    payload["github_activity_score"] = float(
        signals.get("github_activity_score", -1) or -1
    )
    payload["recruiter_response_rate"] = float(
        signals.get("recruiter_response_rate", 0) or 0
    )
    payload["interview_completion_rate"] = float(
        signals.get("interview_completion_rate", 0) or 0
    )
    payload["profile_completeness_score"] = float(
        signals.get("profile_completeness_score", 0) or 0
    )
    payload["verified_email"] = bool(signals.get("verified_email", False))
    payload["verified_phone"] = bool(signals.get("verified_phone", False))
    payload["linkedin_connected"] = bool(
        signals.get("linkedin_connected", False)
    )
    payload["open_to_work_flag"] = bool(
        candidate.get("open_to_work", False)
    )

    # Freshness
    last_active = candidate.get("last_active_date")
    if last_active:
        try:
            if isinstance(last_active, str):
                la_date = datetime.fromisoformat(
                    last_active.replace("Z", "+00:00")
                ).date()
            else:
                la_date = last_active
            payload["days_since_active"] = (REFERENCE_DATE - la_date).days
        except Exception:
            payload["days_since_active"] = 180
    else:
        payload["days_since_active"] = 180

    # Honeypot detection (simple heuristic)
    payload["honeypot_flag"] = False
    if payload["total_skill_count"] > 30 and payload["years_of_experience"] < 2:
        payload["honeypot_flag"] = True

    return payload


# =============================================================================
# SCORING FUNCTIONS (from ranker.py — kept identical for reproducibility)
# =============================================================================

def compute_experience_penalty(years: float) -> float:
    BAND_LOW, BAND_HIGH, SIGMA = 5.0, 9.0, 3.0
    if BAND_LOW <= years <= BAND_HIGH:
        return 1.0
    distance = min(abs(years - BAND_LOW), abs(years - BAND_HIGH))
    return max(math.exp(-(distance ** 2) / (2 * SIGMA ** 2)), 0.05)


def compute_industry_penalty(
    is_pure_services: bool, has_product_experience: bool,
    services_fraction: float,
) -> float:
    if is_pure_services and not has_product_experience:
        return 0.10
    if not has_product_experience:
        return max(0.15, 1.0 - services_fraction * 0.85)
    return max(0.40, 1.0 - services_fraction * 0.60)


def compute_title_penalty(title_lower: str) -> float:
    title_clean = title_lower.strip()
    if title_clean in AI_RELEVANT_TITLES:
        return 1.0
    ai_keywords = {"ai", "ml", "machine learning", "data scien", "nlp",
                   "deep learning", "research scien", "applied scien"}
    for kw in ai_keywords:
        if kw in title_clean:
            return 0.95
    tech_keywords = {"engineer", "developer", "architect", "programmer",
                     "scientist", "analyst", "technical", "devops", "sre",
                     "platform", "infrastructure"}
    for kw in tech_keywords:
        if kw in title_clean:
            return 0.75
    if title_clean in NON_TECH_TITLES:
        return 0.30
    return 0.50


def compute_notice_penalty(notice_days: int) -> float:
    MIDPOINT, STEEPNESS = 45.0, 15.0
    exponent = (notice_days - MIDPOINT) / STEEPNESS
    return max(1.0 / (1.0 + math.exp(exponent)), 0.05)


def compute_location_penalty(
    country: str, willing_to_relocate: bool, preferred_work_mode: str,
) -> float:
    is_india = country.strip().lower() == "india"
    if is_india:
        return 0.90 if preferred_work_mode == "remote" else 1.0
    if willing_to_relocate:
        return 0.80 if preferred_work_mode in ("onsite", "hybrid", "flexible") else 0.70
    return 0.50


def compute_staleness_penalty(days_since_active: int) -> float:
    return max(math.exp(-days_since_active / 180.0), 0.05)


def compute_skills_bonus(
    must_have_count: int, nice_to_have_count: int, total_skill_count: int,
) -> float:
    must_score = min(must_have_count / 4.0, 1.0)
    nice_score = min(nice_to_have_count / 6.0, 1.0)
    combined = must_score * 0.70 + nice_score * 0.30
    bonus = 1.0 + combined * 0.30
    if total_skill_count > 15:
        relevant_ratio = (must_have_count + nice_to_have_count) / total_skill_count
        if relevant_ratio < 0.3:
            bonus *= 0.90
    return bonus


def generate_reasoning(row: dict) -> str:
    parts = []
    title = row.get("current_title", "Unknown")
    yoe = row.get("years_of_experience", 0)
    parts.append(f"{title} with {yoe:.1f} yrs exp")

    sem = row.get("semantic_score", 0)
    if sem >= 0.6:
        parts.append(f"strong semantic match ({sem:.2f})")
    elif sem >= 0.4:
        parts.append(f"moderate semantic match ({sem:.2f})")
    else:
        parts.append(f"weak semantic match ({sem:.2f})")

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


# =============================================================================
# MAIN PIPELINE (in-memory, no Qdrant)
# =============================================================================

def run_pipeline(candidates_raw: list[dict], model: SentenceTransformer,
                 top_k: int) -> pl.DataFrame:
    """Run the full ranking pipeline in-memory."""

    progress = st.progress(0, text="Extracting payloads...")

    # Step 1: Build texts and extract payloads
    texts = []
    payloads = []
    for cand in candidates_raw:
        text = build_composite_text(cand)
        payload = extract_payload(cand)
        texts.append(text)
        payloads.append(payload)

    progress.progress(20, text="Computing embeddings...")

    # Step 2: Embed JD + all candidates
    jd_embedding = model.encode(JD_TEXT, normalize_embeddings=True)
    candidate_embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False,
        batch_size=64,
    )

    progress.progress(50, text="Computing semantic scores...")

    # Step 3: Cosine similarity (embeddings are already normalised → dot product)
    import numpy as np
    similarities = np.dot(candidate_embeddings, jd_embedding)

    for i, payload in enumerate(payloads):
        payload["semantic_score"] = float(similarities[i])

    progress.progress(60, text="Applying penalty multipliers...")

    # Step 4: Penalty multipliers
    for c in payloads:
        if c.get("honeypot_flag"):
            c["semantic_score"] = 0.0
        c["experience_penalty"] = compute_experience_penalty(
            c.get("years_of_experience", 0.0))
        c["industry_penalty"] = compute_industry_penalty(
            c.get("is_pure_services", False),
            c.get("has_product_experience", True),
            c.get("services_fraction", 0.0))
        c["title_penalty"] = compute_title_penalty(
            c.get("current_title_lower", ""))
        c["notice_penalty"] = compute_notice_penalty(
            c.get("notice_period_days", 90))
        c["location_penalty"] = compute_location_penalty(
            c.get("country", ""),
            c.get("willing_to_relocate", False),
            c.get("preferred_work_mode", "remote"))
        c["staleness_penalty"] = compute_staleness_penalty(
            c.get("days_since_active", 999))
        c["skills_bonus"] = compute_skills_bonus(
            c.get("must_have_skill_count", 0),
            c.get("nice_to_have_count", 0),
            c.get("total_skill_count", 0))

    progress.progress(75, text="Computing behavioral scores...")

    # Step 5: Build DataFrame and compute behavioral + final scores
    df = pl.DataFrame(payloads)

    w = BEHAVIORAL_WEIGHTS
    df = df.with_columns([
        pl.when(pl.col("github_activity_score") < 0)
          .then(pl.lit(0.3))
          .otherwise(pl.col("github_activity_score").clip(0, 100) / 100.0)
          .alias("github_norm"),
        pl.col("recruiter_response_rate").clip(0.0, 1.0).alias("response_norm"),
        (-pl.col("days_since_active").cast(pl.Float64) / 180.0)
          .exp().clip(0.0, 1.0).alias("freshness_norm"),
        pl.col("interview_completion_rate").clip(0.0, 1.0).alias("interview_norm"),
        (pl.col("profile_completeness_score") / 100.0)
          .clip(0.0, 1.0).alias("completeness_norm"),
        ((pl.col("verified_email").cast(pl.Float64)
          + pl.col("verified_phone").cast(pl.Float64)
          + pl.col("linkedin_connected").cast(pl.Float64)) / 3.0)
          .alias("verification_norm"),
        pl.when(pl.col("open_to_work_flag") == True)
          .then(pl.lit(1.0)).otherwise(pl.lit(0.3))
          .alias("open_to_work_norm"),
    ])

    df = df.with_columns(
        (pl.col("github_norm") * w["github_activity"]
         + pl.col("response_norm") * w["recruiter_response"]
         + pl.col("freshness_norm") * w["freshness"]
         + pl.col("interview_norm") * w["interview_completion"]
         + pl.col("completeness_norm") * w["profile_completeness"]
         + pl.col("verification_norm") * w["verification"]
         + pl.col("open_to_work_norm") * w["open_to_work"]
        ).alias("behavioral_score")
    )

    progress.progress(85, text="Fusing scores...")

    # Penalty product
    df = df.with_columns(
        (pl.col("experience_penalty") * pl.col("industry_penalty")
         * pl.col("title_penalty") * pl.col("notice_penalty")
         * pl.col("location_penalty") * pl.col("staleness_penalty")
        ).alias("penalty_product")
    )

    # Final score
    df = df.with_columns(
        (pl.col("semantic_score") * pl.col("penalty_product")
         * pl.col("skills_bonus")
         * (1.0 + pl.col("behavioral_score") * BEHAVIORAL_BOOST_CAP)
        ).alias("final_score")
    )

    df = df.sort("final_score", descending=True)

    progress.progress(95, text="Generating output...")

    # Step 6: Build output
    top = df.head(top_k)
    rows = []
    for rank, row_dict in enumerate(top.iter_rows(named=True), start=1):
        reasoning = generate_reasoning(row_dict)
        rows.append({
            "candidate_id": row_dict["candidate_id"],
            "rank": rank,
            "score": round(row_dict["final_score"], 4),
            "reasoning": reasoning,
        })

    progress.progress(100, text="Done!")
    return pl.DataFrame(rows), df


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.title("🎯 Candidate Reranker")
    st.markdown(
        "**Redrob Hackathon — Hybrid Semantic + Soft-Penalty Ranking Engine**"
    )
    st.markdown("---")

    st.markdown("""
    This sandbox demonstrates the full candidate reranking pipeline:
    1. **Upload** a JSON/JSONL candidate file (≤100 candidates)
    2. **Rank** candidates against the AI Engineer JD using semantic similarity + soft penalties
    3. **Download** the ranked CSV submission file
    """)

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        top_k = st.slider(
            "Top-K output", min_value=10, max_value=100,
            value=TOP_K_OUTPUT, step=10,
        )
        st.markdown("---")
        st.markdown("### 📄 JD Summary")
        st.text_area(
            "Job Description (read-only)", value=JD_TEXT, height=200,
            disabled=True,
        )
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "Uses `all-MiniLM-L6-v2` embeddings with cosine similarity "
            "and a soft-penalty scoring architecture. "
            "No external API calls. CPU only."
        )

    # ── File Upload ──────────────────────────────────────────────────────
    st.subheader("📁 Upload Candidate Data")

    use_sample = st.checkbox(
        "Use pre-loaded sample (sample_candidates.json)", value=False,
    )

    candidates_raw = None

    if use_sample:
        import os
        sample_path = os.path.join(
            "data_and_problem_statement", "sample_candidates.json"
        )
        if os.path.exists(sample_path):
            with open(sample_path, "r") as f:
                candidates_raw = json.load(f)
            st.success(f"Loaded {len(candidates_raw)} candidates from sample file.")
        else:
            st.error(f"Sample file not found at `{sample_path}`.")
    else:
        uploaded_file = st.file_uploader(
            "Upload JSON or JSONL file",
            type=["json", "jsonl"],
            help="Max 100 candidates. JSON array or line-delimited JSONL.",
        )
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8")
            if uploaded_file.name.endswith(".jsonl"):
                candidates_raw = []
                for line in content.strip().split("\n"):
                    if line.strip():
                        candidates_raw.append(json.loads(line))
            else:
                data = json.loads(content)
                candidates_raw = data if isinstance(data, list) else [data]
            st.success(f"Loaded {len(candidates_raw)} candidates from upload.")

    if candidates_raw is not None and len(candidates_raw) > 100:
        st.warning(
            f"⚠️ {len(candidates_raw)} candidates loaded. "
            f"Truncating to first 100 for sandbox demo."
        )
        candidates_raw = candidates_raw[:100]

    # ── Run Pipeline ─────────────────────────────────────────────────────
    st.markdown("---")
    if candidates_raw is not None:
        if st.button("🚀 Run Ranking Pipeline", type="primary"):
            start_time = time.time()

            with st.spinner("Loading embedding model..."):
                model = load_model()

            output_df, full_df = run_pipeline(candidates_raw, model, top_k)
            elapsed = time.time() - start_time

            st.success(
                f"✅ Ranking complete! Processed {len(candidates_raw)} "
                f"candidates in {elapsed:.1f}s"
            )

            # ── Results ──────────────────────────────────────────────────
            st.subheader("📊 Ranked Results")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Candidates Ranked", len(output_df))
            with col2:
                top_score = output_df["score"][0]
                st.metric("Top Score", f"{top_score:.4f}")
            with col3:
                bottom_score = output_df["score"][-1]
                st.metric("Bottom Score", f"{bottom_score:.4f}")
            with col4:
                st.metric("Runtime", f"{elapsed:.1f}s")

            # Display table
            st.dataframe(
                output_df.to_pandas(),
                use_container_width=True,
                hide_index=True,
            )

            # ── Download CSV ─────────────────────────────────────────────
            csv_buffer = io.BytesIO()
            output_df.write_csv(csv_buffer)
            csv_bytes = csv_buffer.getvalue()

            st.download_button(
                label="⬇️ Download submission.csv",
                data=csv_bytes,
                file_name="submission.csv",
                mime="text/csv",
            )

            # ── Score Distribution ───────────────────────────────────────
            with st.expander("📈 Score Distribution Details"):
                st.markdown("#### Semantic Score Distribution")
                sem_stats = full_df.select(
                    pl.col("semantic_score").mean().alias("mean"),
                    pl.col("semantic_score").min().alias("min"),
                    pl.col("semantic_score").max().alias("max"),
                    pl.col("semantic_score").std().alias("std"),
                )
                st.dataframe(sem_stats.to_pandas(), hide_index=True)

                st.markdown("#### Penalty Product Distribution")
                pen_stats = full_df.select(
                    pl.col("penalty_product").mean().alias("mean"),
                    pl.col("penalty_product").min().alias("min"),
                    pl.col("penalty_product").max().alias("max"),
                    pl.col("penalty_product").std().alias("std"),
                )
                st.dataframe(pen_stats.to_pandas(), hide_index=True)

                st.markdown("#### Final Score Distribution")
                final_stats = full_df.select(
                    pl.col("final_score").mean().alias("mean"),
                    pl.col("final_score").min().alias("min"),
                    pl.col("final_score").max().alias("max"),
                    pl.col("final_score").std().alias("std"),
                )
                st.dataframe(final_stats.to_pandas(), hide_index=True)

    else:
        st.info("👆 Upload a candidate file or select the sample to get started.")


if __name__ == "__main__":
    main()
