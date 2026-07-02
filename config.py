"""
config.py — Shared Constants & JD Constraint Definitions
=========================================================

This module is the single source of truth for:
  1. Qdrant connection settings
  2. Embedding model configuration
  3. The deterministic JD constraint schema (hard filters + soft weights)
  4. Scoring formula constants

Every magic number in the pipeline traces back to a constant defined here,
making it trivial to tune without hunting through multiple scripts.
"""

from datetime import date

# =============================================================================
# QDRANT CONNECTION
# =============================================================================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333          # REST port; gRPC is 6334
COLLECTION_NAME = "candidates"

# =============================================================================
# EMBEDDING MODEL
# =============================================================================
# all-MiniLM-L6-v2: 384-dim, ~80 MB, ideal for 8 GB RAM machines.
# Max sequence length: 256 word-pieces (we concat and truncate).
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# =============================================================================
# INGESTION SETTINGS
# =============================================================================
EMBED_BATCH_SIZE = 64       # Candidates per embedding batch (controls peak RAM)
UPSERT_BATCH_SIZE = 100     # Points per Qdrant upsert call
MAX_TEXT_CHARS = 2000        # Truncate composite text before tokenisation

# =============================================================================
# RETRIEVAL SETTINGS
# =============================================================================
RETRIEVAL_LIMIT = 2000       # How many candidates to pull from Qdrant
TOP_K_OUTPUT = 100           # Final CSV row count (submission spec)

# =============================================================================
# JD CONSTRAINT SCHEMA
# =============================================================================
# Hard constraints — candidates violating these are dropped before scoring.
# Derived from the AI Engineer JD (see job_description.docx).

JD_HARD_CONSTRAINTS = {
    # ── Experience ──────────────────────────────────────────────────────
    # JD says "5-9 years" but also "we'll consider outside the band".
    # We use a generous 3-12 band to avoid false negatives, then penalise
    # distance from the sweet-spot (5-9) in the soft scoring phase.
    "experience_min": 3.0,
    "experience_max": 12.0,

    # ── Notice Period ───────────────────────────────────────────────────
    # "Sub-30 day preferred; can buy out up to 30 days; 30+ still in scope
    # but bar gets higher."  We hard-cut at 90 days (anything beyond is
    # practically un-hireable), and penalise 30-90 in soft scoring.
    "notice_period_hard_max": 90,

    # ── Location ────────────────────────────────────────────────────────
    # India-based preferred; outside India OK if willing_to_relocate.
    # We don't hard-filter on country — we penalise instead, because
    # the JD says "case-by-case" for non-India.
    "preferred_countries": ["India"],

    # ── Pure Services Disqualification ──────────────────────────────────
    # "People who have ONLY worked at consulting firms in their ENTIRE
    # career" — this is the only absolute disqualification besides pure
    # academic researchers.
    "services_companies": [
        "tcs", "infosys", "wipro", "accenture", "cognizant",
        "capgemini", "hcl", "tech mahindra", "mindtree", "mphasis",
        "ltimindtree", "hexaware", "persistent", "zensar",
        "l&t infotech", "lti", "cyient",
    ],

    # ── Pure Academic Disqualification ──────────────────────────────────
    "academic_industries": [
        "academia", "research", "higher education", "university",
    ],
}

# =============================================================================
# SOFT SCORING WEIGHTS
# =============================================================================
# These weights control how the Behavioral Score is composed.
# They sum to 1.0 by design.

BEHAVIORAL_WEIGHTS = {
    "github_activity":        0.15,   # Open-source signal (JD: "open-source contributions")
    "recruiter_response":     0.25,   # Responsiveness = reachability
    "freshness":              0.20,   # Recency of platform activity
    "interview_completion":   0.15,   # Follow-through signal
    "profile_completeness":   0.10,   # Effort & seriousness
    "verification":           0.10,   # Trust signals (email, phone, LinkedIn)
    "open_to_work":           0.05,   # Explicit availability signal
}

# Bounded multiplier cap: behavioral score can boost semantic score by at most
# this fraction.  Final = semantic × (1 + behavioral × BEHAVIORAL_BOOST_CAP)
BEHAVIORAL_BOOST_CAP = 0.25

# =============================================================================
# PENALTY FACTORS
# =============================================================================
# Each penalty is a multiplier ∈ (0, 1].  A value of 1.0 means "no penalty".

PENALTY_PURE_SERVICES = 0.0        # Hard drop — pure services with zero product exp
PENALTY_MOSTLY_SERVICES = 0.65     # Has product exp but majority is services
PENALTY_NOTICE_PERIOD_MAX = 0.55   # Floor for notice-period penalty
PENALTY_TITLE_MISMATCH = 0.40     # Non-technical title (Accountant, Marketing Mgr)
PENALTY_STALE_PROFILE = 0.60      # Inactive > 180 days
PENALTY_NON_INDIA = 0.75          # Non-India & not willing to relocate
PENALTY_REMOTE_ONLY = 0.85        # Remote-only when JD prefers hybrid/onsite
PENALTY_HONEYPOT = 0.0            # Detected honeypots get zeroed out

# =============================================================================
# TITLE CLASSIFICATION
# =============================================================================
# Titles that indicate the candidate is clearly NOT in a tech/AI/ML/Data role.
# Used for title_mismatch penalty.
NON_TECH_TITLES = {
    "accountant", "marketing manager", "operations manager",
    "sales executive", "hr manager", "content writer",
    "graphic designer", "customer support", "civil engineer",
    "mechanical engineer", "brand manager", "copy writer",
    "financial analyst", "recruitment specialist",
    "administrative assistant", "office manager",
}

# Titles that strongly signal AI/ML/Data relevance.
AI_RELEVANT_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "data scientist", "research scientist", "applied scientist",
    "nlp engineer", "deep learning engineer", "computer vision engineer",
    "senior machine learning engineer", "junior ml engineer",
    "lead data scientist", "principal engineer",
    "staff engineer", "backend engineer", "software engineer",
    "data engineer", "analytics engineer", "research engineer",
    "platform engineer", "infrastructure engineer",
}

# =============================================================================
# SKILLS MATCHING
# =============================================================================
# Core skills the JD explicitly requires or values.
# Grouped by priority (must-have vs nice-to-have).
JD_MUST_HAVE_SKILLS = {
    # "Things you absolutely need"
    "embeddings", "sentence-transformers", "openai embeddings", "bge", "e5",
    "vector database", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "faiss",
    "python",
    "ndcg", "mrr", "evaluation", "ranking", "information retrieval",
    "search", "retrieval", "recommendation",
    "nlp", "natural language processing",
    "transformers", "huggingface", "bert", "gpt",
}

JD_NICE_TO_HAVE_SKILLS = {
    # "Things we'd like you to have"
    "lora", "qlora", "peft", "fine-tuning llms", "llm fine-tuning",
    "xgboost", "learning to rank", "lightgbm",
    "distributed systems", "kubernetes", "docker",
    "open source",
    "rag", "langchain", "llamaindex",
    "pytorch", "tensorflow", "jax",
    "spark", "airflow", "kafka",
    "aws", "gcp", "azure",
    "mlflow", "weights & biases", "wandb",
    "bentoml", "triton", "onnx",
    "gans", "stable diffusion", "image classification",
    "speech recognition", "tts",
}

# =============================================================================
# REFERENCE DATE (for freshness calculations)
# =============================================================================
REFERENCE_DATE = date(2026, 6, 29)

# =============================================================================
# JD TEXT (for embedding)
# =============================================================================
JD_TEXT = """
AI Engineer at Redrob. We need someone to own the intelligence layer:
ranking, retrieval, and matching systems for candidate-job matching.

Requirements: Production experience with embeddings-based retrieval systems
(sentence-transformers, OpenAI embeddings, BGE, E5). Production experience
with vector databases or hybrid search (Pinecone, Weaviate, Qdrant, Milvus,
OpenSearch, Elasticsearch, FAISS). Strong Python. Experience designing
evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing).

Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank models,
HR-tech or marketplace experience, distributed systems, open-source
contributions.

The role involves shipping a v2 ranking system with embeddings, hybrid
retrieval, and LLM-based re-ranking. Setting up offline benchmarks, online
A/B testing, and recruiter-feedback loops. Mentoring engineers.

5-9 years experience preferred. Must have product company experience.
Located in India (Pune/Noida preferred) or willing to relocate.
Sub-30-day notice period preferred.
""".strip()
