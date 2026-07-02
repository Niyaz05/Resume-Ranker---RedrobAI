# Candidate Reranking Pipeline

> **Redrob "India Runs Data & AI" Hackathon — AI Engineer JD Matching**

This project is a comprehensive **Candidate Reranking** system that ingests candidate profiles, computes text embeddings, stores them in a local vector database, and ranks candidates against a specific AI Engineer Job Description using a **hybrid semantic + soft-penalty scoring architecture**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Reproducing the Submission CSV](#reproducing-the-submission-csv)
- [Sandbox / Demo (Streamlit)](#sandbox--demo-streamlit)
- [How Scoring Works](#how-scoring-works)
- [Compute Constraints Compliance](#compute-constraints-compliance)
- [Submission Checklist](#submission-checklist)

---

## Overview

The core philosophy of this ranking engine is a **Soft-Penalty Architecture**. Instead of hard-dropping candidates who don't perfectly meet criteria (like exact years of experience or notice period), it computes a semantic match using embeddings and applies continuous penalty multipliers for constraints. This ensures the system always surfaces the best available matches without risking zero results due to strict filtering.

**Key design principles:**
- **No hard filters** (except honeypots) — every constraint is a continuous multiplier ∈ (0, 1]
- **Multiplicative fusion** — behavioral signals amplify semantic relevance, never replace it
- **Deterministic & reproducible** — no randomness, no external API calls, same input → same output
- **CPU-only, offline** — no GPU, no network calls during ranking

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Phase 1: Semantic Retrieval                                         │
│    • Embed JD text with all-MiniLM-L6-v2 (384-dim)                  │
│    • Query Qdrant for top-2000 by cosine similarity                  │
│    • Only hard filter: honeypot_flag = false                         │
│                                                                      │
│  Phase 2: Penalty Multipliers (in Polars)                            │
│    • Experience penalty   → Gaussian decay outside [5, 9] years      │
│    • Industry penalty     → pure services crushed to 0.10            │
│    • Title penalty        → non-tech titles crushed to 0.30          │
│    • Notice period penalty → smooth sigmoid decay beyond 30d         │
│    • Location penalty     → non-India + won't relocate               │
│    • Staleness penalty    → exponential decay for inactive profiles  │
│    • Skills match bonus   → multiplicative reward for JD skills      │
│                                                                      │
│  Phase 3: Behavioral Score (bounded boost)                           │
│    • GitHub, response rate, freshness, interview completion           │
│    • Normalized to [0, 1], applied as bounded ×1.25 multiplier       │
│                                                                      │
│  Phase 4: Final Score Fusion                                         │
│    Final = Semantic × Π(penalties) × Skills Bonus × (1 + Beh × 0.25)│
│                                                                      │
│  Phase 5: Output                                                     │
│    • Sort descending, take top 100                                   │
│    • Generate per-candidate reasoning                                │
│    • Write CSV: candidate_id, rank, score, reasoning                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

| File | Description |
|---|---|
| `config.py` | Single source of truth for all configurations — Qdrant settings, embedding model (`all-MiniLM-L6-v2`), hard constraints, soft scoring weights, penalty factors, and the target JD text. |
| `setup_db.py` | Bootstraps the local Qdrant collection (`candidates`). Configures 384-dim vector space with cosine distance, on-disk storage, and payload indexes for efficient pre-filtering. |
| `ingest.py` | Streaming ingestion pipeline — reads JSONL/JSON candidate data, builds composite text representations, generates embeddings in batches, extracts structured fields, detects honeypots, and upserts into Qdrant. |
| `ranker.py` | The hybrid ranking engine — retrieves candidates by semantic similarity, applies penalty multipliers via Polars, factors in behavioral scores, and outputs top-K candidates to CSV. |
| `app.py` | **Streamlit sandbox/demo app** — runs the full ranking pipeline in-memory (no Qdrant required). Accepts ≤100 candidates via upload, produces a ranked CSV. See [Sandbox section](#sandbox--demo-streamlit). |
| `requirements.txt` | Python dependencies with pinned minimum versions. |
| `submission.csv` | Final output file generated by the reranking pipeline. |
| `submission_metadata.yaml` | Portal metadata mirror (see `submission_metadata_template.yaml`). |

---

## Prerequisites

- **Python 3.9+**
- **[Docker](https://www.docker.com/)** (to run the Qdrant vector database for full pipeline)
- ~8 GB RAM minimum (pipeline optimised for 8 GB; uses memory-mapped vectors)

---

## Setup Instructions

1. **Clone the repository and navigate to the directory:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO
   ```

2. **Set up a Python virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Qdrant Vector Database:**
   ```bash
   docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
     -v $(pwd)/qdrant_storage:/qdrant/storage:z \
     qdrant/qdrant:latest
   ```

5. **Bootstrap the Qdrant Collection:**
   ```bash
   python setup_db.py
   ```
   *(Use `python setup_db.py --recreate` to drop and rebuild an existing collection.)*

---

## Reproducing the Submission CSV

### Pre-computation Step (Ingestion)

> **Note:** Pre-computation (embedding + ingestion) takes ~15-20 minutes for the full 100K dataset. This is a one-time step. The ranking step that produces the CSV completes within the 5-minute window.

```bash
# Ingest the full candidate dataset into Qdrant
python ingest.py --input data_and_problem_statement/candidates.jsonl --format jsonl
```

For quick testing with the sample dataset:
```bash
python ingest.py --input data_and_problem_statement/sample_candidates.json --format json
```

### Single Reproduce Command

Once data is ingested, produce the submission CSV with:

```bash
python ranker.py --output submission.csv
```

This single command:
- Loads the `all-MiniLM-L6-v2` embedding model (~80 MB)
- Embeds the JD and retrieves 2000 candidates from Qdrant by cosine similarity
- Applies all penalty multipliers and behavioral scoring
- Outputs exactly 100 ranked candidates to `submission.csv`
- **Runtime: ~30 seconds on CPU, well within the 5-minute limit**

Optional arguments:
```bash
python ranker.py --output submission.csv --top-k 100 --retrieve 2000 --verbose
```

---

## Sandbox / Demo (Streamlit)

The Streamlit app provides a **self-contained sandbox** that runs the full ranking pipeline **without requiring Qdrant** (all computation happens in-memory). This satisfies the **Section 10.5** requirement of the submission spec.

### Running Locally

```bash
streamlit run app.py
```

### What the Sandbox Does

1. **Upload** a JSON/JSONL candidate file (≤100 candidates), or use the pre-loaded sample
2. **Rank** candidates using the same scoring pipeline as `ranker.py`
3. **View** ranked results with score breakdowns
4. **Download** the ranked `submission.csv`

### Deploying to Streamlit Cloud

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repository and set `app.py` as the main file
4. Deploy — the app will install dependencies from `requirements.txt` automatically

### Sandbox Constraints Compliance

| Constraint | Status |
|---|---|
| Accepts ≤100 candidate sample | ✅ Enforced (truncates if >100) |
| Runs end-to-end, produces ranked CSV | ✅ Full pipeline in-memory |
| Completes within 5 min on CPU | ✅ ~10-30s for 100 candidates |
| No external API calls | ✅ Fully offline |

---

## How Scoring Works

### 1. Semantic Retrieval
The JD is embedded using `all-MiniLM-L6-v2`, and the top 2000 candidates are retrieved from Qdrant by cosine similarity.

### 2. Penalty Multipliers
Using Polars, JD constraints are applied as continuous multipliers ∈ (0, 1]:

| Penalty | What it checks | Range |
|---|---|---|
| Experience | Gaussian decay outside [5, 9] years | 0.05 – 1.0 |
| Industry | Pure services vs. product experience | 0.10 – 1.0 |
| Title | AI-relevant vs. non-tech titles | 0.30 – 1.0 |
| Notice Period | Sigmoid decay beyond 30 days | 0.05 – 0.95 |
| Location | India-based / willing to relocate | 0.50 – 1.0 |
| Staleness | Exponential decay for inactive profiles | 0.05 – 1.0 |
| Skills Bonus | Reward for JD-relevant skills | 0.85 – 1.30 |

### 3. Behavioral Boost
Factors like GitHub activity, recruiter response rate, profile freshness, and interview completion add a bounded multiplicative boost (max +25%).

### 4. Final Score
```
Final = Semantic × Π(penalties) × Skills_Bonus × (1 + Behavioral × 0.25)
```

---

## Compute Constraints Compliance

As required by the submission spec (Section 6):

| Constraint | Limit | Our System |
|---|---|---|
| Total runtime | ≤ 5 minutes | ~30 seconds for ranking step |
| Memory | ≤ 16 GB RAM | Optimised for 8 GB (on-disk vectors) |
| Compute | CPU only | ✅ No GPU used during ranking |
| Network | Offline | ✅ No external API calls |
| Disk | ≤ 5 GB | ~1.5 GB (vectors + indexes) |

---

## Submission Checklist

Per the submission spec (Section 10), ensure all items are complete:

- [x] **CSV file** — `submission.csv` with top-100 ranked candidates (`candidate_id`, `rank`, `score`, `reasoning`)
- [x] **README.md** — Setup instructions and exact commands to reproduce the submission CSV
- [x] **Full source code** — No hidden steps, no manual edits
- [x] **requirements.txt** — All dependencies with minimum versions
- [x] **submission_metadata.yaml** — Portal metadata mirror (copy from `submission_metadata_template.yaml` and fill in)
- [x] **Sandbox / demo link** — Streamlit app (`app.py`) deployable to Streamlit Cloud
- [x] **Single reproduce command** — `python ranker.py --output submission.csv`
- [x] **Compute constraints** — CPU-only, no network, ≤5 min, ≤16 GB RAM
