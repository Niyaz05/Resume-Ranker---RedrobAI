"""
setup_db.py — Qdrant Collection Bootstrap
==========================================

Creates (or recreates) the `candidates` collection in a local Qdrant instance
with the correct vector configuration and payload indexes.

Usage:
    python setup_db.py              # Create collection (skip if exists)
    python setup_db.py --recreate   # Drop and recreate

What this script does:
    1. Connects to Qdrant at localhost:6333
    2. Creates a collection with:
       - 384-dim vectors (all-MiniLM-L6-v2) using Cosine distance
       - On-disk storage enabled (keeps RAM usage low for 100K vectors)
    3. Creates payload indexes for filterable fields so that Qdrant can
       apply range/match filters BEFORE brute-force similarity search.

Payload Index Strategy:
    We index fields that appear in hard filters or range queries.  Qdrant
    uses these indexes to prune the search space, which means our filtered
    queries run in O(matched) rather than O(total) time.

    ┌─────────────────────────┬──────────┬────────────────────────────────┐
    │ Field                   │ Type     │ Why indexed                    │
    ├─────────────────────────┼──────────┼────────────────────────────────┤
    │ years_of_experience     │ float    │ Range query: 3.0 ≤ yoe ≤ 12.0 │
    │ notice_period_days      │ integer  │ Range query: np ≤ 90           │
    │ country                 │ keyword  │ Match query: "India"           │
    │ preferred_work_mode     │ keyword  │ Match query: hybrid/onsite     │
    │ willing_to_relocate     │ bool     │ Match query: true              │
    │ current_industry        │ keyword  │ Match query: IT Services       │
    │ is_pure_services        │ bool     │ Match query: false             │
    │ is_pure_academic        │ bool     │ Match query: false             │
    │ honeypot_flag           │ bool     │ Match query: false             │
    │ has_product_experience  │ bool     │ Match query: true              │
    │ open_to_work_flag       │ bool     │ Match query: true              │
    │ current_title_lower     │ keyword  │ Match against title sets       │
    └─────────────────────────┴──────────┴────────────────────────────────┘
"""

import sys
import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
    OptimizersConfigDiff,
)

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_DIM,
)


def create_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create the candidates collection with vector config and payload indexes."""

    # ── Check if collection already exists ──────────────────────────────
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            print(f"[setup_db] Dropping existing collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(f"[setup_db] Collection '{COLLECTION_NAME}' already exists. "
                  f"Use --recreate to drop and rebuild.")
            return

    # ── Create collection ───────────────────────────────────────────────
    # on_disk=True tells Qdrant to memory-map vectors rather than holding
    # them entirely in RAM.  For 100K × 384-dim float32 vectors:
    #   In-RAM:  100K × 384 × 4 bytes ≈ 150 MB
    #   On-disk: ~0 MB resident (OS page cache handles hot pages)
    # This is critical for the 8 GB RAM constraint.
    print(f"[setup_db] Creating collection '{COLLECTION_NAME}' "
          f"(dim={EMBEDDING_DIM}, distance=Cosine, on_disk=True)...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
            on_disk=True,          # Memory-map vectors to disk
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,  # Build HNSW index after 20K points
        ),
    )

    # ── Create payload indexes ──────────────────────────────────────────
    # Each index lets Qdrant filter BEFORE similarity search, dramatically
    # reducing the number of distance computations.

    payload_indexes = {
        # Numeric range filters
        "years_of_experience":    PayloadSchemaType.FLOAT,
        "notice_period_days":     PayloadSchemaType.INTEGER,

        # Keyword match filters
        "country":                PayloadSchemaType.KEYWORD,
        "preferred_work_mode":    PayloadSchemaType.KEYWORD,
        "current_industry":       PayloadSchemaType.KEYWORD,
        "current_title_lower":    PayloadSchemaType.KEYWORD,

        # Boolean match filters
        "willing_to_relocate":    PayloadSchemaType.BOOL,
        "is_pure_services":       PayloadSchemaType.BOOL,
        "is_pure_academic":       PayloadSchemaType.BOOL,
        "honeypot_flag":          PayloadSchemaType.BOOL,
        "has_product_experience": PayloadSchemaType.BOOL,
        "open_to_work_flag":      PayloadSchemaType.BOOL,
    }

    for field_name, schema_type in payload_indexes.items():
        print(f"  → Creating index: {field_name} ({schema_type.name})")
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=schema_type,
        )

    print(f"\n[setup_db] ✓ Collection '{COLLECTION_NAME}' ready with "
          f"{len(payload_indexes)} payload indexes.")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap the Qdrant 'candidates' collection."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the existing collection and recreate from scratch.",
    )
    args = parser.parse_args()

    # ── Connect to Qdrant ───────────────────────────────────────────────
    print(f"[setup_db] Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
        # Quick health check
        info = client.get_collections()
        print(f"[setup_db] Connected. Existing collections: "
              f"{[c.name for c in info.collections]}")
    except Exception as e:
        print(f"[setup_db] ✗ Cannot connect to Qdrant: {e}")
        print(f"[setup_db]   Make sure Qdrant is running:")
        print(f"[setup_db]   docker run -d --name qdrant "
              f"-p 6333:6333 -p 6334:6334 "
              f"-v $(pwd)/qdrant_storage:/qdrant/storage:z "
              f"qdrant/qdrant:latest")
        sys.exit(1)

    create_collection(client, recreate=args.recreate)


if __name__ == "__main__":
    main()
