#!/usr/bin/env python3
"""Pre-download FlashRank model into models/flashrank (run once).

Usage (from project root):
  python services/download_reranker.py
  # or
  python -m services.download_reranker
"""

import sys
from pathlib import Path

# Allow running this file directly: python services/download_reranker.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from services.reranker import CACHE_DIR, get_reranker

if __name__ == "__main__":
    settings = get_settings()
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Model: {settings.rerank_model}")
    get_reranker().warmup()
    print("Done. Model is cached and ready.")
