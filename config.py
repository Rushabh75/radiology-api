"""
Central configuration — all tunable settings in one place.
Override any value via environment variable.
"""
import os

# ── Server ────────────────────────────────────────────────────────────────────
PORT    = int(os.environ.get("PORT", 8080))
WORKERS = int(os.environ.get("WORKERS", 4))
TIMEOUT = int(os.environ.get("TIMEOUT", 120))

# ── Classifier ────────────────────────────────────────────────────────────────
# Score >= RELEVANCE_THRESHOLD → predicted relevant
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", 0.5))

# Weights for region overlap vs modality compatibility
REGION_WEIGHT   = float(os.environ.get("REGION_WEIGHT",   0.72))
MODALITY_WEIGHT = float(os.environ.get("MODALITY_WEIGHT", 0.28))

# ── Data ──────────────────────────────────────────────────────────────────────
# The public eval JSON (27,614 labeled pairs, 996 cases) was used ONLY for
# tuning regex patterns and overlap scores.
# Final evaluation uses a separate private split — never seen during tuning.
PUBLIC_EVAL_CASES  = 996
PUBLIC_EVAL_PRIORS = 27614
