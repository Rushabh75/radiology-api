# Relevant Priors — Write-Up

## Approach

The deployed system is a **pure rule-based classifier** — no LLM, no external API calls.
Every prediction is deterministic and runs in under 1 second for a full batch.

### How it works

For each (current exam, prior exam) pair:

1. **Extract body region(s)** from the study description using regex patterns over 25 canonical regions: `brain`, `chest`, `cardiac`, `breast`, `abdomen`, `abdomen_pelvis`, `pelvis`, `spine_cervical`, `spine_thoracic`, `spine_lumbar`, `spine_whole`, `thyroid`, `upper_extremity`, `lower_extremity`, `hip`, `vascular_head`, `vascular_chest`, `vascular_abdomen`, `vascular_peripheral`, `neck_soft_tissue`, `neck_us`, `bone_density`, `whole_body`, `gi_tract`, `msk_general`

2. **Extract modality** (MRI, CT, X-ray, US, NM, mammo, DXA, fluoro, angio)

3. **Score region overlap** via a 60-entry lookup table (same region = 1.0, unrelated = 0.0, partial overlap = 0.2–0.9)

4. **Score modality compatibility** (same modality = 1.0, MRI↔CT = 0.85, XR↔CT = 0.65, etc.)

5. **Combine:** `score = 0.72 × region_overlap + 0.28 × modality_compat`

6. **Threshold at 0.5** → relevant if score ≥ 0.5

All thresholds and weights are configurable via environment variables (see `config.py`).

### Key deduplication rules applied after region extraction

- `abdomen_pelvis` found → drop individual `abdomen` / `pelvis` tags
- Specific spine level found → drop generic `spine_whole`
- `spine_thoracic` alongside `chest` → drop `chest` (distinct studies)
- `breast` alongside `chest` → drop `chest`
- `cardiac` alongside `chest` (no vascular) → drop `chest`
- PET/whole-body → strip spurious `brain` / `lower_extremity` from "skull to thigh"

---

## Data Split Discipline

| Data | Used for |
|---|---|
| Public eval JSON (996 cases, 27,614 pairs) | **Tuning only** — regex patterns, overlap matrix values, deduplication rules |
| Private eval JSON | **Never seen** — final scoring only |

The public eval labels were used iteratively to:
- Identify missing abbreviations (e.g. `MAM screen`, `US ABDOMINAL`, `NMmyo perf`)
- Tune overlap scores for ambiguous region pairs (cardiac/chest, vascular_head/brain)
- Validate deduplication logic

No held-out validation split was used from the public data — all 27,614 pairs were used for tuning. This means the public eval accuracy (92.63%) is optimistic; the private split is the true measure.

---

## Results

| Metric | Value |
|---|---|
| Public eval accuracy | **92.63%** (25,578 / 27,614) |
| True Positives | 6,330 |
| True Negatives | 19,248 |
| False Positives | 1,799 |
| False Negatives | 237 |
| Avg latency (996 cases) | ~8 seconds |
| External API calls | None |

Label distribution: 23.8% relevant, 76.2% not relevant.

---

## Experiments

### Experiment 1 — Pattern coverage
Initial regex patterns gave 63.4% accuracy. The real dataset uses heavy abbreviations not in the initial patterns (`MAM screen`, `LUM TTE`, `DIGITAL SCREENER W CAD`, `NMmyo perf`, `US ABDOMINAL`). Adding these raised accuracy to 90.5%.

### Experiment 2 — Overlap score calibration
Studied FP/FN pairs to tune overlap scores:
- `cardiac` ↔ `chest` set to 0.5 (both FP and FN cases exist in data — genuinely ambiguous)
- `breast` ↔ `chest` set to 0.0 (mammogram is never relevant to chest CT)
- `whole_body` ↔ `breast` set to 0.35 (PET scan not relevant to mammogram)
- Adjacent spine levels (cervical↔thoracic, thoracic↔lumbar) set to 0.25 so MRI modality bonus doesn't push above threshold

### Experiment 3 — LLM fallback (abandoned)
Tested Groq Llama 3.1 8B as a fallback for ambiguous pairs (score 0.43–0.57).
Result: 92.53% — worse than rule-only (92.62%). The model flipped too many already-correct rule predictions.
Decision: removed LLM entirely. The rule-based system is more reliable for this task.

### Experiment 4 — Weight tuning
Tested region:modality weight ratios of 60:40, 70:30, 72:28, 80:20.
72:28 gave best results — region overlap dominates but modality still breaks ties.

---

## Configuration

All runtime settings are in `config.py` and overridable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | 8080 | Server port |
| `WORKERS` | 4 | Gunicorn worker count |
| `TIMEOUT` | 120 | Request timeout (seconds) |
| `RELEVANCE_THRESHOLD` | 0.5 | Score cutoff for relevant |
| `REGION_WEIGHT` | 0.72 | Weight of region overlap score |
| `MODALITY_WEIGHT` | 0.28 | Weight of modality compatibility score |

---

## Running Tests

```bash
python test_classifier.py
```

Runs 60 unit tests covering:
- Region extraction for all 25 region types
- All deduplication rules
- Modality extraction
- End-to-end relevance predictions for known pairs
- Score sanity checks (same study = 1.0, unrelated < 0.3)

---

## Next Improvements

1. **Train overlap weights** — use logistic regression on the public labels to learn optimal region/modality weights rather than hand-tuning
2. **Date recency feature** — priors > 5 years old are less relevant; add `exp(-years/3)` decay factor
3. **Medical text embeddings** — encode descriptions with BioLinkBERT for semantic similarity on rare/unseen study types
4. **Held-out validation split** — reserve 20% of public data for validation so tuning decisions are evaluated on unseen data
5. **Larger LLM** — Llama 70B or Claude Haiku would likely improve over rule-only on ambiguous pairs (Llama 8B did not)
