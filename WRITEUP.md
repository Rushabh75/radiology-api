# Relevant Priors — Experiment Write-Up

## Problem Statement

Given a patient's **current radiology examination** and a list of their **prior examinations**, predict which priors a radiologist would want to see while reading the current study.

Input: `study_description` strings (e.g. `"CT CHEST WITH CONTRAST"`)
Output: `predicted_is_relevant: true | false` for each prior

---

## Approach

### Architecture: Two-Stage Pipeline

```
For each prior:
  1. Rule-based classifier  →  confident? (score ≥ 0.70 or ≤ 0.30)  →  return result
                            →  uncertain? (0.30–0.70)                →  queue for LLM
  2. Batched LLM call (ONE call per case, all uncertain priors)
     LLM failure → rule-based fallback
```

### Stage 1 — Rule-Based Classifier (`classifier.py`)

**Region extraction** via regex over 25 canonical body regions:

| Region | Example descriptions matched |
|---|---|
| `brain` | `CT HEAD`, `MRI BRAIN`, `CT head/brain`, `NE EEG Request` |
| `cardiac` | `ECHO 2D`, `TTE`, `LUM TTE`, `CT FFR`, `NM myo perf`, `CT angio coronary` |
| `chest` | `CT CHEST`, `XR Chest 1V`, `NM pul perfusion`, `THORACENTESIS` |
| `breast` | `MAM screen`, `MAMMOGRAPHY SCREENING BILATERAL`, `DIGITAL SCREENER W CAD`, `Seed Localization` |
| `abdomen` | `CT ABD`, `US ABDOMINAL`, `CHOLANGIOGRAM`, `PARACENTESIS`, `NEPHROSTOMY`, `RENAL COLIC` |
| `spine_lumbar` | `MRI LUMBAR SPINE`, `CT lumbar spine`, `LUMBAR SPINE LIMITED VIEWS` |
| `whole_body` | `PET/CT`, `BONE SCAN`, `NM bone scan`, `PET-CT SKULL TO THIGH` |
| `vascular_head` | `CAROTID ULTRASOUND`, `CT angio carotid`, `VAS transcranial doppler` |

**Key deduplication rules** applied after extraction:
- `abdomen_pelvis` found → drop individual `abdomen`/`pelvis`
- Specific spine level found → drop generic `spine_whole`
- `spine_thoracic` found alongside `chest` → drop `chest` (they're distinct)
- `breast` found alongside `chest` → drop `chest`
- `cardiac` found alongside `chest` (no vascular) → drop `chest`
- PET/whole-body → strip spurious `brain`/`lower_extremity` from "skull to thigh"
- DXA/bone-density → strip spurious `lower_extremity`/`pelvis` hits

**Region overlap scoring** via a curated 60-entry matrix. Key design decisions:

- `cardiac` ↔ `chest`: set to **0.5** (exactly at threshold) → always goes to LLM
- `vascular_head` ↔ `brain`: set to **0.5** → always goes to LLM
- `spine_thoracic` ↔ `chest`: set to **0.5** → always goes to LLM
- `gi_tract` ↔ `chest`: set to **0.5** → always goes to LLM (esophagram sometimes IS relevant to chest CT)
- `whole_body` ↔ `breast`: **0.35** → bone scan not relevant to mammogram
- Different specific spine levels: **0.10–0.35**
- Same region: **1.0** always

**Final score:** `0.72 × region_overlap + 0.28 × modality_compat`

**Confidence gate:** Score ≥ 0.70 or ≤ 0.30 → rule-based only. Middle range → LLM.

**Modality compatibility:** Same modality = 1.0, MRI↔CT = 0.85, XR↔CT = 0.65, breast US↔mammo = 0.80.

### Stage 2 — LLM Fallback (`llm_client.py`)

- Model: **Claude Haiku** (fast, low-cost, accurate for binary classification)
- All uncertain priors for a case batched into **one API call** (avoids timeout)
- 30-second timeout per call with graceful rule-based fallback
- In-memory MD5 cache on `(current_desc, prior_desc)` — no re-scoring on retries

---

## Results on Public Eval (27,614 labeled pairs, 996 cases)

| Stage | Accuracy | FP | FN |
|---|---|---|---|
| Initial code (63 patterns) | 63.4% | — | — |
| After pattern expansion | 90.5% | 2,278 | 342 |
| After targeted overlap fixes | 92.6% | 1,801 | 237 |
| **Estimated with LLM fallback** | **~96.6%** | — | — |

Label distribution: 23.8% relevant, 76.2% not relevant.

---

## Experiments

### Experiment 1 — Pattern Coverage Analysis

Ran full eval, found `63%` accuracy. Identified that real data uses abbreviations not covered by initial patterns:

- `MAM screen BI with tomo` → needed `mam\b` pattern
- `MAMMOGRAPHY SCREENING BILATERAL` → needed `mammography` keyword
- `US ABDOMINAL` → needed `us abdominal` / `abdominal\b`
- `NMmyo perf str/rest SPEC` → needed `myo perf` / `nmmyo`
- `LUM TTE W/DOPPL COMPLETE` → needed `lum tte` / `tte\b`
- `DIGITAL SCREENER W CAD` → needed `digital screener` for breast
- `BIOPSY - ABDOMINL MASS` → needed `abdominl` typo match for abdomen

**Result:** 63% → 90.5% from pattern fixes alone.

### Experiment 2 — Overlap Score Calibration

Studied which pairs were FP vs FN to find the right overlap scores:

- `cardiac` ↔ `chest`: dataset has BOTH relevant and not-relevant cases (echo sometimes compared to chest CT, sometimes not). Set to 0.5 → LLM decides.
- `vascular_head` ↔ `brain`: carotid US sometimes compared to CT head, sometimes not. Set to 0.5 → LLM decides.
- `spine_thoracic` ↔ `chest`: chest X-ray is sometimes pulled for thoracic spine reading. Set to 0.5 → LLM.
- `whole_body` ↔ `breast`: PET scan is NOT relevant to mammogram (0.35).
- `bone_density` ↔ `pelvis`: DXA is NOT relevant to pelvic US (0.25).

**Result:** 90.5% → 92.6%.

### Experiment 3 — LLM Impact Estimate

Uncertain pairs (score 0.30–0.70): 5,041 total, 1,304 currently wrong.
LLM at 85% accuracy on these would fix ~1,108 errors → **~96.6% total accuracy**.

### Experiment 4 — Confidence Threshold

Tested thresholds for routing to LLM:
- ≥0.75 / ≤0.25: fewer LLM calls, but misses some borderline correct pairs
- ≥0.70 / ≤0.30: good balance — catches ambiguous region pairs without over-routing
- ≥0.65 / ≤0.35: too many LLM calls, risks timeout on large batches

**Chose 0.70/0.30.**

---

## Next Improvements

### Short-term (1–2 days)
1. **Logistic regression on public labels** — train a simple LR on `[region_overlap, modality_compat, days_since_prior, same_modality_bool]` features using the 27,614 labeled pairs to tune weights and threshold automatically
2. **Date recency feature** — priors > 5 years old slightly less relevant; add `recency_score = exp(-years/3)`
3. **Exact description match cache** — pre-compute results for all 827 unique descriptions seen in training

### Medium-term
4. **Medical text embeddings** — encode descriptions with BioLinkBERT / ClinicalBERT, use cosine similarity as additional feature for unknown/rare descriptions
5. **CPT code integration** — if CPT codes available, perfect region/modality classification with no parsing
6. **Study type ontology** — formalize the region hierarchy (e.g. vascular_head IS-A brain_region) for better transitivity

### Long-term
7. **Reader preference modeling** — different subspecialties have different relevance norms; model per-reader or per-department preferences
8. **Report-text NER** — if prior report text available, extract anatomical findings to improve relevance beyond study type
