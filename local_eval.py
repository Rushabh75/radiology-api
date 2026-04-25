"""
Local evaluator - tests your running server against the public eval JSON.
Place this file in the same folder as relevant_priors_public.json and run:
    python local_eval.py
Make sure your server is running first: python app.py
"""
import json
import requests
import sys
import time

SERVER = "http://127.0.0.1:8080"
EVAL_FILE = "relevant_priors_public.json"

print("=" * 60)
print("Radiology Prior Relevance — Local Evaluator")
print("=" * 60)

# ── 1. Check server is up ────────────────────────────────────────
print("\n[1] Checking server health...")
try:
    r = requests.get(f"{SERVER}/health", timeout=5)
    print(f"    Server OK: {r.json()}")
except Exception as e:
    print(f"    ERROR: Cannot reach server at {SERVER}")
    print(f"    Make sure 'python app.py' is running first.")
    print(f"    Detail: {e}")
    sys.exit(1)

# ── 2. Load eval data ────────────────────────────────────────────
print(f"\n[2] Loading {EVAL_FILE}...")
try:
    with open(EVAL_FILE) as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"    ERROR: {EVAL_FILE} not found.")
    print(f"    Place it in the same folder as this script.")
    sys.exit(1)

cases = data["cases"]
truth_list = data["truth"]
truth_map = {(t["case_id"], t["study_id"]): t["is_relevant_to_current"] for t in truth_list}

total_priors = sum(len(c["prior_studies"]) for c in cases)
relevant_count = sum(1 for t in truth_list if t["is_relevant_to_current"])
print(f"    Cases: {len(cases)}")
print(f"    Total priors: {total_priors}")
print(f"    Relevant: {relevant_count} ({relevant_count/total_priors*100:.1f}%)")
print(f"    Not relevant: {total_priors - relevant_count} ({(total_priors-relevant_count)/total_priors*100:.1f}%)")

# ── 3. Send request ──────────────────────────────────────────────
print(f"\n[3] Sending all {len(cases)} cases to {SERVER}/predict ...")

payload = {
    "challenge_id": data["challenge_id"],
    "schema_version": data["schema_version"],
    "cases": cases,
}

t0 = time.time()
try:
    resp = requests.post(f"{SERVER}/predict", json=payload, timeout=360)
    resp.raise_for_status()
except Exception as e:
    print(f"    ERROR: Request failed: {e}")
    sys.exit(1)

elapsed = round(time.time() - t0, 1)
print(f"    Done in {elapsed}s")

# ── 4. Score ─────────────────────────────────────────────────────
print(f"\n[4] Scoring predictions...")
predictions = resp.json()["predictions"]
print(f"    Predictions returned: {len(predictions)}")

correct = tp = tn = fp = fn = 0
skipped = 0

for p in predictions:
    key = (p["case_id"], p["study_id"])
    if key not in truth_map:
        skipped += 1
        continue
    expected = truth_map[key]
    predicted = p["predicted_is_relevant"]
    if predicted == expected:
        correct += 1
        if expected: tp += 1
        else: tn += 1
    else:
        if expected: fn += 1   # missed a relevant prior
        else: fp += 1          # wrongly included irrelevant prior

# Check for missing predictions (skipped priors count as wrong)
returned_keys = {(p["case_id"], p["study_id"]) for p in predictions}
missing = 0
for key in truth_map:
    if key not in returned_keys:
        missing += 1
        fn += 1  # skipped = wrong per challenge rules

total_scored = correct + fp + fn
accuracy = correct / total_scored * 100 if total_scored > 0 else 0

print("\n" + "=" * 60)
print(f"  ACCURACY:  {correct}/{total_scored} = {accuracy:.2f}%")
print("=" * 60)
print(f"  True Positives  (correctly marked relevant):     {tp}")
print(f"  True Negatives  (correctly marked irrelevant):   {tn}")
print(f"  False Positives (wrongly marked relevant):       {fp}")
print(f"  False Negatives (wrongly marked irrelevant):     {fn}")
if missing:
    print(f"  Missing predictions (skipped = wrong):          {missing}")
print(f"\n  Time taken: {elapsed}s")

print("\n  Status: GOOD")
