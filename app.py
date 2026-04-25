"""
Radiology Prior Relevance API
POST /predict  — main prediction endpoint
GET  /health   — health check
"""
import logging
import time
import uuid
from flask import Flask, request, jsonify
from classifier import rule_based_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

app = Flask(__name__)

THRESHOLD = 0.5


def process_case(case: dict) -> list:
    case_id = str(case["case_id"])
    cur_desc = case["current_study"].get("study_description", "")
    priors = case.get("prior_studies", [])

    predictions = []
    for prior in priors:
        prior_id = str(prior["study_id"])
        pri_desc = prior.get("study_description", "")
        score, _ = rule_based_score(cur_desc, pri_desc)
        predictions.append({
            "case_id": case_id,
            "study_id": prior_id,
            "predicted_is_relevant": score >= THRESHOLD,
        })
    return predictions


@app.route("/predict", methods=["POST"])
def predict():
    req_id = str(uuid.uuid4())[:8]
    t0 = time.time()

    try:
        body = request.get_json(force=True)
    except Exception as e:
        log.error(f"[{req_id}] JSON parse error: {e}")
        return jsonify({"error": "invalid JSON"}), 400

    cases = body.get("cases", [])
    total_priors = sum(len(c.get("prior_studies", [])) for c in cases)
    log.info(f"[{req_id}] {len(cases)} cases, {total_priors} priors")

    all_predictions = []
    for case in cases:
        case_id = case.get("case_id", "?")
        try:
            all_predictions.extend(process_case(case))
        except Exception as e:
            log.error(f"[{req_id}] Error on case {case_id}: {e}", exc_info=True)
            # Never skip — emit False for all priors in failed case
            for prior in case.get("prior_studies", []):
                all_predictions.append({
                    "case_id": str(case_id),
                    "study_id": str(prior["study_id"]),
                    "predicted_is_relevant": False,
                })

    elapsed = round(time.time() - t0, 2)
    log.info(f"[{req_id}] Done: {len(all_predictions)} predictions in {elapsed}s")
    return jsonify({"predictions": all_predictions})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": "radiology-prior-relevance", "version": "1.0"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
