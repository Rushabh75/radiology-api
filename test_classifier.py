"""
Unit tests for the rule-based classifier.
Run with: python test_classifier.py
"""
import sys
import traceback

sys.path.insert(0, ".")
from classifier import extract_regions, extract_modality, rule_based_score, predict_relevance

PASS = 0
FAIL = 0


def check(name, got, expected):
    global PASS, FAIL
    if got == expected:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}")
        print(f"        expected: {expected}")
        print(f"        got:      {got}")
        FAIL += 1


def approx(a, b, tol=0.05):
    return abs(a - b) <= tol


# ── Region extraction ─────────────────────────────────────────────────────────
print("\n=== Region Extraction ===")
check("CT HEAD",                extract_regions("CT HEAD WITHOUT CONTRAST"),          ["brain"])
check("MRI BRAIN",              extract_regions("MRI BRAIN WITHOUT CONTRAST"),        ["brain"])
check("CT CHEST",               extract_regions("CT CHEST WITH CONTRAST"),            ["chest"])
check("XR Chest",               extract_regions("XR Chest 1V Frontal Only"),          ["chest"])
check("MAM screen",             extract_regions("MAM screen BI with tomo"),           ["breast"])
check("MAMMOGRAPHY SCREENING",  extract_regions("MAMMOGRAPHY SCREENING BILATERAL"),   ["breast"])
check("DIGITAL SCREENER W CAD", extract_regions("DIGITAL SCREENER W CAD"),            ["breast"])
check("US ABDOMINAL",           extract_regions("US ABDOMINAL"),                      ["abdomen"])
check("CT ABD/PEL",             extract_regions("CT ABD/PEL WITH CNTRST"),            ["abdomen_pelvis"])
check("MRI LUMBAR SPINE",       extract_regions("MRI LUMBAR SPINE WITHOUT CNTRS"),    ["spine_lumbar"])
check("MRI CERVICAL SPINE",     extract_regions("MRI CERVICAL SPINE WITHOUT CNTR"),   ["spine_cervical"])
check("MRI THORACIC SPINE",     extract_regions("MRI thoracic spine wo con"),         ["spine_thoracic"])
check("ECHO TTE",               extract_regions("ECHO 2D Mmode transthorac TTE"),     ["cardiac"])
check("NM myo perf",            extract_regions("NMmyo perf str/rest SPEC-no p"),     ["cardiac"])
check("CT coronary",            extract_regions("CT angio coronary artery"),          ["cardiac"])
check("CAROTID US",             extract_regions("CAROTID ULTRASOUND"),                ["vascular_head"])
check("PET CT skull to thigh",  extract_regions("PET-CT SKULL TO THIGH SUBSQNT"),     ["whole_body"])
check("NM pul perfusion",       extract_regions("NM pul perfusion"),                  ["chest"])
check("THYROID US",             extract_regions("US thyroid"),                        ["thyroid"])
r = set(extract_regions("BONE DENSITY (HIP/SPINE)"))
check("BONE DENSITY", r == {"bone_density", "spine_whole", "hip"}, True)
check("GI SERIES",              extract_regions("GI SERIES, SINGLE CONTRAST"),        ["gi_tract"])
check("PARACENTESIS",           extract_regions("US PARACENTESIS"),                   ["abdomen"])

# ── Deduplication rules ───────────────────────────────────────────────────────
print("\n=== Deduplication Rules ===")
# Thoracic spine should NOT also tag as chest
r = extract_regions("MRI thoracic spine wo con")
check("thoracic spine != chest",     "chest" not in r,     True)
check("thoracic spine tagged",       "spine_thoracic" in r, True)

# Breast should NOT also tag as chest
r = extract_regions("MAM screen BI with tomo")
check("breast != chest",             "chest" not in r,     True)
check("breast tagged",               "breast" in r,        True)

# Cardiac should NOT also tag as chest
r = extract_regions("ECHO 2D Mmode transthorac TTE")
check("cardiac != chest",            "chest" not in r,     True)
check("cardiac tagged",              "cardiac" in r,        True)

# PET skull-to-thigh should NOT tag brain or lower_extremity
r = extract_regions("PET-CT SKULL TO THIGH SUBSQNT")
check("pet != brain",                "brain" not in r,         True)
check("pet != lower_extremity",      "lower_extremity" not in r, True)
check("pet == whole_body",           "whole_body" in r,         True)

# Abd+pelvis combined should drop individual tags
r = extract_regions("CT ABD/PEL WITH CNTRST")
check("abd_pelvis drops abdomen",    "abdomen" not in r,       True)
check("abd_pelvis drops pelvis",     "pelvis" not in r,        True)
check("abd_pelvis tagged",           "abdomen_pelvis" in r,    True)

# Specific spine level drops spine_whole
r = extract_regions("MRI LUMBAR SPINE WITHOUT CNTRS")
check("lumbar drops spine_whole",    "spine_whole" not in r,   True)
check("lumbar tagged",               "spine_lumbar" in r,      True)

# ── Modality extraction ───────────────────────────────────────────────────────
print("\n=== Modality Extraction ===")
check("MRI",    extract_modality("MRI BRAIN WITHOUT CONTRAST"), "mri")
check("CT",     extract_modality("CT CHEST WITH CONTRAST"),     "ct")
check("XR",     extract_modality("XR Chest 1V Frontal Only"),   "xray")
check("US",     extract_modality("ULTRASOUND ABDOMEN"),         "us")
check("MAMMO",  extract_modality("MAM screen BI with tomo"),    "mammo")
check("NM",     extract_modality("NM bone scan whole body"),    "nm")

# ── Relevance predictions ─────────────────────────────────────────────────────
print("\n=== Relevance Predictions ===")
def rel(cur, pri):
    pred, score, _ = predict_relevance(cur, pri)
    return pred

# Same region → relevant
check("CT chest vs CT chest",           rel("CT CHEST WITH CONTRAST",          "CT CHEST WITHOUT CONTRAST"),            True)
check("MRI brain vs CT head",           rel("MRI BRAIN WITHOUT CONTRAST",       "CT HEAD WITHOUT CNTRST"),               True)
check("MAM screen vs MAMMOGRAPHY",      rel("MAM screen BI with tomo",          "MAMMOGRAPHY SCREENING BILATERAL"),      True)
check("US abdomen vs CT abd/pel",       rel("US ABDOMINAL",                     "CT ABD/PEL WITH CNTRST"),               True)
check("MRI lumbar vs XR lumbar",        rel("MRI LUMBAR SPINE WITHOUT CNTRS",   "LUMBAR SPINE, LIMITED VIEWS"),          True)
check("Echo vs Echo",                   rel("ECHO 2D Mmode transthorac TTE",    "ECHO definity study w/full ech"),       True)
check("MRI left knee vs MRI right knee",rel("MRI RIGHT KNEE WITHOUT CONTRAST",  "MRI LEFT KNEE WITHOUT CONTRAST"),       True)

# Different region → not relevant
check("CT chest vs MRI brain",          rel("CT CHEST WITH CONTRAST",           "MRI BRAIN WITHOUT CONTRAST"),           False)
check("MAM screen vs CT chest",         rel("MAM screen BI with tomo",          "CT CHEST WITH CNTRST"),                 False)
check("MRI brain vs MRI lumbar",        rel("MRI BRAIN WITHOUT CONTRAST",       "MRI LUMBAR SPINE WITHOUT CNTRS"),       False)
check("CT chest vs CT abd/pel",         rel("CT CHEST WITH CNTRST",             "CT ABD/PEL WITH CNTRST"),               False)
check("PET whole body vs MAM",          rel("PET-CT SKULL TO THIGH SUBSQNT",    "MAMMOGRAPHY SCREENING BILATERAL"),      False)

# Different spine levels → not relevant
check("lumbar vs cervical",             rel("MRI LUMBAR SPINE WITHOUT CNTRS",   "MRI CERVICAL SPINE WITHOUT CNTR"),      False)
check("thoracic vs lumbar",             rel("MRI thoracic spine wo con",        "MRI LUMBAR SPINE WITHOUT CNTRS"),       False)

# Adjacent spine levels → not relevant (score < 0.5)
check("cervical vs thoracic",           rel("MRI CERVICAL SPINE WITHOUT CNTR",  "MRI thoracic spine wo con"),            False)

# ── Score ranges ──────────────────────────────────────────────────────────────
print("\n=== Score Sanity Checks ===")
def score(cur, pri):
    _, s, _ = predict_relevance(cur, pri)
    return s

check("same study = 1.0",   approx(score("CT CHEST WITH CONTRAST", "CT CHEST WITHOUT CONTRAST"), 1.0), True)
check("unrelated < 0.3",    score("CT CHEST WITH CONTRAST", "MRI BRAIN WITHOUT CONTRAST") < 0.3,       True)
check("breast vs chest < 0.2", score("MAM screen BI with tomo", "CT CHEST WITH CNTRST") < 0.2,        True)

# ── Summary ───────────────────────────────────────────────────────────────────
total = PASS + FAIL
print(f"\n{'='*50}")
print(f"Results: {PASS}/{total} passed", "✓" if FAIL == 0 else f"— {FAIL} FAILED")
print(f"{'='*50}")
sys.exit(0 if FAIL == 0 else 1)
