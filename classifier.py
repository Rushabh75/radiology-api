"""
Rule-based relevance classifier for radiology prior studies.
Tuned against 27,614 labeled pairs from the public eval set.
"""
import re
from config import RELEVANCE_THRESHOLD, REGION_WEIGHT, MODALITY_WEIGHT

REGION_PATTERNS = {
    # ── Brain / Head ──────────────────────────────────────────────────────────
    "brain": (
        r"\b(brain|cerebr|intracran|cranial|skull|sella|pituitary|"
        r"orbit|orbits|iac|internal auditory|acoustic|temporal bone|mastoid|"
        r"facial|maxillofacial|jaw|mandible|tmj|paranasal|sinus|sinuses|"
        r"maxfacial|ct head|mr head|mri head|"
        r"brain spect|datscan|eeg|"
        r"ir cerebral|cerebral stroke|brain perfusion|"
        r"mr angio head|mra head)\b"
        r"|(?<!\w)(head)(?!\s+and\s+neck)(?!\s*&\s*neck)\b"
    ),
    # ── Neck soft tissue ──────────────────────────────────────────────────────
    "neck_soft_tissue": (
        r"\b(soft tissue neck|neck soft tissue|ct neck|ct soft tissue neck|"
        r"xr neck soft tissue|neck soft tissue)\b"
    ),
    # ── Neck US (thyroid/neck ultrasound) ─────────────────────────────────────
    "neck_us": (
        r"\b(us head and neck|head and neck.*soft|us neck|"
        r"us.*soft tissue neck|soft tissue neck.*us)\b"
    ),
    # ── Spine ─────────────────────────────────────────────────────────────────
    "spine_cervical": (
        r"\b(cervical spine|c-spine|c spine|cerv.*spine|spine.*cerv|"
        r"cervicl spine|mri cerv|ct cerv|xr cerv|cervical|"
        r"cervicl spine)\b"
    ),
    "spine_thoracic": (
        r"\b(thoracic spine|t-spine|t spine|thor.*spine|spine.*thor|"
        r"xr thoracic spine|mri thoracic spine|ct thoracic spine|"
        r"xr thoracic spine)\b"
    ),
    "spine_lumbar": (
        r"\b(lumbar spine|l-spine|l spine|lumbosacral|ls spine|"
        r"lumbar|lum spine|mri lum|ct lum|xr lum|lumbar spne|"
        r"ct lumbar spine)\b"
    ),
    "spine_whole": (
        r"\b(whole spine|entire spine|total spine|spine|scoliosis)\b"
    ),
    # ── Chest / Thorax ────────────────────────────────────────────────────────
    "chest": (
        r"\b(chest|thorax|lung|lungs|pulmon|pleura|mediastin|rib\b|ribs|sternum|"
        r"chest.*view|chest.*frontal|chest pa|chest ap|chest 1v|chest 2v|chest 3v|"
        r"chest 1 view|chest 2 view|chest frontal|xr chest|"
        r"nm pul|pul perfusion|pulmonary perfusion|lung perfusion|"
        r"ventilation.*perfusion|v/q scan|thoracentesis|"
        r"chest wo|chest w con|ct chest)\b"
    ),
    # ── Cardiac / Echo ────────────────────────────────────────────────────────
    "cardiac": (
        r"\b(cardiac|coronary|echo\b|echocardiograph|tte\b|tee\b|"
        r"transesophageal|transthorac|lum tte|chemo tte|mmode|m-mode|"
        r"ct angio.*coron|ct coronary|coronary calc|ct ffr|ffr\b|"
        r"nm myo|myo perf|myocardial|nmmyo|nm.*perf|myo.*perf|perf.*spect|"
        r"echo definity|echo 2d|3d transesophageal|stress.*echo|"
        r"mr cardiac|mri cardiac|cardiac mri|"
        r"ct angio coronary|angio coronary|coronary angio|"
        r"ct angiogram.*coron|ct angiogram.*cardiac)\b"
    ),
    # ── Vascular chest ────────────────────────────────────────────────────────
    "vascular_chest": (
        r"\b(cta chest|ct angio.*chest|ct angiogram.*chest|"
        r"pulmonary angio|pulmonary embol|ct angio chest)\b"
    ),
    # ── Abdomen ───────────────────────────────────────────────────────────────
    "abdomen": (
        r"\b(abdomen|abdom|abdomin|us abdominal|abdominal us|abdominal\b|"
        r"liver|hepat|spleen|splenic|pancrea|"
        r"renal|kidney|kidneys|adrenal|gallbladd|biliary|"
        r"ivc|mesentery|retroperiton|bowel|colon|appendix|stomach|gastric|"
        r"enterography|colonograph|mrcp|cholangiogram|cholangio|ercp|"
        r"paracentesis|peritoneal|nephrostomy|nephro|"
        r"abdl\b|abdld\b|abd1v\b|abdominal arteriogram|"
        r"ct urogram|urogram|hematuria|renal colic|"
        r"biopsy.*abdom|abdom.*biopsy|abdominl|"
        r"ct biopsy abdomen|drainage.*periton|drainage.*retroperiton|"
        r"ir nephrostomy|collection.*peritoneal)\b"
    ),
    # ── Abdomen + Pelvis combined ─────────────────────────────────────────────
    "abdomen_pelvis": (
        r"\b(abd.*pel|pel.*abd|abdomen.*pelvis|pelvis.*abdomen|"
        r"abd/pel|abd_pel|abdomen pelvis|ct cap\b|"
        r"ct abd/pel|ct abd_pel|ct abd pel|"
        r"ct angiogram.*abd.*pel|ct angiogram.*abd_pel)\b"
    ),
    # ── Pelvis ────────────────────────────────────────────────────────────────
    "pelvis": (
        r"\b(pelvis|pelvic|bladder|prostate|uterus|uterine|"
        r"ovari|ovary|rectum|rectal|sigmoid|"
        r"sacrum|sacral|sacroiliac|si joint|coccyx|"
        r"endovaginal|transvaginal|gyn\b|us pelvic|ct pelvis|mr pelvis|mri pelvis|"
        r"xr pelvis|pelvic.*us|us.*pelvic)\b"
    ),
    # ── Breast / Mammography ──────────────────────────────────────────────────
    "breast": (
        r"\b(breast|mammogram|mammograph|mammography|tomosynthesis|"
        r"mam\b|mam |3d mammo|mam screen|mam diag|mam us|"
        r"mam stereo|stereo bx|stereotactic biopsy|breast biopsy|"
        r"breast ultrasound|breast locali|breast specimen|"
        r"breast fine needle|avma|bilat.*mammo|mammo.*bilat|"
        r"r2 mammo|digitized film|combo.*hd|screen.*bilat|bilat.*screen|"
        r"digital screener|screener.*cad|mammo dx|dx.*mammo|"
        r"seed locali.*mammo|mammo.*seed|wire locali.*mammo|"
        r"guided breast biopsy|mammo.*biopsy|biopsy.*mammo|"
        r"standard screening combo|screening combo)\b"
    ),
    # ── Thyroid ───────────────────────────────────────────────────────────────
    "thyroid": (
        r"\b(thyroid|thyroid scan|us thyroid|thyroid/soft tissue|"
        r"thyroid uptake|thyroid ablat|thyroid nodule|parathyroid)\b"
    ),
    # ── Upper extremity ───────────────────────────────────────────────────────
    "upper_extremity": (
        r"\b(shoulder|humerus|elbow|forearm|wrist|hand|finger|thumb|"
        r"upper extremity|upper arm|upper extrem|"
        r"radius\b|ulna\b|clavicle|scapula|acromioclavicular|ac joint|"
        r"ct ue|mr ue|xr ue|ue\b|up venous|upper.*venous|venous.*arm|"
        r"ct uppr|uppr.*extrem)\b"
    ),
    # ── Lower extremity ───────────────────────────────────────────────────────
    "lower_extremity": (
        r"\b(femur|knee|tibia|fibula|ankle|foot\b|feet|toe\b|toes|"
        r"lower extremity|lower extrem|lower leg|thigh|"
        r"patella|calcaneus|heel|metatars|"
        r"ct le|mr le|xr le|le\b|claudication|arterial.*leg|arterial.*lower|"
        r"arterial imaging.*leg|arterial flow.*lower|"
        r"ct lower|lower.*extrem)\b"
    ),
    # ── Hip (overlaps both lower extremity and pelvis) ────────────────────────
    "hip": r"\b(hip\b|hips\b|femoral head)\b",
    # ── Vascular peripheral ───────────────────────────────────────────────────
    "vascular_peripheral": (
        r"\b(venous doppler|venous imaging|vas venous|doppler.*venous|"
        r"arterial flow study|arterial imaging bilat|"
        r"peripheral.*angio|ankle brachial|"
        r"arterial imaging right|arterial imaging left)\b"
    ),
    # ── Vascular head / neck ──────────────────────────────────────────────────
    "vascular_head": (
        r"\b(mra brain|mra head|cta head|cta brain|"
        r"cerebral angio|carotid|carotid us|carotid ultrasound|"
        r"transcranial doppler|vas transcranial|ct angio carotid|"
        r"ct angiogram.*carotid|carotid angio|angio.*carotid)\b"
    ),
    # ── Vascular abdomen ──────────────────────────────────────────────────────
    "vascular_abdomen": (
        r"\b(ct angiogram.*abd|cta abdomen|mesenteric|renal arteri|"
        r"abdominal arteriogram|aorta)\b"
    ),
    # ── Bone density ──────────────────────────────────────────────────────────
    "bone_density": (
        r"\b(bone density|dxa\b|dexa\b|bmd\b|bmdcwo|densitometry)\b"
    ),
    # ── Whole body / nuclear ──────────────────────────────────────────────────
    "whole_body": (
        r"\b(whole body|total body|bone scan|pet scan|pet/ct|pet-ct|"
        r"nuclear med|bone scan total|skull to thigh|skull.*thigh|"
        r"nm bone scan)\b"
    ),
    # ── GI tract ─────────────────────────────────────────────────────────────
    "gi_tract": (
        r"\b(swallow|esophagram|upper gi|small bowel follow|"
        r"barium enema|gi series|cookie swallow|modified barium)\b"
    ),
    # ── MSK general ───────────────────────────────────────────────────────────
    "msk_general": r"\b(musculoskeletal|msk|arthritis)\b",
}

MODALITY_PATTERNS = {
    "mri":    r"\b(mri\b|mr\b|magnetic resonance|flair|dwi|dti|mrcp|mra|wo con|wo/w|w con)\b",
    "ct":     r"\b(ct\b|cta\b|computed tomography|cat scan)\b",
    "xray":   r"\b(x-ray|xray|radiograph|plain film|kub|view\b|views\b|frontal\b|lateral\b|pa\b|ap\b|portable)\b",
    "us":     r"\b(ultrasound|us\b|echo\b|sonograph|doppler|tte\b|tee\b|transthorac|transesophag)\b",
    "nm":     r"\b(nuclear|pet\b|spect\b|bone scan|thyroid scan|tagged rbc|myo perf|nmmyo)\b",
    "mammo":  r"\b(mammo|mammograph|mam\b|mam |tomosynthesis|3d mammo)\b",
    "fluoro": r"\b(fluoro|barium|swallow|esophagram|upper gi|cookie)\b",
    "dxa":    r"\b(dxa\b|dexa\b|bone density|bmd\b)\b",
    "angio":  r"\b(angiograph|angiogram|interventional)\b",
}

MODALITY_COMPAT = {
    ("mri",   "mri"):   1.0,
    ("ct",    "ct"):    1.0,
    ("xray",  "xray"):  0.9,
    ("us",    "us"):    0.9,
    ("nm",    "nm"):    0.9,
    ("mammo", "mammo"): 1.0,
    ("dxa",   "dxa"):   1.0,
    ("mri",   "ct"):    0.85, ("ct",    "mri"):   0.85,
    ("xray",  "ct"):    0.65, ("ct",    "xray"):  0.65,
    ("xray",  "mri"):   0.5,  ("mri",   "xray"):  0.5,
    ("us",    "ct"):    0.65, ("ct",    "us"):    0.65,
    ("us",    "mri"):   0.65, ("mri",   "us"):    0.65,
    ("nm",    "ct"):    0.7,  ("ct",    "nm"):    0.7,
    ("nm",    "mri"):   0.6,  ("mri",   "nm"):    0.6,
    ("mammo", "us"):    0.8,  ("us",    "mammo"): 0.8,
    ("mammo", "mri"):   0.7,  ("mri",   "mammo"): 0.7,
    ("xray",  "us"):    0.45, ("us",    "xray"):  0.45,
}

# ─── Region overlap matrix ───────────────────────────────────────────────────
# Key insight: cardiac↔chest and vascular_head↔brain are set at EXACTLY 0.5
# so that LLM is always consulted for these ambiguous pairs (both FP and FN exist).
REGION_OVERLAP = {
    # Brain / head
    ("brain",             "vascular_head"):     0.5,   # intentionally at threshold → LLM
    ("vascular_head",     "brain"):             0.5,
    ("brain",             "neck_soft_tissue"):  0.3,
    ("neck_soft_tissue",  "brain"):             0.3,
    ("brain",             "neck_us"):           0.2,
    ("neck_us",           "brain"):             0.2,
    ("neck_us",           "thyroid"):           0.7,
    ("thyroid",           "neck_us"):           0.7,
    ("neck_us",           "neck_soft_tissue"):  0.8,
    ("neck_soft_tissue",  "neck_us"):           0.8,
    ("neck_us",           "vascular_head"):     0.5,
    ("vascular_head",     "neck_us"):           0.5,
    # Spine inter-level
    ("spine_cervical",    "spine_thoracic"):    0.35,
    ("spine_thoracic",    "spine_cervical"):    0.35,
    ("spine_thoracic",    "spine_lumbar"):      0.35,
    ("spine_lumbar",      "spine_thoracic"):    0.35,
    ("spine_cervical",    "spine_lumbar"):      0.1,
    ("spine_lumbar",      "spine_cervical"):    0.1,
    # Spine ↔ spine_whole
    ("spine_whole",       "spine_cervical"):    0.9,
    ("spine_whole",       "spine_thoracic"):    0.9,
    ("spine_whole",       "spine_lumbar"):      0.9,
    ("spine_cervical",    "spine_whole"):       0.9,
    ("spine_thoracic",    "spine_whole"):       0.9,
    ("spine_lumbar",      "spine_whole"):       0.9,
    # Thoracic spine ↔ chest — intentionally at threshold → LLM
    ("spine_thoracic",    "chest"):             0.5,
    ("chest",             "spine_thoracic"):    0.5,
    # Cervical spine ↔ neck / brain
    ("spine_cervical",    "brain"):             0.3,
    ("brain",             "spine_cervical"):    0.3,
    ("spine_cervical",    "neck_soft_tissue"):  0.55,
    ("neck_soft_tissue",  "spine_cervical"):    0.55,
    # Chest / cardiac — intentionally at threshold → LLM
    ("chest",             "cardiac"):           0.5,
    ("cardiac",           "chest"):             0.5,
    ("chest",             "vascular_chest"):    0.85,
    ("vascular_chest",    "chest"):             0.85,
    ("cardiac",           "vascular_chest"):    0.6,
    ("vascular_chest",    "cardiac"):           0.6,
    # Abdomen / pelvis
    ("abdomen",           "abdomen_pelvis"):    0.9,
    ("pelvis",            "abdomen_pelvis"):    0.9,
    ("abdomen_pelvis",    "abdomen"):           0.9,
    ("abdomen_pelvis",    "pelvis"):            0.9,
    ("abdomen",           "pelvis"):            0.45,
    ("pelvis",            "abdomen"):           0.45,
    ("abdomen",           "vascular_abdomen"):  0.85,
    ("vascular_abdomen",  "abdomen"):           0.85,
    ("pelvis",            "vascular_abdomen"):  0.7,
    ("vascular_abdomen",  "pelvis"):            0.7,
    ("abdomen_pelvis",    "vascular_abdomen"):  0.85,
    ("vascular_abdomen",  "abdomen_pelvis"):    0.85,
    # Lumbar ↔ abdomen/pelvis
    ("spine_lumbar",      "abdomen"):           0.4,
    ("abdomen",           "spine_lumbar"):      0.4,
    ("spine_lumbar",      "pelvis"):            0.4,
    ("pelvis",            "spine_lumbar"):      0.4,
    ("spine_lumbar",      "abdomen_pelvis"):    0.45,
    ("abdomen_pelvis",    "spine_lumbar"):      0.45,
    # Chest ↔ abdomen (distinct)
    ("chest",             "abdomen"):           0.2,
    ("abdomen",           "chest"):             0.2,
    ("chest",             "abdomen_pelvis"):    0.15,
    ("abdomen_pelvis",    "chest"):             0.15,
    # Extremities
    ("upper_extremity",   "msk_general"):       0.55,
    ("lower_extremity",   "msk_general"):       0.55,
    ("msk_general",       "upper_extremity"):   0.55,
    ("msk_general",       "lower_extremity"):   0.55,
    ("lower_extremity",   "vascular_peripheral"): 0.7,
    ("vascular_peripheral","lower_extremity"):   0.7,
    ("upper_extremity",   "vascular_peripheral"): 0.55,
    ("vascular_peripheral","upper_extremity"):   0.55,
    # Hip
    ("hip",               "lower_extremity"):   0.9,
    ("lower_extremity",   "hip"):               0.9,
    ("hip",               "pelvis"):            0.6,
    ("pelvis",            "hip"):               0.6,
    # Whole body — generous but not total overlap with specific regions
    ("whole_body",        "chest"):             0.55,
    ("whole_body",        "abdomen"):           0.55,
    ("whole_body",        "pelvis"):            0.55,
    ("whole_body",        "abdomen_pelvis"):    0.55,
    ("whole_body",        "cardiac"):           0.45,
    ("whole_body",        "breast"):            0.35,   # PET doesn't include breast
    ("whole_body",        "spine_lumbar"):      0.5,
    ("whole_body",        "spine_cervical"):    0.5,
    ("whole_body",        "spine_thoracic"):    0.5,
    ("chest",             "whole_body"):        0.55,
    ("abdomen",           "whole_body"):        0.55,
    ("pelvis",            "whole_body"):        0.55,
    ("abdomen_pelvis",    "whole_body"):        0.55,
    ("cardiac",           "whole_body"):        0.45,
    ("breast",            "whole_body"):        0.35,
    # GI tract
    ("gi_tract",          "chest"):             0.5,    # esophagram ↔ chest → LLM
    ("chest",             "gi_tract"):          0.5,
    ("gi_tract",          "abdomen"):           0.55,
    ("abdomen",           "gi_tract"):          0.55,
    ("gi_tract",          "abdomen_pelvis"):    0.55,
    ("abdomen_pelvis",    "gi_tract"):          0.55,
    # Bone density: specific — only relevant to bone density exams and spine/hip
    ("bone_density",      "spine_lumbar"):      0.6,
    ("spine_lumbar",      "bone_density"):      0.6,
    ("bone_density",      "spine_whole"):       0.6,
    ("spine_whole",       "bone_density"):      0.6,
    ("bone_density",      "hip"):               0.6,
    ("hip",               "bone_density"):      0.6,
    ("bone_density",      "lower_extremity"):   0.3,
    ("lower_extremity",   "bone_density"):      0.3,
    ("bone_density",      "pelvis"):            0.25,
    ("pelvis",            "bone_density"):      0.25,
}

SPECIFIC_SPINE = {"spine_cervical", "spine_thoracic", "spine_lumbar"}


def normalize(text: str) -> str:
    return text.lower().strip()


def extract_regions(desc: str) -> list:
    d = normalize(desc)
    found = []
    for region, pat in REGION_PATTERNS.items():
        if re.search(pat, d):
            found.append(region)

    # ── Deduplication rules ──────────────────────────────────────────────────
    if "abdomen_pelvis" in found:
        found = [r for r in found if r not in ("abdomen", "pelvis")]
    if any(r in SPECIFIC_SPINE for r in found) and "spine_whole" in found:
        found = [r for r in found if r != "spine_whole"]
    # Thoracic spine keyword shouldn't also pull in "chest"
    if "spine_thoracic" in found and "chest" in found:
        found = [r for r in found if r != "chest"]
    # Breast != chest
    if "breast" in found and "chest" in found:
        found = [r for r in found if r != "chest"]
    # Cardiac echo != chest (let LLM decide)
    if "cardiac" in found and "chest" in found and "vascular_chest" not in found:
        found = [r for r in found if r != "chest"]
    # Thyroid absorbs neck_soft_tissue
    if "thyroid" in found and "neck_soft_tissue" in found:
        found = [r for r in found if r != "neck_soft_tissue"]
    # Hip merges into lower_extremity
    if "hip" in found and "lower_extremity" in found:
        found = [r for r in found if r != "hip"]
    # PET whole-body: strip spurious sub-region hits from "skull" or "thigh"
    if "whole_body" in found:
        found = [r for r in found if r not in ("brain", "lower_extremity", "upper_extremity")]
    # DXA: strip lower_extremity if bone_density is present (DXA hip/spine != extremity)
    if "bone_density" in found:
        found = [r for r in found if r not in ("lower_extremity", "pelvis")]
        # keep hip as a distinct signal for DXA
    # CT upper extremity labeled as "CT UPPR" — strip spine_cervical if upper_extremity
    if "upper_extremity" in found and "spine_cervical" in found:
        found = [r for r in found if r != "spine_cervical"]

    return found if found else ["unknown"]


def extract_modality(desc: str) -> str:
    d = normalize(desc)
    for mod, pat in MODALITY_PATTERNS.items():
        if re.search(pat, d):
            return mod
    return "unknown"


def region_overlap_score(regions_a: list, regions_b: list) -> float:
    if not regions_a or not regions_b:
        return 0.5
    if regions_a == ["unknown"] and regions_b == ["unknown"]:
        return 0.5

    # Specific spine-level mismatch
    specific_a = SPECIFIC_SPINE & set(regions_a)
    specific_b = SPECIFIC_SPINE & set(regions_b)
    if specific_a and specific_b and not (specific_a & specific_b):
        adj = {frozenset({"spine_cervical", "spine_thoracic"}),
               frozenset({"spine_thoracic", "spine_lumbar"})}
        if frozenset(specific_a | specific_b) in adj:
            return 0.25
        return 0.1

    best = 0.0
    for ra in regions_a:
        for rb in regions_b:
            if ra == rb:
                best = max(best, 1.0)
            elif ra == "unknown" or rb == "unknown":
                best = max(best, 0.4)
            elif ra == "whole_body" or rb == "whole_body":
                s = REGION_OVERLAP.get((ra, rb), REGION_OVERLAP.get((rb, ra), 0.5))
                best = max(best, s)
            elif ra in ("procedural", "msk_general") or rb in ("procedural", "msk_general"):
                best = max(best, REGION_OVERLAP.get((ra, rb), REGION_OVERLAP.get((rb, ra), 0.3)))
            else:
                best = max(best, REGION_OVERLAP.get((ra, rb), REGION_OVERLAP.get((rb, ra), 0.0)))
    return best


def modality_compat_score(mod_a: str, mod_b: str) -> float:
    if mod_a == "unknown" or mod_b == "unknown":
        return 0.6
    if mod_a == mod_b:
        return 1.0
    return MODALITY_COMPAT.get((mod_a, mod_b), 0.35)


def rule_based_score(current_desc: str, prior_desc: str) -> tuple:
    cur_regions = extract_regions(current_desc)
    pri_regions = extract_regions(prior_desc)
    cur_mod = extract_modality(current_desc)
    pri_mod = extract_modality(prior_desc)

    region_score = region_overlap_score(cur_regions, pri_regions)
    mod_score = modality_compat_score(cur_mod, pri_mod)
    combined = REGION_WEIGHT * region_score + MODALITY_WEIGHT * mod_score

    debug = {
        "cur_regions": cur_regions, "pri_regions": pri_regions,
        "cur_mod": cur_mod, "pri_mod": pri_mod,
        "region_score": round(region_score, 3),
        "mod_score": round(mod_score, 3),
        "combined": round(combined, 3),
    }
    return combined, debug


def is_confident(score: float) -> bool:
    """True if rule-based is confident enough to skip LLM fallback."""
    return score >= 0.57 or score <= 0.43


def predict_relevance(current_desc: str, prior_desc: str) -> tuple:
    score, debug = rule_based_score(current_desc, prior_desc)
    return score >= RELEVANCE_THRESHOLD, score, debug
