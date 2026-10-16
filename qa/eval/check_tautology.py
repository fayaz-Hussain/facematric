import inspect
import sys
import os

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add backend to path
sys.path.insert(0, os.path.abspath("backend"))

# Let's inspect test functions from the 4 targeted areas:
# Scorer, Classifier, Percentile, Matcher

files_to_check = [
    ("backend/tests/unit/test_golden_ratio_scorer.py", [
        "test_golden_ratio_scorer_initialization",
        "test_measure_face_length_to_width",
        "test_measure_cheekbone_to_jawline_width",
        "test_measure_eye_spacing_to_eye_width",
        "test_calculate_feature_score_exact_match",
        "test_calculate_feature_score_with_deviation"
    ]),
    ("backend/tests/unit/test_symmetry_classification.py", [
        "test_classify_oval_shape",
        "test_classify_round_shape",
        "test_classify_square_shape"
    ]),
    ("backend/tests/unit/test_percentile_engine.py", [
        "test_calculate_percentile_exact_mean",
        "test_calculate_percentile_one_sd_above",
        "test_calculate_percentile_two_sd_below"
    ]),
    ("backend/tests/unit/test_celebrity_matcher.py", [
        "test_cosine_similarity_identical_vectors",
        "test_rank_celebrities_sorting",
        "test_find_best_match_above_threshold"
    ])
]

# Read source code of each test function
for fpath, funcs in files_to_check:
    print(f"=== {fpath} ===")
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    for fn in funcs:
        print(f"\n--- {fn} ---")
        start = content.find(f"def {fn}")
        if start != -1:
            end = content.find("\n    def ", start + 1)
            if end == -1:
                end = content.find("\nclass ", start + 1)
            if end == -1:
                end = start + 800
            print(content[start:end][:600])
        else:
            print("NOT FOUND")
