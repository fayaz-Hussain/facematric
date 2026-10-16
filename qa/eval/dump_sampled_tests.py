import sys
import os

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sampled = [
    ("backend/tests/unit/test_golden_ratio_scorer.py", "test_measure_face_length_to_width"),
    ("backend/tests/unit/test_golden_ratio_scorer.py", "test_measure_face_to_jaw_width"),
    ("backend/tests/unit/test_golden_ratio_scorer.py", "test_measure_eye_spacing_to_width"),
    ("backend/tests/unit/test_golden_ratio_scorer.py", "test_measure_vertical_thirds"),
    ("backend/tests/unit/test_symmetry_classification.py", "test_classify_oval_shape"),
    ("backend/tests/unit/test_symmetry_classification.py", "test_classify_round_shape"),
    ("backend/tests/unit/test_symmetry_classification.py", "test_classify_rectangle_shape"),
    ("backend/tests/unit/test_symmetry_classification.py", "test_round_not_square"),
    ("backend/tests/unit/test_percentile_engine.py", "test_initialize_with_default_parameters"),
    ("backend/tests/unit/test_percentile_engine.py", "test_score_to_percentile_uses_cdf"),
    ("backend/tests/unit/test_percentile_engine.py", "test_percentile_bounds_are_enforced"),
    ("backend/tests/unit/test_percentile_engine.py", "test_score_clamping_before_conversion"),
    ("backend/tests/unit/test_celebrity_matcher.py", "test_embeddings_are_512_dimensional"),
    ("backend/tests/unit/test_celebrity_matcher.py", "test_rank_celebrities_uses_cosine_similarity"),
    ("backend/tests/unit/test_celebrity_matcher.py", "test_identical_embedding_high_similarity"),
]

for idx, (fpath, fname) in enumerate(sampled, 1):
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read()
    start = text.find(f"def {fname}")
    if start == -1:
        print(f"[{idx}] {fname}: NOT FOUND in {fpath}")
        continue
    # find next def or class
    end = text.find("\n    def ", start + 1)
    if end == -1:
        end = text.find("\nclass ", start + 1)
    if end == -1:
        end = start + 1000
    print(f"==================================================")
    print(f"[{idx}] {fname} ({fpath})")
    print(f"==================================================")
    print(text[start:end].strip())
    print("\n")
