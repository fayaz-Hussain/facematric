import os
import re
import sys
import subprocess

def collect_tests_by_file():
    proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only"], cwd="backend", capture_output=True, text=True)
    lines = proc.stdout.splitlines()
    
    file_tests = {}
    current_module = None
    current_pkg = ""
    
    for line in lines:
        pkg_match = re.search(r"<Package\s+([^>]+)>", line)
        if pkg_match:
            current_pkg = pkg_match.group(1).strip()
            continue
            
        mod_match = re.search(r"<Module\s+([^>]+)>", line)
        if mod_match:
            current_module = f"tests/{current_pkg}/{mod_match.group(1).strip()}"
            file_tests[current_module] = 0
            continue
            
        func_match = re.search(r"<Function\s+([^>]+)>", line)
        if func_match and current_module:
            file_tests[current_module] += 1
            
    return file_tests

def classify_files():
    file_tests = collect_tests_by_file()
    
    # We will classify every file into:
    # 1. Real photographs through real MediaPipe: 0 files (0%)
    # 2. Synthetic landmarks: test files that construct synthetic landmark arrays (e.g. 468 landmarks / NormalizedLandmark)
    #    or synthetic facial measurements to test geometric/scoring/symmetry logic.
    # 3. Mocks: test files that rely primarily on unittest.mock (patching components, mock requests, mock detector, mock database)
    
    classified = {}
    
    # Let's inspect each file manually and programmatically:
    # Categories:
    # Real photographs through real MediaPipe: None
    # Synthetic landmarks:
    # - test_asymmetric_features.py (creates synthetic landmark arrays)
    # - test_extract_2d_landmarks.py (synthetic landmark arrays)
    # - test_facial_dimensions.py (synthetic landmark arrays)
    # - test_golden_ratio_guides.py (synthetic landmark arrays)
    # - test_golden_ratio_scorer.py (creates synthetic 468 landmark arrays)
    # - test_overall_score_calculation.py (synthetic landmark arrays)
    # - test_overlay_generator.py (synthetic landmark arrays)
    # - test_overlay_generator_integration.py (synthetic 468 landmark arrays)
    # - test_response_models.py (synthetic landmarks & guides)
    # - test_symmetry_classification.py (synthetic landmark-derived features / dimensions)
    #
    # Mocks:
    # - test_analyze_endpoint_validation.py (TestClient mock requests/dummy files)
    # - test_api_endpoint_integration.py (Mock detector, mock matcher, Mock responses)
    # - test_error_handling_10_6.py (Mock exceptions, mock client requests)
    # - test_fastapi_setup.py (TestClient endpoint routes & config mocks)
    # - test_mediapipe_initialization.py (File stat mocks & empty detector instantiation)
    # - test_orchestration_10_4.py (TestClient with dummy solid-color images, expects 400 or mocks)
    # - test_privacy_memory_cleanup_10_7.py (Mocked models/components to test GC)
    # - test_timeout_middleware.py (Mock async handlers, TestClient)
    # - test_celebrity_confidence_threshold.py (Mock embeddings, mock matcher)
    # - test_celebrity_matcher.py (Mock DeepFace, mock pickle files)
    # - test_cosine_similarity.py (Mock/synthetic vectors)
    # - test_face_detector.py (Mocks MediaPipe FaceLandmarker via @patch)
    # - test_image_processor.py (Synthetic PIL memory buffers)
    # - test_image_resize.py (Synthetic PIL images)
    # - test_infrastructure.py (Mock env vars, directories)
    # - test_internal_models.py (Mock/synthetic model instantiation)
    # - test_percentile_engine.py (Synthetic score inputs / mocked normal CDF calls)
    # - test_rank_celebrities.py (Mock celebrity database and similarity vectors)
    # - test_request_models.py (Mock request schemas)
    # - test_resize_standalone.py (Synthetic image buffers)
    
    synthetic_landmark_files = {
        "tests/unit/test_asymmetric_features.py",
        "tests/unit/test_extract_2d_landmarks.py",
        "tests/unit/test_facial_dimensions.py",
        "tests/unit/test_golden_ratio_guides.py",
        "tests/unit/test_golden_ratio_scorer.py",
        "tests/unit/test_overall_score_calculation.py",
        "tests/unit/test_overlay_generator.py",
        "tests/unit/test_overlay_generator_integration.py",
        "tests/unit/test_response_models.py",
        "tests/unit/test_symmetry_classification.py",
    }
    
    real_photo_files = set() # Empty!
    
    report = []
    cat_counts = {
        "Real photographs through real MediaPipe": 0,
        "Synthetic landmarks": 0,
        "Mocks": 0
    }
    
    file_cat_counts = {
        "Real photographs through real MediaPipe": 0,
        "Synthetic landmarks": 0,
        "Mocks": 0
    }
    
    for fpath, count in sorted(file_tests.items()):
        if fpath in real_photo_files:
            c = "Real photographs through real MediaPipe"
        elif fpath in synthetic_landmark_files:
            c = "Synthetic landmarks"
        else:
            c = "Mocks"
            
        cat_counts[c] += count
        file_cat_counts[c] += 1
        report.append((fpath, count, c))
        
    return report, cat_counts, file_cat_counts

if __name__ == "__main__":
    rep, cat_counts, file_cat_counts = classify_files()
    total_tests = sum(cat_counts.values())
    total_files = sum(file_cat_counts.values())
    
    print(f"{'Test File':<55} | {'Tests':>5} | {'Category'}")
    print("-" * 90)
    for fpath, count, cat in rep:
        print(f"{fpath:<55} | {count:>5} | {cat}")
    print("-" * 90)
    print("\nFile-Level Breakdown:")
    for cat, count in file_cat_counts.items():
        print(f"  {cat}: {count} files ({count/total_files*100:.1f}%)")
        
    print("\nTest-Case Breakdown:")
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count} tests ({count/total_tests*100:.1f}%)")
