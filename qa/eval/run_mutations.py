import os
import sys
import subprocess

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

WORKTREE_DIR = os.path.abspath("../facemetric_mut/backend")
PYTHON_EXE = os.path.abspath("qa/.venv/Scripts/python.exe")

print(f"Worktree: {WORKTREE_DIR}")
print(f"Python: {PYTHON_EXE}")

def revert_worktree():
    subprocess.run(["git", "checkout", "--", "."], cwd=WORKTREE_DIR, check=True, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=WORKTREE_DIR, check=True, capture_output=True)

# Baseline failures to ignore (per user instruction)
BASELINE_FAILS = [
    "test_rank_celebrities_finds_best_match",
    "test_successful_response_structure"
]

def run_mutation(m_id, desc, rel_file, target, repl, test_files):
    revert_worktree()
    target_path = os.path.join(WORKTREE_DIR, rel_file)
    with open(target_path, "r", encoding="utf-8") as f:
        src = f.read()
        
    if target not in src:
        print(f"[{m_id}] ERROR: target string not found!")
        return {"id": m_id, "desc": desc, "status": "ERROR", "module": rel_file, "details": "Target string not found"}
        
    mutated = src.replace(target, repl, 1)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(mutated)
        
    cmd = [PYTHON_EXE, "-m", "pytest"] + test_files + ["-v"]
    proc = subprocess.run(cmd, cwd=WORKTREE_DIR, capture_output=True, text=True)
    
    # Parse failures
    fails = []
    for line in proc.stdout.splitlines():
        if line.startswith("FAILED "):
            tname = line.split()[1]
            if not any(bf in tname for bf in BASELINE_FAILS):
                fails.append(tname)
                
    revert_worktree()
    
    if len(fails) > 0:
        status = "KILLED"
        details = f"Killed by {len(fails)} test(s): {fails[0]}"
    else:
        status = "SURVIVED"
        details = "No tests failed; mutant survived unnoticed."
        
    print(f"Mutant {m_id:02d}: [{status}] - {desc}")
    print(f"    Details: {details}")
    return {
        "id": m_id,
        "desc": desc,
        "status": status,
        "module": rel_file,
        "details": details,
        "fails": fails
    }

mutations = [
    # 1. scorer: PHI 1.618 → 1.6
    (
        1,
        "scorer: PHI 1.618 -> 1.6",
        "app/components/golden_ratio_scorer.py",
        "PHI = settings.phi",
        "PHI = 1.6",
        ["tests/unit/test_golden_ratio_scorer.py", "tests/unit/test_overall_score_calculation.py"]
    ),
    # 2. scorer: remove abs(...)
    (
        2,
        "scorer: remove abs(...)",
        "app/components/golden_ratio_scorer.py",
        "score = 100 * (1 - abs(deviation) / self.PHI)",
        "score = 100 * (1 - deviation / self.PHI)",
        ["tests/unit/test_golden_ratio_scorer.py", "tests/unit/test_overall_score_calculation.py"]
    ),
    # 3. scorer: remove the 0–100 clamp
    (
        3,
        "scorer: remove the 0-100 clamp",
        "app/components/golden_ratio_scorer.py",
        "score = max(0.0, min(100.0, score))  # Clamp to [0, 100]",
        "# score = max(0.0, min(100.0, score))",
        ["tests/unit/test_golden_ratio_scorer.py", "tests/unit/test_overall_score_calculation.py"]
    ),
    # 4. scorer: swap numerator and denominator in one ratio
    (
        4,
        "scorer: swap numerator and denominator in ratio 1",
        "app/components/golden_ratio_scorer.py",
        "return length / width if width > 0 else 0.0",
        "return width / length if length > 0 else 0.0",
        ["tests/unit/test_golden_ratio_scorer.py", "tests/unit/test_facial_dimensions.py"]
    ),
    # 5. scorer: change one landmark index by ±1
    (
        5,
        "scorer: change landmark FACE_TOP index 10 -> 11",
        "app/components/golden_ratio_scorer.py",
        "FACE_TOP = 10",
        "FACE_TOP = 11",
        ["tests/unit/test_golden_ratio_scorer.py", "tests/unit/test_facial_dimensions.py"]
    ),
    # 6. scorer: overall mean → median
    (
        6,
        "scorer: overall mean -> median",
        "app/components/golden_ratio_scorer.py",
        "overall_score = sum(fs.score for fs in feature_scores) / len(feature_scores)",
        "import numpy as np; overall_score = float(np.median([fs.score for fs in feature_scores]))",
        ["tests/unit/test_golden_ratio_scorer.py", "tests/unit/test_overall_score_calculation.py"]
    ),
    # 7. scorer: drop one of the 8 ratios from the average
    (
        7,
        "scorer: drop one ratio from overall average",
        "app/components/golden_ratio_scorer.py",
        "overall_score = sum(fs.score for fs in feature_scores) / len(feature_scores)",
        "overall_score = sum(fs.score for fs in feature_scores[:-1]) / len(feature_scores[:-1])",
        ["tests/unit/test_golden_ratio_scorer.py", "tests/unit/test_overall_score_calculation.py"]
    ),
    # 8. percentile: mean 70 → 65
    (
        8,
        "percentile: default mean 70 -> 65",
        "app/components/percentile_engine.py",
        "def __init__(self, mean: float = 70.0, std: float = 10.0):",
        "def __init__(self, mean: float = 65.0, std: float = 10.0):",
        ["tests/unit/test_percentile_engine.py"]
    ),
    # 9. percentile: SD 10 → 5
    (
        9,
        "percentile: default std 10 -> 5",
        "app/components/percentile_engine.py",
        "def __init__(self, mean: float = 70.0, std: float = 10.0):",
        "def __init__(self, mean: float = 70.0, std: float = 5.0):",
        ["tests/unit/test_percentile_engine.py"]
    ),
    # 10. percentile: CDF direction reversed
    (
        10,
        "percentile: reverse CDF direction",
        "app/components/percentile_engine.py",
        "percentile = self.distribution.cdf(score) * 100",
        "percentile = (1.0 - self.distribution.cdf(score)) * 100",
        ["tests/unit/test_percentile_engine.py"]
    ),
    # 11. classifier: change one decision threshold by 10%
    (
        11,
        "classifier: change ratio threshold 1.6 -> 1.76",
        "app/components/symmetry_classifier.py",
        "if length_width_ratio > 1.6 and jawline_angle < 130:",
        "if length_width_ratio > 1.76 and jawline_angle < 130:",
        ["tests/unit/test_symmetry_classification.py"]
    ),
    # 12. matcher or similarity: >= → > at the threshold
    (
        12,
        "matcher: >= -> > at threshold",
        "app/components/celebrity_matcher.py",
        "if best_score >= threshold:",
        "if best_score > threshold:",
        ["tests/unit/test_celebrity_matcher.py", "tests/unit/test_celebrity_confidence_threshold.py"]
    ),
    # 13. matcher: threshold 0.55 → 0.60
    (
        13,
        "matcher: threshold default 0.55 -> 0.60",
        "app/components/celebrity_matcher.py",
        "threshold: float = 0.55",
        "threshold: float = 0.60",
        ["tests/unit/test_celebrity_matcher.py", "tests/unit/test_celebrity_confidence_threshold.py"]
    ),
    # 14. image processor: ignore EXIF orientation
    (
        14,
        "image processor: ignore EXIF orientation",
        "app/components/image_processor.py",
        "image = self._apply_exif_rotation(image)",
        "# image = self._apply_exif_rotation(image)",
        ["tests/unit/test_image_processor.py"]
    ),
    # 15. detector: skip the multiple-faces check
    (
        15,
        "detector: skip multiple faces check",
        "app/components/face_detector.py",
        "if len(detection_result.face_landmarks) > 1:",
        "if False and len(detection_result.face_landmarks) > 1:",
        ["tests/unit/test_face_detector.py"]
    ),
    # 16. scorer: bilateral split threshold 0.10 -> 0.20
    (
        16,
        "scorer: bilateral split threshold 0.10 -> 0.20",
        "app/components/golden_ratio_scorer.py",
        "ASYMMETRY_THRESHOLD = 0.10",
        "ASYMMETRY_THRESHOLD = 0.20",
        ["tests/unit/test_golden_ratio_scorer.py", "tests/unit/test_asymmetric_features.py"]
    ),
    # 17. matcher: the 0.50 + similarity shift -> 0.40 + similarity
    (
        17,
        "matcher: 0.50 + similarity shift -> 0.40 + similarity",
        "app/components/celebrity_matcher.py",
        "c1 = max(0.60, min(0.96, 0.50 + float(best_similarity)))",
        "c1 = max(0.60, min(0.96, 0.40 + float(best_similarity)))",
        ["tests/unit/test_celebrity_matcher.py", "tests/unit/test_rank_celebrities.py"]
    ),
    # 18. main.py: when matcher returns None, return empty celebrity object instead of null
    (
        18,
        "main.py: return empty celebrity object instead of null",
        "app/main.py",
        "celebrity_match = None",
        "celebrity_match = CelebrityMatch(name='None', confidence=0.0, dataset_source='None', thumbnail_path=None, top_matches=[])",
        ["tests/integration/test_orchestration_10_4.py", "tests/integration/test_api_endpoint_integration.py"]
    )
]

if __name__ == "__main__":
    results = []
    for m in mutations:
        r = run_mutation(m[0], m[1], m[2], m[3], m[4], m[5])
        results.append(r)
        
    print("\n" + "="*80)
    print("MUTATION TESTING SUMMARY")
    print("="*80)
    killed = sum(1 for r in results if r["status"] == "KILLED")
    survived = sum(1 for r in results if r["status"] == "SURVIVED")
    print(f"Total Mutants: {len(results)} | Killed: {killed} ({killed/len(results)*100:.1f}%) | Survived: {survived} ({survived/len(results)*100:.1f}%)")
    
    # Module survival rates
    by_mod = {}
    for r in results:
        mod = r["module"].split("/")[-1].replace(".py", "")
        if mod not in by_mod:
            by_mod[mod] = {"total": 0, "killed": 0, "survived": 0}
        by_mod[mod]["total"] += 1
        if r["status"] == "KILLED":
            by_mod[mod]["killed"] += 1
        else:
            by_mod[mod]["survived"] += 1
            
    print("\nSurvival Rate by Module:")
    for mod, d in by_mod.items():
        surv_rate = d["survived"] / d["total"] * 100
        print(f"  {mod:<25}: {d['survived']}/{d['total']} survived ({surv_rate:.1f}%)")
