# FaceMetric — Validation & Test Plan (v2, 2026-10-13)

Audience: an AI coding agent doing QA work, and the human who reviews it.
Source documents: the FaceMetric PRD (Draft v4) and this plan. If they disagree with each other or with the code, **stop and ask** (see "SPEC-CONFLICT" below).

The project root also contains AI-generated status files (QUICK_TEST_GUIDE, TESTING_GUIDE, TESTING_SUMMARY, PROJECT_STATUS, PROGRESS_STATUS, INTEGRATION_COMPLETE and several *_SUMMARY files). **Treat everything in them as claims to verify, not evidence.** Some were written earlier than later fixes, so they may be stale. "399 tests passing" says the tests pass, not that the tests check the right things.

Two different questions are being answered. Keep them separate in every report:

- **Verification**: does the code do what the PRD says? (math, contracts, error handling)
- **Validation**: does the PRD's design produce numbers worth showing? (stability, calibration, bias)

A passing API test is never evidence of ML accuracy. Unit tests, integration tests and model evaluations are reported separately.

---

## 0. How to work (rules for the agent)

1. **Work one phase at a time.** End each phase by writing `qa/reports/NN_<phase>.md` and then stop until the human says "continue".
2. **Never edit production code or existing tests** (the app code and everything already under `backend/tests/`). New tests go only under `qa/`. When you find a bug:
   - write a failing test marked `@pytest.mark.xfail(strict=True, reason="FM-NNN")`,
   - add an entry to `qa/reports/BUGS.md`,
   - report it. Fixes happen only after the human approves them, in a separate step.
3. **Never loosen a tolerance, threshold or assertion to make a test pass.** Tolerances come from measured data (Phase 3) and are locked in only after the human approves them.
4. **SPEC-CONFLICT**: if the PRD, this plan and the code disagree, do not pick a winner. Write the conflict into the report under "Decisions needed" and stop.
5. **Evidence rule**: every claim in a report must point to a command, file or CSV that reproduces it. If you did not run it, say "not run".
6. **Privacy**: never commit, upload or print face images, embeddings or datasets. `qa/data/` and `qa/reports/overlays/` go in `.gitignore`. Volunteer photos require consent and stay local.
7. **Environment**: this is a Windows machine, so use PowerShell-compatible commands and `venv\Scripts\activate`. Use a virtual environment. Ask before downloading large datasets or installing system packages. Fix random seeds. Pin the package versions you tested with in `qa/reports/00_recon.md`.
8. **Markers**: `unit`, `metamorphic`, `integration`, `e2e`, `eval` (slow, model-dependent). `pytest -m "unit or integration"` must stay fast.
9. **BLOCKED, not skipped**: if a phase depends on something that is not implemented yet (for example the matcher or `/api/analyze`), mark it BLOCKED in the report with the reason and move to the parts that can run.

### Suggested layout

```
qa/
  data/                 # NOT committed
    frontal_set/        # 20–30 clear frontal photos (consenting volunteers / licensed images)
    same_person/        # >=10 volunteers x >=6 photos each (optional but preferred)
    unknown/            # faces of people NOT among the celebrity identities
    faceshape_labels.csv
  tests/
    unit/  metamorphic/  integration/  e2e/
  eval/                 # scripts that write CSV + report, not pass/fail
  reports/
    00_recon.md ... 08_api_e2e.md  FINAL_REPORT.md  BUGS.md
    overlays/           # NOT committed
```

---

## 1. Hypotheses to confirm or refute (not facts)

These came from reading the PRD. In Phase 0, mark each **confirmed / refuted / can't tell** with file+line evidence.

| ID | Hypothesis |
|---|---|
| H1 | The PRD's API example scores (ratio 1.52 → 82.3, ratio 1.70 → 74.1) cannot be produced by the stated formula, which gives 93.94 and 94.93. |
| H2 | MediaPipe landmarks are normalized by image width and height. If distances are computed on raw normalized x/y, ratios are distorted on non-square images. |
| H3 | Ratio 6 uses a "hairline", but MediaPipe has no hairline landmark. The code uses some proxy (probably landmark 10, which sits on the upper forehead). The status files still describe ratio 6 as hairline to nose. |
| H4 | Ratio 4 (nose ÷ mouth) and ratio 5 (mouth ÷ face width) are structurally far from φ in real faces, which caps the achievable overall score well below 100 (likely near 85). The PRD's "85+ is rare, most users 60–85" may be unreachable. |
| H5 | Feature scores are not clamped, so ratios above 2φ give negative scores. |
| H6 | Reference celebrity embeddings were built offline with different preprocessing (detector, alignment, resize, JPEG) than the production request path. |
| H7 | Per-celebrity embeddings are averaged without L2-normalizing each embedding first; the 0.55 threshold is uncalibrated; "confidence" is just cosine similarity. The status files mention 512-dimensional embeddings, which suggests `Facenet512`, whose usual same-person cut-off is well above 0.55. |
| H8 | "Occlusion detection" has no real mechanism: Face Mesh usually returns a full mesh on masked or sunglassed faces. |
| H9 | The celebrity result shape is inconsistent: PRD Step 5 says `null` below threshold, the API example returns an object with `name: null`. |
| H10 | EXIF orientation is ignored and/or HEIC is unsupported, despite the "any format" claim. |
| H11 | The 0.55 comparison is made on a rounded value, so the UI can show "0.55" next to "No match". |
| H12 | The percentile engine uses an assumed "bootstrap" Normal(mean 70, SD 10) (per the status files), not real data, so "You score higher than X% of people" is unsupported. |
| H13 | The three status guides are stale: they say the API gateway, celebrity matching and frontend integration were not implemented, but later-dated files in the root (INTEGRATION_COMPLETE, OCCLUSION_FIX_SUMMARY, QUALITY_CHECK_FIX_SUMMARY) suggest more was done afterwards. |
| H14 | The ~399 existing unit tests are component-level with synthetic or mocked inputs, so they can pass while wrong landmark indices, the PRD example conflict (H1) or a bad percentile assumption (H12) go undetected. Some tests may be tautological (the expected value is computed with the same code or indices as production). |

---

## Phase 0 — Recon (no tests, no code changes)

Write `qa/reports/00_recon.md` with:

- [ ] Map of where these live: landmark extraction, ratio computation, scoring, face-shape classification, percentile, celebrity matching, `/api/analyze`, frontend results screen, overlay drawing.
- [ ] **Ratio spec table** for each of the 8 ratios as *implemented*: name, landmark indices (both endpoints, both sides if paired), numerator/denominator, 2D or 3D distance, whether coordinates are converted to pixels, how left/right pairs are combined (mean / min / other), what happens when a denominator is ~0.
- [ ] Constants: PHI value (1.618 vs exact), 0.55 threshold and where it is compared, FaceNet variant (128-d `Facenet` vs `Facenet512`), detector backend, alignment, resize/JPEG settings, timeout.
- [ ] Where the percentile reference distribution comes from.
- [ ] Existing tests and the exact commands to run them (the status files say `cd backend`, then `pytest tests/unit/ -v`). Run them once and record passed / failed / skipped / xfail counts, duration, Python and package versions.
- [ ] **Claims reconciliation.** Using `git log` (last 30 commits with `--stat`), file timestamps and the code, mark each claim in the status files as **current / stale / false**: what is implemented vs not (API gateway, FaceNet embeddings, celebrity database, matcher, frontend integration), the test counts, and what each *_FIX_SUMMARY says was fixed.
- [ ] State which pieces exist right now: `/api/analyze`, celebrity embeddings file and its location, matcher, frontend result screen. Phases that depend on missing pieces are BLOCKED (rule 9).
- [ ] Verdict on H1–H14.
- [ ] Anything this plan assumes that does not exist in the repo.

**Stop.** Wait for "continue".

---

## Phase 0.5 — Audit the existing test suite

Goal: find out whether "399 passing" is worth anything. Passing tests only matter if they would fail when the code is wrong.

- [ ] **Coverage**: run the existing suite with line and branch coverage (`--cov=app --cov-branch`). Report per module for the scorer, symmetry classifier, percentile engine, face detector, image processor and matcher. Note uncovered branches.
- [ ] **Test inventory**: per test file, count tests that use (a) real photographs through real MediaPipe, (b) synthetic or hand-made landmark arrays, (c) mocks of MediaPipe or DeepFace. Report the percentages. Real-image tests are the only ones that can catch a wrong landmark index.
- [ ] **Tautology check**: sample 15 tests from the scorer and classifier files. For each, say whether the expected value is independent (hard-coded from the PRD or hand calculation) or derived from the same function or indices it tests.
- [ ] **Hand-applied mutation testing** on the small pure modules. Do this in a throwaway copy (for example `git worktree add ../facemetric_mut`), never in the real working tree. Apply **one** change at a time, run only the relevant test file, and record whether the suite *failed* (mutant killed) or *still passed* (mutant survived). At least these 15:
  1. scorer: PHI 1.618 → 1.6
  2. scorer: remove `abs(...)`
  3. scorer: remove the 0–100 clamp
  4. scorer: swap numerator and denominator in one ratio
  5. scorer: change one landmark index by ±1
  6. scorer: overall mean → median
  7. scorer: drop one of the 8 ratios from the average
  8. percentile: mean 70 → 65
  9. percentile: SD 10 → 5
  10. percentile: CDF direction reversed
  11. classifier: change one decision threshold by 10%
  12. matcher or similarity: `>=` → `>` at the threshold
  13. matcher: threshold 0.55 → 0.60
  14. image processor: ignore EXIF orientation
  15. detector: skip the multiple-faces check
- [ ] Report the mutant survival rate per module. **Every surviving mutant is a missing test** and becomes a task in Phases 1–4. Do not fix the tests; list them.
- [ ] Also report: tests that cannot fail (no assertions, `assert True`, assertions on mocks only), skipped tests, and tests that depend on file paths or the clock.

Output `qa/reports/00b_test_audit.md`. **Stop.** Wait for "continue".

---

## Phase 1 — Scoring math (unit tests)

Goal: one consistent mathematical definition, proven against an independent oracle.

Formula under test: `score_i = 100 × (1 − |r_i − φ| / φ)`; overall = arithmetic mean of the 8 feature scores (equal weights).

Reference values (computed with φ = 1.618; verify against the code's constant and allow `abs=0.01` if the code uses the exact φ):

| ratio | expected score | note |
|---|---|---|
| 1.618 | 100.00 | exact target |
| 1.500 | 92.71 | below target |
| 1.520 | 93.94 | PRD example ratio |
| 1.700 | 94.93 | PRD example ratio |
| 1.000 | 61.80 | |
| 0.000 | 0.00 | lower boundary |
| 3.236 | 0.00 | upper zero-score boundary |
| 4.000 | −47.22 before clamping | must clamp to 0 or be rejected |

(An earlier draft elsewhere listed 61.68 for ratio 1.000. That is an arithmetic error; the formula gives 61.80.)

Tasks:
- [ ] Parametrized test of the table above against the production function.
- [ ] Clamping: every returned score is within [0, 100]. Negative, NaN, inf and ≈0-denominator inputs give a controlled error, never a 500.
- [ ] Overall score equals the mean of the 8 feature scores (independent recomputation in the test, not a call to the production function).
- [ ] Equal-weight behavior; weights, if configurable, sum correctly.
- [ ] Rounding: define where rounding happens (feature score, overall, API). Test that displayed values agree with underlying values within the documented rounding.
- [ ] H1: compare the PRD example against the formula. If they disagree, log **SPEC-CONFLICT-1** and **stop for a human decision** before encoding either value as truth.
- [ ] Determinism: same input → bit-identical output over 10 runs.
- [ ] Check whether any existing test pins the PRD example numbers (82.3 / 74.1) or contradicts the formula values above. If an existing test asserts a value the formula does not give, log it as a bug.

---

## Phase 2 — Measurement verification (are the inputs to the formula right?)

Goal: prove the 8 measured ratios are what the ratio spec table says they are.

- [ ] **Independent oracle.** In `qa/eval/oracle.py`, recompute each ratio from the raw landmark array using only the indices in the Phase 0 spec table, with explicit pixel-space Euclidean distance `sqrt((x2−x1)² + (y2−y1)²)` after `x *= image_width`, `y *= image_height`. Compare with production for every image in `frontal_set/`. They must agree to 1e-6 relative.
- [ ] **H2 non-square test.** Take a frontal image, pad it to square (and separately to 16:9) without scaling the face. Ratios must match the original within the Phase 3 resize tolerance. Also verify what the code does with the z coordinate (2D vs 3D) and that the choice is documented.
- [ ] **Endpoint overlays.** For each image and each of the 8 ratios, draw the exact endpoints used with labels, saved to `qa/reports/overlays/`. Produce a contact sheet of 20 images per ratio for the human to eyeball. A mesh that looks right does not prove the right indices are used.
- [ ] **H3 hairline.** Show where the "hairline" point lands on at least 20 faces, including hair covering the forehead, hats and receding hairlines. Report how often it is anatomically wrong.
- [ ] **Ratio audit (H4).** Compute all 8 ratios on every usable image of `frontal_set/` and on a sample of at least 1,000 celebrity images. Per ratio, report mean, SD, 5th/95th percentiles and the share within ±10% of φ; save histograms with φ marked. Compute the *maximum achievable overall score* if the six best-fitting ratios were exactly φ and the others sat at their observed 95th-percentile values closest to φ. Compare with the PRD's "85+ is rare, most users 60–85".
- [ ] **Left/right pairs.** Verify how paired measurements are combined and test with a synthetic landmark set where only the left eye is distorted.
- [ ] **Synthetic landmark tests.** Build landmark arrays where every distance is set by hand and assert the extractor returns the intended ratios (this isolates the geometry code from MediaPipe).

---

## Phase 3 — Invariance (metamorphic) tests

Goal: learn how much the score moves when nothing about the face changes. There is no ground truth for a "correct" golden-ratio score, so test properties the score must have.

Take the clean frontal set. For each transform, compute `|score(transformed) − score(original)|` and also each ratio's change.

| Transform | Expectation | Starting tolerance (hypothesis only) |
|---|---|---|
| Same image, 10 runs | identical | 0 (≤ 1e-6) |
| Mirror flip | same | ≤ 1.0 pt |
| Resize 0.6×–2× (stay above the 300px minimum) | same | ≤ 1.0 |
| Pad to square / 16:9 | same | ≤ 1.0 |
| In-plane rotation ±10° | same | ≤ 2.0 |
| Grayscale and hue shift | same (the "color-blind" claim at the scoring layer) | ≤ 1.5 |
| Brightness 60–140%, gamma 0.7–1.4 | same until detection degrades | ≤ 2.0 |
| JPEG quality 40–95 | same | ≤ 1.5 |
| Blur (Gaussian σ 1–4), downscale to 300px | degrades | measure |
| Yaw 10° / 20° / 30° (real photos, or a 3D head-pose estimate from landmarks) | degrades | measure |
| Neutral vs smiling (same person) | mouth ratios change | measure |

Procedure:
1. Run the harness and write `qa/reports/03_invariance.csv` (image, transform, parameter, score, Δ per ratio, detected?, landmark failures).
2. Report per transform: median, p95, max of |Δ|, and plots of Δ against severity.
3. **Propose** tolerances and **numeric quality-gate thresholds** (minimum face size, blur, brightness, yaw/pitch/roll limits) as the points where drift exceeds tolerance. Do not lock them in. The human approves them, then they become regression tests.

Note for the report: the scoring formulas use no color, but the landmark *detector* is pixel-based. Detection failures and landmark shifts under lighting and skin-tone differences are tested in Phase 5.

---

## Phase 4 — Quality gates and error handling

Using the thresholds the human approved after Phase 3:

- [ ] No face → `NO_FACE_DETECTED`; group photo → `MULTIPLE_FACES`; profile/extreme yaw → `POOR_QUALITY`; face too small; very low resolution (below 300×300); poor lighting; blur.
- [ ] Occlusion (H8): sunglasses, clear glasses, medical mask, hand over mouth, hair over eyes. Report what the code actually does. Expected-reject cases that pass through are bugs, not test failures to hide. Read OCCLUSION_FIX_SUMMARY and QUALITY_CHECK_FIX_SUMMARY, reproduce the original failure each one describes, and confirm the fix holds on real photographs, not only on synthetic inputs.
- [ ] Non-human: pet, cartoon, statue, poster of a face. Fail gracefully.
- [ ] Input handling: corrupt file, empty body, `data:image/jpeg;base64,` prefix, RGBA PNG, grayscale, animated GIF, HEIC (needs `pillow-heif`), WebP, BMP, **EXIF-rotated phone photo** (H10), 20 MB+ image.
- [ ] Every error response matches the contract (`status`, `error_code`, `message`, `retry`) and the message gives actionable guidance.
- [ ] Per the PRD's own rule: if a measurement is unreliable the app must reject or flag it, not show a precise-looking score.

---

## Phase 5 — Repeatability, validity, percentile, face shape

**Repeatability and signal vs noise**
- [ ] Same-person study (`same_person/`): per person, SD of overall score across good near-frontal photos. Proposed engineering target: **SD ≤ 3 points** (a proposal, not an industry standard). Per-ratio SD too.
- [ ] If volunteers are unavailable, use celebrity images grouped by identity, filtered to near-frontal using a landmark-based head-pose estimate.
- [ ] **Within-person vs between-person variance**: one-way ANOVA / ICC(1) of the overall score across identities. If within-person variance is comparable to between-person variance, the score mostly reflects pose, expression and lens rather than the face. Report this plainly; it belongs in the limitations section.

**Bias slice**
- [ ] On a labelled set (if one is available and licensed), report detection rate and landmark error by skin tone, gender and age band. Gaps in detection rate or landmark error are bugs. Differences in score distributions are findings to explain, not automatically bugs.
- [ ] Re-run Phase 3 brightness and gamma transforms per group.

**Percentile**
- [ ] **Assumed vs empirical (H12).** Compare the engine's assumed Normal(70, 10) with the empirical score distribution from the Phase 2 audit images. For the scores 55, 60, 65, 70, 75, 80, report the percentile each would assign. If they differ materially, "You score higher than X% of people" is unsupported; recommend calibrating on real data or labelling the figure an estimate. The existing percentile tests only check CDF arithmetic against the assumption, so say so.
- [ ] Reference distribution is frozen (stored quantiles), versioned, and documented (what population it represents).
- [ ] Monotonic: higher score never gives a lower percentile. Below-range and above-range scores behave sensibly. Ties follow a documented convention. Result is within 0–100.
- [ ] **Signal-to-noise**: using the Phase 3 noise level and the reference SD, report how many percentile points 1 score point moves near the median. If measurement noise is not well below the distribution SD, flag that the percentile is unstable.
- [ ] Percentiles computed against celebrity or test data must not be described as population-wide.

**Face shape (informational only)**
- [ ] Write the numeric rules the code uses for oval/rectangle/round/square/heart/diamond (the PRD gives none). Boundary tests at each threshold.
- [ ] Generate `qa/data/faceshape_labels.csv` as a template for the human and two other reviewers to label independently. Report inter-rater agreement (Cohen's kappa), then per-class precision/recall and a confusion matrix against the consensus label. Do not expect perfect agreement; the categories are fuzzy.
- [ ] Assert the face-shape result has **no effect** on the golden-ratio score (same landmarks, force a different shape label, score unchanged).
- [ ] Stability: same person, different photos → same class rate.

---

## Phase 6 — Celebrity dataset audit

Run before any evaluation. Output `qa/reports/06_dataset_audit.md`. This phase needs only the images in `celebrity_dataset/` and does not depend on the matcher being finished.

- [ ] Exactly the expected identities after merging; no leftover `pins_` prefixes; no duplicate or near-duplicate names (spelling variants). Count per source (Pins, Bollywood).
- [ ] Images per identity; list identities with too few usable images.
- [ ] Per image: no face, multiple faces, corrupt file, failed embedding. Report counts instead of silently skipping.
- [ ] Duplicate and near-duplicate images (exact hash, perceptual hash, and embedding similarity > 0.98), including across identities.
- [ ] Spot-check 30 random images per source for mislabeled identity (contact sheet for the human).
- [ ] Every identity has a thumbnail referenced by the API response.
- [ ] No NaN or zero vectors in `celebrity_embeddings.pkl`; embedding dimension matches the model variant.

---

## Phase 7 — Celebrity matching evaluation

### 7.1 Setup
- If the app's matcher or embeddings are not implemented yet (the status files said they were blocked at the time), build the evaluation harness in `qa/eval/` **independently of app code**, using the configuration the PRD specifies (DeepFace FaceNet; confirm 512-d `Facenet512` against the code). Thresholds can then be chosen now, and Phase 7.2 parity is checked once the matcher exists.
- Freeze the embedding config (model variant, detector, alignment, normalization) and record it in the report.
- Group near-duplicates so they always land in the same split.
- Per identity with enough images: split into **reference / validation / test** (e.g. 60/20/20). Reference images build the stored embeddings; validation picks the threshold; **the test split is used once, at the end**. Save the file lists (`qa/reports/splits.csv`) so leakage can be audited.
- Start with a pilot of 30–50 identities (balanced between Pins and Bollywood), then scale up if time allows. The pilot is an engineering check, not evidence of production-level accuracy.
- Unknown set: faces of people who are in none of the celebrity identities. Aim for at least 200 faces across at least 50 people (for example LFW people not in the dataset, or consenting volunteers).

### 7.2 Pipeline parity (H6)
- [ ] Take 50 reference images. Embed each through the offline script and through the production request path (resize to 640, JPEG 85, detector, crop/align). Report the cosine similarity between the two embeddings per image (proposed expectation: ≥ 0.95). If parity fails, thresholds tuned offline do not apply to real users.

### 7.3 Aggregation comparison (H7)
Compare on the **validation** split, same queries for each method:
- A. production method: mean of reference embeddings (report whether each is L2-normalized first)
- B. mean of L2-normalized embeddings, re-normalized
- C. max similarity over individual reference embeddings
- D. mean of top-k similarities

### 7.4 Metrics
For each method, over thresholds 0.30 → 0.80 in steps of 0.025:

- **Top-1 accuracy** (threshold-free) and top-5, on known-celebrity queries
- **Coverage**: fraction of queries that return a match
- **Precision of returned matches**: correct returned / all returned
- **False-match rate**: unknown-person queries that return any name / all unknown queries
- **Margin test**: add a rule requiring `top1 − top2 ≥ m`; report whether it improves precision at equal coverage

Report every proportion with a **Wilson 95% interval** and the numbers of identities and photos behind it.

Save `qa/reports/celebrity_predictions.csv` with: image path, true identity (or "UNKNOWN"), top-1 identity, top-1 similarity, top-2 identity, top-2 similarity, margin, returned/rejected, correct?, split.

### 7.5 Threshold decision
- Declare the acceptable false-match rate **before** looking at test results (proposal: ≤ 5%).
- Choose the threshold on validation data only. Show where 0.55 falls on the curve. For reference only (verify in the installed deepface source): its default same-person cosine-distance cut-offs are reportedly 0.40 for `Facenet` and 0.30 for `Facenet512`, i.e. similarity ≈ 0.60 and ≈ 0.70. If the embeddings are 512-d as the status files suggest, the usual cut-off sits near 0.70, so 0.55 is much looser than the model's own same-person setting; the sweep must show what that costs in false matches.
- Run the test split once with the chosen threshold and report.

### 7.6 What the UI is allowed to claim
- [ ] Cosine similarity is not a probability. Either relabel "confidence" as "similarity", or calibrate (isotonic or Platt on validation data) and show a reliability table (binned similarity vs fraction correct). Report which one the code does today.

### 7.7 Behavior tests with mocked similarity (fast, no model)
- [ ] 0.5499 → no match; 0.5500 → match; 0.5500001 → match; comparison uses the unrounded value (H11).
- [ ] Contract: when below threshold the response shape is exactly what the contract says (H9).
- [ ] Celebrity module raising an exception → still HTTP 200 with the golden-ratio result (progressive degradation).
- [ ] `include_celebrity_match=false` → the celebrity module is not called.
- [ ] Returned name and similarity belong to the actual top candidate.

### 7.8 Optional slices
- Match and no-match rate by gender and ethnicity labels, if available. Report patterns (for example, all South Asian faces landing on the Bollywood set) as findings.
- Small human study: 10–15 consenting raters score 1–5 "does this look like them?" for the top match vs a random celebrity.

---

## Phase 8 — API integration and end-to-end

**API (FastAPI `TestClient`)**
- [ ] `POST /api/analyze` with base64 JSON and with multipart. Success shape matches the contract field by field (types, 8 breakdown entries, 468 landmarks, guides present).
- [ ] Every documented error code is reachable and returns the documented HTTP status.
- [ ] Timeout: monkeypatch a slow step past the threshold → HTTP 408, `PROCESSING_TIMEOUT`, `retry: true`.
- [ ] Two simultaneous requests return correct, independent results.
- [ ] Performance: P50/P95 for the P0 path (landmarks + scoring) on the test machine; PRD target < 3 s. Celebrity matching timed separately.

**Privacy**
- [ ] Snapshot the temp/upload directories before and after a request: no image files remain.
- [ ] Capture logs during requests: only metadata (timestamp, processing time, error code); no base64, no landmark arrays, no embeddings.

**Frontend (Playwright)**
- [ ] Upload a fixture; intercept the API response; assert the displayed score, percentile, face type, 8-feature breakdown and celebrity card equal the response values.
- [ ] Overlay alignment: landmark positions drawn on the canvas match the expected pixel positions within 2 px at two viewport sizes.
- [ ] Error screens show the right message and a working "Try Again".
- [ ] Failed celebrity match still shows the score.
- [ ] The disclaimer from the PRD is visible on every screen (landing, analyzing, results) at mobile and desktop widths.

---

## Acceptance criteria (proposed targets, not claims)

| Area | Criterion |
|---|---|
| Scoring math | All unit tests pass; matches the independent oracle to 1e-6 |
| Score range | Every valid response has all scores within 0–100 |
| Measurements | All 8 ratios have visually verified endpoints and a written definition |
| Determinism | Same image → identical output |
| Invariance | Each transform within the human-approved tolerance from Phase 3 |
| Same-person stability | Overall-score SD ≤ 3 on good near-frontal photos (proposal) |
| Percentile | Monotonic, bounded, documented reference; signal-to-noise reported |
| Celebrity top-1 | ≥ 90% on the held-out test split of known identities (proposal; report whatever is measured) |
| Returned-match precision | ≥ 95% on a known + unknown mixture at the chosen threshold, with intervals |
| False-match rate | Meets the limit declared before testing |
| Errors | Every documented error returns a controlled, contract-correct response |
| Privacy | No image retention; no embeddings or images in logs |
| Performance | P0 path < 3 s on the test machine |

Do not state that a target is met unless the CSVs in `qa/reports/` show it.

---

## Final report (`qa/reports/FINAL_REPORT.md`)

1. Pass/fail table for every acceptance criterion with links to evidence.
2. Verdict on H1–H12.
3. Bug list from `BUGS.md` (ID, severity S1–S4, repro command, expected vs actual).
4. **Decisions needed** from the human (spec conflicts, thresholds, definitions).
5. Limitations: what the evaluation does *not* show (for example, that proximity to φ says nothing about attractiveness, and that celebrity photos are not the general population).
6. Reproduction: environment, package versions, seeds, commands.

---

## Appendix A — Reference snippets

```python
# qa/eval/oracle.py  (independent of production code)
import math

PHI = 1.618

def feature_score(r: float) -> float:
    return max(0.0, min(100.0, 100 * (1 - abs(r - PHI) / PHI)))

def px(pt, w, h):                      # MediaPipe landmarks are normalized to [0, 1]
    return (pt[0] * w, pt[1] * h)

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
```

```python
# Wilson 95% interval for a proportion
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)
```

```python
# Threshold sweep (celebrity eval): known = (true, pred, sim), unknown = [sim, ...]
import numpy as np
for thr in np.arange(0.30, 0.8001, 0.025):
    returned = [(t, p) for t, p, s in known if s >= thr]
    coverage = len(returned) / len(known)
    precision = np.mean([t == p for t, p in returned]) if returned else float("nan")
    fmr = np.mean([s >= thr for s in unknown])
```
