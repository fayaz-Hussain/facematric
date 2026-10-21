# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: frontend.spec.js >> Frontend E2E Integration Suite >> Displayed values on Results screen equal API response
- Location: tests\frontend.spec.js:68:3

# Error details

```
Error: expect(locator).toContainText(expected) failed

Locator: locator('#percentileStatus')
Expected substring: "7%"
Received string:    "below typical (compared with 959 celebrity photos)"
Timeout: 5000ms

Call log:
  - Expect "toContainText" locator('#percentileStatus') with timeout 5000ms
  - waiting for locator('#percentileStatus')
    14 × locator resolved to <div id="percentileStatus" class="golden-ratio-status">below typical (compared with 959 celebrity photos)</div>
       - unexpected value "below typical (compared with 959 celebrity photos)"

```

```yaml
- text: below typical (compared with 959 celebrity photos)
```

# Test source

```ts
  1   | // @ts-check
  2   | const { test, expect } = require('@playwright/test');
  3   | 
  4   | const FRONTEND_URL = 'http://localhost:3000';
  5   | const BACKEND_8000 = 'http://127.0.0.1:8000';
  6   | const BACKEND_8001 = 'http://127.0.0.1:8001';
  7   | 
  8   | // Sample API response from port 8000 (Adriana Lima with 96% match)
  9   | const MOCK_RESPONSE_WITH_CELEBRITY = {
  10  |   status: "success",
  11  |   processing_time_ms: 220,
  12  |   results: {
  13  |     golden_ratio_score: 51.9,
  14  |     percentile_rank: 7,
  15  |     face_symmetry_type: "heart",
  16  |     feature_breakdown: [
  17  |       { feature: "Face Length ÷ Face Width", measured_ratio: 1.174, ideal_ratio: 1.618, score: 72.5, deviation: -0.444 },
  18  |       { feature: "Cheekbone to Jaw Width", measured_ratio: 1.306, ideal_ratio: 1.618, score: 80.7, deviation: -0.312 },
  19  |       { feature: "Eye Spacing ÷ Eye Width", measured_ratio: 1.237, ideal_ratio: 1.618, score: 76.5, deviation: -0.381 },
  20  |       { feature: "Nose Width ÷ Mouth Width", measured_ratio: 0.641, ideal_ratio: 1.618, score: 39.6, deviation: -0.977 },
  21  |       { feature: "Mouth Width ÷ Face Width", measured_ratio: 0.351, ideal_ratio: 1.618, score: 21.7, deviation: -1.267 },
  22  |       { feature: "Nose-Chin ÷ Hairline-Nose (Vertical Thirds)", measured_ratio: 1.684, ideal_ratio: 1.618, score: 95.9, deviation: 0.066 },
  23  |       { feature: "Eye-Eyebrow Distance ÷ Eye Height", measured_ratio: 2.781, ideal_ratio: 1.618, score: 28.2, deviation: 1.163 },
  24  |       { feature: "Chin Width ÷ Nose Width", measured_ratio: 3.421, ideal_ratio: 1.618, score: 0.0, deviation: 1.803 }
  25  |     ],
  26  |     overlay_data: {
  27  |       midline_x: 274.8,
  28  |       horizontal_thirds: [99.4, 295.4, 416.5, 612.6],
  29  |       phi_spiral_center: [277.5, 290.6],
  30  |       landmarks: Array.from({ length: 468 }, (_, i) => [200 + (i % 20) * 5, 100 + Math.floor(i / 20) * 15]),
  31  |       golden_ratio_guides: {}
  32  |     },
  33  |     celebrity_match: {
  34  |       name: "Adriana Lima",
  35  |       confidence: 0.96,
  36  |       rank: 1,
  37  |       dataset_source: "Western",
  38  |       thumbnail_url: "/api/celebrity-thumbnail/pins_Adriana_Lima/0.jpg",
  39  |       top_matches: [
  40  |         { rank: 1, name: "Adriana Lima", confidence: 0.96, thumbnail_url: "/api/celebrity-thumbnail/pins_Adriana_Lima/0.jpg" },
  41  |         { rank: 2, name: "Megan Fox", confidence: 0.74, thumbnail_url: "" },
  42  |         { rank: 3, name: "Angelina Jolie", confidence: 0.68, thumbnail_url: "" }
  43  |       ]
  44  |     }
  45  |   }
  46  | };
  47  | 
  48  | // Response from port 8001 (Null celebrity match due to missing deepface)
  49  | const MOCK_RESPONSE_NULL_CELEBRITY = {
  50  |   status: "success",
  51  |   processing_time_ms: 180,
  52  |   results: {
  53  |     golden_ratio_score: 51.9,
  54  |     percentile_rank: 7,
  55  |     face_symmetry_type: "heart",
  56  |     feature_breakdown: MOCK_RESPONSE_WITH_CELEBRITY.results.feature_breakdown,
  57  |     overlay_data: MOCK_RESPONSE_WITH_CELEBRITY.results.overlay_data,
  58  |     celebrity_match: null
  59  |   }
  60  | };
  61  | 
  62  | // 1x1 transparent dummy image data URI
  63  | const DUMMY_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
  64  | 
  65  | 
  66  | test.describe('Frontend E2E Integration Suite', () => {
  67  | 
  68  |   test('Displayed values on Results screen equal API response', async ({ page }) => {
  69  |     // Inject session storage with mock analysis result
  70  |     await page.goto(`${FRONTEND_URL}/index.html`);
  71  |     await page.evaluate(({ res, img }) => {
  72  |       sessionStorage.setItem('analysisResults', JSON.stringify(res));
  73  |       sessionStorage.setItem('uploadedImageData', img);
  74  |     }, { res: MOCK_RESPONSE_WITH_CELEBRITY, img: DUMMY_IMAGE });
  75  | 
  76  |     await page.goto(`${FRONTEND_URL}/results.html`);
  77  | 
  78  |     // Verify Golden Ratio Score display
  79  |     const scoreEl = page.locator('#goldenRatioValue');
  80  |     await expect(scoreEl).toBeVisible();
  81  |     await expect(scoreEl).toHaveText('52', { timeout: 3000 }); // Math.round(51.9) = 52
  82  | 
  83  |     // Verify Percentile display in status
  84  |     const pctStatus = page.locator('#percentileStatus');
  85  |     await expect(pctStatus).toBeVisible();
> 86  |     await expect(pctStatus).toContainText('7%');
      |                             ^ Error: expect(locator).toContainText(expected) failed
  87  | 
  88  |     // Verify Face Symmetry Type display
  89  |     const symmetryEl = page.locator('#symmetryType');
  90  |     await expect(symmetryEl).toBeVisible();
  91  |     await expect(symmetryEl).toHaveText('Heart');
  92  | 
  93  |     // Verify Feature Breakdown display
  94  |     const featureItems = page.locator('.feature-item, .proportion-row');
  95  |     const count = await featureItems.count();
  96  |     expect(count).toBeGreaterThanOrEqual(8);
  97  | 
  98  |     // Verify Celebrity Card display
  99  |     const celebName = page.locator('#celebrityName');
  100 |     await expect(celebName).toBeVisible();
  101 |     await expect(celebName).toHaveText('Adriana Lima');
  102 | 
  103 |     const celebConfidence = page.locator('#celebrityConfidence');
  104 |     await expect(celebConfidence).toBeVisible();
  105 |     await expect(celebConfidence).toHaveText('96%');
  106 |   });
  107 | 
  108 |   test('Overlay alignment within 2 px of image boundary at two viewport sizes', async ({ page }) => {
  109 |     await page.goto(`${FRONTEND_URL}/index.html`);
  110 |     await page.evaluate(({ res, img }) => {
  111 |       sessionStorage.setItem('analysisResults', JSON.stringify(res));
  112 |       sessionStorage.setItem('uploadedImageData', img);
  113 |     }, { res: MOCK_RESPONSE_WITH_CELEBRITY, img: DUMMY_IMAGE });
  114 | 
  115 |     await page.goto(`${FRONTEND_URL}/results.html`);
  116 | 
  117 |     const canvas = page.locator('#overlayCanvas');
  118 |     const image = page.locator('#analyzedImage');
  119 | 
  120 |     await expect(canvas).toBeVisible();
  121 |     await expect(image).toBeVisible();
  122 | 
  123 |     // Check canvas and image dimensions alignment
  124 |     const canvasBox = await canvas.boundingBox();
  125 |     const imageBox = await image.boundingBox();
  126 | 
  127 |     expect(canvasBox).not.toBeNull();
  128 |     expect(imageBox).not.toBeNull();
  129 | 
  130 |     if (canvasBox && imageBox) {
  131 |       // Differences must be within 2 pixels
  132 |       const widthDiff = Math.abs(canvasBox.width - imageBox.width);
  133 |       const heightDiff = Math.abs(canvasBox.height - imageBox.height);
  134 |       expect(widthDiff).toBeLessThanOrEqual(2.0);
  135 |       expect(heightDiff).toBeLessThanOrEqual(2.0);
  136 |     }
  137 |   });
  138 | 
  139 |   test('Error screen displays message and working Try Again on analysis failure', async ({ page }) => {
  140 |     // Intercept backend /api/analyze to return HTTP 400
  141 |     await page.route('**/api/analyze', route => {
  142 |       route.fulfill({
  143 |         status: 400,
  144 |         contentType: 'application/json',
  145 |         body: JSON.stringify({
  146 |           status: 'error',
  147 |           error_code: 'NO_FACE_DETECTED',
  148 |           message: 'No face detected in photo. Please ensure face is clearly visible.',
  149 |           retry: true
  150 |         })
  151 |       });
  152 |     });
  153 | 
  154 |     await page.goto(`${FRONTEND_URL}/index.html`);
  155 |     await page.evaluate((img) => {
  156 |       sessionStorage.setItem('uploadedImageData', img);
  157 |     }, DUMMY_IMAGE);
  158 | 
  159 |     await page.goto(`${FRONTEND_URL}/analyzing.html`);
  160 | 
  161 |     // Wait for analyzing.js to catch the error and show error modal
  162 |     const errorModal = page.locator('#facemetric-error-modal');
  163 |     await expect(errorModal).toBeVisible({ timeout: 6000 });
  164 | 
  165 |     // Assert "Take New Photo" / Retry button exists and navigates back to index.html
  166 |     const tryAgainBtn = page.locator('#modal-retry-btn');
  167 |     await expect(tryAgainBtn).toBeVisible();
  168 |     await tryAgainBtn.click();
  169 |     await expect(page).toHaveURL(/index\.html/);
  170 |   });
  171 | 
  172 |   test('PRD disclaimer is present on every screen (Expected failure: FM-BUG-012)', async ({ page }) => {
  173 |     const PRD_DISCLAIMER_SNIPPET = "This tool analyzes geometric facial proportions for educational purposes. Scores do not reflect beauty, worth, or health. Results are influenced by photo angle and quality.";
  174 | 
  175 |     // 1. Landing screen
  176 |     await page.goto(`${FRONTEND_URL}/index.html`);
  177 |     const indexBody = await page.textContent('body');
  178 |     expect(indexBody).toContain(PRD_DISCLAIMER_SNIPPET);
  179 | 
  180 |     // 2. Analyzing screen
  181 |     await page.goto(`${FRONTEND_URL}/analyzing.html`);
  182 |     const analyzingBody = await page.textContent('body');
  183 |     expect(analyzingBody).toContain(PRD_DISCLAIMER_SNIPPET);
  184 | 
  185 |     // 3. Results screen
  186 |     await page.goto(`${FRONTEND_URL}/results.html`);
```