# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: frontend.spec.js >> Frontend E2E Integration Suite >> PRD disclaimer is present on every screen (Expected failure: FM-BUG-012)
- Location: tests\frontend.spec.js:172:3

# Error details

```
Error: expect(received).toContain(expected) // indexOf

Expected substring: "This tool analyzes geometric facial proportions for educational purposes. Scores do not reflect beauty, worth, or health. Results are influenced by photo angle and quality."
Received string:    "······················································································································································································································································································································································································································································································································
                FaceMetric
                Facial Symmetry & Golden Ratio Analysis·······························
            About Project·············
                ☀️·················
                🌙·····································································
                SCIENCE. SYMMETRY. YOU.···························
                Discover Your
                Facial Geometry···························
                Upload an image to receive a comprehensive, scientific analysis of your facial proportions, symmetry, and alignment with the Golden Ratio using advanced AI diagnostics.······································································································································································
                Upload a Photo or Take a Picture
                JPG, PNG, WEBP • Maximum 10MB······················································································································
                        Upload Image······················································································································································
                        Use Camera···························································································································
                    Your image is processed securely and is not stored.··························································································································································
                This analysis is based on facial geometry and proportions, not skin color, ethnicity, or beauty standards.························································
"
```

# Page snapshot

```yaml
- generic [active] [ref=e1]:
  - banner [ref=e2]:
    - generic [ref=e3]:
      - img "FaceMetric Logo" [ref=e5]
      - generic [ref=e6]:
        - heading "FaceMetric" [level=1] [ref=e7]
        - paragraph [ref=e8]: Facial Symmetry & Golden Ratio Analysis
    - generic [ref=e9]:
      - link "About Project" [ref=e10] [cursor=pointer]:
        - /url: "#about"
      - generic [ref=e11]:
        - generic [ref=e12]: ☀️
        - generic "Toggle theme" [ref=e13] [cursor=pointer]
        - generic [ref=e14]: 🌙
  - main [ref=e15]:
    - generic [ref=e16]:
      - generic [ref=e17]: ◈ SCIENCE. SYMMETRY. YOU.
      - heading "Discover Your Facial Geometry" [level=2] [ref=e18]
      - paragraph [ref=e19]: Upload an image to receive a comprehensive, scientific analysis of your facial proportions, symmetry, and alignment with the Golden Ratio using advanced AI diagnostics.
      - generic [ref=e20]:
        - heading "Upload a Photo or Take a Picture" [level=3] [ref=e25]
        - paragraph [ref=e26]: JPG, PNG, WEBP • Maximum 10MB
        - generic [ref=e27]:
          - button "Upload image" [ref=e28] [cursor=pointer]: Upload Image
          - button "Use camera" [ref=e31] [cursor=pointer]: Use Camera
        - generic [ref=e35]: Your image is processed securely and is not stored.
      - generic [ref=e38]: This analysis is based on facial geometry and proportions, not skin color, ethnicity, or beauty standards.
```

# Test source

```ts
  80  |     await expect(scoreEl).toBeVisible();
  81  |     await expect(scoreEl).toHaveText('52', { timeout: 3000 }); // Math.round(51.9) = 52
  82  | 
  83  |     // Verify Percentile display in status
  84  |     const pctStatus = page.locator('#percentileStatus');
  85  |     await expect(pctStatus).toBeVisible();
  86  |     await expect(pctStatus).toContainText('7%');
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
  173 |     test.fail(true, 'FM-BUG-012: The PRD Section 15 disclaimer is missing from analyzing.html and diverges in index/results.');
  174 | 
  175 |     const PRD_DISCLAIMER_SNIPPET = "This tool analyzes geometric facial proportions for educational purposes. Scores do not reflect beauty, worth, or health. Results are influenced by photo angle and quality.";
  176 | 
  177 |     // 1. Landing screen
  178 |     await page.goto(`${FRONTEND_URL}/index.html`);
  179 |     const indexBody = await page.textContent('body');
> 180 |     expect(indexBody).toContain(PRD_DISCLAIMER_SNIPPET);
      |                       ^ Error: expect(received).toContain(expected) // indexOf
  181 | 
  182 |     // 2. Analyzing screen
  183 |     await page.goto(`${FRONTEND_URL}/analyzing.html`);
  184 |     const analyzingBody = await page.textContent('body');
  185 |     expect(analyzingBody).toContain(PRD_DISCLAIMER_SNIPPET);
  186 | 
  187 |     // 3. Results screen
  188 |     await page.goto(`${FRONTEND_URL}/results.html`);
  189 |     const resultsBody = await page.textContent('body');
  190 |     expect(resultsBody).toContain(PRD_DISCLAIMER_SNIPPET);
  191 |   });
  192 | 
  193 |   test('When celebrity_match is null, page must NOT show Tom Holland placeholder (Expected failure: FM-BUG-010)', async ({ page }) => {
  194 |     test.fail(true, 'FM-BUG-010: When celebrity_match is null, results.html fails to update the card and leaves Tom Holland (92%) visible.');
  195 | 
  196 |     // Inject null celebrity result (as produced by Server 2 on port 8001)
  197 |     await page.goto(`${FRONTEND_URL}/index.html`);
  198 |     await page.evaluate(({ res, img }) => {
  199 |       sessionStorage.setItem('analysisResults', JSON.stringify(res));
  200 |       sessionStorage.setItem('uploadedImageData', img);
  201 |     }, { res: MOCK_RESPONSE_NULL_CELEBRITY, img: DUMMY_IMAGE });
  202 | 
  203 |     await page.goto(`${FRONTEND_URL}/results.html`);
  204 | 
  205 |     const celebName = page.locator('#celebrityName');
  206 |     await expect(celebName).toBeVisible();
  207 | 
  208 |     const nameText = await celebName.textContent();
  209 |     // In buggy production code, nameText is still "Tom Holland"
  210 |     // Expectation: If celebrity_match is null, Tom Holland must NOT be displayed
  211 |     expect(nameText?.trim()).not.toBe('Tom Holland');
  212 |   });
  213 | 
  214 | });
  215 | 
```