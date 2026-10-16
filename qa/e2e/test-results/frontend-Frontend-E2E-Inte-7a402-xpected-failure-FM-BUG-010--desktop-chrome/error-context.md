# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: frontend.spec.js >> Frontend E2E Integration Suite >> When celebrity_match is null, page must NOT show Tom Holland placeholder (Expected failure: FM-BUG-010)
- Location: tests\frontend.spec.js:193:3

# Error details

```
Error: expect(received).not.toBe(expected) // Object.is equality

Expected: not "Tom Holland"
```

# Page snapshot

```yaml
- generic [active] [ref=f1e1]:
  - banner [ref=f1e2]:
    - generic [ref=f1e3]:
      - img "FaceMetric Logo" [ref=f1e5]
      - generic [ref=f1e6]:
        - heading "FaceMetric" [level=1] [ref=f1e7]
        - paragraph [ref=f1e8]: Facial Symmetry & Golden Ratio Analysis
    - generic [ref=f1e9]:
      - link "← New Analysis" [ref=f1e10] [cursor=pointer]:
        - /url: index.html
      - generic [ref=f1e11]:
        - generic [ref=f1e12]: ☀️
        - generic "Toggle theme" [ref=f1e13] [cursor=pointer]
        - generic [ref=f1e14]: 🌙
  - main [ref=f1e15]:
    - generic [ref=f1e16]:
      - generic [ref=f1e17]:
        - generic [ref=f1e18]: SCIENCE. SYMMETRY. YOU.
        - heading "Your Analysis Results" [level=2] [ref=f1e19]
      - generic [ref=f1e20]:
        - generic [ref=f1e21]:
          - generic [ref=f1e22]: Overall Score
          - generic [ref=f1e27]: 3 /100
          - generic [ref=f1e32]: Great Balance!
        - generic [ref=f1e33]:
          - generic [ref=f1e34]: Face Shape
          - generic [ref=f1e42]: Heart
          - generic [ref=f1e43]: Graceful & Sculpted Contour
        - generic [ref=f1e44]:
          - generic [ref=f1e45]: Golden Ratio Score
          - generic [ref=f1e50]: 3 /100
          - generic [ref=f1e54]: Higher than 7% of people
        - generic [ref=f1e55]:
          - generic [ref=f1e56]: Quality Checks
          - generic [ref=f1e61]:
            - generic [ref=f1e62]:
              - generic [ref=f1e65]: Face Quality
              - generic [ref=f1e66]: Good
            - generic [ref=f1e67]:
              - generic [ref=f1e71]: Lighting
              - generic [ref=f1e72]: Good
            - generic [ref=f1e73]:
              - generic [ref=f1e76]: Face Angle
              - generic [ref=f1e77]: Good
            - generic [ref=f1e78]:
              - generic [ref=f1e84]: Landmarks
              - generic [ref=f1e85]: 468 / 468
      - generic [ref=f1e86]:
        - generic [ref=f1e87]:
          - heading "Symmetry Analysis" [level=3] [ref=f1e89]
          - generic [ref=f1e93]:
            - generic [ref=f1e94]: Midline
            - generic [ref=f1e97]: Your Face
            - generic [ref=f1e100]: Mirror
          - img "Analyzed face" [ref=f1e104]
        - generic [ref=f1e105]:
          - heading "Proportion (Golden Ratio) Breakdown" [level=3] [ref=f1e107]
          - generic [ref=f1e111]:
            - generic [ref=f1e112]: Your Ratio
            - generic [ref=f1e113]: Golden Ratio (φ=1.618)
          - generic [ref=f1e114]:
            - generic [ref=f1e115]:
              - generic [ref=f1e116]: Face Length ÷ Face Width
              - generic [ref=f1e122]: 73%
            - generic [ref=f1e127]:
              - generic [ref=f1e128]: Cheekbone to Jaw Width
              - generic [ref=f1e134]: 81%
            - generic [ref=f1e139]:
              - generic [ref=f1e140]: Eye Spacing ÷ Eye Width
              - generic [ref=f1e146]: 77%
            - generic [ref=f1e151]:
              - generic [ref=f1e152]: Nose Width ÷ Mouth Width
              - generic [ref=f1e158]: 40%
            - generic [ref=f1e163]:
              - generic [ref=f1e164]: Mouth Width ÷ Face Width
              - generic [ref=f1e170]: 22%
            - generic [ref=f1e175]:
              - generic [ref=f1e176]: Nose-Chin ÷ Hairline-Nose (Vertical Thirds)
              - generic [ref=f1e182]: 96%
            - generic [ref=f1e187]:
              - generic [ref=f1e188]: Eye-Eyebrow Distance ÷ Eye Height
              - generic [ref=f1e194]: 28%
            - generic [ref=f1e199]:
              - generic [ref=f1e200]: Chin Width ÷ Nose Width
              - generic [ref=f1e206]: 0%
        - generic [ref=f1e210]:
          - heading "Look-Alike Match" [level=3] [ref=f1e212]
          - generic [ref=f1e216]:
            - generic [ref=f1e217]: "#1"
            - generic [ref=f1e218]:
              - img "Top Match" [ref=f1e219]
              - generic [ref=f1e220]: 92%
            - generic [ref=f1e221]: Tom Holland
            - generic [ref=f1e222]: Match Confidence
      - paragraph [ref=f1e228]: This analysis is based on facial geometry and proportions, not skin color, ethnicity, or beauty standards.
```

# Test source

```ts
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
  180 |     expect(indexBody).toContain(PRD_DISCLAIMER_SNIPPET);
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
> 211 |     expect(nameText?.trim()).not.toBe('Tom Holland');
      |                                  ^ Error: expect(received).not.toBe(expected) // Object.is equality
  212 |   });
  213 | 
  214 | });
  215 | 
```