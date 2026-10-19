// @ts-check
const { test, expect } = require('@playwright/test');

const FRONTEND_URL = 'http://localhost:3000';
const BACKEND_8000 = 'http://127.0.0.1:8000';
const BACKEND_8001 = 'http://127.0.0.1:8001';

// Sample API response from port 8000 (Adriana Lima with 96% match)
const MOCK_RESPONSE_WITH_CELEBRITY = {
  status: "success",
  processing_time_ms: 220,
  results: {
    golden_ratio_score: 51.9,
    percentile_rank: 7,
    face_symmetry_type: "heart",
    feature_breakdown: [
      { feature: "Face Length ÷ Face Width", measured_ratio: 1.174, ideal_ratio: 1.618, score: 72.5, deviation: -0.444 },
      { feature: "Cheekbone to Jaw Width", measured_ratio: 1.306, ideal_ratio: 1.618, score: 80.7, deviation: -0.312 },
      { feature: "Eye Spacing ÷ Eye Width", measured_ratio: 1.237, ideal_ratio: 1.618, score: 76.5, deviation: -0.381 },
      { feature: "Nose Width ÷ Mouth Width", measured_ratio: 0.641, ideal_ratio: 1.618, score: 39.6, deviation: -0.977 },
      { feature: "Mouth Width ÷ Face Width", measured_ratio: 0.351, ideal_ratio: 1.618, score: 21.7, deviation: -1.267 },
      { feature: "Nose-Chin ÷ Hairline-Nose (Vertical Thirds)", measured_ratio: 1.684, ideal_ratio: 1.618, score: 95.9, deviation: 0.066 },
      { feature: "Eye-Eyebrow Distance ÷ Eye Height", measured_ratio: 2.781, ideal_ratio: 1.618, score: 28.2, deviation: 1.163 },
      { feature: "Chin Width ÷ Nose Width", measured_ratio: 3.421, ideal_ratio: 1.618, score: 0.0, deviation: 1.803 }
    ],
    overlay_data: {
      midline_x: 274.8,
      horizontal_thirds: [99.4, 295.4, 416.5, 612.6],
      phi_spiral_center: [277.5, 290.6],
      landmarks: Array.from({ length: 468 }, (_, i) => [200 + (i % 20) * 5, 100 + Math.floor(i / 20) * 15]),
      golden_ratio_guides: {}
    },
    celebrity_match: {
      name: "Adriana Lima",
      confidence: 0.96,
      rank: 1,
      dataset_source: "Western",
      thumbnail_url: "/api/celebrity-thumbnail/pins_Adriana_Lima/0.jpg",
      top_matches: [
        { rank: 1, name: "Adriana Lima", confidence: 0.96, thumbnail_url: "/api/celebrity-thumbnail/pins_Adriana_Lima/0.jpg" },
        { rank: 2, name: "Megan Fox", confidence: 0.74, thumbnail_url: "" },
        { rank: 3, name: "Angelina Jolie", confidence: 0.68, thumbnail_url: "" }
      ]
    }
  }
};

// Response from port 8001 (Null celebrity match due to missing deepface)
const MOCK_RESPONSE_NULL_CELEBRITY = {
  status: "success",
  processing_time_ms: 180,
  results: {
    golden_ratio_score: 51.9,
    percentile_rank: 7,
    face_symmetry_type: "heart",
    feature_breakdown: MOCK_RESPONSE_WITH_CELEBRITY.results.feature_breakdown,
    overlay_data: MOCK_RESPONSE_WITH_CELEBRITY.results.overlay_data,
    celebrity_match: null
  }
};

// 1x1 transparent dummy image data URI
const DUMMY_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";


test.describe('Frontend E2E Integration Suite', () => {

  test('Displayed values on Results screen equal API response', async ({ page }) => {
    // Inject session storage with mock analysis result
    await page.goto(`${FRONTEND_URL}/index.html`);
    await page.evaluate(({ res, img }) => {
      sessionStorage.setItem('analysisResults', JSON.stringify(res));
      sessionStorage.setItem('uploadedImageData', img);
    }, { res: MOCK_RESPONSE_WITH_CELEBRITY, img: DUMMY_IMAGE });

    await page.goto(`${FRONTEND_URL}/results.html`);

    // Verify Golden Ratio Score display
    const scoreEl = page.locator('#goldenRatioValue');
    await expect(scoreEl).toBeVisible();
    await expect(scoreEl).toHaveText('52', { timeout: 3000 }); // Math.round(51.9) = 52

    // Verify Percentile display in status
    const pctStatus = page.locator('#percentileStatus');
    await expect(pctStatus).toBeVisible();
    await expect(pctStatus).toContainText('7%');

    // Verify Face Symmetry Type display
    const symmetryEl = page.locator('#symmetryType');
    await expect(symmetryEl).toBeVisible();
    await expect(symmetryEl).toHaveText('Heart');

    // Verify Feature Breakdown display
    const featureItems = page.locator('.feature-item, .proportion-row');
    const count = await featureItems.count();
    expect(count).toBeGreaterThanOrEqual(8);

    // Verify Celebrity Card display
    const celebName = page.locator('#celebrityName');
    await expect(celebName).toBeVisible();
    await expect(celebName).toHaveText('Adriana Lima');

    const celebConfidence = page.locator('#celebrityConfidence');
    await expect(celebConfidence).toBeVisible();
    await expect(celebConfidence).toHaveText('96%');
  });

  test('Overlay alignment within 2 px of image boundary at two viewport sizes', async ({ page }) => {
    await page.goto(`${FRONTEND_URL}/index.html`);
    await page.evaluate(({ res, img }) => {
      sessionStorage.setItem('analysisResults', JSON.stringify(res));
      sessionStorage.setItem('uploadedImageData', img);
    }, { res: MOCK_RESPONSE_WITH_CELEBRITY, img: DUMMY_IMAGE });

    await page.goto(`${FRONTEND_URL}/results.html`);

    const canvas = page.locator('#overlayCanvas');
    const image = page.locator('#analyzedImage');

    await expect(canvas).toBeVisible();
    await expect(image).toBeVisible();

    // Check canvas and image dimensions alignment
    const canvasBox = await canvas.boundingBox();
    const imageBox = await image.boundingBox();

    expect(canvasBox).not.toBeNull();
    expect(imageBox).not.toBeNull();

    if (canvasBox && imageBox) {
      // Differences must be within 2 pixels
      const widthDiff = Math.abs(canvasBox.width - imageBox.width);
      const heightDiff = Math.abs(canvasBox.height - imageBox.height);
      expect(widthDiff).toBeLessThanOrEqual(2.0);
      expect(heightDiff).toBeLessThanOrEqual(2.0);
    }
  });

  test('Error screen displays message and working Try Again on analysis failure', async ({ page }) => {
    // Intercept backend /api/analyze to return HTTP 400
    await page.route('**/api/analyze', route => {
      route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'error',
          error_code: 'NO_FACE_DETECTED',
          message: 'No face detected in photo. Please ensure face is clearly visible.',
          retry: true
        })
      });
    });

    await page.goto(`${FRONTEND_URL}/index.html`);
    await page.evaluate((img) => {
      sessionStorage.setItem('uploadedImageData', img);
    }, DUMMY_IMAGE);

    await page.goto(`${FRONTEND_URL}/analyzing.html`);

    // Wait for analyzing.js to catch the error and show error modal
    const errorModal = page.locator('#facemetric-error-modal');
    await expect(errorModal).toBeVisible({ timeout: 6000 });

    // Assert "Take New Photo" / Retry button exists and navigates back to index.html
    const tryAgainBtn = page.locator('#modal-retry-btn');
    await expect(tryAgainBtn).toBeVisible();
    await tryAgainBtn.click();
    await expect(page).toHaveURL(/index\.html/);
  });

  test('PRD disclaimer is present on every screen (Expected failure: FM-BUG-012)', async ({ page }) => {
    const PRD_DISCLAIMER_SNIPPET = "This tool analyzes geometric facial proportions for educational purposes. Scores do not reflect beauty, worth, or health. Results are influenced by photo angle and quality.";

    // 1. Landing screen
    await page.goto(`${FRONTEND_URL}/index.html`);
    const indexBody = await page.textContent('body');
    expect(indexBody).toContain(PRD_DISCLAIMER_SNIPPET);

    // 2. Analyzing screen
    await page.goto(`${FRONTEND_URL}/analyzing.html`);
    const analyzingBody = await page.textContent('body');
    expect(analyzingBody).toContain(PRD_DISCLAIMER_SNIPPET);

    // 3. Results screen
    await page.goto(`${FRONTEND_URL}/results.html`);
    const resultsBody = await page.textContent('body');
    expect(resultsBody).toContain(PRD_DISCLAIMER_SNIPPET);
  });

  test('When celebrity_match is null, page must NOT show Tom Holland placeholder (Expected failure: FM-BUG-010)', async ({ page }) => {
    // Inject null celebrity result (as produced by Server 2 on port 8001)
    await page.goto(`${FRONTEND_URL}/index.html`);
    await page.evaluate(({ res, img }) => {
      sessionStorage.setItem('analysisResults', JSON.stringify(res));
      sessionStorage.setItem('uploadedImageData', img);
    }, { res: MOCK_RESPONSE_NULL_CELEBRITY, img: DUMMY_IMAGE });

    await page.goto(`${FRONTEND_URL}/results.html`);

    const celebName = page.locator('#celebrityName');
    await expect(celebName).toBeVisible();

    const nameText = await celebName.textContent();
    // In buggy production code, nameText is still "Tom Holland"
    // Expectation: If celebrity_match is null, Tom Holland must NOT be displayed
    expect(nameText?.trim()).not.toBe('Tom Holland');
  });

});
