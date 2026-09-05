# FaceMetric Frontend

This is the frontend for the FaceMetric facial geometry analysis application. The UI is built with static HTML, Tailwind CSS, and vanilla JavaScript.

## Structure

```
frontend/
├── index.html                                   # Root redirect to upload screen
├── js/
│   ├── api-client.js                           # API communication layer
│   ├── upload-handler.js                       # Upload screen logic
│   ├── analyzing-handler.js                    # Analyzing screen logic
│   └── results-handler.js                      # Results screen logic
├── facemetric_upload_photo_updated_mint/
│   └── code.html                               # Upload screen
├── facemetric_analyzing_unified/
│   └── code.html                               # Analyzing screen
└── facemetric_analysis_results_dark_brand_theme/
    └── code.html                               # Results screen
```

## Setup

### 1. Configure API URL

Edit `js/api-client.js` and update the `API_BASE_URL`:

```javascript
// For local development
const API_BASE_URL = 'http://localhost:8000';

// For production
const API_BASE_URL = 'https://your-backend-url.railway.app';
```

### 2. Run Locally

Since this is static HTML, you can use any local server:

**Option 1: Python HTTP Server**
```bash
cd frontend
python -m http.server 3000
```

**Option 2: Node.js HTTP Server**
```bash
npm install -g http-server
cd frontend
http-server -p 3000
```

**Option 3: VS Code Live Server**
- Install "Live Server" extension
- Right-click on `index.html` → "Open with Live Server"

Then open http://localhost:3000 in your browser.

### 3. Deploy to Vercel/Netlify

**Vercel:**
```bash
cd frontend
vercel deploy
```

**Netlify:**
```bash
cd frontend
netlify deploy --prod
```

Or simply drag the `frontend` folder into the Vercel/Netlify dashboard.

## Features

✅ **File Upload**: Drag-and-drop or click to select image
✅ **Camera Capture**: Take photo with device camera
✅ **Real-time Analysis**: Shows progress through 5 steps
✅ **Results Display**: 
  - Overall golden ratio score with circular progress
  - Face symmetry type classification
  - Feature breakdown with progress bars
  - Celebrity look-alike matching
  - Visual overlay with landmarks

✅ **Error Handling**: User-friendly error messages with retry options
✅ **Privacy**: Images stored in sessionStorage only (cleared after analysis)

## Flow

1. **Upload Screen** → User uploads or captures photo
2. **Analyzing Screen** → Photo sent to backend API, progress shown
3. **Results Screen** → Analysis results displayed with visualizations

## Testing Without Backend

To test the frontend UI without a running backend:

1. Comment out the API call in `analyzing-handler.js`
2. Create mock data:

```javascript
// In analyzing-handler.js, replace API call with:
const result = {
    results: {
        golden_ratio_score: 78.4,
        percentile_rank: 73,
        face_symmetry_type: "oval",
        feature_breakdown: [
            {
                feature: "Face Length ÷ Face Width",
                measured_ratio: 1.52,
                ideal_ratio: 1.618,
                score: 82.3,
                deviation: -0.098
            }
            // ... more features
        ],
        celebrity_match: {
            name: "Test Celebrity",
            confidence: 0.67
        }
    }
};
```

## Browser Support

- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support (iOS 14.5+ for camera)

## Troubleshooting

**Camera not working:**
- Ensure HTTPS (required for getUserMedia API)
- Check browser permissions
- Fallback to file upload if camera unavailable

**CORS errors:**
- Update backend CORS configuration to allow your frontend domain
- For local dev, backend should allow `http://localhost:3000`

**sessionStorage cleared:**
- Some browsers clear sessionStorage on page refresh
- Use Incognito/Private mode to avoid extensions interfering
