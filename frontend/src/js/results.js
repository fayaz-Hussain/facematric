/**
 * results.js
 * Displays real analysis results from backend
 */

class ResultsManager {
    constructor() {
        this.analyzedImage = null;
        this.overlayCanvas = null;
        this.ctx = null;
        this.analysisData = null;
        this.init();
    }

    init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupElements());
        } else {
            this.setupElements();
        }
    }

    setupElements() {
        this.analyzedImage = document.getElementById('analyzedImage');
        this.overlayCanvas = document.getElementById('overlayCanvas');
        
        if (this.overlayCanvas) {
            this.ctx = this.overlayCanvas.getContext('2d');
        }

        this.loadAnalysisResults();
    }

    loadAnalysisResults() {
        try {
            const resultsJson = sessionStorage.getItem('analysisResults');
            if (!resultsJson) {
                console.error('No analysis results found');
                this.showError('No analysis results found');
                return;
            }

            const response = JSON.parse(resultsJson);
            if (response.status !== 'success') {
                throw new Error(response.detail?.message || 'Analysis failed');
            }

            this.analysisData = response.results;
            console.log('Analysis results:', this.analysisData);

            const imageData = sessionStorage.getItem('uploadedImageData');
            if (imageData && this.analyzedImage) {
                this.analyzedImage.src = imageData;
                this.analyzedImage.onload = () => {
                    if (this.overlayCanvas) {
                        const container = this.analyzedImage.parentElement;
                        this.overlayCanvas.width = container.offsetWidth;
                        this.overlayCanvas.height = container.offsetHeight;
                        this.drawOverlay();
                    }
                };
            }

            this.displayResults();

        } catch (error) {
            console.error('Error loading results:', error);
            this.showError(`Error: ${error.message}`);
        }
    }

    displayResults() {
        if (!this.analysisData) return;

        const scoreValue = this.analysisData.golden_ratio_score;
        this.animateScoreCircle(scoreValue);

        const percentileValue = this.analysisData.percentile_rank;
        const percentileElement = document.getElementById('percentileValue');
        if (percentileElement) {
            this.animateNumber(percentileElement, 0, percentileValue, 1500);
        }

        const symmetryType = this.analysisData.face_symmetry_type;
        const symmetryElement = document.getElementById('symmetryType');
        if (symmetryElement) {
            symmetryElement.textContent = symmetryType.toUpperCase();
        }

        this.displayFeatureBreakdown();

        if (this.analysisData.celebrity_match) {
            this.displayCelebrityMatch();
        }
    }

    displayFeatureBreakdown() {
        const features = this.analysisData.feature_breakdown;
        const breakdownContainer = document.querySelector('.feature-breakdown') || 
                                   document.querySelector('.features-grid') ||
                                   document.getElementById('featuresContainer');

        if (!breakdownContainer || !features) return;

        breakdownContainer.innerHTML = '';

        features.forEach(feature => {
            const featureDiv = document.createElement('div');
            featureDiv.className = 'feature-item';
            featureDiv.innerHTML = `
                <div class="feature-header">
                    <h4>${feature.feature}</h4>
                    <span class="feature-score">${feature.score.toFixed(1)}/100</span>
                </div>
                <div class="feature-bar">
                    <div class="feature-fill" style="width: ${feature.score}%"></div>
                </div>
                <div class="feature-details">
                    <small>Measured: ${feature.measured_ratio.toFixed(2)} | Ideal: ${feature.ideal_ratio.toFixed(2)}</small>
                </div>
            `;
            breakdownContainer.appendChild(featureDiv);
        });

        setTimeout(() => this.animateProportionBars(), 300);
    }

    displayCelebrityMatch() {
        const celebrity = this.analysisData.celebrity_match;
        const celebrityContainer = document.querySelector('.celebrity-match') ||
                                   document.getElementById('celebrityMatch');

        if (!celebrityContainer) return;

        const matchDiv = document.createElement('div');
        matchDiv.className = 'celebrity-match-result';
        matchDiv.innerHTML = `
            <h3>Celebrity Match</h3>
            <h4>${celebrity.name}</h4>
            <div class="confidence-badge">${(celebrity.confidence * 100).toFixed(1)}%</div>
            <p>${celebrity.message || 'Similar facial proportions detected'}</p>
        `;
        celebrityContainer.innerHTML = '';
        celebrityContainer.appendChild(matchDiv);
    }

    animateScoreCircle(score) {
        const scoreCircle = document.getElementById('scoreCircle');
        const scoreValue = document.getElementById('overallScore');
        
        if (!scoreCircle || !scoreValue) return;

        const circumference = 2 * Math.PI * 80;
        const offset = circumference - (score / 100) * circumference;

        setTimeout(() => {
            scoreCircle.style.strokeDashoffset = offset;
        }, 100);

        this.animateNumber(scoreValue, 0, score, 1500);
    }

    animateNumber(element, start, end, duration) {
        const startTime = performance.now();
        
        const updateNumber = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const current = Math.floor(start + (end - start) * easeOutQuart);
            
            element.textContent = current;
            
            if (progress < 1) {
                requestAnimationFrame(updateNumber);
            }
        };
        
        requestAnimationFrame(updateNumber);
    }

    animateProportionBars() {
        const proportionBars = document.querySelectorAll('.feature-fill');
        proportionBars.forEach((bar, index) => {
            const originalWidth = bar.style.width;
            bar.style.width = '0%';
            setTimeout(() => {
                bar.style.width = originalWidth;
            }, index * 100);
        });
    }

    drawOverlay() {
        if (!this.ctx || !this.analysisData || !this.analysisData.overlay_data) return;

        const canvas = this.overlayCanvas;
        this.ctx.clearRect(0, 0, canvas.width, canvas.height);

        const overlayData = this.analysisData.overlay_data;
        const landmarks = overlayData.landmarks;
        const guides = overlayData.golden_ratio_guides;

        if (landmarks && landmarks.length > 0) {
            this.drawLandmarks(landmarks);
        }

        if (guides) {
            this.drawGoldenRatioGuides(guides);
        }
    }

    drawLandmarks(landmarks) {
        if (!this.ctx) return;

        this.ctx.fillStyle = 'rgba(64, 204, 162, 0.9)';
        this.ctx.lineWidth = 2;

        landmarks.forEach(([x, y]) => {
            this.ctx.beginPath();
            this.ctx.arc(x, y, 3, 0, 2 * Math.PI);
            this.ctx.fill();
        });
    }

    drawGoldenRatioGuides(guides) {
        if (!this.ctx) return;

        this.ctx.strokeStyle = 'rgba(64, 204, 162, 0.5)';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([8, 8]);

        if (guides.midline) {
            this.ctx.beginPath();
            this.ctx.moveTo(guides.midline, 0);
            this.ctx.lineTo(guides.midline, this.overlayCanvas.height);
            this.ctx.stroke();
        }

        if (guides.horizontal_thirds && guides.horizontal_thirds.length > 0) {
            guides.horizontal_thirds.forEach(y => {
                this.ctx.beginPath();
                this.ctx.moveTo(0, y);
                this.ctx.lineTo(this.overlayCanvas.width, y);
                this.ctx.stroke();
            });
        }

        this.ctx.setLineDash([]);
    }

    showError(message) {
        const container = document.querySelector('main') || document.body;
        container.innerHTML = `
            <div style="text-align: center; padding: 2rem; color: red;">
                <h2>Error</h2>
                <p>${message}</p>
                <a href="index.html" style="color: var(--accent-teal);">Back to Home</a>
            </div>
        `;
    }
}

const resultsManager = new ResultsManager();

window.addEventListener('resize', () => {
    if (resultsManager.overlayCanvas && resultsManager.analyzedImage) {
        const container = resultsManager.overlayCanvas.parentElement;
        resultsManager.overlayCanvas.width = container.offsetWidth;
        resultsManager.overlayCanvas.height = container.offsetHeight;
        resultsManager.drawOverlay();
    }
});
