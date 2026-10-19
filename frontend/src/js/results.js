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

        const goldenRatioElement = document.getElementById('goldenRatioValue');
        if (goldenRatioElement) {
            this.animateNumber(goldenRatioElement, 0, Math.round(scoreValue), 1500);
        }

        const percentileValue = this.analysisData.percentile_rank;
        const percentileElement = document.getElementById('percentileValue');
        if (percentileElement) {
            this.animateNumber(percentileElement, 0, percentileValue, 1500);
        }

        const percentileStatus = document.getElementById('percentileStatus');
        if (percentileStatus) {
            percentileStatus.style.display = 'none';
        }

        const symmetryType = this.analysisData.face_symmetry_type;
        const symmetryElement = document.getElementById('symmetryType');
        if (symmetryElement && symmetryType) {
            symmetryElement.textContent = symmetryType.charAt(0).toUpperCase() + symmetryType.slice(1);
        }

        const symmetryDesc = document.getElementById('symmetryDesc');
        if (symmetryDesc && symmetryType) {
            const descriptions = {
                oval: 'Balanced & Proportioned',
                rectangle: 'Strong & Defined Structure',
                round: 'Soft & Harmonious Curves',
                square: 'Bold & Symmetrical Jawline',
                heart: 'Graceful & Sculpted Contour',
                diamond: 'Dramatic & Angular Cheekbones'
            };
            symmetryDesc.textContent = descriptions[symmetryType.toLowerCase()] || 'Harmonious Proportions';
        }

        this.displayFeatureBreakdown();
        this.displayCelebrityMatch();
    }

    displayFeatureBreakdown() {
        const features = this.analysisData.feature_breakdown;
        const breakdownContainer = document.getElementById('proportionsContainer') ||
                                   document.querySelector('.proportions-list') ||
                                   document.querySelector('.feature-breakdown') || 
                                   document.querySelector('.features-grid');

        if (!breakdownContainer || !features) return;

        breakdownContainer.innerHTML = '';

        features.forEach((feature, idx) => {
            const rowDiv = document.createElement('div');
            rowDiv.className = 'proportion-row';
            const scorePct = Math.min(100, Math.max(0, Math.round(feature.score)));
            
            rowDiv.innerHTML = `
                <div class="proportion-header-row">
                    <div class="proportion-icon-wrapper">
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="2"/>
                            <path d="M12 8V16M8 12H16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                        </svg>
                    </div>
                    <span class="proportion-label">${feature.feature}</span>
                </div>
                <div class="proportion-comparison">
                    <div class="proportion-bar-wrapper">
                        <div class="proportion-bar">
                            <div class="proportion-fill feature-fill" style="width: ${scorePct}%"></div>
                        </div>
                    </div>
                    <span class="proportion-percentage">${scorePct}%</span>
                </div>
            `;
            breakdownContainer.appendChild(rowDiv);
        });

        setTimeout(() => this.animateProportionBars(), 300);
    }

    displayCelebrityMatch() {
        const celebrity = this.analysisData ? this.analysisData.celebrity_match : null;
        const nameEl = document.getElementById('celebrityName');
        const badgeEl = document.getElementById('celebrityConfidence');
        const subtitleEl = document.getElementById('celebritySubtitle');
        const imageEl = document.getElementById('celebrityImage');
        const imageContainer = document.getElementById('celebrityImageContainer') || (imageEl ? imageEl.parentElement : null);
        const rankBadge = document.getElementById('celebrityRankBadge') || document.querySelector('.lookalike-rank-badge');
        const othersContainer = document.getElementById('lookalikeOthers');
        const backendBase = window.API_BASE_URL || (window.ENV && window.ENV.API_BASE_URL) || 'http://localhost:8000';

        if (celebrity && celebrity.name) {
            if (rankBadge) rankBadge.style.display = 'inline-flex';
            if (imageContainer) imageContainer.style.display = 'block';
            if (nameEl) {
                nameEl.textContent = celebrity.name.replace(/_/g, ' ');
                nameEl.style.display = 'block';
            }
            const simVal = (celebrity.similarity !== null && celebrity.similarity !== undefined) 
                ? celebrity.similarity 
                : celebrity.confidence;
            const strengthLabel = celebrity.strength 
                ? (celebrity.strength.charAt(0).toUpperCase() + celebrity.strength.slice(1).toLowerCase())
                : '';
                
            if (badgeEl && simVal !== null && simVal !== undefined) {
                badgeEl.style.display = 'block';
                const pct = Math.round(simVal * 100);
                badgeEl.textContent = strengthLabel 
                    ? `Similarity: ${pct}% (${strengthLabel})`
                    : `Similarity: ${pct}%`;
            } else if (badgeEl) {
                badgeEl.style.display = 'none';
                badgeEl.textContent = '';
            }
            if (subtitleEl) {
                subtitleEl.textContent = `${celebrity.dataset_source || 'Celebrity'} Match`;
                subtitleEl.style.display = 'block';
            }
            if (imageEl) {
                if (celebrity.thumbnail_url) {
                    let imgUrl = celebrity.thumbnail_url;
                    if (imgUrl.startsWith('/')) {
                        imgUrl = backendBase + imgUrl;
                    }
                    imageEl.src = encodeURI(imgUrl);
                    imageEl.alt = celebrity.name.replace(/_/g, ' ');
                    imageEl.style.display = 'block';
                } else {
                    imageEl.style.display = 'none';
                    imageEl.src = '';
                }
            }

            // Populate #2 to #5 matches
            if (othersContainer) {
                othersContainer.innerHTML = '';
                const otherMatches = (celebrity.top_matches && celebrity.top_matches.length > 1)
                    ? celebrity.top_matches.slice(1, 5)
                    : [];

                otherMatches.forEach(match => {
                    let matchImg = match.thumbnail_url || '';
                    if (matchImg && matchImg.startsWith('/')) {
                        matchImg = backendBase + matchImg;
                    }
                    const safeImgSrc = matchImg ? encodeURI(matchImg) : '';
                    const matchSim = (match.similarity !== null && match.similarity !== undefined)
                        ? match.similarity
                        : match.confidence;
                    const matchStrength = match.strength
                        ? ` (${match.strength.charAt(0).toUpperCase() + match.strength.slice(1).toLowerCase()})`
                        : '';
                    const itemDiv = document.createElement('div');
                    itemDiv.className = 'lookalike-item';
                    itemDiv.innerHTML = `
                        <span class="lookalike-item-rank">${match.rank}</span>
                        <div class="lookalike-item-image-wrapper">
                            ${safeImgSrc ? `<img src="${safeImgSrc}" alt="${match.name}" class="lookalike-item-image" onerror="this.onerror=null;this.parentElement.style.display='none';"/>` : ''}
                        </div>
                        <div class="lookalike-item-info">
                            <span class="lookalike-item-name">${match.name}</span>
                            <span class="lookalike-item-match">Similarity: ${Math.round(matchSim * 100)}%${matchStrength}</span>
                        </div>
                    `;
                    othersContainer.appendChild(itemDiv);
                });
            }
        } else {
            // When celebrity_match is null, has no name, or weak resemblance
            if (rankBadge) rankBadge.style.display = 'none';
            if (imageContainer) imageContainer.style.display = 'none';
            if (imageEl) {
                imageEl.src = '';
                imageEl.style.display = 'none';
            }
            if (badgeEl) {
                badgeEl.style.display = 'none';
                badgeEl.textContent = '';
            }
            if (nameEl) {
                if (!celebrity) {
                    nameEl.textContent = 'Look-alike matching is unavailable';
                } else {
                    nameEl.textContent = celebrity.message || 'No close resemblance found';
                }
                nameEl.style.display = 'block';
            }
            if (subtitleEl) {
                subtitleEl.textContent = '';
                subtitleEl.style.display = 'none';
            }
            if (othersContainer) {
                othersContainer.innerHTML = '';
            }
        }
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
            }, index * 80);
        });
    }

    drawOverlay() {
        if (!this.ctx || !this.analysisData || !this.analysisData.overlay_data || !this.analyzedImage) return;

        const canvas = this.overlayCanvas;
        const img = this.analyzedImage;
        
        // Ensure canvas dimensions match rendered image dimensions
        const rect = img.getBoundingClientRect();
        const displayWidth = img.clientWidth || canvas.parentElement.offsetWidth;
        const displayHeight = img.clientHeight || canvas.parentElement.offsetHeight;

        canvas.width = displayWidth;
        canvas.height = displayHeight;
        this.ctx.clearRect(0, 0, canvas.width, canvas.height);

        const overlayData = this.analysisData.overlay_data;
        const landmarks = overlayData.landmarks;
        const guides = overlayData.golden_ratio_guides;

        // Landmarks coordinates are in pre-processed image dimensions (max 640px)
        const naturalWidth = img.naturalWidth || displayWidth;
        const naturalHeight = img.naturalHeight || displayHeight;
        const scaleX = displayWidth / naturalWidth;
        const scaleY = displayHeight / naturalHeight;

        if (landmarks && landmarks.length > 0) {
            this.drawLandmarks(landmarks, scaleX, scaleY);
        }

        if (guides) {
            this.drawGoldenRatioGuides(guides, scaleX, scaleY);
        }
    }

    drawLandmarks(landmarks, scaleX = 1, scaleY = 1) {
        if (!this.ctx) return;

        this.ctx.fillStyle = 'rgba(64, 204, 162, 0.9)';
        this.ctx.lineWidth = 1.5;

        landmarks.forEach(([x, y]) => {
            this.ctx.beginPath();
            this.ctx.arc(x * scaleX, y * scaleY, 2.5, 0, 2 * Math.PI);
            this.ctx.fill();
        });
    }

    drawGoldenRatioGuides(guides, scaleX = 1, scaleY = 1) {
        if (!this.ctx) return;

        this.ctx.strokeStyle = 'rgba(64, 204, 162, 0.6)';
        this.ctx.lineWidth = 1.5;
        this.ctx.setLineDash([6, 6]);

        const midline = guides.vertical_midline !== undefined ? guides.vertical_midline : guides.midline;
        if (midline !== undefined && midline !== null) {
            const mx = midline * scaleX;
            this.ctx.beginPath();
            this.ctx.moveTo(mx, 0);
            this.ctx.lineTo(mx, this.overlayCanvas.height);
            this.ctx.stroke();
        }

        const thirds = guides.horizontal_thirds;
        if (thirds && thirds.length > 0) {
            thirds.forEach(y => {
                const my = y * scaleY;
                this.ctx.beginPath();
                this.ctx.moveTo(0, my);
                this.ctx.lineTo(this.overlayCanvas.width, my);
                this.ctx.stroke();
            });
        }

        if (guides.phi_spiral_center && guides.phi_spiral_center.length >= 2) {
            const [cx, cy] = guides.phi_spiral_center;
            this.ctx.setLineDash([]);
            this.ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)';
            this.ctx.beginPath();
            this.ctx.arc(cx * scaleX, cy * scaleY, 6, 0, 2 * Math.PI);
            this.ctx.stroke();
        }

        this.ctx.setLineDash([]);
    }

    showError(message) {
        const disclaimerEl = document.querySelector('.results-disclaimer');
        const disclaimerHtml = disclaimerEl ? disclaimerEl.outerHTML : `
            <div class="results-disclaimer">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                    <path d="M12 8V12M12 16H12.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <p>This tool analyzes geometric facial proportions for educational purposes. Scores do not reflect beauty, worth, or health. Results are influenced by photo angle and quality.</p>
            </div>
        `;
        const container = document.querySelector('main') || document.body;
        container.innerHTML = `
            <div class="results-container">
                <div style="text-align: center; padding: 2rem; color: red;">
                    <h2>Error</h2>
                    <p>${message}</p>
                    <a href="index.html" style="color: var(--accent-teal);">Back to Home</a>
                </div>
                ${disclaimerHtml}
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
