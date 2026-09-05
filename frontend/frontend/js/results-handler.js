// Results Handler for Results Screen

class ResultsHandler {
    constructor() {
        this.results = null;
    }

    initialize() {
        // Get results from session storage
        const resultsJSON = sessionStorage.getItem('analysisResults');
        if (!resultsJSON) {
            // No results found, redirect to upload
            window.location.href = '../facemetric_upload_photo_updated_mint/code.html';
            return;
        }

        this.results = JSON.parse(resultsJSON);

        // Populate UI with results
        this.populateResults();

        // Attach event listener to "NEW ANALYSIS" button
        this.attachNewAnalysisHandler();
    }

    populateResults() {
        // Overall Score
        this.updateOverallScore(this.results.golden_ratio_score);

        // Face Shape
        this.updateFaceShape(this.results.face_symmetry_type);

        // Golden Ratio Score (might be duplicate of overall, update both)
        this.updateGoldenRatioDisplay(this.results.golden_ratio_score);

        // Percentile
        if (this.results.percentile_rank !== undefined) {
            this.updatePercentile(this.results.percentile_rank);
        }

        // Feature Breakdown
        if (this.results.feature_breakdown) {
            this.updateFeatureBreakdown(this.results.feature_breakdown);
        }

        // Celebrity Match
        if (this.results.celebrity_match) {
            this.updateCelebrityMatch(this.results.celebrity_match);
        }

        // Symmetry Photo (use uploaded image with overlay)
        const uploadedImage = sessionStorage.getItem('uploadedImage');
        if (uploadedImage) {
            this.updateSymmetryPhoto(uploadedImage);
        }
    }

    updateOverallScore(score) {
        // Find the score display elements
        const scoreDisplays = document.querySelectorAll('.font-display, [class*="text-5xl"], [class*="text-\\[48px\\]"]');
        
        scoreDisplays.forEach(el => {
            // Check if this element contains a number that looks like a score
            if (el.textContent.match(/^\d+$/)) {
                el.textContent = Math.round(score);
            }
        });

        // Update circular progress if exists
        const progressCircles = document.querySelectorAll('svg circle:last-child');
        progressCircles.forEach(circle => {
            const radius = parseFloat(circle.getAttribute('r') || 45);
            const circumference = 2 * Math.PI * radius;
            const offset = circumference - (score / 100) * circumference;
            circle.setAttribute('stroke-dasharray', circumference);
            circle.setAttribute('stroke-dashoffset', offset);
        });

        // Update label
        const label = this.getScoreLabel(score);
        const labelElements = document.querySelectorAll('.font-headline');
        labelElements.forEach(el => {
            if (el.textContent.match(/balance|proportioned|harmony/i)) {
                el.textContent = label;
            }
        });
    }

    getScoreLabel(score) {
        if (score >= 90) return 'Exceptional';
        if (score >= 80) return 'Great Balance';
        if (score >= 70) return 'Well Proportioned';
        if (score >= 60) return 'Good Harmony';
        return 'Unique Character';
    }

    updateFaceShape(symmetryType) {
        // Find elements that might contain face shape text
        const shapeElements = document.querySelectorAll('.font-headline, h2, h3');
        shapeElements.forEach(el => {
            if (el.textContent.match(/oval|rectangle|round|square|heart|diamond/i)) {
                el.textContent = this.capitalizeFirst(symmetryType);
            }
        });
    }

    updateGoldenRatioDisplay(score) {
        // Update any golden ratio specific displays
        const ratioElements = document.querySelectorAll('[class*="text-\\[48px\\]"], .font-display');
        ratioElements.forEach(el => {
            // If it has a /100 notation, update it
            if (el.innerHTML.includes('/100')) {
                el.innerHTML = `${Math.round(score)}<span class="text-[24px] text-[#25272C]/70">/100</span>`;
            }
        });
    }

    updatePercentile(percentile) {
        // Add or update percentile display
        console.log(`Percentile rank: ${percentile}% - User scores higher than ${percentile}% of people`);
        
        // Try to find a place to display percentile
        const cards = document.querySelectorAll('.card-bg');
        // Could add percentile to one of the summary cards if space exists
    }

    updateFeatureBreakdown(features) {
        // Find the breakdown container
        const breakdownContainers = document.querySelectorAll('.flex.flex-col.gap-6, .flex.flex-col.gap-4');
        
        breakdownContainers.forEach(container => {
            // Check if this looks like a metrics container
            if (container.innerHTML.includes('Face Length') || container.innerHTML.includes('Eye Distance')) {
                // Clear and repopulate
                container.innerHTML = '';
                
                // Display up to 3 features
                features.slice(0, 3).forEach(feature => {
                    const metricHTML = this.createFeatureMetricHTML(feature);
                    container.insertAdjacentHTML('beforeend', metricHTML);
                });
            }
        });
    }

    createFeatureMetricHTML(feature) {
        const percentage = feature.score;

        return `
            <div>
                <div class="flex justify-between font-body text-base text-[#25272C] mb-1">
                    <span>${feature.feature}</span>
                    <span class="text-[#25272C]/70">Your Ratio: ${feature.measured_ratio.toFixed(2)} <span class="mx-2">|</span> Ideal: ${feature.ideal_ratio.toFixed(3)}</span>
                </div>
                <div class="w-full h-2 bg-[#25272C]/10 rounded-full overflow-hidden relative">
                    <div class="absolute top-0 bottom-0 left-0 bg-[#25272C]/80 rounded-full transition-all duration-1000" style="width: ${percentage}%"></div>
                    <div class="absolute top-0 bottom-0 left-[97%] w-1 bg-[#25272C] z-10" title="Golden Ratio (φ = 1.618)"></div>
                </div>
            </div>
        `;
    }

    updateCelebrityMatch(match) {
        if (!match || !match.name) {
            // No match found - update UI
            const celebritySections = document.querySelectorAll('.card-bg');
            celebritySections.forEach(section => {
                if (section.innerHTML.includes('Geometric Matches') || section.innerHTML.includes('Match')) {
                    section.innerHTML = `
                        <div class="font-label text-xs font-semibold text-[#25272C]/70 uppercase tracking-widest mb-6">Geometric Matches</div>
                        <div class="text-center py-8">
                            <p class="font-body text-base text-[#25272C]/70">${match.message || 'No close match found in our dataset'}</p>
                        </div>
                    `;
                }
            });
            return;
        }

        // Update celebrity name and confidence
        const nameElements = document.querySelectorAll('.font-headline.text-xl, h3');
        nameElements.forEach(el => {
            // Look for celebrity name placeholders
            if (el.textContent.includes('Holland') || el.innerHTML.includes('celebrity')) {
                el.textContent = match.name;
            }
        });

        // Update confidence percentage
        const confidenceElements = document.querySelectorAll('span');
        confidenceElements.forEach(el => {
            if (el.textContent.includes('Match') || el.textContent.includes('%')) {
                el.textContent = `${Math.round(match.confidence * 100)}% Match`;
            }
        });

        // Update image if thumbnail available
        if (match.thumbnail_url) {
            const celebImages = document.querySelectorAll('.rounded-full img, img[class*="rounded"]');
            celebImages.forEach(img => {
                // Update primary match image (typically larger)
                if (img.closest('.w-32') || img.closest('[class*="w-"]')) {
                    img.src = match.thumbnail_url;
                    img.alt = match.name;
                }
            });
        }
    }

    updateSymmetryPhoto(imageBase64) {
        // Find the symmetry analysis image
        const symmetryImages = document.querySelectorAll('img');
        symmetryImages.forEach(img => {
            // Look for the image in the symmetry section
            const parent = img.closest('.relative');
            if (parent && parent.querySelector('svg')) {
                // This is likely the symmetry overlay image
                img.src = imageBase64;
            }
        });
    }

    attachNewAnalysisHandler() {
        // Find "NEW ANALYSIS" button or similar
        const buttons = document.querySelectorAll('button');
        buttons.forEach(btn => {
            if (btn.textContent.includes('NEW ANALYSIS') || btn.textContent.includes('Try Again') || btn.querySelector('[data-icon="add"]')) {
                btn.addEventListener('click', () => {
                    sessionStorage.clear();
                    window.location.href = '../facemetric_upload_photo_updated_mint/code.html';
                });
            }
        });
    }

    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const resultsHandler = new ResultsHandler();
    resultsHandler.initialize();
});
