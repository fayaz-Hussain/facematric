/**
 * analyzing.js
 * Handles the analyzing page animation and progress
 */

class AnalyzingManager {
    constructor() {
        this.progressFill = null;
        this.progressText = null;
        this.uploadedImage = null;
        this.steps = [];
        this.currentStep = 0;
        this.progress = 0;
        this.init();
    }

    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupElements());
        } else {
            this.setupElements();
        }
    }

    setupElements() {
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.uploadedImage = document.getElementById('uploadedImage');
        this.steps = [
            document.getElementById('step1'),
            document.getElementById('step2'),
            document.getElementById('step3'),
            document.getElementById('step4')
        ];

        // Load the uploaded image from sessionStorage
        this.loadUploadedImage();

        // Start the analysis animation
        this.startAnalysis();
    }

    loadUploadedImage() {
        const imageData = sessionStorage.getItem('uploadedImageData');
        if (imageData && this.uploadedImage) {
            this.uploadedImage.src = imageData;
        } else {
            // If no image data, redirect back to home
            console.warn('No image data found, redirecting to home');
            setTimeout(() => {
                window.location.href = 'index.html';
            }, 1000);
        }
    }

    startAnalysis() {
        // Simulate analysis process
        const stepDuration = 2500; // 2.5 seconds per step
        const totalSteps = this.steps.length;

        // Activate first step immediately
        if (this.steps[0]) {
            this.steps[0].classList.add('active');
        }

        // Progress through each step
        const stepInterval = setInterval(() => {
            if (this.currentStep < totalSteps) {
                // Complete current step
                if (this.steps[this.currentStep]) {
                    this.steps[this.currentStep].classList.remove('active');
                    this.steps[this.currentStep].classList.add('completed');
                }

                // Move to next step
                this.currentStep++;
                
                if (this.currentStep < totalSteps) {
                    // Activate next step
                    if (this.steps[this.currentStep]) {
                        this.steps[this.currentStep].classList.add('active');
                    }
                } else {
                    // All steps complete
                    clearInterval(stepInterval);
                    clearInterval(progressInterval);
                    this.completeAnalysis();
                }
            }
        }, stepDuration);

        // Update progress bar smoothly
        const progressInterval = setInterval(() => {
            if (this.progress < 100) {
                this.progress += 0.5;
                this.updateProgress(this.progress);
            } else {
                clearInterval(progressInterval);
            }
        }, 50);
    }

    updateProgress(percent) {
        if (this.progressFill) {
            this.progressFill.style.width = percent + '%';
        }
        if (this.progressText) {
            this.progressText.textContent = Math.round(percent) + '%';
        }
    }

    completeAnalysis() {
        // Ensure progress is at 100%
        this.updateProgress(100);

        // Wait a moment to show completion, then navigate to results
        setTimeout(() => {
            console.log('Analysis complete! Navigating to results...');
            window.location.href = 'results.html';
        }, 1000);
    }

    showCompletionMessage() {
        const statusHeader = document.querySelector('.status-header');
        if (statusHeader) {
            statusHeader.innerHTML = `
                <h2>Analysis Complete! ✨</h2>
                <p class="subtitle">Your facial analysis report is ready.</p>
            `;
        }

        // Show a button to view results (placeholder for now)
        const securityNote = document.querySelector('.security-note');
        if (securityNote) {
            securityNote.innerHTML = `
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                    <path d="M9 12L11 14L15 10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span>Analysis completed successfully!</span>
            `;
            securityNote.style.background = 'rgba(64, 204, 162, 0.15)';
        }
    }
}

// Initialize analyzing manager
const analyzingManager = new AnalyzingManager();
