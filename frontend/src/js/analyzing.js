/**
 * analyzing.js
 * Handles the analyzing page - sends image to backend for analysis
 */

const API_BASE_URL = 'http://localhost:8000';

class AnalyzingManager {
    constructor() {
        this.progressFill = null;
        this.progressText = null;
        this.uploadedImage = null;
        this.steps = [];
        this.currentStep = 0;
        this.progress = 0;
        this.analysisResult = null;
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
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.uploadedImage = document.getElementById('uploadedImage');
        this.steps = [
            document.getElementById('step1'),
            document.getElementById('step2'),
            document.getElementById('step3'),
            document.getElementById('step4')
        ];

        this.loadUploadedImage();
        this.startAnalysis();
    }

    loadUploadedImage() {
        const imageData = sessionStorage.getItem('uploadedImageData');
        if (imageData && this.uploadedImage) {
            this.uploadedImage.src = imageData;
        } else {
            setTimeout(() => {
                window.location.href = 'index.html';
            }, 1000);
        }
    }

    async startAnalysis() {
        try {
            if (this.steps[0]) {
                this.steps[0].classList.add('active');
            }
            this.updateProgress(10);

            const imageData = sessionStorage.getItem('uploadedImageData');
            if (!imageData) {
                throw new Error('No image data found');
            }

            this.updateProgress(20);
            if (this.steps[0]) {
                this.steps[0].classList.remove('active');
                this.steps[0].classList.add('completed');
            }
            if (this.steps[1]) {
                this.steps[1].classList.add('active');
            }

            const response = await fetch(imageData);
            const blob = await response.blob();

            const formData = new FormData();
            formData.append('file', blob, 'image.jpg');

            this.updateProgress(40);
            if (this.steps[1]) {
                this.steps[1].classList.remove('active');
                this.steps[1].classList.add('completed');
            }
            if (this.steps[2]) {
                this.steps[2].classList.add('active');
            }

            const analysisResponse = await fetch(API_BASE_URL + '/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!analysisResponse.ok) {
                const errorData = await analysisResponse.json();
                throw new Error(errorData.detail?.message || 'Backend error: ' + analysisResponse.status);
            }

            this.analysisResult = await analysisResponse.json();
            
            this.updateProgress(80);
            if (this.steps[2]) {
                this.steps[2].classList.remove('active');
                this.steps[2].classList.add('completed');
            }
            if (this.steps[3]) {
                this.steps[3].classList.add('active');
            }

            sessionStorage.setItem('analysisResults', JSON.stringify(this.analysisResult));

            this.updateProgress(100);
            
            setTimeout(() => {
                if (this.steps[3]) {
                    this.steps[3].classList.remove('active');
                    this.steps[3].classList.add('completed');
                }
                setTimeout(() => {
                    window.location.href = 'results.html';
                }, 500);
            }, 1000);

        } catch (error) {
            console.error('Analysis error:', error);
            alert('Analysis failed: ' + error.message + '\n\nMake sure backend is running at http://localhost:8000');
            setTimeout(() => {
                window.location.href = 'index.html';
            }, 2000);
        }
    }

    updateProgress(percent) {
        if (this.progressFill) {
            this.progressFill.style.width = percent + '%';
        }
        if (this.progressText) {
            this.progressText.textContent = Math.round(percent) + '%';
        }
    }
}

const analyzingManager = new AnalyzingManager();
