/**
 * analyzing.js
 * Handles the analyzing page - sends image to backend for analysis
 */

const API_BASE_URL = window.API_BASE_URL || (window.ENV && window.ENV.API_BASE_URL) || 'http://localhost:8000';

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

            // Convert data URL to blob
            const blob = await this.dataUrlToBlob(imageData);

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
                let errorMsg = 'Backend error: ' + analysisResponse.status;
                try {
                    const errorData = await analysisResponse.json();
                    if (errorData) {
                        if (typeof errorData.detail === 'string') {
                            errorMsg = errorData.detail;
                        } else if (errorData.detail?.message) {
                            errorMsg = errorData.detail.message;
                        } else if (Array.isArray(errorData.detail) && errorData.detail.length > 0) {
                            errorMsg = errorData.detail.map(d => d.msg || JSON.stringify(d)).join(', ');
                        } else if (errorData.message) {
                            errorMsg = errorData.message;
                        }
                    }
                } catch (e) {
                    // Fall back to default error status text
                }
                throw new Error(errorMsg);
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
            const isNetworkError = error.message.toLowerCase().includes('failed to fetch') || error.message.toLowerCase().includes('network');
            let userTitle = 'Analysis Notice';
            let userMessage = error.message;

            if (isNetworkError) {
                userTitle = 'Connection Error';
                userMessage = 'Unable to connect to the backend server. Please verify the service is running.';
            } else if (error.message.includes('We detected')) {
                userTitle = 'Photo Adjustment Needed';
                userMessage = error.message;
            } else if (error.message.includes('face the camera') || error.message.includes('head level')) {
                userTitle = 'Pose Adjustment Needed';
                userMessage = error.message;
            } else if (error.message.includes('resolution too low') || error.message.includes('quality is too low')) {
                userTitle = 'Lighting or Quality Issue';
                userMessage = error.message;
            }

            this.showThemedErrorModal(userTitle, userMessage);
        }
    }

    showThemedErrorModal(title, message) {
        // Remove existing error modal if any
        const existing = document.getElementById('facemetric-error-modal');
        if (existing) existing.remove();

        const modal = document.createElement('div');
        modal.id = 'facemetric-error-modal';
        modal.style.cssText = `
            position: fixed;
            inset: 0;
            background: rgba(10, 20, 24, 0.85);
            backdrop-filter: blur(8px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 99999;
            padding: 1.5rem;
            animation: fadeIn 0.25s ease-out;
        `;

        modal.innerHTML = `
            <div style="
                background: var(--surface-bg, #122329);
                border: 1px solid var(--accent-teal, #40CCA2);
                border-radius: 16px;
                max-width: 440px;
                width: 100%;
                padding: 2rem;
                text-align: center;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6), 0 0 25px rgba(64, 204, 162, 0.2);
                color: var(--text-primary, #FFFFFF);
                font-family: inherit;
            ">
                <div style="
                    width: 56px;
                    height: 56px;
                    margin: 0 auto 1.25rem;
                    border-radius: 50%;
                    background: rgba(64, 204, 162, 0.15);
                    border: 1px solid var(--accent-teal, #40CCA2);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: var(--accent-teal, #40CCA2);
                ">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="8" x2="12" y2="12"></line>
                        <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                </div>
                <h3 style="
                    margin: 0 0 0.75rem;
                    font-size: 1.35rem;
                    font-weight: 600;
                    color: var(--text-primary, #FFFFFF);
                    letter-spacing: -0.02em;
                "></h3>
                <p style="
                    margin: 0 0 1.75rem;
                    font-size: 0.95rem;
                    line-height: 1.55;
                    color: var(--text-secondary, rgba(255, 255, 255, 0.75));
                "></p>
                <div style="display: flex; gap: 0.75rem; justify-content: center;">
                    <button id="modal-retry-btn" style="
                        flex: 1;
                        padding: 0.85rem 1.5rem;
                        background: var(--accent-teal, #40CCA2);
                        color: #0b191e;
                        font-weight: 700;
                        font-size: 0.9rem;
                        border: none;
                        border-radius: 10px;
                        cursor: pointer;
                        transition: all 0.2s ease;
                        letter-spacing: 0.02em;
                    ">TAKE NEW PHOTO</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        const titleEl = modal.querySelector('h3');
        const messageEl = modal.querySelector('p');
        if (titleEl) titleEl.textContent = title;
        if (messageEl) messageEl.textContent = message;

        const btn = document.getElementById('modal-retry-btn');
        if (btn) {
            btn.addEventListener('mouseenter', () => {
                btn.style.filter = 'brightness(1.1)';
                btn.style.transform = 'translateY(-1px)';
            });
            btn.addEventListener('mouseleave', () => {
                btn.style.filter = 'none';
                btn.style.transform = 'none';
            });
            btn.addEventListener('click', () => {
                window.location.href = 'index.html';
            });
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

    async dataUrlToBlob(dataUrl) {
        /**
         * Convert data URL to blob for sending to API.
         * Handles both base64 and regular data URLs.
         */
        try {
            // Split the data URL to get mime type and data
            const arr = dataUrl.split(',');
            const mime = arr[0].match(/:(.*?);/)[1];
            const bstr = atob(arr[1]);
            const n = bstr.length;
            const u8arr = new Uint8Array(n);
            
            for (let i = 0; i < n; i++) {
                u8arr[i] = bstr.charCodeAt(i);
            }
            
            return new Blob([u8arr], { type: mime });
        } catch (error) {
            console.error('Error converting data URL to blob:', error);
            throw new Error('Failed to process image data');
        }
    }
}

const analyzingManager = new AnalyzingManager();
