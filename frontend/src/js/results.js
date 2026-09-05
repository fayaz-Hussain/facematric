/**
 * results.js
 * Handles the results page functionality
 */

class ResultsManager {
    constructor() {
        this.analyzedImage = null;
        this.overlayCanvas = null;
        this.ctx = null;
        this.overallScore = 85;
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
        this.analyzedImage = document.getElementById('analyzedImage');
        this.overlayCanvas = document.getElementById('overlayCanvas');
        
        if (this.overlayCanvas) {
            this.ctx = this.overlayCanvas.getContext('2d');
        }

        // Load the analyzed image
        this.loadAnalyzedImage();

        // Animate the score circle
        this.animateScoreCircle();

        // Draw facial landmarks overlay
        setTimeout(() => this.drawFacialLandmarks(), 500);
    }

    loadAnalyzedImage() {
        const imageData = sessionStorage.getItem('uploadedImageData');
        if (imageData && this.analyzedImage) {
            this.analyzedImage.src = imageData;
            
            // Set canvas size when image loads
            this.analyzedImage.onload = () => {
                if (this.overlayCanvas) {
                    this.overlayCanvas.width = this.analyzedImage.offsetWidth;
                    this.overlayCanvas.height = this.analyzedImage.offsetHeight;
                    this.drawFacialLandmarks();
                }
            };
        } else {
            // If no image data, redirect back to home
            console.warn('No image data found, redirecting to home');
            setTimeout(() => {
                window.location.href = 'index.html';
            }, 1000);
        }
    }

    animateScoreCircle() {
        const scoreCircle = document.getElementById('scoreCircle');
        const scoreValue = document.getElementById('overallScore');
        
        if (!scoreCircle || !scoreValue) return;

        // Calculate stroke-dashoffset based on score
        const circumference = 2 * Math.PI * 90; // 2πr where r=90
        const offset = circumference - (this.overallScore / 100) * circumference;

        // Animate from 0 to final score
        setTimeout(() => {
            scoreCircle.style.strokeDashoffset = offset;
        }, 100);

        // Animate the number
        this.animateNumber(scoreValue, 0, this.overallScore, 1500);
    }

    animateNumber(element, start, end, duration) {
        const startTime = performance.now();
        
        const updateNumber = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const current = Math.floor(start + (end - start) * easeOutQuart);
            
            element.textContent = current;
            
            if (progress < 1) {
                requestAnimationFrame(updateNumber);
            }
        };
        
        requestAnimationFrame(updateNumber);
    }

    drawFacialLandmarks() {
        if (!this.ctx || !this.analyzedImage.complete) return;

        const canvas = this.overlayCanvas;
        const img = this.analyzedImage;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Get image dimensions
        const imgRect = img.getBoundingClientRect();
        const canvasRect = canvas.getBoundingClientRect();
        
        // Calculate scaling
        const scaleX = canvas.width / imgRect.width;
        const scaleY = canvas.height / imgRect.height;

        // Set drawing style
        this.ctx.strokeStyle = 'rgba(64, 204, 162, 0.8)';
        this.ctx.fillStyle = 'rgba(64, 204, 162, 0.9)';
        this.ctx.lineWidth = 2;

        // Draw example landmarks (in a real app, these would come from face detection)
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const faceWidth = canvas.width * 0.4;
        const faceHeight = canvas.height * 0.6;

        // Draw vertical center line
        this.ctx.setLineDash([5, 5]);
        this.ctx.beginPath();
        this.ctx.moveTo(centerX, centerY - faceHeight / 2);
        this.ctx.lineTo(centerX, centerY + faceHeight / 2);
        this.ctx.stroke();

        // Draw horizontal guidelines
        const thirds = [
            centerY - faceHeight / 3,
            centerY,
            centerY + faceHeight / 3
        ];

        thirds.forEach(y => {
            this.ctx.beginPath();
            this.ctx.moveTo(centerX - faceWidth / 2, y);
            this.ctx.lineTo(centerX + faceWidth / 2, y);
            this.ctx.stroke();
        });

        // Reset line dash
        this.ctx.setLineDash([]);

        // Draw key points
        const keyPoints = [
            // Eyes
            { x: centerX - faceWidth * 0.25, y: centerY - faceHeight * 0.15 },
            { x: centerX + faceWidth * 0.25, y: centerY - faceHeight * 0.15 },
            // Nose
            { x: centerX, y: centerY + faceHeight * 0.05 },
            // Mouth corners
            { x: centerX - faceWidth * 0.2, y: centerY + faceHeight * 0.25 },
            { x: centerX + faceWidth * 0.2, y: centerY + faceHeight * 0.25 },
            // Chin
            { x: centerX, y: centerY + faceHeight * 0.45 }
        ];

        keyPoints.forEach(point => {
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
            this.ctx.fill();
        });

        // Draw face outline
        this.ctx.strokeStyle = 'rgba(64, 204, 162, 0.5)';
        this.ctx.beginPath();
        this.ctx.ellipse(
            centerX,
            centerY,
            faceWidth / 2,
            faceHeight / 2,
            0,
            0,
            2 * Math.PI
        );
        this.ctx.stroke();
    }
}

// Initialize results manager
const resultsManager = new ResultsManager();

// Export function for download button
function downloadReport() {
    alert('Download functionality will generate a PDF report with your analysis results.');
    // TODO: Implement PDF generation
}

// Handle window resize
window.addEventListener('resize', () => {
    if (resultsManager.overlayCanvas && resultsManager.analyzedImage) {
        resultsManager.overlayCanvas.width = resultsManager.analyzedImage.offsetWidth;
        resultsManager.overlayCanvas.height = resultsManager.analyzedImage.offsetHeight;
        resultsManager.drawFacialLandmarks();
    }
});
