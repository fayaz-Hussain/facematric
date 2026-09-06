/**
 * results.js
 * Handles the results page functionality - FaceMetric Design System
 * Preserves all existing functionality with new UI
 */

class ResultsManager {
    constructor() {
        this.analyzedImage = null;
        this.overlayCanvas = null;
        this.ctx = null;
        this.overallScore = 86;
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

        // Animate proportion bars
        setTimeout(() => this.animateProportionBars(), 300);

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
                    const container = this.analyzedImage.parentElement;
                    this.overlayCanvas.width = container.offsetWidth;
                    this.overlayCanvas.height = container.offsetHeight;
                    this.drawFacialLandmarks();
                }
            };
        } else {
            // If no image data, use placeholder
            console.warn('No image data found, using placeholder');
            this.analyzedImage.src = 'https://via.placeholder.com/400x600/E8F7F2/40CCA2?text=Face+Image';
        }
    }

    animateScoreCircle() {
        const scoreCircle = document.getElementById('scoreCircle');
        const scoreValue = document.getElementById('overallScore');
        
        if (!scoreCircle || !scoreValue) return;

        // Calculate stroke-dashoffset based on score (radius = 80)
        const circumference = 2 * Math.PI * 80; // 2πr where r=80
        const offset = circumference - (this.overallScore / 100) * circumference;

        // Animate from 0 to final score
        setTimeout(() => {
            scoreCircle.style.strokeDashoffset = offset;
        }, 100);

        // Animate the number
        this.animateNumber(scoreValue, 0, this.overallScore, 1500);
    }

    animateProportionBars() {
        const proportionBars = document.querySelectorAll('.proportion-fill');
        proportionBars.forEach((bar, index) => {
            const originalWidth = bar.style.width;
            bar.style.width = '0%';
            setTimeout(() => {
                bar.style.width = originalWidth;
            }, index * 100);
        });
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
        const container = canvas.parentElement;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Get container dimensions
        const containerWidth = container.offsetWidth;
        const containerHeight = container.offsetHeight;

        // Calculate center
        const centerX = containerWidth / 2;
        const centerY = containerHeight / 2;
        const faceWidth = containerWidth * 0.35;
        const faceHeight = containerHeight * 0.55;

        // Set drawing style
        this.ctx.strokeStyle = 'rgba(64, 204, 162, 0.8)';
        this.ctx.fillStyle = 'rgba(64, 204, 162, 0.9)';
        this.ctx.lineWidth = 2;

        // Draw vertical center line (mirror axis)
        this.ctx.setLineDash([8, 8]);
        this.ctx.beginPath();
        this.ctx.moveTo(centerX, centerY - faceHeight / 2);
        this.ctx.lineTo(centerX, centerY + faceHeight / 2);
        this.ctx.stroke();

        // Draw horizontal guidelines (facial thirds)
        this.ctx.strokeStyle = 'rgba(64, 204, 162, 0.4)';
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
        this.ctx.strokeStyle = 'rgba(64, 204, 162, 0.8)';

        // Draw key facial points
        const keyPoints = [
            // Eyes
            { x: centerX - faceWidth * 0.25, y: centerY - faceHeight * 0.18, size: 5 },
            { x: centerX + faceWidth * 0.25, y: centerY - faceHeight * 0.18, size: 5 },
            // Nose tip
            { x: centerX, y: centerY + faceHeight * 0.08, size: 5 },
            // Mouth corners
            { x: centerX - faceWidth * 0.22, y: centerY + faceHeight * 0.28, size: 4 },
            { x: centerX + faceWidth * 0.22, y: centerY + faceHeight * 0.28, size: 4 },
            // Eyebrow points
            { x: centerX - faceWidth * 0.3, y: centerY - faceHeight * 0.25, size: 3 },
            { x: centerX + faceWidth * 0.3, y: centerY - faceHeight * 0.25, size: 3 },
            // Chin
            { x: centerX, y: centerY + faceHeight * 0.45, size: 5 }
        ];

        // Draw points with glow effect
        keyPoints.forEach(point => {
            // Glow
            this.ctx.shadowColor = 'rgba(64, 204, 162, 0.6)';
            this.ctx.shadowBlur = 10;
            
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, point.size, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // Reset shadow
            this.ctx.shadowBlur = 0;
        });

        // Draw face outline (ellipse)
        this.ctx.strokeStyle = 'rgba(64, 204, 162, 0.5)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.ellipse(
            centerX,
            centerY + faceHeight * 0.05,
            faceWidth / 2,
            faceHeight / 2,
            0,
            0,
            2 * Math.PI
        );
        this.ctx.stroke();

        // Draw connecting lines between symmetrical points
        this.ctx.strokeStyle = 'rgba(64, 204, 162, 0.3)';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([4, 4]);
        
        // Connect eyes
        this.ctx.beginPath();
        this.ctx.moveTo(keyPoints[0].x, keyPoints[0].y);
        this.ctx.lineTo(keyPoints[1].x, keyPoints[1].y);
        this.ctx.stroke();
        
        // Connect mouth corners
        this.ctx.beginPath();
        this.ctx.moveTo(keyPoints[3].x, keyPoints[3].y);
        this.ctx.lineTo(keyPoints[4].x, keyPoints[4].y);
        this.ctx.stroke();
        
        this.ctx.setLineDash([]);
    }
}

// Initialize results manager
const resultsManager = new ResultsManager();

// Handle window resize
window.addEventListener('resize', () => {
    if (resultsManager.overlayCanvas && resultsManager.analyzedImage) {
        const container = resultsManager.overlayCanvas.parentElement;
        resultsManager.overlayCanvas.width = container.offsetWidth;
        resultsManager.overlayCanvas.height = container.offsetHeight;
        resultsManager.drawFacialLandmarks();
    }
});
