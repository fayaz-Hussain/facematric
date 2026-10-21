/**
 * upload.js
 * Handles file upload and drag-and-drop functionality
 */

class UploadManager {
    constructor() {
        this.dropZone = null;
        this.fileInput = null;
        this.maxFileSize = 10 * 1024 * 1024; // 10MB
        this.allowedTypes = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/webp',
            'image/heic', 'image/heif', 'image/bmp', 'image/gif',
            'image/jfif', 'image/pjpeg', 'image/x-png'
        ];
        this.allowedExtensions = ['.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.jfif', '.bmp', '.gif'];
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
        this.dropZone = document.getElementById('dropZone');
        this.fileInput = document.getElementById('fileInput');

        if (this.dropZone) {
            this.setupDragAndDrop();
        }
    }

    setupDragAndDrop() {
        this.dropZone.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.dropZone.addEventListener('dragleave', () => this.handleDragLeave());
        this.dropZone.addEventListener('drop', (e) => this.handleDrop(e));
    }

    handleDragOver(e) {
        e.preventDefault();
        this.dropZone.style.borderColor = 'var(--accent-teal)';
    }

    handleDragLeave() {
        this.dropZone.style.borderColor = 'var(--border-teal)';
    }

    handleDrop(e) {
        e.preventDefault();
        this.dropZone.style.borderColor = 'var(--border-teal)';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    isValidImage(file) {
        if (!file) return false;
        const type = (file.type || '').toLowerCase();
        if (type && this.allowedTypes.includes(type)) {
            return true;
        }
        const filename = (file.name || '').toLowerCase();
        return this.allowedExtensions.some(ext => filename.endsWith(ext));
    }

    showError(title, message) {
        const existing = document.getElementById('upload-error-modal');
        if (existing) existing.remove();

        const modal = document.createElement('div');
        modal.id = 'upload-error-modal';
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
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: var(--accent-teal, #40CCA2);
                ">
                    <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                </div>
                <h3 style="font-size: 1.25rem; margin-bottom: 0.75rem; color: #FFFFFF;">${title}</h3>
                <p style="font-size: 0.95rem; line-height: 1.5; color: var(--text-secondary, #A0B2B8); margin-bottom: 1.5rem;">${message}</p>
                <button onclick="document.getElementById('upload-error-modal').remove()" style="
                    background: var(--accent-teal, #40CCA2);
                    color: #0A1418;
                    border: none;
                    border-radius: 8px;
                    padding: 0.75rem 1.75rem;
                    font-weight: 600;
                    cursor: pointer;
                    font-size: 0.95rem;
                ">OK</button>
            </div>
        `;
        document.body.appendChild(modal);
    }

    processFile(file) {
        // Validate file type
        if (!this.isValidImage(file)) {
            this.showError('Invalid File Format', 'Please upload a supported image file (JPEG, PNG, WebP, HEIC, BMP, GIF).');
            return;
        }

        // Validate file size
        if (file.size > this.maxFileSize) {
            this.showError('File Too Large', 'File size exceeds the 10MB limit. Please upload a smaller image.');
            return;
        }

        console.log('File selected:', file.name);
        
        // Convert file to data URL and store in sessionStorage
        const reader = new FileReader();
        reader.onload = (e) => {
            // Store image data for the analyzing page
            sessionStorage.setItem('uploadedImageData', e.target.result);
            sessionStorage.setItem('uploadedFileName', file.name);
            sessionStorage.setItem('uploadedFileSize', (file.size / 1024 / 1024).toFixed(2));
            
            // Navigate to analyzing page
            window.location.href = 'analyzing.html';
        };
        reader.readAsDataURL(file);
    }

    async openCamera() {
        try {
            // Request camera access
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'user' } 
            });
            
            // Create a modal for camera preview
            this.createCameraModal(stream);
        } catch (error) {
            console.error('Camera access error:', error);
            if (error.name === 'NotAllowedError') {
                alert('Camera access was denied. Please allow camera access to use this feature.');
            } else if (error.name === 'NotFoundError') {
                alert('No camera found on this device.');
            } else {
                alert('Unable to access camera. Please try uploading an image instead.');
            }
        }
    }

    createCameraModal(stream) {
        // Create modal elements
        const modal = document.createElement('div');
        modal.className = 'camera-modal';
        modal.innerHTML = `
            <div class="camera-container">
                <div class="camera-header">
                    <h3>Take a Photo</h3>
                    <button class="close-btn" onclick="uploadManager.closeCameraModal()">✕</button>
                </div>
                <video id="cameraPreview" autoplay playsinline></video>
                <div class="camera-controls">
                    <button class="btn btn-secondary" onclick="uploadManager.closeCameraModal()">Cancel</button>
                    <button class="btn btn-primary" onclick="uploadManager.capturePhoto()">
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="12" cy="12" r="10" stroke="white" stroke-width="2"/>
                        </svg>
                        Capture
                    </button>
                </div>
            </div>
        `;

        // Add styles for the modal
        const style = document.createElement('style');
        style.textContent = `
            .camera-modal {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.9);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 1000;
            }
            .camera-container {
                background: var(--surface-bg);
                border-radius: 20px;
                padding: 1.5rem;
                max-width: 600px;
                width: 90%;
            }
            .camera-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }
            .camera-header h3 {
                color: var(--text-primary);
                margin: 0;
            }
            .close-btn {
                background: none;
                border: none;
                color: var(--text-secondary);
                font-size: 1.5rem;
                cursor: pointer;
                padding: 0.25rem 0.5rem;
            }
            #cameraPreview {
                width: 100%;
                border-radius: 12px;
                margin-bottom: 1rem;
            }
            .camera-controls {
                display: flex;
                gap: 1rem;
                justify-content: center;
            }
        `;
        document.head.appendChild(style);

        // Add modal to page
        document.body.appendChild(modal);
        
        // Start video stream
        const video = document.getElementById('cameraPreview');
        video.srcObject = stream;
        this.currentStream = stream;
    }

    capturePhoto() {
        const video = document.getElementById('cameraPreview');
        if (!video) return;

        // Create canvas to capture the frame
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // Convert to blob and process
        canvas.toBlob((blob) => {
            // Create a file from the blob
            const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
            
            // Close camera modal
            this.closeCameraModal();
            
            // Process the captured image
            this.processFile(file);
        }, 'image/jpeg', 0.95);
    }

    closeCameraModal() {
        // Stop video stream
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
            this.currentStream = null;
        }

        // Remove modal
        const modal = document.querySelector('.camera-modal');
        if (modal) {
            modal.remove();
        }
    }
}

// Initialize upload manager
const uploadManager = new UploadManager();

// Export functions for use in HTML
function handleFileSelect(event) {
    uploadManager.handleFileSelect(event);
}

function openCamera() {
    uploadManager.openCamera();
}
