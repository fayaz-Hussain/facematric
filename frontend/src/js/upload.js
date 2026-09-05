/**
 * upload.js
 * Handles file upload and drag-and-drop functionality
 */

class UploadManager {
    constructor() {
        this.dropZone = null;
        this.fileInput = null;
        this.maxFileSize = 10 * 1024 * 1024; // 10MB
        this.allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
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

    processFile(file) {
        // Validate file type
        if (!this.allowedTypes.includes(file.type)) {
            alert('Please upload an image file (JPG, PNG, WEBP)');
            return;
        }

        // Validate file size
        if (file.size > this.maxFileSize) {
            alert('File size exceeds 10MB limit');
            return;
        }

        console.log('File selected:', file.name);
        console.log('File size:', (file.size / 1024 / 1024).toFixed(2) + 'MB');
        
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
