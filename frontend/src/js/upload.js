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
        
        // TODO: Navigate to analyzing page or process file
        alert('File selected: ' + file.name + '\n\nNext: Navigate to analyzing page');
    }

    openCamera() {
        // TODO: Implement camera capture functionality
        alert('Camera functionality will open the device camera for photo capture');
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
