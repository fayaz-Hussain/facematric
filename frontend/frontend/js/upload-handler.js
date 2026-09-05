// Upload Handler for Upload Screen

class UploadHandler {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.uploadButton = null;
        this.cameraButton = null;
        this.fileInput = null;
    }

    initialize() {
        // Get buttons - using more flexible selectors
        const buttons = document.querySelectorAll('button');
        buttons.forEach(btn => {
            const iconSpan = btn.querySelector('.material-symbols-outlined');
            if (iconSpan) {
                const iconText = iconSpan.getAttribute('data-icon') || iconSpan.textContent.trim();
                if (iconText === 'upload') {
                    this.uploadButton = btn;
                } else if (iconText === 'photo_camera') {
                    this.cameraButton = btn;
                }
            }
        });

        // Create hidden file input
        this.fileInput = document.createElement('input');
        this.fileInput.type = 'file';
        this.fileInput.accept = 'image/jpeg,image/png,image/webp,image/heic,image/bmp,image/gif';
        this.fileInput.style.display = 'none';
        document.body.appendChild(this.fileInput);

        // Attach event listeners
        if (this.uploadButton) {
            this.uploadButton.addEventListener('click', () => this.handleUploadClick());
        }
        if (this.cameraButton) {
            this.cameraButton.addEventListener('click', () => this.handleCameraClick());
        }
        this.fileInput.addEventListener('change', (e) => this.handleFileSelected(e));

        // Drag and drop support
        const dropZone = document.querySelector('.border-dashed');
        if (dropZone) {
            dropZone.addEventListener('dragover', (e) => this.handleDragOver(e));
            dropZone.addEventListener('drop', (e) => this.handleDrop(e));
            dropZone.addEventListener('click', () => this.handleUploadClick());
        }
    }

    handleUploadClick() {
        this.fileInput.click();
    }

    async handleCameraClick() {
        try {
            // Request camera access
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'user' } 
            });

            // Create video element and capture
            const video = document.createElement('video');
            video.srcObject = stream;
            video.play();

            // Wait for video to be ready
            await new Promise(resolve => {
                video.onloadedmetadata = resolve;
            });

            // Capture frame
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);

            // Stop camera
            stream.getTracks().forEach(track => track.stop());

            // Convert to base64
            const imageBase64 = canvas.toDataURL('image/jpeg', 0.9);

            // Process image
            await this.processImage(imageBase64);

        } catch (error) {
            console.error('Camera error:', error);
            alert('Could not access camera. Please upload an image instead.');
        }
    }

    handleFileSelected(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            alert('File size exceeds 10MB. Please choose a smaller image.');
            return;
        }

        // Read file as base64
        const reader = new FileReader();
        reader.onload = (e) => {
            this.processImage(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    handleDragOver(event) {
        event.preventDefault();
        event.stopPropagation();
    }

    handleDrop(event) {
        event.preventDefault();
        event.stopPropagation();

        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            // Trigger file input with dropped file
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            this.fileInput.files = dataTransfer.files;
            this.handleFileSelected({ target: this.fileInput });
        }
    }

    async processImage(imageBase64) {
        // Store image in session storage
        sessionStorage.setItem('uploadedImage', imageBase64);

        // Navigate to analyzing screen
        window.location.href = '../facemetric_analyzing_unified/code.html';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const uploadHandler = new UploadHandler(apiClient);
    uploadHandler.initialize();
});
