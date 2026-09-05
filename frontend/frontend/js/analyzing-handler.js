// Analyzing Handler for Analyzing Screen

class AnalyzingHandler {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.progressSteps = [];
        this.progressBar = null;
    }

    initialize() {
        // Get UI elements
        this.progressSteps = Array.from(document.querySelectorAll('main .flex.items-center.gap-4'));
        this.progressBar = document.querySelector('.bg-dark-graphite.h-full.rounded-full');

        // Get uploaded image from session storage
        const imageBase64 = sessionStorage.getItem('uploadedImage');
        if (!imageBase64) {
            // No image found, redirect back to upload
            window.location.href = '../facemetric_upload_photo_updated_mint/code.html';
            return;
        }

        // Display uploaded image
        this.displayImage(imageBase64);

        // Start analysis
        this.startAnalysis(imageBase64);
    }

    displayImage(imageBase64) {
        const imgElement = document.querySelector('main img');
        if (imgElement) {
            imgElement.src = imageBase64;
        }
    }

    async startAnalysis(imageBase64) {
        try {
            // Update progress: step 3 (analyzing)
            this.updateProgress(2, 'active'); // 0-indexed
            this.updateProgressBar(60);

            // Call API
            const result = await this.apiClient.analyzeFace(imageBase64, true);

            // Update progress: step 4 (calculating)
            this.updateProgress(3, 'completed');
            this.updateProgress(3, 'active');
            this.updateProgressBar(80);

            // Simulate brief delay for smooth transition
            await this.delay(500);

            // Update progress: step 5 (finding matches)
            this.updateProgress(3, 'completed');
            this.updateProgress(4, 'active');
            this.updateProgressBar(100);

            // Simulate brief delay
            await this.delay(500);

            // Complete all steps
            this.updateProgress(4, 'completed');

            // Store results in session storage
            sessionStorage.setItem('analysisResults', JSON.stringify(result.results));

            // Navigate to results screen
            window.location.href = '../facemetric_analysis_results_dark_brand_theme/code.html';

        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(error);
        }
    }

    updateProgress(stepIndex, state) {
        // state: 'completed', 'active', 'pending'
        if (!this.progressSteps[stepIndex]) return;

        const step = this.progressSteps[stepIndex];
        const icon = step.querySelector('.material-symbols-outlined');
        const text = step.querySelector('span:not(.material-symbols-outlined)');

        if (state === 'completed') {
            icon.textContent = 'check_circle';
            icon.classList.remove('animate-spin');
            step.classList.remove('opacity-50');
            step.classList.add('opacity-100');
            if (text) text.classList.remove('font-bold');
        } else if (state === 'active') {
            icon.textContent = 'sync';
            icon.classList.add('animate-spin');
            step.classList.remove('opacity-50');
            step.classList.add('opacity-100');
            if (text) text.classList.add('font-bold');
        } else {
            icon.textContent = 'radio_button_unchecked';
            icon.classList.remove('animate-spin');
            step.classList.add('opacity-50');
        }
    }

    updateProgressBar(percentage) {
        if (this.progressBar) {
            this.progressBar.style.width = `${percentage}%`;
        }
    }

    showError(error) {
        // Replace analyzing UI with error message
        const mainContent = document.querySelector('main');
        if (!mainContent) return;

        mainContent.innerHTML = `
            <div class="w-full max-w-[500px] bg-sky-mint rounded-xl p-card-padding flex flex-col gap-6 shadow-[0_0_8px_rgba(184,247,228,0.1)] border border-dark-graphite/15">
                <div class="flex items-center gap-4 text-dark-graphite">
                    <span class="material-symbols-outlined text-[48px]">error</span>
                    <div>
                        <h2 class="font-headline-md text-headline-md mb-2">Analysis Failed</h2>
                        <p class="font-body-md text-body-md">${error.message || 'An unexpected error occurred.'}</p>
                    </div>
                </div>
                ${error.retry ? `
                    <button onclick="window.location.href='../facemetric_upload_photo_updated_mint/code.html'" 
                            class="bg-dark-graphite text-sky-mint px-6 py-3 rounded font-semibold hover:opacity-90 transition-opacity">
                        Try Again
                    </button>
                ` : ''}
            </div>
        `;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const analyzingHandler = new AnalyzingHandler(apiClient);
    analyzingHandler.initialize();
});
