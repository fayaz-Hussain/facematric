// API Configuration
const API_BASE_URL = 'http://localhost:8000'; // Change to production URL when deployed

// API Client Class
class FaceMetricAPI {
    constructor(baseURL = API_BASE_URL) {
        this.baseURL = baseURL;
    }

    /**
     * Analyze face image
     * @param {string} imageBase64 - Base64-encoded image string (with or without data URL prefix)
     * @param {boolean} includeCelebrity - Whether to perform celebrity matching
     * @returns {Promise<Object>} Analysis results
     */
    async analyzeFace(imageBase64, includeCelebrity = true) {
        // Remove data URL prefix if present
        const base64Data = imageBase64.replace(/^data:image\/[a-z]+;base64,/, '');

        try {
            const response = await fetch(`${this.baseURL}/api/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_base64: base64Data,
                    include_celebrity_match: includeCelebrity
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new APIError(
                    data.error_code,
                    data.message,
                    data.retry,
                    response.status
                );
            }

            return data;
        } catch (error) {
            if (error instanceof APIError) {
                throw error;
            }
            // Network error or other unexpected error
            throw new APIError(
                'NETWORK_ERROR',
                'Unable to connect to server. Please check your internet connection.',
                true,
                0
            );
        }
    }
}

// Custom Error Class
class APIError extends Error {
    constructor(errorCode, message, retry, statusCode) {
        super(message);
        this.name = 'APIError';
        this.errorCode = errorCode;
        this.retry = retry;
        this.statusCode = statusCode;
    }
}

// Export singleton instance
const apiClient = new FaceMetricAPI();
