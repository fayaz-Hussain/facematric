/**
 * dom-utils.js
 * Shared DOM utility functions used across all pages
 */

/**
 * Shared initialization pattern - waits for DOM ready before setup
 * @param {Function} setupCallback - Function to call when DOM is ready
 */
export function initializeWhenReady(setupCallback) {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupCallback);
    } else {
        setupCallback();
    }
}

/**
 * Load HTML component from external file
 * @param {string} componentPath - Path to the HTML component file
 * @param {string} targetSelector - CSS selector for where to inject component
 * @returns {Promise<void>}
 */
export async function loadComponent(componentPath, targetSelector) {
    try {
        const response = await fetch(componentPath);
        if (!response.ok) {
            throw new Error(`Failed to load component: ${componentPath}`);
        }
        const html = await response.text();
        const target = document.querySelector(targetSelector);
        if (target) {
            target.innerHTML = html;
        }
    } catch (error) {
        console.error('Error loading component:', error);
    }
}

/**
 * Show toast notification (replaces alert())
 * @param {string} message - Message to display
 * @param {string} type - 'success', 'error', 'warning', 'info'
 * @param {number} duration - Duration in ms (default: 3000)
 */
export function showToast(message, type = 'info', duration = 3000) {
    // Remove existing toasts
    const existing = document.querySelector('.toast-notification');
    if (existing) {
        existing.remove();
    }

    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast-notification toast-${type}`;
    toast.textContent = message;
    
    // Add styles
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : type === 'warning' ? '#f59e0b' : 'var(--accent-teal)'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 9999;
        animation: slideIn 0.3s ease-out;
        font-family: var(--font-primary);
        font-size: 0.9rem;
        max-width: 400px;
    `;

    document.body.appendChild(toast);

    // Auto remove after duration
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/**
 * Add toast animation styles to document
 */
function addToastStyles() {
    if (!document.querySelector('#toast-styles')) {
        const style = document.createElement('style');
        style.id = 'toast-styles';
        style.textContent = `
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(100%);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
}

// Initialize toast styles
initializeWhenReady(addToastStyles);
