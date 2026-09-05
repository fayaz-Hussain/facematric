/**
 * theme.js
 * Handles theme switching between light and dark modes
 */

class ThemeManager {
    constructor() {
        this.THEME_KEY = 'theme';
        this.DARK_THEME = 'dark';
        this.LIGHT_THEME = '';
        this.init();
    }

    init() {
        this.loadSavedTheme();
    }

    toggleTheme() {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === this.DARK_THEME ? this.LIGHT_THEME : this.DARK_THEME;
        
        html.setAttribute('data-theme', newTheme);
        this.saveTheme(newTheme);
    }

    saveTheme(theme) {
        localStorage.setItem(this.THEME_KEY, theme);
    }

    loadSavedTheme() {
        const savedTheme = localStorage.getItem(this.THEME_KEY);
        if (savedTheme) {
            document.documentElement.setAttribute('data-theme', savedTheme);
        }
    }

    getCurrentTheme() {
        return document.documentElement.getAttribute('data-theme') || this.LIGHT_THEME;
    }
}

// Initialize theme manager
const themeManager = new ThemeManager();

// Export for use in HTML
function toggleTheme() {
    themeManager.toggleTheme();
}
