// Theme Manager - Dark/Light Mode Toggle for All Pages

class ThemeManager {
    constructor() {
        this.currentTheme = this.getSavedTheme() || 'dark';
        this.init();
    }

    init() {
        // Apply saved theme on load
        this.applyTheme(this.currentTheme);
        
        // Attach toggle listeners
        this.attachToggleListeners();
    }

    getSavedTheme() {
        return localStorage.getItem('facemetric-theme');
    }

    saveTheme(theme) {
        localStorage.setItem('facemetric-theme', theme);
    }

    applyTheme(theme) {
        const html = document.documentElement;
        
        if (theme === 'dark') {
            html.classList.add('dark');
            html.classList.remove('light');
        } else {
            html.classList.add('light');
            html.classList.remove('dark');
        }
        
        this.currentTheme = theme;
        this.saveTheme(theme);
        this.updateToggleIcon();
    }

    toggleTheme() {
        const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.applyTheme(newTheme);
    }

    updateToggleIcon() {
        const toggleButtons = document.querySelectorAll('[aria-label="Toggle Dark Mode"], .theme-toggle');
        toggleButtons.forEach(btn => {
            const icon = btn.querySelector('.material-symbols-outlined');
            if (icon) {
                icon.textContent = this.currentTheme === 'dark' ? 'light_mode' : 'dark_mode';
                icon.setAttribute('data-icon', this.currentTheme === 'dark' ? 'light_mode' : 'dark_mode');
            }
        });
    }

    attachToggleListeners() {
        // Find all theme toggle buttons
        const toggleButtons = document.querySelectorAll('[aria-label="Toggle Dark Mode"], .theme-toggle, button:has([data-icon="dark_mode"]), button:has([data-icon="light_mode"])');
        
        toggleButtons.forEach(btn => {
            // Remove existing listeners
            const newBtn = btn.cloneNode(true);
            btn.parentNode.replaceChild(newBtn, btn);
            
            // Add new listener
            newBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.toggleTheme();
            });
            
            // Mark as theme toggle
            newBtn.classList.add('theme-toggle');
        });
    }
}

// Initialize theme manager when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.themeManager = new ThemeManager();
});
