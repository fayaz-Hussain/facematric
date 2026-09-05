# FaceMetric - Frontend

A modern web application for facial symmetry and golden ratio analysis using AI diagnostics.

## 📁 Project Structure

```
frontend/
├── index.html              # Main landing page
├── src/
│   ├── css/                # Stylesheets (modular)
│   │   ├── variables.css   # CSS custom properties & theme colors
│   │   ├── base.css        # Base styles & resets
│   │   ├── header.css      # Header & logo styles
│   │   ├── theme-toggle.css # Dark/light mode toggle
│   │   ├── main-content.css # Main content area styles
│   │   ├── upload-section.css # File upload area
│   │   ├── buttons.css     # Button styles
│   │   └── footer.css      # Footer styles
│   ├── js/                 # JavaScript modules
│   │   ├── theme.js        # Theme management (light/dark mode)
│   │   └── upload.js       # File upload & drag-and-drop
│   └── assets/             # Static assets
│       ├── images/         # Image files
│       └── logo/           # Logo files
│           └── logo.png    # FaceMetric logo
├── public/                 # Public static files (if needed)
└── README.md              # This file
```

## 🚀 Features

- **Responsive Design** - Works on desktop, tablet, and mobile
- **Dark/Light Mode** - Automatic theme switching with localStorage persistence
- **Drag & Drop Upload** - Easy file upload with drag-and-drop support
- **File Validation** - Client-side validation for file type and size
- **Modular CSS** - Separated concerns for easy maintenance
- **Clean JavaScript** - Object-oriented approach with ES6 classes
- **Accessible** - ARIA labels and semantic HTML

## 🎨 Design System

### Colors

**Light Mode:**
- Background: Mint gradient (#e8f5f1 to #d4f0ea)
- Primary Text: Dark (#1a1a1a)
- Accent: Teal (#40cca2)

**Dark Mode:**
- Background: Dark gradient (#0a0e12 to #1a2430)
- Primary Text: White (#ffffff)
- Accent: Teal (#40cca2) with glow effect

### Typography

- Headings: Georgia, Times New Roman (serif)
- Body: System fonts (San Francisco, Segoe UI, Roboto)

## 🛠️ Development

### Prerequisites

- Modern web browser (Chrome, Firefox, Safari, Edge)
- Local web server (Python HTTP server, Live Server, etc.)

### Running Locally

#### Option 1: Python HTTP Server
```bash
# Navigate to the frontend folder
cd frontend

# Start server on port 3000
python -m http.server 3000

# Open browser to http://localhost:3000
```

#### Option 2: Live Server (VS Code Extension)
1. Install "Live Server" extension in VS Code
2. Right-click index.html
3. Select "Open with Live Server"

### File Organization

**CSS Modules:**
- Each CSS file handles a specific component or feature
- Variables are centralized in ariables.css
- Import order matters (variables → base → components)

**JavaScript Modules:**
- 	heme.js - Handles theme switching and localStorage
- upload.js - Manages file upload, validation, and drag-and-drop

## 📝 Code Style

### CSS
- Use CSS custom properties for theme values
- BEM-like naming convention for classes
- Mobile-first responsive design
- Smooth transitions (0.3s ease)

### JavaScript
- ES6 classes for organization
- Clear method names and comments
- Error handling and validation
- No global variables (except exported functions)

## 🔧 Configuration

### File Upload Settings

Located in src/js/upload.js:

```javascript
maxFileSize: 10 * 1024 * 1024,  // 10MB
allowedTypes: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
```

### Theme Settings

Located in src/js/theme.js:

```javascript
THEME_KEY: 'theme',
DARK_THEME: 'dark',
LIGHT_THEME: ''
```

## 🌐 Browser Support

- Chrome/Edge: Latest 2 versions
- Firefox: Latest 2 versions
- Safari: Latest 2 versions
- Mobile browsers: iOS Safari 12+, Chrome Android

## 📦 Dependencies

None! This is a vanilla JavaScript project with no external dependencies.

## 🎯 Future Enhancements

- [ ] Add analyzing page with progress animation
- [ ] Add results page with facial analysis visualization
- [ ] Implement camera capture functionality
- [ ] Add image preview before upload
- [ ] Progressive Web App (PWA) support
- [ ] Multi-language support

## 👨‍💻 Development Guidelines

### Adding New Styles

1. Create a new CSS file in src/css/
2. Add <link> tag in index.html
3. Follow existing naming conventions
4. Use CSS custom properties for colors

### Adding New JavaScript Features

1. Create a new JS file in src/js/
2. Use ES6 class-based structure
3. Export functions needed in HTML
4. Add <script> tag in index.html

### Testing Theme Changes

1. Toggle between light/dark modes
2. Check localStorage persistence
3. Verify all components adapt correctly
4. Test on different screen sizes

## 🐛 Known Issues

None currently. Please report issues on the project repository.

## 📄 License

[Your License Here]

## 🤝 Contributing

Contributions are welcome! Please follow the code style guidelines and test thoroughly before submitting.

---

Built with ❤️ for facial geometry analysis
