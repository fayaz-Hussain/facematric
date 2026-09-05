import React from 'react'
import ReactDOM from 'react-dom/client'
// import { Agentation } from 'agentation'  // Temporarily commented out
import './frontend/src/css/variables.css'
import './frontend/src/css/base.css'
import './frontend/src/css/header.css'
import './frontend/src/css/theme-toggle.css'
import './frontend/src/css/main-content.css'
import './frontend/src/css/upload-section.css'
import './frontend/src/css/buttons.css'
import './frontend/src/css/footer.css'

console.log('main.jsx loaded!')

// Simple test component
function TestApp() {
  return <div><h1>React is working!</h1></div>
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <TestApp />
    {/* <Agentation /> */}
  </React.StrictMode>,
)
