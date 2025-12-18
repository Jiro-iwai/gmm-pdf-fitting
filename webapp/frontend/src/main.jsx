import React from 'react'
import ReactDOM from 'react-dom/client'
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material'
import App from './App.jsx'
import './index.css'

// Get theme preference from localStorage or default to light mode
const getInitialThemeMode = () => {
  const savedMode = localStorage.getItem('themeMode')
  return savedMode === 'dark' ? 'dark' : 'light'
}

// Create theme function
const createAppTheme = (mode) => {
  return createTheme({
    palette: {
      mode,
      ...(mode === 'dark'
        ? {
            // Dark mode palette
            background: {
              default: '#121212',
              paper: '#1e1e1e',
            },
            text: {
              primary: '#ffffff',
              secondary: 'rgba(255, 255, 255, 0.7)',
            },
          }
        : {
            // Light mode palette
            background: {
              default: '#f5f5f5',
              paper: '#ffffff',
            },
            text: {
              primary: 'rgba(0, 0, 0, 0.87)',
              secondary: 'rgba(0, 0, 0, 0.6)',
            },
          }),
    },
  })
}

// Initialize theme mode
const initialMode = getInitialThemeMode()
const theme = createAppTheme(initialMode)

// Create a wrapper component to manage theme state
function AppWithTheme() {
  const [mode, setMode] = React.useState(initialMode)
  const [currentTheme, setCurrentTheme] = React.useState(theme)

  const toggleColorMode = () => {
    const newMode = mode === 'light' ? 'dark' : 'light'
    setMode(newMode)
    setCurrentTheme(createAppTheme(newMode))
    localStorage.setItem('themeMode', newMode)
  }

  // Pass toggleColorMode to App via context or props
  // For simplicity, we'll use a custom event or pass it through props
  // Actually, let's use React Context for this
  return (
    <ThemeProvider theme={currentTheme}>
      <CssBaseline />
      <App toggleColorMode={toggleColorMode} mode={mode} />
    </ThemeProvider>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppWithTheme />
  </React.StrictMode>,
)

