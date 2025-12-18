import React, { useState, useEffect } from 'react'
import { Container, Box, Typography, Paper, IconButton, Tooltip } from '@mui/material'
import Brightness4Icon from '@mui/icons-material/Brightness4'
import Brightness7Icon from '@mui/icons-material/Brightness7'
import ParameterForm from './components/ParameterForm'
import ResultDisplay from './components/ResultDisplay'
import './App.css'

const PLOT_SETTINGS_KEY = 'gmm-fitting-plot-settings'

const defaultPlotSettings = {
  scaleMode: 'linear',
  xRangeMin: null,
  xRangeMax: null,
  yRangeLinearMin: null,
  yRangeLinearMax: null,
  yRangeLogMin: null,
  yRangeLogMax: null,
  showGridPoints: true,
  showGmmComponents: false,
  truePdfColor: '#1f77b4',
  gmmColor: '#d62728',
  gridColor: '#1f77b4',
  lineWidth: 2,
  gridPointSize: 5,
  truePdfLineStyle: 'solid',
  gmmLineStyle: 'dash',
}

// Load saved plot settings from localStorage
const loadSavedPlotSettings = () => {
  try {
    const saved = localStorage.getItem(PLOT_SETTINGS_KEY)
    if (saved) {
      const parsed = JSON.parse(saved)
      return { ...defaultPlotSettings, ...parsed }
    }
  } catch (e) {
    console.warn('Failed to load saved plot settings:', e)
  }
  return defaultPlotSettings
}

function App({ toggleColorMode, mode }) {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  // Plot settings state - persist across result changes and browser sessions
  const [plotSettings, setPlotSettings] = useState(loadSavedPlotSettings)
  
  // Save plot settings to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(PLOT_SETTINGS_KEY, JSON.stringify(plotSettings))
    } catch (e) {
      console.warn('Failed to save plot settings:', e)
    }
  }, [plotSettings])

  const handleCompute = async (params) => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/compute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      })

      if (!response.ok) {
        let errorMessage = 'Computation failed'
        try {
          const errorData = await response.json()
          // Handle different error response formats
          if (typeof errorData === 'string') {
            errorMessage = errorData
          } else if (errorData.detail) {
            // Pydantic validation errors
            if (Array.isArray(errorData.detail)) {
              errorMessage = errorData.detail.map((e) => {
                if (typeof e === 'string') return e
                if (e.msg) return `${e.loc ? e.loc.join('.') + ': ' : ''}${e.msg}`
                return JSON.stringify(e)
              }).join('\n')
            } else if (typeof errorData.detail === 'string') {
              errorMessage = errorData.detail
            } else {
              errorMessage = JSON.stringify(errorData.detail)
            }
          } else if (errorData.message) {
            errorMessage = errorData.message
          } else {
            errorMessage = JSON.stringify(errorData)
          }
        } catch (e) {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`
        }
        throw new Error(errorMessage)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      // Handle error object properly
      let errorMessage = 'An error occurred'
      if (err instanceof Error) {
        errorMessage = err.message
      } else if (typeof err === 'string') {
        errorMessage = err
      } else {
        errorMessage = JSON.stringify(err)
      }
      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
        <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box>
            <Typography variant="h3" component="h1" gutterBottom>
              GMM Fitting
            </Typography>
            <Typography variant="subtitle1" color="text.secondary">
              Approximate PDF of max(X, Y) using Gaussian Mixture Models
            </Typography>
          </Box>
          <Tooltip title={mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}>
            <IconButton onClick={toggleColorMode} color="inherit" sx={{ ml: 2 }}>
              {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
            </IconButton>
          </Tooltip>
        </Box>

      <Box sx={{ display: 'flex', gap: 3, flexDirection: { xs: 'column', lg: 'row' }, height: { lg: 'calc(100vh - 200px)' } }}>
        <Paper sx={{ 
          p: 3, 
          flex: { xs: '1', lg: '0 0 400px' },
          maxHeight: { lg: 'calc(100vh - 200px)' },
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <ParameterForm onSubmit={handleCompute} loading={loading} />
        </Paper>

        <Box sx={{ 
          flex: 1,
          maxHeight: { lg: 'calc(100vh - 200px)' },
          overflowY: 'auto'
        }}>
          {error && (
            <Paper sx={{ p: 3, mb: 3, bgcolor: 'error.light', color: 'error.contrastText' }}>
              <Typography variant="h6">Error</Typography>
              <Typography>{error}</Typography>
            </Paper>
          )}

          <ResultDisplay 
            result={result} 
            plotSettings={plotSettings}
            setPlotSettings={setPlotSettings}
          />

          {!result && !error && !loading && (
            <Paper sx={{ p: 3 }}>
              <Typography variant="body1" color="text.secondary" align="center">
                Configure parameters and click "Compute" to see results
              </Typography>
            </Paper>
          )}
        </Box>
      </Box>
    </Container>
  )
}

export default App

