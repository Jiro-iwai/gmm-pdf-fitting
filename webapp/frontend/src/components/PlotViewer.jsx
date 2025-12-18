import React, { useState, useMemo, useEffect } from 'react'
import Plot from 'react-plotly.js'
import {
  Box,
  Typography,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Button,
  Divider,
  useTheme,
} from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import SettingsIcon from '@mui/icons-material/Settings'

// Component to render line style preview
const LineStylePreview = ({ style, width = 60, height = 20 }) => {
  const theme = useTheme()
  const strokeColor = theme.palette.mode === 'dark' ? '#ffffff' : '#000000'
  const getDashArray = (style) => {
    switch (style) {
      case 'solid':
        return 'none'
      case 'dash':
        return '5,5'
      case 'dot':
        return '2,2'
      case 'dashdot':
        return '5,2,2,2'
      case 'longdash':
        return '10,5'
      case 'longdashdot':
        return '10,5,2,5'
      default:
        return 'none'
    }
  }

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
      <svg width={width} height={height} style={{ overflow: 'visible' }}>
        <line
          x1="0"
          y1={height / 2}
          x2={width}
          y2={height / 2}
          stroke={strokeColor}
          strokeWidth="2"
          strokeDasharray={getDashArray(style)}
        />
      </svg>
      <Typography variant="body2" sx={{ flex: 1 }}>
        {style === 'solid' ? 'Solid' :
         style === 'dash' ? 'Dash' :
         style === 'dot' ? 'Dot' :
         style === 'dashdot' ? 'Dash Dot' :
         style === 'longdash' ? 'Long Dash' :
         style === 'longdashdot' ? 'Long Dash Dot' : style}
      </Typography>
    </Box>
  )
}

const PlotViewer = ({ result, plotSettings: externalPlotSettings, setPlotSettings: setExternalPlotSettings }) => {
  const theme = useTheme()
  
  // Use external state if provided, otherwise use local state
  const [localPlotSettings, setLocalPlotSettings] = useState({
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
  })
  
  // Colors for GMM components (cycle through these)
  const componentColors = [
    '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2',
    '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
  ]
  
  const plotSettings = externalPlotSettings || localPlotSettings
  const setPlotSettings = setExternalPlotSettings || setLocalPlotSettings
  const scaleMode = plotSettings.scaleMode || 'linear'
  const setScaleMode = (value) => {
    setPlotSettings(prev => ({ ...prev, scaleMode: value }))
  }
  
  // Track revision for Plotly updates - increment when range settings change
  const [revision, setRevision] = useState(0)
  
  // Increment revision when range settings change to force Plotly update
  useEffect(() => {
    setRevision(prev => prev + 1)
  }, [
    plotSettings.xRangeMin, 
    plotSettings.xRangeMax,
    plotSettings.yRangeLinearMin,
    plotSettings.yRangeLinearMax,
    plotSettings.yRangeLogMin,
    plotSettings.yRangeLogMax,
    scaleMode
  ])

  if (!result || !result.z || !result.f_true || !result.f_hat) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="text.secondary">Plot data not available</Typography>
      </Box>
    )
  }

  const { z, f_true, f_hat } = result
  const zArray = Array.isArray(z) ? z : []
  const fTrueArray = Array.isArray(f_true) ? f_true : []
  const fHatArray = Array.isArray(f_hat) ? f_hat : []

  // Validate data arrays
  if (zArray.length === 0 || fTrueArray.length === 0 || fHatArray.length === 0) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="error">
          Invalid data: z={zArray.length}, f_true={fTrueArray.length}, f_hat={fHatArray.length}
        </Typography>
      </Box>
    )
  }

  // Prepare grid points for display (memoized to recalculate when settings change)
  const { zDisplay, fTrueDisplay } = useMemo(() => {
    const maxGridPoints = 200
    if (plotSettings.showGridPoints && zArray.length > maxGridPoints) {
      const step = Math.ceil(zArray.length / maxGridPoints)
      return {
        zDisplay: zArray.filter((_, i) => i % step === 0),
        fTrueDisplay: fTrueArray.filter((_, i) => i % step === 0),
      }
    }
    return {
      zDisplay: zArray,
      fTrueDisplay: fTrueArray,
    }
  }, [zArray, fTrueArray, plotSettings.showGridPoints])

  // Get GMM components from result
  const gmmComponents = result?.gmm_components || []
  
  // Helper function to compute Gaussian PDF
  const gaussianPdf = (x, mu, sigma) => {
    const coefficient = 1 / (sigma * Math.sqrt(2 * Math.PI))
    const exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coefficient * Math.exp(exponent)
  }

  // Prepare plot data based on scale mode
  const plotData = useMemo(() => {
    const MIN_PDF_VALUE = 1e-10
    
    // For log scale, ensure values are positive
    const fTrue = scaleMode === 'log' 
      ? fTrueArray.map(v => Math.max(v, MIN_PDF_VALUE))
      : fTrueArray
    const fHat = scaleMode === 'log'
      ? fHatArray.map(v => Math.max(v, MIN_PDF_VALUE))
      : fHatArray
    const fTrueDisplayProcessed = scaleMode === 'log'
      ? fTrueDisplay.map(v => Math.max(v, MIN_PDF_VALUE))
      : fTrueDisplay

    const truePdfLineStyle = plotSettings.truePdfLineStyle === 'solid' ? undefined : plotSettings.truePdfLineStyle
    const gmmLineStyle = plotSettings.gmmLineStyle === 'solid' ? undefined : plotSettings.gmmLineStyle

    const data = [
      {
        x: zArray,
        y: fTrue,
        type: 'scatter',
        mode: 'lines',
        name: 'True PDF',
        line: { 
          color: plotSettings.truePdfColor, 
          width: plotSettings.lineWidth,
          dash: truePdfLineStyle,
        },
      },
      {
        x: zArray,
        y: fHat,
        type: 'scatter',
        mode: 'lines',
        name: 'GMM approximation',
        line: { 
          color: plotSettings.gmmColor, 
          width: plotSettings.lineWidth,
          dash: gmmLineStyle,
        },
      },
    ]

    // Add GMM component curves if enabled
    if (plotSettings.showGmmComponents && gmmComponents.length > 0) {
      gmmComponents.forEach((comp, idx) => {
        const { pi, mu, sigma } = comp
        if (typeof pi === 'number' && typeof mu === 'number' && typeof sigma === 'number' && sigma > 0) {
          // Calculate weighted Gaussian PDF for this component
          const componentPdf = zArray.map(z => {
            const pdf = pi * gaussianPdf(z, mu, sigma)
            return scaleMode === 'log' ? Math.max(pdf, MIN_PDF_VALUE) : pdf
          })
          
          data.push({
            x: zArray,
            y: componentPdf,
            type: 'scatter',
            mode: 'lines',
            name: `Component ${idx + 1} (π=${pi.toFixed(3)})`,
            line: {
              color: componentColors[idx % componentColors.length],
              width: Math.max(1, plotSettings.lineWidth - 0.5),
              dash: 'dot',
            },
            opacity: 0.7,
          })
        }
      })
    }

    if (plotSettings.showGridPoints) {
      data.push({
        x: zDisplay,
        y: fTrueDisplayProcessed,
        type: 'scatter',
        mode: 'markers',
        name: `Grid points (n=${zArray.length})`,
        marker: {
          color: plotSettings.gridColor,
          size: plotSettings.gridPointSize,
          opacity: 0.6,
        },
      })
    }

    return data
  }, [zArray, fTrueArray, fHatArray, zDisplay, fTrueDisplay, plotSettings, scaleMode, gmmComponents, componentColors])

  const plotLayout = useMemo(() => {
    const isDark = theme.palette.mode === 'dark'
    const textColor = isDark ? '#ffffff' : '#000000'
    const bgColor = isDark ? '#1e1e1e' : '#ffffff'
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
    
    const layout = {
      title: {
        text: `PDF Comparison (${scaleMode === 'log' ? 'Log' : 'Linear'} Scale)`,
        font: { color: textColor },
      },
      xaxis: {
        title: 'z',
        titlefont: { color: textColor },
        tickfont: { color: textColor },
        gridcolor: gridColor,
        linecolor: textColor,
      },
      yaxis: {
        title: scaleMode === 'log' ? 'Probability Density (log scale)' : 'Probability Density',
        type: scaleMode === 'log' ? 'log' : 'linear',
        titlefont: { color: textColor },
        tickfont: { color: textColor },
        gridcolor: gridColor,
        linecolor: textColor,
      },
      hovermode: 'closest',
      showlegend: true,
      legend: { 
        x: 0.02, 
        y: 0.98,
        font: { color: textColor },
        bgcolor: 'transparent',
      },
      margin: { l: 60, r: 20, t: 40, b: 50 },
      height: 500,
      paper_bgcolor: bgColor,
      plot_bgcolor: bgColor,
    }
    
    // Set X axis range
    const xMin = plotSettings.xRangeMin !== null && plotSettings.xRangeMin !== undefined && plotSettings.xRangeMin !== '' 
      ? Number(plotSettings.xRangeMin) 
      : null
    const xMax = plotSettings.xRangeMax !== null && plotSettings.xRangeMax !== undefined && plotSettings.xRangeMax !== '' 
      ? Number(plotSettings.xRangeMax) 
      : null
    if (xMin !== null && xMax !== null && !isNaN(xMin) && !isNaN(xMax) && isFinite(xMin) && isFinite(xMax) && xMin < xMax) {
      layout.xaxis.range = [xMin, xMax]
    }
    
    // Set Y axis range
    const yMin = scaleMode === 'log' 
      ? (plotSettings.yRangeLogMin !== null && plotSettings.yRangeLogMin !== undefined && plotSettings.yRangeLogMin !== '' 
          ? Number(plotSettings.yRangeLogMin) 
          : null)
      : (plotSettings.yRangeLinearMin !== null && plotSettings.yRangeLinearMin !== undefined && plotSettings.yRangeLinearMin !== '' 
          ? Number(plotSettings.yRangeLinearMin) 
          : null)
    const yMax = scaleMode === 'log'
      ? (plotSettings.yRangeLogMax !== null && plotSettings.yRangeLogMax !== undefined && plotSettings.yRangeLogMax !== '' 
          ? Number(plotSettings.yRangeLogMax) 
          : null)
      : (plotSettings.yRangeLinearMax !== null && plotSettings.yRangeLinearMax !== undefined && plotSettings.yRangeLinearMax !== '' 
          ? Number(plotSettings.yRangeLinearMax) 
          : null)
    if (yMin !== null && yMax !== null && !isNaN(yMin) && !isNaN(yMax) && isFinite(yMin) && isFinite(yMax) && yMin < yMax) {
      layout.yaxis.range = [yMin, yMax]
    }
    
    return layout
  }, [
    plotSettings.xRangeMin,
    plotSettings.xRangeMax,
    plotSettings.yRangeLinearMin,
    plotSettings.yRangeLinearMax,
    plotSettings.yRangeLogMin,
    plotSettings.yRangeLogMax,
    scaleMode,
    theme.palette.mode,
  ])


  const handleSettingChange = (field) => (event) => {
    const value = event.target.value
    setPlotSettings((prev) => ({
      ...prev,
      [field]: value === '' ? null : value,
    }))
  }
  
  const handleRangeChange = (field) => (event) => {
    const value = event.target.value
    const numValue = value === '' ? null : (isNaN(Number(value)) ? null : Number(value))
    setPlotSettings((prev) => ({
      ...prev,
      [field]: numValue,
    }))
  }

  const resetSettings = () => {
    setPlotSettings({
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
    })
  }

  // Get title from result (if available from request context, otherwise use default)
  const title = 'PDF Comparison'

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Scale Mode</InputLabel>
          <Select
            value={scaleMode}
            onChange={(e) => setScaleMode(e.target.value)}
            label="Scale Mode"
          >
            <MenuItem value="linear">Linear Scale</MenuItem>
            <MenuItem value="log">Log Scale</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SettingsIcon />
            <Typography>Plot Settings</Typography>
          </Box>
        </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  X Axis Range
                </Typography>
              </Grid>
              <Grid item xs={6} md={3}>
                <TextField
                  fullWidth
                  label="Min"
                  type="number"
                  value={plotSettings.xRangeMin !== null && plotSettings.xRangeMin !== undefined ? plotSettings.xRangeMin : ''}
                  onChange={handleRangeChange('xRangeMin')}
                  placeholder="Auto"
                  size="small"
                  inputProps={{ step: 'any' }}
                />
              </Grid>
              <Grid item xs={6} md={3}>
                <TextField
                  fullWidth
                  label="Max"
                  type="number"
                  value={plotSettings.xRangeMax !== null && plotSettings.xRangeMax !== undefined ? plotSettings.xRangeMax : ''}
                  onChange={handleRangeChange('xRangeMax')}
                  placeholder="Auto"
                  size="small"
                  inputProps={{ step: 'any' }}
                />
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                  Y Axis Range ({scaleMode === 'log' ? 'Log' : 'Linear'} Scale)
                </Typography>
              </Grid>
              <Grid item xs={6} md={3}>
                <TextField
                  fullWidth
                  label="Min"
                  type="number"
                  value={
                    scaleMode === 'log' 
                      ? (plotSettings.yRangeLogMin !== null && plotSettings.yRangeLogMin !== undefined ? plotSettings.yRangeLogMin : '')
                      : (plotSettings.yRangeLinearMin !== null && plotSettings.yRangeLinearMin !== undefined ? plotSettings.yRangeLinearMin : '')
                  }
                  onChange={handleRangeChange(scaleMode === 'log' ? 'yRangeLogMin' : 'yRangeLinearMin')}
                  placeholder="Auto"
                  size="small"
                  inputProps={{ step: 'any' }}
                />
              </Grid>
              <Grid item xs={6} md={3}>
                <TextField
                  fullWidth
                  label="Max"
                  type="number"
                  value={
                    scaleMode === 'log'
                      ? (plotSettings.yRangeLogMax !== null && plotSettings.yRangeLogMax !== undefined ? plotSettings.yRangeLogMax : '')
                      : (plotSettings.yRangeLinearMax !== null && plotSettings.yRangeLinearMax !== undefined ? plotSettings.yRangeLinearMax : '')
                  }
                  onChange={handleRangeChange(scaleMode === 'log' ? 'yRangeLogMax' : 'yRangeLinearMax')}
                  placeholder="Auto"
                  size="small"
                  inputProps={{ step: 'any' }}
                />
              </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth size="small">
                <InputLabel>Show Grid Points</InputLabel>
                <Select
                  value={plotSettings.showGridPoints ? 'true' : 'false'}
                  onChange={(e) => setPlotSettings(prev => ({ ...prev, showGridPoints: e.target.value === 'true' }))}
                  label="Show Grid Points"
                >
                  <MenuItem value="true">Yes</MenuItem>
                  <MenuItem value="false">No</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth size="small">
                <InputLabel>Show GMM Components</InputLabel>
                <Select
                  value={plotSettings.showGmmComponents ? 'true' : 'false'}
                  onChange={(e) => setPlotSettings(prev => ({ ...prev, showGmmComponents: e.target.value === 'true' }))}
                  label="Show GMM Components"
                >
                  <MenuItem value="true">Yes</MenuItem>
                  <MenuItem value="false">No</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="True PDF Color"
                type="color"
                value={plotSettings.truePdfColor}
                onChange={handleSettingChange('truePdfColor')}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="GMM Color"
                type="color"
                value={plotSettings.gmmColor}
                onChange={handleSettingChange('gmmColor')}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Grid Points Color"
                type="color"
                value={plotSettings.gridColor}
                onChange={handleSettingChange('gridColor')}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Line Width"
                type="number"
                value={plotSettings.lineWidth}
                onChange={(e) => setPlotSettings(prev => ({ ...prev, lineWidth: parseFloat(e.target.value) || 2 }))}
                inputProps={{ min: 0.5, max: 10, step: 0.5 }}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Grid Point Size"
                type="number"
                value={plotSettings.gridPointSize}
                onChange={(e) => setPlotSettings(prev => ({ ...prev, gridPointSize: parseFloat(e.target.value) || 5 }))}
                inputProps={{ min: 1, max: 20, step: 1 }}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth size="small">
                <InputLabel>True PDF Line Style</InputLabel>
                <Select
                  value={plotSettings.truePdfLineStyle || 'solid'}
                  onChange={(e) => setPlotSettings(prev => ({ ...prev, truePdfLineStyle: e.target.value }))}
                  label="True PDF Line Style"
                  renderValue={(value) => (
                    <LineStylePreview style={value || 'solid'} width={80} />
                  )}
                >
                  <MenuItem value="solid">
                    <LineStylePreview style="solid" />
                  </MenuItem>
                  <MenuItem value="dash">
                    <LineStylePreview style="dash" />
                  </MenuItem>
                  <MenuItem value="dot">
                    <LineStylePreview style="dot" />
                  </MenuItem>
                  <MenuItem value="dashdot">
                    <LineStylePreview style="dashdot" />
                  </MenuItem>
                  <MenuItem value="longdash">
                    <LineStylePreview style="longdash" />
                  </MenuItem>
                  <MenuItem value="longdashdot">
                    <LineStylePreview style="longdashdot" />
                  </MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth size="small">
                <InputLabel>GMM Line Style</InputLabel>
                <Select
                  value={plotSettings.gmmLineStyle || 'dash'}
                  onChange={(e) => setPlotSettings(prev => ({ ...prev, gmmLineStyle: e.target.value }))}
                  label="GMM Line Style"
                  renderValue={(value) => (
                    <LineStylePreview style={value || 'dash'} width={80} />
                  )}
                >
                  <MenuItem value="solid">
                    <LineStylePreview style="solid" />
                  </MenuItem>
                  <MenuItem value="dash">
                    <LineStylePreview style="dash" />
                  </MenuItem>
                  <MenuItem value="dot">
                    <LineStylePreview style="dot" />
                  </MenuItem>
                  <MenuItem value="dashdot">
                    <LineStylePreview style="dashdot" />
                  </MenuItem>
                  <MenuItem value="longdash">
                    <LineStylePreview style="longdash" />
                  </MenuItem>
                  <MenuItem value="longdashdot">
                    <LineStylePreview style="longdashdot" />
                  </MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <Button variant="outlined" onClick={resetSettings} size="small">
                Reset to Defaults
              </Button>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      <Divider sx={{ my: 2 }} />

      <Box sx={{ minHeight: 500, width: '100%' }}>
        {plotData && plotData.length > 0 ? (
          <Plot
            key={`plot-${revision}`}
            data={plotData}
            layout={plotLayout}
            config={{ 
              responsive: true, 
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['select2d', 'lasso2d']
            }}
            style={{ width: '100%', minHeight: 500 }}
            useResizeHandler={true}
          />
        ) : (
          <Typography color="error">Failed to prepare plot data</Typography>
        )}
      </Box>
    </Box>
  )
}

export default PlotViewer
