import React, { useState, useEffect } from 'react'
import {
  Box,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid,
  Switch,
  FormControlLabel,
  Divider,
} from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'

const STORAGE_KEY = 'gmm-fitting-params'

const defaultFormData = {
  // Bivariate normal parameters
  mu_x: 0.0,
  sigma_x: 0.8,
  mu_y: 0.0,
  sigma_y: 1.6,
  rho: 0.9,
  
  // Grid parameters
  z_range_min: -6.0,
  z_range_max: 8.0,
  z_npoints: 2500,
  
  // GMM parameters
  K: 3,
  method: 'em',
  
  // EM parameters
  max_iter: 400,
  tol: 1e-10,
  reg_var: 1e-6,
  n_init: 8,
  seed: 1,
  init: 'quantile',
  use_moment_matching: false,
  qp_mode: 'hard',
  soft_lambda: 1e4,
  mdn_model_path: './ml_init/checkpoints/mdn_init_v1_N64_K5.pt',
  mdn_device: 'auto',
  
  // LP parameters
  L: 5,
  objective_mode: 'pdf',
  solver: 'highs',
  sigma_min_scale: 0.1,
  sigma_max_scale: 3.0,
  pdf_tolerance: null,
  lambda_raw: null,
  objective_form: 'A',
  
  // Hybrid parameters
  dict_J: null,
  dict_L: 10,
  mu_mode: 'quantile',
  tail_focus: 'none',
  tail_alpha: 1.0,
  
  // Display options
  show_grid_points: true,
  max_grid_points_display: 200,
}

// Load saved parameters from localStorage
const loadSavedParams = () => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      const parsed = JSON.parse(saved)
      // Merge with defaults to ensure all fields exist
      return { ...defaultFormData, ...parsed }
    }
  } catch (e) {
    console.warn('Failed to load saved parameters:', e)
  }
  return defaultFormData
}

const ParameterForm = ({ onSubmit, loading }) => {
  // Track which fields are being edited (to allow temporary string values)
  const [editingFields, setEditingFields] = useState({})
  
  const [formData, setFormData] = useState(loadSavedParams)
  
  // Save parameters to localStorage whenever formData changes
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(formData))
    } catch (e) {
      console.warn('Failed to save parameters:', e)
    }
  }, [formData])

  const handleChange = (field) => (event) => {
    const value = event.target.type === 'checkbox' ? event.target.checked : event.target.value
    setFormData((prev) => {
      const prevValue = prev[field]
      if (typeof prevValue === 'number') {
        // Mark field as being edited
        setEditingFields((prevEditing) => ({ ...prevEditing, [field]: true }))
        
        // Allow empty string and minus sign during editing (store as string temporarily)
        if (value === '' || value === '-') {
          return {
            ...prev,
            [field]: value, // Store as string temporarily
          }
        }
        const numValue = parseFloat(value)
        // Update if it's a valid finite number (including 0 and negative numbers)
        if (!isNaN(numValue) && isFinite(numValue)) {
          return {
            ...prev,
            [field]: numValue,
          }
        }
        // If invalid, keep previous value
        return prev
      }
      const newData = {
        ...prev,
        [field]: value,
      }
      // If init is changed to 'mdn', set n_init to 1
      if (field === 'init' && value === 'mdn') {
        newData.n_init = 1
      }
      // If init is changed from 'mdn' to something else, restore n_init to default
      if (field === 'init' && value !== 'mdn' && prev.init === 'mdn') {
        newData.n_init = 8
      }
      return newData
    })
  }

  const handleBlur = (field, defaultValue) => (event) => {
    // Mark field as not being edited
    setEditingFields((prevEditing) => {
      const newEditing = { ...prevEditing }
      delete newEditing[field]
      return newEditing
    })
    
    // Field-specific validation
    const validateValue = (fieldName, value) => {
      if (fieldName === 'tol' || fieldName === 'reg_var' || fieldName === 'soft_lambda' || 
          fieldName === 'sigma_min_scale' || fieldName === 'sigma_max_scale' || 
          fieldName === 'sigma_x' || fieldName === 'sigma_y') {
        // These fields must be > 0
        return value > 0 ? value : defaultValue
      }
      return value
    }
    
    // Convert to number on blur
    setFormData((prev) => {
      const value = prev[field]
      if (typeof value === 'string') {
        const numValue = parseFloat(value)
        if (!isNaN(numValue) && isFinite(numValue)) {
          const validatedValue = validateValue(field, numValue)
          return {
            ...prev,
            [field]: validatedValue,
          }
        } else {
          // Invalid value, use default
          return {
            ...prev,
            [field]: defaultValue,
          }
        }
      } else if (typeof value === 'number') {
        // Validate existing number value
        const validatedValue = validateValue(field, value)
        return {
          ...prev,
          [field]: validatedValue,
        }
      }
      return prev
    })
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    
    // Normalize all numeric fields before submitting
    // This ensures that any temporary string values are converted to numbers
    const defaults = {
      mu_x: 0.0,
      sigma_x: 0.8,
      mu_y: 0.0,
      sigma_y: 1.6,
      rho: 0.9,
      z_range_min: -6.0,
      z_range_max: 8.0,
      z_npoints: 2500,
      K: 3,
      max_iter: 400,
      tol: 1e-10,
      reg_var: 1e-6,
      n_init: 8,
      seed: 1,
      soft_lambda: 1e4,
      L: 5,
      sigma_min_scale: 0.1,
      sigma_max_scale: 3.0,
      max_grid_points_display: 200,
    }
    
    // Normalize formData values directly (don't update state, just use normalized values)
    const normalizedData = { ...formData }
    Object.keys(defaults).forEach((field) => {
      const value = formData[field]
      if (typeof value === 'string') {
        const numValue = parseFloat(value)
        normalizedData[field] = !isNaN(numValue) && isFinite(numValue) ? numValue : defaults[field]
      } else if (typeof value === 'number') {
        normalizedData[field] = isNaN(value) || !isFinite(value) ? defaults[field] : value
      } else if (value === null || value === undefined || value === '') {
        normalizedData[field] = defaults[field]
      }
    })
    
    // Update formData state with normalized values (for display)
    setFormData(normalizedData)
    
    // Also clear editing fields
    setEditingFields({})
    
    // Ensure all numeric values are properly converted to numbers
    const ensureNumber = (val, defaultValue = 0) => {
      // Handle empty string, null, or undefined
      if (val === '' || val === null || val === undefined) {
        return defaultValue
      }
      // If it's already a number, return it
      if (typeof val === 'number') {
        return isNaN(val) || !isFinite(val) ? defaultValue : val
      }
      // If it's a string (shouldn't happen after normalization, but safety check)
      if (typeof val === 'string') {
        const num = parseFloat(val)
        return isNaN(num) || !isFinite(num) ? defaultValue : num
      }
      // Fallback to default value
      return defaultValue
    }
    
    const request = {
      bivariate_params: {
        mu_x: ensureNumber(normalizedData.mu_x, 0.0),
        sigma_x: ensureNumber(normalizedData.sigma_x, 0.8),
        mu_y: ensureNumber(normalizedData.mu_y, 0.0),
        sigma_y: ensureNumber(normalizedData.sigma_y, 1.6),
        rho: ensureNumber(normalizedData.rho, 0.9),
      },
      grid_params: {
        z_range: [ensureNumber(normalizedData.z_range_min, -6.0), ensureNumber(normalizedData.z_range_max, 8.0)],
        z_npoints: ensureNumber(normalizedData.z_npoints, 2500),
      },
      K: ensureNumber(normalizedData.K, 3),
      method: normalizedData.method,
      show_grid_points: normalizedData.show_grid_points,
      max_grid_points_display: ensureNumber(normalizedData.max_grid_points_display, 200),
    }

    // Add method-specific parameters
    if (normalizedData.method === 'em') {
      request.em_params = {
        max_iter: ensureNumber(normalizedData.max_iter, 400),
        tol: ensureNumber(normalizedData.tol, 1e-10),
        reg_var: ensureNumber(normalizedData.reg_var, 1e-6),
        n_init: ensureNumber(normalizedData.n_init, 8),
        seed: ensureNumber(normalizedData.seed, 1),
        init: normalizedData.init,
        use_moment_matching: normalizedData.use_moment_matching,
        qp_mode: normalizedData.qp_mode,
        soft_lambda: ensureNumber(normalizedData.soft_lambda, 1e4),
      }
      // Add MDN parameters if init is 'mdn'
      if (normalizedData.init === 'mdn') {
        request.em_params.mdn_params = {
          model_path: normalizedData.mdn_model_path || './ml_init/checkpoints/mdn_init_v1_N64_K5.pt',
          device: normalizedData.mdn_device || 'auto',
        }
      }
    } else if (normalizedData.method === 'lp') {
      request.lp_params = {
        L: ensureNumber(normalizedData.L, 5),
        objective_mode: normalizedData.objective_mode,
        solver: normalizedData.solver,
        sigma_min_scale: ensureNumber(normalizedData.sigma_min_scale, 0.1),
        sigma_max_scale: ensureNumber(normalizedData.sigma_max_scale, 3.0),
        pdf_tolerance: normalizedData.pdf_tolerance ? ensureNumber(normalizedData.pdf_tolerance) : null,
        lambda_raw: normalizedData.lambda_raw || null,
        objective_form: normalizedData.objective_form || 'A',
      }
    } else if (normalizedData.method === 'hybrid') {
      request.hybrid_params = {
        dict_L: ensureNumber(normalizedData.L, 5),
        objective_mode: 'raw_moments',
      }
      request.em_params = {
        max_iter: ensureNumber(normalizedData.max_iter, 400),
        tol: ensureNumber(normalizedData.tol, 1e-10),
        reg_var: ensureNumber(normalizedData.reg_var, 1e-6),
        n_init: ensureNumber(normalizedData.n_init, 8),
        seed: ensureNumber(normalizedData.seed, 1),
        use_moment_matching: normalizedData.use_moment_matching,
        qp_mode: normalizedData.qp_mode,
        soft_lambda: ensureNumber(normalizedData.soft_lambda, 1e4),
      }
    }

    onSubmit(request)
  }

  const handleLoadConfig = async (event) => {
    const file = event.target.files[0]
    if (!file) return

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('/api/load-config', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to load config file')
      }

      const config = await response.json()

      // Update form data with loaded config
      setFormData((prev) => {
        const newData = {
          ...prev,
          // Bivariate normal parameters
          mu_x: config.bivariate_params?.mu_x ?? prev.mu_x,
          sigma_x: config.bivariate_params?.sigma_x ?? prev.sigma_x,
          mu_y: config.bivariate_params?.mu_y ?? prev.mu_y,
          sigma_y: config.bivariate_params?.sigma_y ?? prev.sigma_y,
          rho: config.bivariate_params?.rho ?? prev.rho,
          
          // Grid parameters
          z_range_min: config.grid_params?.z_range?.[0] ?? prev.z_range_min,
          z_range_max: config.grid_params?.z_range?.[1] ?? prev.z_range_max,
          z_npoints: config.grid_params?.z_npoints ?? prev.z_npoints,
          
          // GMM parameters
          K: config.K ?? prev.K,
          method: config.method ?? prev.method,
          show_grid_points: config.show_grid_points ?? prev.show_grid_points,
          max_grid_points_display: config.max_grid_points_display ?? prev.max_grid_points_display,
        }

        // Method-specific parameters
        if (config.method === 'em' && config.em_params) {
          Object.assign(newData, {
            max_iter: config.em_params.max_iter ?? prev.max_iter,
            tol: config.em_params.tol ?? prev.tol,
            reg_var: config.em_params.reg_var ?? prev.reg_var,
            n_init: config.em_params.n_init ?? prev.n_init,
            seed: config.em_params.seed ?? prev.seed,
            init: config.em_params.init ?? prev.init,
            use_moment_matching: config.em_params.use_moment_matching ?? prev.use_moment_matching,
            qp_mode: config.em_params.qp_mode ?? prev.qp_mode,
            soft_lambda: config.em_params.soft_lambda ?? prev.soft_lambda,
          })
          // Add MDN parameters if present (from em_params.mdn_params or config.mdn)
          if (config.em_params.mdn_params) {
            Object.assign(newData, {
              mdn_model_path: config.em_params.mdn_params.model_path ?? prev.mdn_model_path,
              mdn_device: config.em_params.mdn_params.device ?? prev.mdn_device,
            })
          } else if (config.mdn) {
            Object.assign(newData, {
              mdn_model_path: config.mdn.model_path ?? prev.mdn_model_path,
              mdn_device: config.mdn.device ?? prev.mdn_device,
            })
          }
        } else if (config.method === 'lp' && config.lp_params) {
          Object.assign(newData, {
            L: config.lp_params.L ?? prev.L,
            objective_mode: config.lp_params.objective_mode ?? prev.objective_mode,
            solver: config.lp_params.solver ?? prev.solver,
            sigma_min_scale: config.lp_params.sigma_min_scale ?? prev.sigma_min_scale,
            sigma_max_scale: config.lp_params.sigma_max_scale ?? prev.sigma_max_scale,
            lambda_raw: config.lp_params.lambda_raw ?? prev.lambda_raw,
            objective_form: config.lp_params.objective_form ?? prev.objective_form,
            pdf_tolerance: config.lp_params.pdf_tolerance ?? prev.pdf_tolerance,
          })
        } else if (config.method === 'hybrid' && config.hybrid_params) {
          Object.assign(newData, {
            dict_J: config.hybrid_params.dict_J ?? prev.dict_J,
            dict_L: config.hybrid_params.dict_L ?? prev.dict_L,
            objective_mode: config.hybrid_params.objective_mode ?? prev.objective_mode,
            mu_mode: config.hybrid_params.mu_mode ?? prev.mu_mode,
            tail_focus: config.hybrid_params.tail_focus ?? prev.tail_focus,
            tail_alpha: config.hybrid_params.tail_alpha ?? prev.tail_alpha,
            sigma_min_scale: config.hybrid_params.sigma_min_scale ?? prev.sigma_min_scale,
            sigma_max_scale: config.hybrid_params.sigma_max_scale ?? prev.sigma_max_scale,
            pdf_tolerance: config.hybrid_params.pdf_tolerance ?? prev.pdf_tolerance,
            lambda_raw: config.hybrid_params.lambda_raw ?? prev.lambda_raw,
          })
          // Also set EM params for hybrid
          if (config.em_params) {
            Object.assign(newData, {
              max_iter: config.em_params.max_iter ?? prev.max_iter,
              tol: config.em_params.tol ?? prev.tol,
              reg_var: config.em_params.reg_var ?? prev.reg_var,
              n_init: config.em_params.n_init ?? prev.n_init,
              seed: config.em_params.seed ?? prev.seed,
              use_moment_matching: config.em_params.use_moment_matching ?? prev.use_moment_matching,
              qp_mode: config.em_params.qp_mode ?? prev.qp_mode,
              soft_lambda: config.em_params.soft_lambda ?? prev.soft_lambda,
            })
          }
        }

        return newData
      })

      // Reset file input
      event.target.value = ''
    } catch (error) {
      alert(`Error loading config: ${error.message}`)
      event.target.value = ''
    }
  }

  const handleExportConfig = () => {
    // Normalize all numeric fields before exporting
    const defaults = {
      mu_x: 0.0,
      sigma_x: 0.8,
      mu_y: 0.0,
      sigma_y: 1.6,
      rho: 0.9,
      z_range_min: -6.0,
      z_range_max: 8.0,
      z_npoints: 2500,
      K: 3,
      max_iter: 400,
      tol: 1e-10,
      reg_var: 1e-6,
      n_init: 8,
      seed: 1,
      soft_lambda: 1e4,
      L: 5,
      sigma_min_scale: 0.1,
      sigma_max_scale: 3.0,
      max_grid_points_display: 200,
    }
    
    // Normalize formData values
    const normalizedData = { ...formData }
    Object.keys(defaults).forEach((field) => {
      const value = formData[field]
      if (typeof value === 'string') {
        const numValue = parseFloat(value)
        normalizedData[field] = !isNaN(numValue) && isFinite(numValue) ? numValue : defaults[field]
      } else if (typeof value === 'number') {
        normalizedData[field] = isNaN(value) || !isFinite(value) ? defaults[field] : value
      } else if (value === null || value === undefined || value === '') {
        normalizedData[field] = defaults[field]
      }
    })
    
    // Build config JSON structure
    const config = {
      mu_x: normalizedData.mu_x,
      sigma_x: normalizedData.sigma_x,
      mu_y: normalizedData.mu_y,
      sigma_y: normalizedData.sigma_y,
      rho: normalizedData.rho,
      z_range: [normalizedData.z_range_min, normalizedData.z_range_max],
      z_npoints: normalizedData.z_npoints,
      K: normalizedData.K,
      method: normalizedData.method,
      show_grid_points: normalizedData.show_grid_points,
      max_grid_points_display: normalizedData.max_grid_points_display,
      output_path: "pdf_comparison",
    }
    
    // Add method-specific parameters
    if (normalizedData.method === 'em') {
      config.max_iter = normalizedData.max_iter
      config.tol = normalizedData.tol
      config.reg_var = normalizedData.reg_var
      config.n_init = normalizedData.n_init
      config.seed = normalizedData.seed
      config.init = normalizedData.init
      config.use_moment_matching = normalizedData.use_moment_matching
      config.qp_mode = normalizedData.qp_mode
      config.soft_lambda = normalizedData.soft_lambda
      // Add MDN parameters if init is 'mdn'
      if (normalizedData.init === 'mdn') {
        config.mdn = {
          model_path: normalizedData.mdn_model_path || './ml_init/checkpoints/mdn_init_v1_N64_K5.pt',
          device: normalizedData.mdn_device || 'auto',
        }
      }
    } else if (normalizedData.method === 'lp') {
      config.L = normalizedData.L
      config.objective_mode = normalizedData.objective_mode
      config.lp_params = {
        solver: normalizedData.solver,
        sigma_min_scale: normalizedData.sigma_min_scale,
        sigma_max_scale: normalizedData.sigma_max_scale,
      }
      if (normalizedData.objective_mode === 'raw_moments') {
        if (normalizedData.pdf_tolerance) {
          config.lp_params.pdf_tolerance = normalizedData.pdf_tolerance
        }
        if (normalizedData.lambda_raw) {
          config.lp_params.lambda_raw = normalizedData.lambda_raw
        }
        config.lp_params.objective_form = normalizedData.objective_form || 'A'
      }
    } else if (normalizedData.method === 'hybrid') {
      config.dict_L = normalizedData.L
      config.objective_mode = 'raw_moments'
      config.max_iter = normalizedData.max_iter
      config.tol = normalizedData.tol
      config.reg_var = normalizedData.reg_var
      config.n_init = normalizedData.n_init
      config.seed = normalizedData.seed
      config.use_moment_matching = normalizedData.use_moment_matching
      config.qp_mode = normalizedData.qp_mode
      config.soft_lambda = normalizedData.soft_lambda
    }
    
    // Convert to JSON string with pretty formatting
    const jsonString = JSON.stringify(config, null, 2)
    
    // Create blob and download
    const blob = new Blob([jsonString], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'config.json'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <Typography variant="h5" gutterBottom>
        Parameters
      </Typography>

      {/* Bivariate Normal Parameters */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Bivariate Normal Distribution</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="μ_X"
                type="number"
                value={editingFields.mu_x ? formData.mu_x : (typeof formData.mu_x === 'number' ? formData.mu_x : '')}
                onChange={handleChange('mu_x')}
                onBlur={handleBlur('mu_x', 0.0)}
                margin="normal"
                inputProps={{ step: 0.1 }}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="σ_X"
                type="number"
                value={editingFields.sigma_x ? formData.sigma_x : (typeof formData.sigma_x === 'number' ? formData.sigma_x : '')}
                onChange={handleChange('sigma_x')}
                onBlur={handleBlur('sigma_x', 0.8)}
                margin="normal"
                inputProps={{ step: 0.01, min: 0.01 }}
                onInvalid={(e) => {
                  e.preventDefault()
                }}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="μ_Y"
                type="number"
                value={editingFields.mu_y ? formData.mu_y : (typeof formData.mu_y === 'number' ? formData.mu_y : '')}
                onChange={handleChange('mu_y')}
                onBlur={handleBlur('mu_y', 0.0)}
                margin="normal"
                inputProps={{ step: 0.1 }}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="σ_Y"
                type="number"
                value={editingFields.sigma_y ? formData.sigma_y : (typeof formData.sigma_y === 'number' ? formData.sigma_y : '')}
                onChange={handleChange('sigma_y')}
                onBlur={handleBlur('sigma_y', 1.6)}
                margin="normal"
                inputProps={{ step: 0.01, min: 0.01 }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="ρ (Correlation)"
                type="number"
                value={editingFields.rho ? formData.rho : (typeof formData.rho === 'number' ? formData.rho : '')}
                onChange={handleChange('rho')}
                onBlur={handleBlur('rho', 0.9)}
                margin="normal"
                inputProps={{ step: 0.01, min: -1, max: 1 }}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Grid Parameters */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Grid Settings</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="z_min"
                type="number"
                value={editingFields.z_range_min ? formData.z_range_min : (typeof formData.z_range_min === 'number' ? formData.z_range_min : '')}
                onChange={handleChange('z_range_min')}
                onBlur={handleBlur('z_range_min', -6.0)}
                margin="normal"
                inputProps={{ step: 0.1 }}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="z_max"
                type="number"
                value={editingFields.z_range_max ? formData.z_range_max : (typeof formData.z_range_max === 'number' ? formData.z_range_max : '')}
                onChange={handleChange('z_range_max')}
                onBlur={handleBlur('z_range_max', 8.0)}
                margin="normal"
                inputProps={{ step: 0.1 }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Number of Grid Points"
                type="number"
                value={editingFields.z_npoints ? formData.z_npoints : (typeof formData.z_npoints === 'number' ? formData.z_npoints : '')}
                onChange={handleChange('z_npoints')}
                onBlur={handleBlur('z_npoints', 2500)}
                margin="normal"
                inputProps={{ min: 8, max: 100000 }}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Method Selection */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Fitting Method</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <FormControl fullWidth margin="normal">
                <InputLabel>Method</InputLabel>
                <Select
                  value={formData.method}
                  onChange={handleChange('method')}
                  label="Method"
                >
                  <MenuItem value="em">EM Algorithm</MenuItem>
                  <MenuItem value="lp">LP Algorithm</MenuItem>
                  <MenuItem value="hybrid">Hybrid (LP→EM→QP)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="K (Number of Components)"
                type="number"
                value={editingFields.K ? formData.K : (typeof formData.K === 'number' ? formData.K : '')}
                onChange={handleChange('K')}
                onBlur={handleBlur('K', 3)}
                margin="normal"
                inputProps={{ min: 1, max: 50 }}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* EM Parameters */}
      {(formData.method === 'em' || formData.method === 'hybrid') && (
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">EM Algorithm Parameters</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Max Iterations"
                  type="number"
                  value={editingFields.max_iter ? formData.max_iter : (typeof formData.max_iter === 'number' ? formData.max_iter : '')}
                  onChange={handleChange('max_iter')}
                  onBlur={handleBlur('max_iter', 400)}
                  margin="normal"
                  inputProps={{ min: 1 }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Tolerance"
                  type="number"
                  value={editingFields.tol ? formData.tol : (typeof formData.tol === 'number' ? formData.tol : '')}
                  onChange={handleChange('tol')}
                  onBlur={handleBlur('tol', 1e-10)}
                  margin="normal"
                  inputProps={{ step: 'any', min: 0.0000000001 }}
                  helperText="Must be greater than 0"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="reg_var"
                  type="number"
                  value={editingFields.reg_var ? formData.reg_var : (typeof formData.reg_var === 'number' ? formData.reg_var : '')}
                  onChange={handleChange('reg_var')}
                  onBlur={handleBlur('reg_var', 1e-6)}
                  margin="normal"
                  inputProps={{ step: 'any' }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="n_init"
                  type="number"
                  value={editingFields.n_init ? formData.n_init : (typeof formData.n_init === 'number' ? formData.n_init : '')}
                  onChange={handleChange('n_init')}
                  onBlur={handleBlur('n_init', 8)}
                  margin="normal"
                  inputProps={{ min: 1 }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Seed"
                  type="number"
                  value={editingFields.seed ? formData.seed : (typeof formData.seed === 'number' ? formData.seed : '')}
                  onChange={handleChange('seed')}
                  onBlur={handleBlur('seed', 1)}
                  margin="normal"
                />
              </Grid>
              {formData.method === 'hybrid' && (
                <Grid item xs={12}>
                  <Typography variant="body2" color="textSecondary" sx={{ mt: 1, mb: 1 }}>
                    Note: Hybrid method uses LP solution as initial values for EM algorithm.
                    Initialization method selection is not applicable.
                  </Typography>
                </Grid>
              )}
              {formData.method !== 'hybrid' && (
                <Grid item xs={12}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Initialization</InputLabel>
                    <Select
                      value={formData.init}
                      onChange={handleChange('init')}
                      label="Initialization"
                    >
                      <MenuItem value="quantile">Quantile</MenuItem>
                      <MenuItem value="random">Random</MenuItem>
                      <MenuItem value="qmi">QMI</MenuItem>
                      <MenuItem value="wqmi">WQMI</MenuItem>
                      <MenuItem value="mdn">MDN (Machine Learning)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              )}
              {formData.init === 'mdn' && formData.method !== 'hybrid' && (
                <>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="MDN Model Path"
                      value={formData.mdn_model_path || ''}
                      onChange={handleChange('mdn_model_path')}
                      helperText="Path to MDN model file (default: ./ml_init/checkpoints/mdn_init_v1_N64_K5.pt)"
                      margin="normal"
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl fullWidth margin="normal">
                      <InputLabel>MDN Device</InputLabel>
                      <Select
                        value={formData.mdn_device || 'auto'}
                        onChange={handleChange('mdn_device')}
                        label="MDN Device"
                      >
                        <MenuItem value="auto">Auto</MenuItem>
                        <MenuItem value="cpu">CPU</MenuItem>
                        <MenuItem value="cuda">CUDA</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                </>
              )}
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={formData.use_moment_matching}
                      onChange={handleChange('use_moment_matching')}
                    />
                  }
                  label="Use Moment Matching"
                />
              </Grid>
              {formData.use_moment_matching && (
                <>
                  <Grid item xs={6}>
                    <FormControl fullWidth margin="normal">
                      <InputLabel>QP Mode</InputLabel>
                      <Select
                        value={formData.qp_mode}
                        onChange={handleChange('qp_mode')}
                        label="QP Mode"
                      >
                        <MenuItem value="hard">Hard</MenuItem>
                        <MenuItem value="soft">Soft</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="Soft Lambda"
                      type="number"
                      value={editingFields.soft_lambda ? formData.soft_lambda : (typeof formData.soft_lambda === 'number' ? formData.soft_lambda : '')}
                      onChange={handleChange('soft_lambda')}
                      onBlur={handleBlur('soft_lambda', 1e4)}
                      margin="normal"
                    />
                  </Grid>
                </>
              )}
            </Grid>
          </AccordionDetails>
        </Accordion>
      )}

      {/* LP Parameters */}
      {formData.method === 'lp' && (
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">LP Algorithm Parameters</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="L (Sigma Levels)"
                  type="number"
                  value={editingFields.L ? formData.L : (typeof formData.L === 'number' ? formData.L : '')}
                  onChange={handleChange('L')}
                  onBlur={handleBlur('L', 5)}
                  margin="normal"
                  inputProps={{ min: 1 }}
                />
              </Grid>
              <Grid item xs={6}>
                <FormControl fullWidth margin="normal">
                  <InputLabel>Objective Mode</InputLabel>
                  <Select
                    value={formData.objective_mode}
                    onChange={handleChange('objective_mode')}
                    label="Objective Mode"
                  >
                    <MenuItem value="pdf">PDF Error</MenuItem>
                    <MenuItem value="raw_moments">Raw Moments</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={6}>
                <FormControl fullWidth margin="normal">
                  <InputLabel>Solver</InputLabel>
                  <Select
                    value={formData.solver}
                    onChange={handleChange('solver')}
                    label="Solver"
                  >
                    <MenuItem value="highs">Highs</MenuItem>
                    <MenuItem value="interior-point">Interior Point</MenuItem>
                    <MenuItem value="revised simplex">Revised Simplex</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Sigma Min Scale"
                  type="number"
                  value={editingFields.sigma_min_scale ? formData.sigma_min_scale : (typeof formData.sigma_min_scale === 'number' ? formData.sigma_min_scale : '')}
                  onChange={handleChange('sigma_min_scale')}
                  onBlur={handleBlur('sigma_min_scale', 0.1)}
                  margin="normal"
                  inputProps={{ step: 'any', min: 0.01 }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Sigma Max Scale"
                  type="number"
                  value={editingFields.sigma_max_scale ? formData.sigma_max_scale : (typeof formData.sigma_max_scale === 'number' ? formData.sigma_max_scale : '')}
                  onChange={handleChange('sigma_max_scale')}
                  onBlur={handleBlur('sigma_max_scale', 3.0)}
                  margin="normal"
                  inputProps={{ step: 'any', min: 1 }}
                />
              </Grid>
              {formData.objective_mode === 'raw_moments' && (
                <>
                  <Grid item xs={6}>
                    <FormControl fullWidth margin="normal">
                      <InputLabel>Objective Form</InputLabel>
                      <Select
                        value={formData.objective_form}
                        onChange={handleChange('objective_form')}
                        label="Objective Form"
                      >
                        <MenuItem value="A">Form A (PDF constraint)</MenuItem>
                        <MenuItem value="B">Form B (Weighted sum)</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="PDF Tolerance (optional)"
                      type="number"
                      value={formData.pdf_tolerance || ''}
                      onChange={handleChange('pdf_tolerance')}
                      margin="normal"
                      inputProps={{ step: 'any', min: 0 }}
                      placeholder="None"
                    />
                  </Grid>
                </>
              )}
            </Grid>
          </AccordionDetails>
        </Accordion>
      )}

      <Divider sx={{ my: 2 }} />

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Button
          type="submit"
          variant="contained"
          color="primary"
          fullWidth
          size="large"
          disabled={loading}
          sx={{ flex: 1, minWidth: '200px' }}
        >
          {loading ? 'Computing...' : 'Compute'}
        </Button>
        <Button
          variant="outlined"
          color="secondary"
          size="large"
          component="label"
          disabled={loading}
        >
          Load Config
          <input
            type="file"
            accept=".json"
            hidden
            onChange={handleLoadConfig}
          />
        </Button>
        <Button
          variant="outlined"
          color="secondary"
          size="large"
          onClick={handleExportConfig}
          disabled={loading}
        >
          Export Config
        </Button>
        <Button
          variant="outlined"
          color="warning"
          size="large"
          onClick={() => {
            setFormData(defaultFormData)
            setEditingFields({})
          }}
          disabled={loading}
        >
          Reset
        </Button>
      </Box>
    </Box>
  )
}

export default ParameterForm

