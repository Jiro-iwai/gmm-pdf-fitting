import React from 'react'
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Grid,
  Card,
  CardContent,
  Divider,
} from '@mui/material'
import PlotViewer from './PlotViewer'
import StatisticsTable from './StatisticsTable'

const ResultDisplay = ({ result, plotSettings, setPlotSettings }) => {
  // Safety checks for nested objects
  const errorMetrics = result?.error_metrics || {}
  const executionTime = result?.execution_time || {}
  const gmmComponents = result?.gmm_components || []
  const statisticsTrue = result?.statistics_true
  const statisticsHat = result?.statistics_hat
  
  // Show statistics if both true and hat statistics exist
  const hasStatistics = statisticsTrue && statisticsHat

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* Plot - Only show if result exists */}
      {result && (
        <Paper sx={{ p: 2, flexShrink: 0 }}>
          <PlotViewer result={result} plotSettings={plotSettings} setPlotSettings={setPlotSettings} />
        </Paper>
      )}

      {/* Statistics Comparison */}
      {hasStatistics && (
        <Paper sx={{ p: 2, flexShrink: 0 }}>
          <Typography variant="h6" gutterBottom>
            Statistics Comparison
          </Typography>
          <StatisticsTable
            statisticsTrue={statisticsTrue}
            statisticsHat={statisticsHat}
          />
        </Paper>
      )}

      {/* Error Metrics - Only show if result exists */}
      {result && errorMetrics.linf_pdf !== undefined && (
        <Paper sx={{ p: 2, flexShrink: 0 }}>
          <Typography variant="h6" gutterBottom>
            Error Metrics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    PDF L∞ Error
                  </Typography>
                  <Typography variant="h6">
                    {typeof errorMetrics.linf_pdf === 'number' 
                      ? errorMetrics.linf_pdf.toExponential(3)
                      : 'N/A'}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    CDF L∞ Error
                  </Typography>
                  <Typography variant="h6">
                    {typeof errorMetrics.linf_cdf === 'number'
                      ? errorMetrics.linf_cdf.toExponential(3)
                      : 'N/A'}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Tail L1 Error
                  </Typography>
                  <Typography variant="h6">
                    {typeof errorMetrics.tail_l1_error === 'number'
                      ? errorMetrics.tail_l1_error.toExponential(3)
                      : 'N/A'}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Total Time
                  </Typography>
                  <Typography variant="h6">
                    {typeof executionTime.total_time === 'number'
                      ? `${executionTime.total_time.toFixed(3)}s`
                      : 'N/A'}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>
      )}

      {/* GMM Components - Only show if result exists */}
      {result && gmmComponents.length > 0 && (
        <Paper sx={{ p: 2, flexShrink: 0 }}>
          <Typography variant="h6" gutterBottom>
            GMM Components ({gmmComponents.length})
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Component</TableCell>
                  <TableCell align="right">π (Weight)</TableCell>
                  <TableCell align="right">μ (Mean)</TableCell>
                  <TableCell align="right">σ (Std Dev)</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {gmmComponents.map((comp, idx) => (
                  <TableRow key={idx}>
                    <TableCell>{idx + 1}</TableCell>
                    <TableCell align="right">
                      {typeof comp.pi === 'number' ? comp.pi.toFixed(6) : 'N/A'}
                    </TableCell>
                    <TableCell align="right">
                      {typeof comp.mu === 'number' ? comp.mu.toFixed(6) : 'N/A'}
                    </TableCell>
                    <TableCell align="right">
                      {typeof comp.sigma === 'number' ? comp.sigma.toFixed(6) : 'N/A'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}

      {/* Execution Info - Only show if result exists */}
      {result && (
        <Paper sx={{ p: 2, flexShrink: 0 }}>
          <Typography variant="h6" gutterBottom>
            Execution Information
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6} md={3}>
              <Typography variant="body2" color="textSecondary">
                Method
              </Typography>
              <Typography variant="body1">
                {result.method ? result.method.toUpperCase() : 'N/A'}
              </Typography>
            </Grid>
          
          {/* LP-specific information */}
          {result.method === 'lp' && result.diagnostics && (
            <>
              {result.diagnostics.t_pdf !== null && result.diagnostics.t_pdf !== undefined && (
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="textSecondary">
                    PDF L∞ Error
                  </Typography>
                  <Typography variant="body1">
                    {typeof result.diagnostics.t_pdf === 'number'
                      ? result.diagnostics.t_pdf.toExponential(3)
                      : 'N/A'}
                  </Typography>
                </Grid>
              )}
              {result.diagnostics.n_nonzero !== null && result.diagnostics.n_nonzero !== undefined && (
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="textSecondary">
                    Non-zero Components
                  </Typography>
                  <Typography variant="body1">
                    {result.diagnostics.n_nonzero} / {result.diagnostics.n_bases || 'N/A'}
                  </Typography>
                </Grid>
              )}
            </>
          )}
          
          {/* EM/Hybrid-specific information */}
          {(result.method === 'em' || result.method === 'hybrid') && (
            <>
              {result.log_likelihood !== null && result.log_likelihood !== undefined && (
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="textSecondary">
                    Log-Likelihood
                  </Typography>
                  <Typography variant="body1">
                    {typeof result.log_likelihood === 'number'
                      ? result.log_likelihood.toFixed(6)
                      : 'N/A'}
                  </Typography>
                </Grid>
              )}
              {result.n_iterations && (
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="textSecondary">
                    Iterations
                  </Typography>
                  <Typography variant="body1">{result.n_iterations}</Typography>
                </Grid>
              )}
              {executionTime.em_time !== null && executionTime.em_time !== undefined && (
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="textSecondary">
                    EM Time
                  </Typography>
                  <Typography variant="body1">
                    {typeof executionTime.em_time === 'number'
                      ? `${executionTime.em_time.toFixed(3)}s`
                      : 'N/A'}
                  </Typography>
                </Grid>
              )}
              {executionTime.init_time !== null && executionTime.init_time !== undefined && (
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="textSecondary">
                    Init Time
                  </Typography>
                  <Typography variant="body1">
                    {typeof executionTime.init_time === 'number'
                      ? `${executionTime.init_time.toFixed(3)}s`
                      : 'N/A'}
                  </Typography>
                </Grid>
              )}
            </>
          )}
          
          {/* LP Time - shown for both LP and Hybrid */}
          {executionTime.lp_time !== null && executionTime.lp_time !== undefined && (
            <Grid item xs={6} md={3}>
              <Typography variant="body2" color="textSecondary">
                LP Time
              </Typography>
              <Typography variant="body1">
                {typeof executionTime.lp_time === 'number'
                  ? `${executionTime.lp_time.toFixed(3)}s`
                  : 'N/A'}
              </Typography>
            </Grid>
          )}
          
          {/* QP Time - shown for EM/Hybrid with moment matching */}
          {executionTime.qp_time !== null && executionTime.qp_time !== undefined && (
            <Grid item xs={6} md={3}>
              <Typography variant="body2" color="textSecondary">
                QP Time
              </Typography>
              <Typography variant="body1">
                {typeof executionTime.qp_time === 'number'
                  ? `${executionTime.qp_time.toFixed(3)}s`
                  : 'N/A'}
              </Typography>
            </Grid>
          )}
          
          {/* Total Time */}
          {executionTime.total_time !== null && executionTime.total_time !== undefined && (
            <Grid item xs={6} md={3}>
              <Typography variant="body2" color="textSecondary">
                Total Time
              </Typography>
              <Typography variant="body1">
                {typeof executionTime.total_time === 'number'
                  ? `${executionTime.total_time.toFixed(3)}s`
                  : 'N/A'}
              </Typography>
            </Grid>
          )}
          </Grid>
        </Paper>
      )}
    </Box>
  )
}

export default ResultDisplay

