import React, { useMemo } from 'react'
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
} from '@mui/material'
import { useTheme } from '@mui/material/styles'

const StatisticsTable = ({ statisticsTrue, statisticsHat }) => {
  const theme = useTheme()
  
  // Always show the table, even if data is incomplete
  // Use empty objects as fallback to ensure table structure is always displayed
  // Use useMemo to prevent unnecessary re-renders
  const safeStatisticsTrue = useMemo(() => {
    return statisticsTrue || { mean: 0, std: 0, skewness: 0, kurtosis: 0 }
  }, [statisticsTrue])
  
  const safeStatisticsHat = useMemo(() => {
    return statisticsHat || { mean: 0, std: 0, skewness: 0, kurtosis: 0 }
  }, [statisticsHat])

  // Calculate relative error as a number (percentage)
  const calcRelativeErrorValue = (trueVal, hatVal) => {
    if (typeof trueVal !== 'number' || typeof hatVal !== 'number') {
      return null
    }
    if (isNaN(trueVal) || isNaN(hatVal)) {
      return null
    }
    if (Math.abs(trueVal) < 1e-10) {
      return Math.abs(hatVal) > 1e-10 ? Infinity : 0
    }
    const error = ((hatVal - trueVal) / Math.abs(trueVal) * 100)
    return isNaN(error) ? null : error
  }

  const calcRelativeError = (trueVal, hatVal) => {
    const errorValue = calcRelativeErrorValue(trueVal, hatVal)
    if (errorValue === null) {
      return 'N/A'
    }
    if (errorValue === Infinity) {
      return '∞'
    }
    return `${errorValue.toFixed(2)}%`
  }

  // Get row background color based on error magnitude
  const getRowBackgroundColor = (errorValue) => {
    if (errorValue === null || errorValue === Infinity) {
      return 'transparent'
    }
    
    const absError = Math.abs(errorValue)
    
    // Thresholds: small (blue) < 1%, medium (yellow) < 5%, large (red) >= 5%
    if (absError < 1.0) {
      // Small error: blue (light blue for dark mode, darker blue for light mode)
      return theme.palette.mode === 'dark' 
        ? 'rgba(66, 165, 245, 0.15)'  // Light blue with transparency
        : 'rgba(33, 150, 243, 0.1)'   // Blue with transparency
    } else if (absError < 5.0) {
      // Medium error: yellow
      return theme.palette.mode === 'dark'
        ? 'rgba(255, 235, 59, 0.15)'   // Light yellow with transparency
        : 'rgba(255, 193, 7, 0.1)'     // Yellow with transparency
    } else {
      // Large error: red
      return theme.palette.mode === 'dark'
        ? 'rgba(244, 67, 54, 0.2)'     // Light red with transparency
        : 'rgba(244, 67, 54, 0.1)'     // Red with transparency
    }
  }

  const formatNumber = (val) => {
    if (typeof val !== 'number' || isNaN(val)) {
      return 'N/A'
    }
    return val.toFixed(6)
  }

  const rows = [
    { label: 'Mean', true: safeStatisticsTrue.mean, hat: safeStatisticsHat.mean },
    { label: 'Std Dev', true: safeStatisticsTrue.std, hat: safeStatisticsHat.std },
    { label: 'Skewness', true: safeStatisticsTrue.skewness, hat: safeStatisticsHat.skewness },
    { label: 'Kurtosis', true: safeStatisticsTrue.kurtosis, hat: safeStatisticsHat.kurtosis },
  ]

  return (
    <TableContainer component={Paper} variant="outlined">
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Statistic</TableCell>
            <TableCell align="right">True PDF</TableCell>
            <TableCell align="right">GMM Approx</TableCell>
            <TableCell align="right">Rel Error (%)</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => {
            const errorValue = calcRelativeErrorValue(row.true, row.hat)
            const backgroundColor = getRowBackgroundColor(errorValue)
            return (
              <TableRow 
                key={row.label}
                sx={{
                  backgroundColor: backgroundColor,
                  '&:hover': {
                    backgroundColor: backgroundColor,
                    opacity: 0.8,
                  },
                }}
              >
                <TableCell component="th" scope="row">
                  {row.label}
                </TableCell>
                <TableCell align="right">{formatNumber(row.true)}</TableCell>
                <TableCell align="right">{formatNumber(row.hat)}</TableCell>
                <TableCell align="right">{calcRelativeError(row.true, row.hat)}</TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

export default StatisticsTable

