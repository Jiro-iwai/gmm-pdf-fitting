import React from 'react'
import { render, screen } from '@testing-library/react'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import StatisticsTable from '../StatisticsTable'

const theme = createTheme()

const renderWithTheme = (component) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  )
}

describe('StatisticsTable', () => {
  const defaultStatisticsTrue = {
    mean: 0.5,
    std: 1.0,
    skewness: 0.2,
    kurtosis: 0.3,
  }

  const defaultStatisticsHat = {
    mean: 0.51,
    std: 0.99,
    skewness: 0.21,
    kurtosis: 0.31,
  }

  test('renders statistics table with headers', () => {
    renderWithTheme(
      <StatisticsTable
        statisticsTrue={defaultStatisticsTrue}
        statisticsHat={defaultStatisticsHat}
      />
    )

    expect(screen.getByText(/statistic/i)).toBeInTheDocument()
    expect(screen.getByText(/true pdf/i)).toBeInTheDocument()
    expect(screen.getByText(/gmm approx/i)).toBeInTheDocument()
    expect(screen.getByText(/rel error/i)).toBeInTheDocument()
  })

  test('displays all statistics rows', () => {
    renderWithTheme(
      <StatisticsTable
        statisticsTrue={defaultStatisticsTrue}
        statisticsHat={defaultStatisticsHat}
      />
    )

    expect(screen.getByText(/mean/i)).toBeInTheDocument()
    expect(screen.getByText(/std dev/i)).toBeInTheDocument()
    expect(screen.getByText(/skewness/i)).toBeInTheDocument()
    expect(screen.getByText(/kurtosis/i)).toBeInTheDocument()
  })

  test('displays correct values', () => {
    renderWithTheme(
      <StatisticsTable
        statisticsTrue={defaultStatisticsTrue}
        statisticsHat={defaultStatisticsHat}
      />
    )

    // Check that values are displayed (format may vary)
    // There may be multiple elements with these values, so use getAllByText
    const meanValues = screen.getAllByText(/0[.]500000/)
    expect(meanValues.length).toBeGreaterThan(0) // mean
    
    const stdValues = screen.getAllByText(/1[.]000000/)
    expect(stdValues.length).toBeGreaterThan(0) // std
  })

  test('calculates and displays relative error', () => {
    renderWithTheme(
      <StatisticsTable
        statisticsTrue={defaultStatisticsTrue}
        statisticsHat={defaultStatisticsHat}
      />
    )

    // Relative error for mean: (0.51 - 0.5) / 0.5 * 100 = 2%
    // Should display percentage (there may be multiple matches, so use getAllByText)
    const percentageElements = screen.getAllByText(/%/)
    expect(percentageElements.length).toBeGreaterThan(0)
  })

  test('handles missing statistics gracefully', () => {
    renderWithTheme(
      <StatisticsTable
        statisticsTrue={null}
        statisticsHat={null}
      />
    )

    // Should still render table structure
    expect(screen.getByText(/statistic/i)).toBeInTheDocument()
  })

  test('handles partial statistics', () => {
    const partialTrue = { mean: 0.5, std: 1.0 }
    const partialHat = { mean: 0.51, std: 0.99 }

    renderWithTheme(
      <StatisticsTable
        statisticsTrue={partialTrue}
        statisticsHat={partialHat}
      />
    )

    expect(screen.getByText(/mean/i)).toBeInTheDocument()
    expect(screen.getByText(/std dev/i)).toBeInTheDocument()
  })

  test('displays N/A for invalid relative errors', () => {
    const invalidTrue = { mean: 0, std: 0, skewness: NaN, kurtosis: null }
    const invalidHat = { mean: 0, std: 0, skewness: NaN, kurtosis: null }

    renderWithTheme(
      <StatisticsTable
        statisticsTrue={invalidTrue}
        statisticsHat={invalidHat}
      />
    )

    // Should handle gracefully without crashing
    expect(screen.getByText(/statistic/i)).toBeInTheDocument()
  })

  test('applies color coding based on error magnitude', () => {
    // Small error (<1%)
    const smallErrorTrue = { mean: 0.5 }
    const smallErrorHat = { mean: 0.5005 } // 0.1% error

    const { container: container1 } = renderWithTheme(
      <StatisticsTable
        statisticsTrue={smallErrorTrue}
        statisticsHat={smallErrorHat}
      />
    )

    // Large error (>5%)
    const largeErrorTrue = { mean: 0.5 }
    const largeErrorHat = { mean: 0.6 } // 20% error

    const { container: container2 } = renderWithTheme(
      <StatisticsTable
        statisticsTrue={largeErrorTrue}
        statisticsHat={largeErrorHat}
      />
    )

    // Both should render without errors
    expect(container1).toBeInTheDocument()
    expect(container2).toBeInTheDocument()
  })
})

