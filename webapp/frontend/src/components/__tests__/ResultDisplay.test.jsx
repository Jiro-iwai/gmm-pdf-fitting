import React from 'react'
import { render, screen } from '@testing-library/react'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import ResultDisplay from '../ResultDisplay'

const theme = createTheme()

const renderWithTheme = (component) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  )
}

describe('ResultDisplay', () => {
  const mockPlotSettings = {
    scaleMode: 'linear',
    showGridPoints: true,
    showGmmComponents: false,
  }

  const mockSetPlotSettings = jest.fn()

  const mockResult = {
    method: 'em',
    z: [0, 1, 2, 3],
    f_true: [0.1, 0.2, 0.3, 0.4],
    f_hat: [0.11, 0.19, 0.31, 0.39],
    statistics_true: {
      mean: 0.5,
      std: 1.0,
      skewness: 0.2,
      kurtosis: 0.3,
    },
    statistics_hat: {
      mean: 0.51,
      std: 0.99,
      skewness: 0.21,
      kurtosis: 0.31,
    },
    error_metrics: {
      linf_pdf: 0.01,
      linf_cdf: 0.02,
      tail_l1_error: 0.005,
    },
    execution_time: {
      total_time: 0.123,
      em_time: 0.1,
    },
    gmm_components: [
      { pi: 0.3, mu: 0.0, sigma: 0.5 },
      { pi: 0.7, mu: 1.0, sigma: 0.8 },
    ],
    log_likelihood: -10.5,
    n_iterations: 15,
  }

  beforeEach(() => {
    mockSetPlotSettings.mockClear()
  })

  test('renders nothing when result is null', () => {
    const { container } = renderWithTheme(
      <ResultDisplay
        result={null}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(container.firstChild).toBeNull()
  })

  test('renders plot when result exists', () => {
    renderWithTheme(
      <ResultDisplay
        result={mockResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.getByText(/pdf comparison/i)).toBeInTheDocument()
  })

  test('renders Statistics Comparison table when statistics exist', () => {
    renderWithTheme(
      <ResultDisplay
        result={mockResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.getByText(/statistics comparison/i)).toBeInTheDocument()
    expect(screen.getByText(/mean/i)).toBeInTheDocument()
  })

  test('does not render Statistics Comparison when statistics are missing', () => {
    const resultWithoutStats = {
      ...mockResult,
      statistics_true: null,
      statistics_hat: null,
    }

    renderWithTheme(
      <ResultDisplay
        result={resultWithoutStats}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.queryByText(/statistics comparison/i)).not.toBeInTheDocument()
  })

  test('renders Error Metrics section', () => {
    renderWithTheme(
      <ResultDisplay
        result={mockResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.getByText(/error metrics/i)).toBeInTheDocument()
    expect(screen.getByText(/pdf l∞ error/i)).toBeInTheDocument()
    expect(screen.getByText(/cdf l∞ error/i)).toBeInTheDocument()
    expect(screen.getByText(/tail l1 error/i)).toBeInTheDocument()
  })

  test('renders GMM Components table', () => {
    renderWithTheme(
      <ResultDisplay
        result={mockResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.getByText(/gmm components/i)).toBeInTheDocument()
    expect(screen.getByText(/component/i)).toBeInTheDocument()
    expect(screen.getByText(/π.*weight/i)).toBeInTheDocument()
  })

  test('displays execution information', () => {
    renderWithTheme(
      <ResultDisplay
        result={mockResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.getByText(/execution information/i)).toBeInTheDocument()
    expect(screen.getByText(/method/i)).toBeInTheDocument()
    expect(screen.getByText(/em/i)).toBeInTheDocument()
  })

  test('handles missing optional fields gracefully', () => {
    const minimalResult = {
      method: 'em',
      z: [0, 1],
      f_true: [0.1, 0.2],
      f_hat: [0.11, 0.19],
    }

    renderWithTheme(
      <ResultDisplay
        result={minimalResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    // Should render plot even with minimal data
    expect(screen.getByText(/pdf comparison/i)).toBeInTheDocument()
  })

  test('displays log-likelihood when available', () => {
    renderWithTheme(
      <ResultDisplay
        result={mockResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.getByText(/log-likelihood/i)).toBeInTheDocument()
  })

  test('displays iteration count when available', () => {
    renderWithTheme(
      <ResultDisplay
        result={mockResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.getByText(/iterations/i)).toBeInTheDocument()
    expect(screen.getByText('15')).toBeInTheDocument()
  })
})

