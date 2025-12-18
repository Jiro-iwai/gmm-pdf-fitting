import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import PlotViewer from '../PlotViewer'

const theme = createTheme()

const renderWithTheme = (component) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  )
}

// Mock react-plotly.js
jest.mock('react-plotly.js', () => {
  return function MockPlot({ data, layout }) {
    return (
      <div data-testid="plotly-plot">
        <div data-testid="plot-data">{JSON.stringify(data?.length || 0)}</div>
        <div data-testid="plot-layout">{layout?.title?.text || 'No title'}</div>
      </div>
    )
  }
})

describe('PlotViewer', () => {
  const mockPlotSettings = {
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

  const mockSetPlotSettings = jest.fn()

  const mockResult = {
    z: [0, 1, 2, 3, 4],
    f_true: [0.1, 0.2, 0.3, 0.2, 0.1],
    f_hat: [0.11, 0.19, 0.31, 0.19, 0.11],
    gmm_components: [
      { pi: 0.5, mu: 1.0, sigma: 0.5 },
      { pi: 0.5, mu: 3.0, sigma: 0.5 },
    ],
  }

  beforeEach(() => {
    mockSetPlotSettings.mockClear()
  })

  test('renders plot title', () => {
    renderWithTheme(
      <PlotViewer
        result={mockResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.getByText(/pdf comparison/i)).toBeInTheDocument()
  })

  test('displays error message when result is null', () => {
    renderWithTheme(
      <PlotViewer
        result={null}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.getByText(/plot data not available/i)).toBeInTheDocument()
  })

  test('displays error message when data arrays are empty', () => {
    const emptyResult = {
      z: [],
      f_true: [],
      f_hat: [],
    }

    renderWithTheme(
      <PlotViewer
        result={emptyResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.getByText(/invalid data/i)).toBeInTheDocument()
  })

  test('renders scale mode selector', () => {
    renderWithTheme(
      <PlotViewer
        result={mockResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.getByLabelText(/scale mode/i)).toBeInTheDocument()
  })

  test('changes scale mode when selector is used', async () => {
    renderWithTheme(
      <PlotViewer
        result={mockResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    const scaleModeSelect = screen.getByLabelText(/scale mode/i)
    fireEvent.mouseDown(scaleModeSelect)
    
    await waitFor(() => {
      const logOption = screen.getByText(/log scale/i)
      fireEvent.click(logOption)
    })

    expect(mockSetPlotSettings).toHaveBeenCalled()
  })

  test('renders plot settings accordion', () => {
    renderWithTheme(
      <PlotViewer
        result={mockResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.getByText(/plot settings/i)).toBeInTheDocument()
  })

  test('renders plot with correct data', () => {
    renderWithTheme(
      <PlotViewer
        result={mockResult}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    const plotElement = screen.getByTestId('plotly-plot')
    expect(plotElement).toBeInTheDocument()
  })

  test('handles missing gmm_components gracefully', () => {
    const resultWithoutComponents = {
      z: [0, 1, 2],
      f_true: [0.1, 0.2, 0.3],
      f_hat: [0.11, 0.19, 0.31],
    }

    renderWithTheme(
      <PlotViewer
        result={resultWithoutComponents}
        plotSettings={mockPlotSettings}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    expect(screen.getByTestId('plotly-plot')).toBeInTheDocument()
  })

  test('shows GMM components when showGmmComponents is true', () => {
    const settingsWithComponents = {
      ...mockPlotSettings,
      showGmmComponents: true,
    }

    renderWithTheme(
      <PlotViewer
        result={mockResult}
        plotSettings={settingsWithComponents}
        setPlotSettings={mockSetPlotSettings}
      />
    )

    // Plot should render with component data
    expect(screen.getByTestId('plotly-plot')).toBeInTheDocument()
  })
})

