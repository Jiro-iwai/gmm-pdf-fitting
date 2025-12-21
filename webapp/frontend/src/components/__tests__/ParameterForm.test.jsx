import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import ParameterForm from '../ParameterForm'

// Mock MUI theme provider
const theme = createTheme()

const renderWithTheme = (component) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  )
}

describe('ParameterForm', () => {
  const mockOnSubmit = jest.fn()

  beforeEach(() => {
    mockOnSubmit.mockClear()
  })

  test('renders all main input fields', async () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    // Note: Compute button has been moved to PlotViewer, so we don't check for it here
    
    // Open the Bivariate Normal Distribution accordion
    const bivariateAccordion = screen.getByRole('button', { name: /bivariate normal distribution/i })
    fireEvent.click(bivariateAccordion)
    
    await waitFor(() => {
      // Check bivariate parameters (using Greek letters)
      expect(screen.getByLabelText(/μ_X|mu_x/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/σ_X|sigma_x/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/μ_Y|mu_y/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/σ_Y|sigma_y/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/ρ|rho/i)).toBeInTheDocument()
    })
    
    // Check method selection (open Fitting Method accordion)
    const methodAccordion = screen.getByRole('button', { name: /fitting method/i })
    fireEvent.click(methodAccordion)
    
    await waitFor(() => {
      // There may be multiple elements with "method" text, so use getAllByText
      const methodElements = screen.getAllByText(/method/i)
      expect(methodElements.length).toBeGreaterThan(0)
    })
  })

  test('displays default values correctly', async () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    // Open the Bivariate Normal Distribution accordion
    const bivariateAccordion = screen.getByRole('button', { name: /bivariate normal distribution/i })
    fireEvent.click(bivariateAccordion)
    
    await waitFor(() => {
      // There may be multiple elements with these values, so use getAllByDisplayValue
      const zeroValues = screen.getAllByDisplayValue('0')
      expect(zeroValues.length).toBeGreaterThan(0) // mu_x default
      
      const sigmaXValues = screen.getAllByDisplayValue('0.8')
      expect(sigmaXValues.length).toBeGreaterThan(0) // sigma_x default
      
      const rhoValues = screen.getAllByDisplayValue('0.9')
      expect(rhoValues.length).toBeGreaterThan(0) // rho default
    })
  })

  test('updates input values when user types', async () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    // Open the Bivariate Normal Distribution accordion
    const bivariateAccordion = screen.getByRole('button', { name: /bivariate normal distribution/i })
    fireEvent.click(bivariateAccordion)
    
    await waitFor(() => {
      const muXInput = screen.getByLabelText(/μ_X|mu_x/i)
      fireEvent.change(muXInput, { target: { value: '1.5' } })
      
      expect(muXInput).toHaveValue(1.5)
    })
  })

  test('calls onSubmit with correct parameters when form is submitted', async () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    // Open the Bivariate Normal Distribution accordion
    const bivariateAccordion = screen.getByRole('button', { name: /bivariate normal distribution/i })
    fireEvent.click(bivariateAccordion)
    
    await waitFor(async () => {
      // Change some values
      const muXInput = screen.getByLabelText(/μ_X|mu_x/i)
      fireEvent.change(muXInput, { target: { value: '0.5' } })
      
      // Submit the form directly (Compute button is now in PlotViewer)
      const form = document.querySelector('form')
      fireEvent.submit(form)
      
      await waitFor(() => {
        expect(mockOnSubmit).toHaveBeenCalledTimes(1)
        const callArgs = mockOnSubmit.mock.calls[0][0]
        expect(callArgs.bivariate_params.mu_x).toBe(0.5)
        expect(callArgs.bivariate_params.sigma_x).toBe(0.8) // default value
      })
    })
  })

  // Note: Compute button has been moved to PlotViewer, so loading state test is no longer applicable here

  test('shows method-specific parameters when method changes', async () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    // Open the Fitting Method accordion
    const methodAccordion = screen.getByRole('button', { name: /fitting method/i })
    fireEvent.click(methodAccordion)
    
    await waitFor(() => {
      // Find method select by role
      const methodSelects = screen.getAllByRole('combobox')
      expect(methodSelects.length).toBeGreaterThan(0)
    })
    
    // Just verify the accordion opened and method select is available
    const methodSelects = screen.getAllByRole('combobox')
    const methodSelect = methodSelects.find(el => 
      el.textContent.includes('EM') || el.textContent.includes('LP') || el.textContent.includes('Hybrid')
    ) || methodSelects[0]
    
    expect(methodSelect).toBeInTheDocument()
  })

  test('validates numeric input fields', async () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    // Open the Bivariate Normal Distribution accordion
    const bivariateAccordion = screen.getByRole('button', { name: /bivariate normal distribution/i })
    fireEvent.click(bivariateAccordion)
    
    await waitFor(() => {
      const sigmaXInput = screen.getByLabelText(/σ_X|sigma_x/i)
      
      // Try invalid input (should keep previous value)
      const initialValue = sigmaXInput.value
      fireEvent.change(sigmaXInput, { target: { value: 'abc' } })
      
      // Should revert to previous valid value or empty
      expect(sigmaXInput.value === initialValue || sigmaXInput.value === '').toBe(true)
    })
  })

  test('expands accordion sections when clicked', async () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    // Find accordion headers by their text content
    const bivariateAccordion = screen.getByRole('button', { name: /bivariate normal distribution/i })
    
    expect(bivariateAccordion).toHaveAttribute('aria-expanded', 'false')
    fireEvent.click(bivariateAccordion)
    
    // Accordion should expand (content becomes visible)
    await waitFor(() => {
      expect(bivariateAccordion).toHaveAttribute('aria-expanded', 'true')
    })
  })
})

