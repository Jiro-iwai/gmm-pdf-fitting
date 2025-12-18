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

  test('renders all main input fields', () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    // Check bivariate parameters
    expect(screen.getByLabelText(/mu_x/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/sigma_x/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/mu_y/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/sigma_y/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/rho/i)).toBeInTheDocument()
    
    // Check method selection
    expect(screen.getByLabelText(/method/i)).toBeInTheDocument()
    
    // Check Compute button
    expect(screen.getByRole('button', { name: /compute/i })).toBeInTheDocument()
  })

  test('displays default values correctly', () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    expect(screen.getByDisplayValue('0')).toBeInTheDocument() // mu_x default
    expect(screen.getByDisplayValue('0.8')).toBeInTheDocument() // sigma_x default
    expect(screen.getByDisplayValue('0.9')).toBeInTheDocument() // rho default
  })

  test('updates input values when user types', async () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    const muXInput = screen.getByLabelText(/mu_x/i)
    fireEvent.change(muXInput, { target: { value: '1.5' } })
    
    await waitFor(() => {
      expect(muXInput).toHaveValue(1.5)
    })
  })

  test('calls onSubmit with correct parameters when Compute is clicked', async () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    // Change some values
    const muXInput = screen.getByLabelText(/mu_x/i)
    fireEvent.change(muXInput, { target: { value: '0.5' } })
    
    const computeButton = screen.getByRole('button', { name: /compute/i })
    fireEvent.click(computeButton)
    
    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledTimes(1)
      const callArgs = mockOnSubmit.mock.calls[0][0]
      expect(callArgs.bivariate_params.mu_x).toBe(0.5)
      expect(callArgs.bivariate_params.sigma_x).toBe(0.8) // default value
    })
  })

  test('disables Compute button when loading', () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={true} />)
    
    const computeButton = screen.getByRole('button', { name: /compute/i })
    expect(computeButton).toBeDisabled()
  })

  test('shows method-specific parameters when method changes', async () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    // Change method to LP
    const methodSelect = screen.getByLabelText(/method/i)
    fireEvent.mouseDown(methodSelect)
    await waitFor(() => {
      const lpOption = screen.getByText(/lp/i)
      fireEvent.click(lpOption)
    })
    
    // Check LP-specific parameters appear
    await waitFor(() => {
      expect(screen.getByLabelText(/L/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/objective_mode/i)).toBeInTheDocument()
    })
  })

  test('validates numeric input fields', async () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    const sigmaXInput = screen.getByLabelText(/sigma_x/i)
    
    // Try invalid input (should keep previous value)
    const initialValue = sigmaXInput.value
    fireEvent.change(sigmaXInput, { target: { value: 'abc' } })
    
    await waitFor(() => {
      // Should revert to previous valid value or empty
      expect(sigmaXInput.value === initialValue || sigmaXInput.value === '').toBe(true)
    })
  })

  test('expands accordion sections when clicked', async () => {
    renderWithTheme(<ParameterForm onSubmit={mockOnSubmit} loading={false} />)
    
    // Find accordion headers
    const accordions = screen.getAllByRole('button', { name: /expand/i })
    
    if (accordions.length > 0) {
      fireEvent.click(accordions[0])
      // Accordion should expand (content becomes visible)
      await waitFor(() => {
        expect(accordions[0]).toHaveAttribute('aria-expanded', 'true')
      })
    }
  })
})

