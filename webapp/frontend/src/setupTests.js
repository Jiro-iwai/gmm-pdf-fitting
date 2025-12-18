import '@testing-library/jest-dom'

// Mock window.URL.createObjectURL for tests
global.URL.createObjectURL = jest.fn(() => 'mocked-url')
global.URL.revokeObjectURL = jest.fn()
