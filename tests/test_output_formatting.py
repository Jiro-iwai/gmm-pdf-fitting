"""
Tests for output formatting functions.
"""
import sys
import numpy as np
from io import StringIO
import pytest
from gmm_fitting import (
    print_section_header,
    print_subsection_header,
    print_em_results,
    print_execution_time,
    print_moment_matching_info,
    calc_relative_error,
    print_statistics_comparison,
    print_gmm_parameters,
    print_plot_output,
    GMM1DParams
)


class TestOutputFormatting:
    """Tests for output formatting functions."""
    
    def test_print_section_header(self):
        """Test section header printing."""
        output = StringIO()
        sys.stdout = output
        
        try:
            print_section_header("TEST SECTION")
            result = output.getvalue()
            
            assert "TEST SECTION" in result
            assert "=" in result
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_subsection_header(self):
        """Test subsection header printing."""
        output = StringIO()
        sys.stdout = output
        
        try:
            print_subsection_header("TEST SUBSECTION")
            result = output.getvalue()
            
            assert "TEST SUBSECTION" in result
            assert "-" in result
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_em_results(self):
        """Test EM results printing."""
        output = StringIO()
        sys.stdout = output
        
        try:
            print_em_results(-0.5, 10, 100)
            result = output.getvalue()
            
            # Note: print_em_results no longer prints section header
            assert "-0.5" in result
            assert "10" in result
            assert "100" in result
            assert "Convergence: Yes" in result
            assert "Best weighted log-likelihood" in result
            assert "Iterations" in result
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_execution_time_without_qp(self):
        """Test execution time printing without QP."""
        output = StringIO()
        sys.stdout = output
        
        try:
            print_execution_time(0.123, use_moment_matching=False)
            result = output.getvalue()
            
            assert "EXECUTION TIME" in result
            assert "0.123" in result
            assert "Total:" in result
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_execution_time_with_qp(self):
        """Test execution time printing with QP."""
        output = StringIO()
        sys.stdout = output
        
        try:
            print_execution_time(0.123, qp_time=0.045, total_time=0.168, use_moment_matching=True)
            result = output.getvalue()
            
            assert "EXECUTION TIME" in result
            assert "0.123" in result
            assert "0.045" in result
            assert "0.168" in result
            assert "QP projection" in result
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_moment_matching_info_hard(self):
        """Test moment matching info printing for hard constraints."""
        qp_info = {
            'method': 'hard',
            'constraint_error': 1e-10,
            'moment_errors': [0.0, 1e-10, -1e-10, 0.0, 0.0]
        }
        
        output = StringIO()
        sys.stdout = output
        
        try:
            print_moment_matching_info(qp_info)
            result = output.getvalue()
            
            assert "MOMENT MATCHING QP PROJECTION" in result
            assert "HARD" in result
            assert "moments matched exactly" in result
            assert "M0" in result
            assert "M1" in result
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_moment_matching_info_soft(self):
        """Test moment matching info printing for soft constraints."""
        qp_info = {
            'method': 'soft',
            'constraint_error': 1e-4,
            'moment_errors': [0.0, 1e-4, -1e-4, 0.0, 0.0]
        }
        
        output = StringIO()
        sys.stdout = output
        
        try:
            print_moment_matching_info(qp_info)
            result = output.getvalue()
            
            assert "SOFT" in result
            assert "moments approximately matched" in result
        finally:
            sys.stdout = sys.__stdout__
    
    def test_calc_relative_error(self):
        """Test relative error calculation."""
        # Normal case
        error = calc_relative_error(100.0, 105.0)
        assert np.isclose(error, 5.0)  # 5% error
        
        # Zero true value
        error = calc_relative_error(0.0, 1.0)
        assert error == float('inf')
        
        # Both zero
        error = calc_relative_error(0.0, 0.0)
        assert error == 0.0
    
    def test_print_statistics_comparison(self):
        """Test statistics comparison printing."""
        stats_true = {
            'mean': 0.5,
            'std': 1.0,
            'skewness': 0.0,
            'kurtosis': 0.0
        }
        stats_hat = {
            'mean': 0.51,
            'std': 0.99,
            'skewness': 0.01,
            'kurtosis': 0.01
        }
        
        output = StringIO()
        sys.stdout = output
        
        try:
            print_statistics_comparison(stats_true, stats_hat)
            result = output.getvalue()
            
            assert "PDF STATISTICS COMPARISON" in result
            assert "Mean" in result
            assert "Std Dev" in result
            assert "Skewness" in result
            assert "Kurtosis" in result
            assert "0.5" in result
            assert "0.51" in result
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_gmm_parameters(self):
        """Test GMM parameters printing."""
        params = GMM1DParams(
            pi=np.array([0.3, 0.7]),
            mu=np.array([-1.0, 1.0]),
            var=np.array([0.5, 0.5])
        )
        
        output = StringIO()
        sys.stdout = output
        
        try:
            print_gmm_parameters(params)
            result = output.getvalue()
            
            assert "GMM PARAMETERS" in result
            assert "Number of components: 2" in result
            assert "Component 1" in result
            assert "Component 2" in result
            assert "π=" in result
            assert "μ=" in result
            assert "σ=" in result
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_plot_output(self):
        """Test plot output printing."""
        output = StringIO()
        sys.stdout = output
        
        try:
            print_plot_output("test_output")
            result = output.getvalue()
            
            assert "PLOT OUTPUT" in result
            assert "test_output.png" in result
        finally:
            sys.stdout = sys.__stdout__

