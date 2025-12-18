"""
Pydantic models for API request/response.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator


class BivariateNormalParams(BaseModel):
    """Parameters for bivariate normal distribution."""
    mu_x: float = Field(default=0.0, description="Mean of X")
    sigma_x: float = Field(default=0.8, gt=0, description="Standard deviation of X")
    mu_y: float = Field(default=0.0, description="Mean of Y")
    sigma_y: float = Field(default=1.6, gt=0, description="Standard deviation of Y")
    rho: float = Field(default=0.9, ge=-1.0, le=1.0, description="Correlation coefficient")


class GridParams(BaseModel):
    """Parameters for PDF grid."""
    z_range: List[float] = Field(default=[-6.0, 8.0], min_length=2, max_length=2, description="PDF range [z_min, z_max]")
    z_npoints: int = Field(default=2500, gt=0, le=100000, description="Number of grid points")

    @field_validator('z_range')
    @classmethod
    def validate_z_range(cls, v):
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("z_range must be [z_min, z_max] with z_min < z_max")
        return v


class EMMethodParams(BaseModel):
    """Parameters specific to EM method."""
    max_iter: int = Field(default=400, gt=0, description="Maximum iterations")
    tol: float = Field(default=1e-10, gt=0, description="Convergence tolerance")
    reg_var: float = Field(default=1e-6, gt=0, description="Variance regularization")
    n_init: int = Field(default=8, gt=0, description="Number of initializations")
    seed: int = Field(default=1, description="Random seed")
    init: Literal["quantile", "random", "qmi", "wqmi", "custom"] = Field(default="quantile", description="Initialization method")
    use_moment_matching: bool = Field(default=False, description="Use moment matching QP projection")
    qp_mode: Literal["hard", "soft"] = Field(default="hard", description="QP projection mode")
    soft_lambda: float = Field(default=1e4, gt=0, description="Soft constraint penalty coefficient")


class LPMethodParams(BaseModel):
    """Parameters specific to LP method."""
    L: int = Field(default=5, gt=0, description="Number of sigma levels per segment")
    objective_mode: Literal["pdf", "raw_moments"] = Field(default="pdf", description="Objective function mode")
    solver: Literal["highs", "interior-point", "revised simplex"] = Field(default="highs", description="LP solver")
    sigma_min_scale: float = Field(default=0.1, gt=0, description="Minimum sigma scale")
    sigma_max_scale: float = Field(default=3.0, gt=0, description="Maximum sigma scale")
    lambda_raw: Optional[List[float]] = Field(default=None, description="Raw moment error weights [λ1, λ2, λ3, λ4]")
    objective_form: Literal["A", "B"] = Field(default="A", description="Objective function form")
    pdf_tolerance: Optional[float] = Field(default=None, gt=0, description="PDF error tolerance")


class HybridMethodParams(BaseModel):
    """Parameters specific to Hybrid method."""
    dict_J: int = Field(default=None, gt=0, description="Dictionary size J (default: 4*K)")
    dict_L: int = Field(default=10, gt=0, description="Dictionary size L")
    objective_mode: Literal["raw_moments"] = Field(default="raw_moments", description="LP objective mode")
    mu_mode: Literal["uniform", "quantile"] = Field(default="quantile", description="Mean placement mode")
    tail_focus: Literal["none", "right", "left", "both"] = Field(default="none", description="Tail focus mode")
    tail_alpha: float = Field(default=1.0, ge=1.0, description="Tail emphasis strength")
    sigma_min_scale: float = Field(default=0.1, gt=0, description="Minimum sigma scale")
    sigma_max_scale: float = Field(default=3.0, gt=0, description="Maximum sigma scale")
    pdf_tolerance: Optional[float] = Field(default=None, gt=0, description="PDF error tolerance")
    lambda_raw: Optional[List[float]] = Field(default=None, description="Raw moment error weights")


class ComputeRequest(BaseModel):
    """Request model for GMM fitting computation."""
    bivariate_params: BivariateNormalParams = Field(default_factory=BivariateNormalParams)
    grid_params: GridParams = Field(default_factory=GridParams)
    K: int = Field(default=3, gt=0, le=50, description="Number of GMM components")
    method: Literal["em", "lp", "hybrid"] = Field(default="em", description="Fitting method")
    em_params: Optional[EMMethodParams] = Field(default=None, description="EM method parameters")
    lp_params: Optional[LPMethodParams] = Field(default=None, description="LP method parameters")
    hybrid_params: Optional[HybridMethodParams] = Field(default=None, description="Hybrid method parameters")
    show_grid_points: bool = Field(default=True, description="Show grid points in plot")
    max_grid_points_display: int = Field(default=200, gt=0, description="Max grid points to display")


class GMMComponent(BaseModel):
    """GMM component parameters."""
    pi: float = Field(description="Mixing weight")
    mu: float = Field(description="Mean")
    sigma: float = Field(description="Standard deviation")


class Statistics(BaseModel):
    """PDF statistics."""
    mean: float
    std: float
    skewness: float
    kurtosis: float


class ErrorMetrics(BaseModel):
    """Error metrics."""
    linf_pdf: float
    linf_cdf: float
    quantile_abs_errors: dict
    tail_l1_error: float


class ExecutionTime(BaseModel):
    """Execution time breakdown."""
    em_time: Optional[float] = None
    lp_time: Optional[float] = None
    qp_time: Optional[float] = None
    total_time: float


class ComputeResponse(BaseModel):
    """Response model for GMM fitting computation."""
    success: bool
    method: str
    z: List[float] = Field(description="Grid points")
    f_true: List[float] = Field(description="True PDF values")
    f_hat: List[float] = Field(description="GMM approximated PDF values")
    gmm_components: List[GMMComponent] = Field(description="GMM component parameters")
    statistics_true: Statistics = Field(description="True PDF statistics")
    statistics_hat: Statistics = Field(description="GMM PDF statistics")
    error_metrics: ErrorMetrics = Field(description="Error metrics")
    execution_time: ExecutionTime = Field(description="Execution time")
    log_likelihood: Optional[float] = None
    n_iterations: Optional[int] = None
    diagnostics: Optional[dict] = None
    message: Optional[str] = None
    plot_data_url: Optional[str] = Field(default=None, description="Base64 encoded plot image")

