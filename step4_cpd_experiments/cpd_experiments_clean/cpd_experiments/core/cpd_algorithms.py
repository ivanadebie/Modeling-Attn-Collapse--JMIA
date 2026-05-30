"""
Change Point Detection (CPD) algorithms.

This module provides wrappers around ruptures library algorithms
with safe error handling and standardized interfaces.
"""

import numpy as np
from typing import List, Optional, Callable
import warnings

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    warnings.warn("ruptures not installed. CPD algorithms unavailable.")


def safe_pelt(signal: np.ndarray,
              penalty: float,
              model: str = "l2",
              min_size: int = 2,
              jump: int = 1) -> List[int]:
    """
    Run PELT algorithm with safe error handling.
    
    Args:
        signal: Input signal (n_samples, n_features) or (n_samples,)
        penalty: Penalty parameter
        model: Cost model ("l2", "l1", "rbf", "linear", "normal", "ar")
        min_size: Minimum segment size
        jump: Subsample step
        
    Returns:
        List of change point indices (excludes final point n)
    """
    if not RUPTURES_AVAILABLE:
        return []
    
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    n = len(signal)
    if n < max(6, 2 * min_size):
        return []
    
    try:
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)
        bkps = algo.predict(pen=penalty)
        # Remove final point (always returned by ruptures)
        cps = [b for b in bkps if b < n]
        return cps
    except Exception:
        return []


def safe_kernel_cpd(signal: np.ndarray,
                    penalty: float,
                    kernel: str = "rbf",
                    min_size: int = 2,
                    jump: int = 1) -> List[int]:
    """
    Run Kernel CPD algorithm with safe error handling.
    
    Args:
        signal: Input signal (n_samples, n_features) or (n_samples,)
        penalty: Penalty parameter (scaled by variance)
        kernel: Kernel type ("rbf", "linear", "cosine")
        min_size: Minimum segment size
        jump: Subsample step
        
    Returns:
        List of change point indices (excludes final point n)
    """
    if not RUPTURES_AVAILABLE:
        return []
    
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    n = len(signal)
    if n < max(6, 2 * min_size):
        return []
    
    # Scale penalty by signal variance
    var = np.nanvar(signal)
    if var > 0:
        scaled_pen = penalty * var * n
    else:
        scaled_pen = penalty * n
    
    try:
        algo = rpt.KernelCPD(kernel=kernel, min_size=min_size, jump=jump).fit(signal)
        bkps = algo.predict(pen=scaled_pen)
        cps = [b for b in bkps if b < n]
        return cps
    except rpt.exceptions.BadSegmentationParameters:
        return []
    except Exception:
        return []


def safe_binseg(signal: np.ndarray,
                n_bkps: int,
                model: str = "l2",
                min_size: int = 2,
                jump: int = 1) -> List[int]:
    """
    Run Binary Segmentation algorithm with safe error handling.
    
    Args:
        signal: Input signal (n_samples, n_features) or (n_samples,)
        n_bkps: Number of breakpoints to detect
        model: Cost model
        min_size: Minimum segment size
        jump: Subsample step
        
    Returns:
        List of change point indices (excludes final point n)
    """
    if not RUPTURES_AVAILABLE:
        return []
    
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    n = len(signal)
    if n < max(6, 2 * min_size):
        return []
    
    try:
        algo = rpt.Binseg(model=model, min_size=min_size, jump=jump).fit(signal)
        bkps = algo.predict(n_bkps=n_bkps)
        cps = [b for b in bkps if b < n]
        return cps
    except Exception:
        return []


def safe_window(signal: np.ndarray,
                penalty: float,
                width: int = 10,
                model: str = "l2") -> List[int]:
    """
    Run Window-based CPD algorithm with safe error handling.
    
    Args:
        signal: Input signal (n_samples, n_features) or (n_samples,)
        penalty: Penalty parameter
        width: Window width
        model: Cost model
        
    Returns:
        List of change point indices (excludes final point n)
    """
    if not RUPTURES_AVAILABLE:
        return []
    
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    n = len(signal)
    if n < 2 * width:
        return []
    
    try:
        algo = rpt.Window(width=width, model=model).fit(signal)
        bkps = algo.predict(pen=penalty)
        cps = [b for b in bkps if b < n]
        return cps
    except Exception:
        return []


def create_cpd_function(algorithm: str = "pelt",
                        model: str = "l2",
                        min_size: int = 2,
                        jump: int = 1,
                        **kwargs) -> Callable[[np.ndarray, float], List[int]]:
    """
    Create a CPD function with specified parameters.
    
    Args:
        algorithm: Algorithm type ("pelt", "kernel", "binseg", "window")
        model: Cost model (for pelt, binseg) or kernel (for kernel)
        min_size: Minimum segment size
        jump: Subsample step
        **kwargs: Additional parameters
        
    Returns:
        Callable that takes (signal, penalty) and returns change points
    """
    if algorithm == "pelt":
        return lambda signal, pen: safe_pelt(signal, pen, model, min_size, jump)
    elif algorithm == "kernel":
        return lambda signal, pen: safe_kernel_cpd(signal, pen, model, min_size, jump)
    elif algorithm == "binseg":
        return lambda signal, n_bkps: safe_binseg(signal, n_bkps, model, min_size, jump)
    elif algorithm == "window":
        width = kwargs.get("width", 10)
        return lambda signal, pen: safe_window(signal, pen, width, model)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


class CPDRunner:
    """
    Unified CPD runner for experiments.
    """
    
    def __init__(self,
                 algorithm: str = "pelt",
                 model: str = "l2",
                 min_size: int = 2,
                 jump: int = 1,
                 **kwargs):
        """
        Initialize CPD runner.
        
        Args:
            algorithm: Algorithm type
            model: Cost model or kernel type
            min_size: Minimum segment size
            jump: Subsample step
            **kwargs: Additional parameters
        """
        self.algorithm = algorithm
        self.model = model
        self.min_size = min_size
        self.jump = jump
        self.kwargs = kwargs
        
        self._cpd_func = create_cpd_function(
            algorithm, model, min_size, jump, **kwargs
        )
    
    def detect(self, signal: np.ndarray, penalty: float) -> List[int]:
        """
        Detect change points in signal.
        
        Args:
            signal: Input signal
            penalty: Penalty parameter
            
        Returns:
            List of change point indices
        """
        return self._cpd_func(signal, penalty)
    
    def detect_first(self, signal: np.ndarray, penalty: float) -> Optional[int]:
        """
        Detect first change point in signal.
        
        Args:
            signal: Input signal
            penalty: Penalty parameter
            
        Returns:
            First change point index or None
        """
        cps = self.detect(signal, penalty)
        return min(cps) if cps else None
    
    def __repr__(self) -> str:
        return f"CPDRunner(algorithm={self.algorithm}, model={self.model})"
