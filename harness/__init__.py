

"""
Reusable test harness for adaptive filter algorithms.

This harness provides:
- Standard interfaces for adaptive filters
- Common test execution engine
- Metrics calculation and reporting
- Output comparison utilities
"""

from .core.interfaces import AdaptiveFilter
from .core.runner import run_test, TestResult
from .core.report import TestReport
from .metrics.comparison import compare_outputs, ComparisonResult, calculate_erle

__version__ = "1.0.0"
__all__ = [
    'AdaptiveFilter',
    'run_test',
    'TestResult',
    'TestReport',
    'compare_outputs',
    'ComparisonResult',
    'calculate_erle',
]
