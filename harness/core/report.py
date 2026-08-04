

"""
Test reporting utilities.

Provides structured report generation for test results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import json

from .runner import TestResult


@dataclass
class TestReport:
    """
    Structured report for test results.
    
    Contains all information needed to reproduce and verify a test run.
    """
    algorithm_name: str
    timestamp: str
    metrics: Dict[str, float]
    passed: bool = True
    config_hash: str = ""
    output_stats: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    
    @classmethod
    def from_result(cls, result: TestResult, passed: bool = True) -> 'TestReport':
        """
        Create a TestReport from a TestResult.
        
        Parameters
        ----------
        result : TestResult
            Test result to convert
        passed : bool
            Whether the test passed
            
        Returns
        -------
        TestReport
        """
        # Calculate output statistics
        output = result.output
        output_stats = {
            'max': float(np.max(np.abs(output))),
            'min': float(np.min(np.abs(output))),
            'mean': float(np.mean(np.abs(output))),
            'std': float(np.std(output)),
            'nonzero_count': int(np.count_nonzero(output)),
        }
        
        # Generate config hash if config exists
        config_hash = ""
        if result.config is not None:
            config_str = json.dumps({
                'fft_size': result.config.fft_size,
                'overlap': result.config.overlap,
                'transient_length': result.config.transient_length,
            }, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return cls(
            algorithm_name=result.algorithm_name,
            timestamp=datetime.now().isoformat(),
            metrics=result.metrics,
            passed=passed,
            config_hash=config_hash,
            output_stats=output_stats,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'algorithm_name': self.algorithm_name,
            'timestamp': self.timestamp,
            'metrics': self.metrics,
            'passed': self.passed,
            'config_hash': self.config_hash,
            'output_stats': self.output_stats,
            'notes': self.notes,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def __str__(self) -> str:
        lines = [
            f"Test Report: {self.algorithm_name}",
            f"  Timestamp: {self.timestamp}",
            f"  Passed: {self.passed}",
            f"  Config Hash: {self.config_hash}",
            f"  Metrics:",
        ]
        for key, value in self.metrics.items():
            lines.append(f"    {key}: {value:.6f}")
        if self.output_stats:
            lines.append(f"  Output Stats:")
            for key, value in self.output_stats.items():
                lines.append(f"    {key}: {value:.6f}")
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)


@dataclass
class ComparisonReport:
    """
    Report comparing two algorithm outputs.
    """
    algorithm1: str
    algorithm2: str
    timestamp: str
    max_difference: float
    correlation: float
    erle_difference: float
    threshold: float
    passed: bool
    notes: str = ""
    
    @classmethod
    def from_comparison(
        cls,
        result1: TestResult,
        result2: TestResult,
        max_diff: float,
        correlation: float,
        threshold: float
    ) -> 'ComparisonReport':
        """
        Create comparison report from two test results.
        
        Parameters
        ----------
        result1 : TestResult
            First algorithm result
        result2 : TestResult
            Second algorithm result
        max_diff : float
            Maximum absolute difference between outputs
        correlation : float
            Correlation coefficient between outputs
        threshold : float
            Pass/fail threshold for max_diff
            
        Returns
        -------
        ComparisonReport
        """
        erle_diff = abs(result1.metrics.get('erle', 0) - result2.metrics.get('erle', 0))
        passed = max_diff < threshold
        
        return cls(
            algorithm1=result1.algorithm_name,
            algorithm2=result2.algorithm_name,
            timestamp=datetime.now().isoformat(),
            max_difference=max_diff,
            correlation=correlation,
            erle_difference=erle_diff,
            threshold=threshold,
            passed=passed,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'algorithm1': self.algorithm1,
            'algorithm2': self.algorithm2,
            'timestamp': self.timestamp,
            'max_difference': self.max_difference,
            'correlation': self.correlation,
            'erle_difference': self.erle_difference,
            'threshold': self.threshold,
            'passed': self.passed,
            'notes': self.notes,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines = [
            f"Comparison Report: {self.algorithm1} vs {self.algorithm2}",
            f"  Timestamp: {self.timestamp}",
            f"  Status: {status}",
            f"  Max Difference: {self.max_difference:.2e}",
            f"  Correlation: {self.correlation:.10f}",
            f"  ERLE Difference: {self.erle_difference:.4f} dB",
            f"  Threshold: {self.threshold:.2e}",
        ]
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)


def generate_comparison_summary(
    results: Dict[str, TestResult],
    threshold: float = 1e-6
) -> str:
    """
    Generate a summary comparison of multiple algorithm results.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping algorithm names to TestResult objects
    threshold : float
        Pass/fail threshold for comparisons
        
    Returns
    -------
    str
        Formatted summary string
    """
    import numpy as np
    
    names = list(results.keys())
    if len(names) < 2:
        return "Need at least 2 algorithms for comparison"
    
    lines = ["", "=" * 60, "Algorithm Comparison Summary", "=" * 60, ""]
    
    # ERLE comparison table
    lines.append("ERLE (dB):")
    lines.append("-" * 40)
    for name in names:
        erle = results[name].metrics.get('erle', 0)
        lines.append(f"  {name:30s}: {erle:8.2f}")
    lines.append("")
    
    # Pairwise comparisons
    lines.append("Pairwise Comparisons:")
    lines.append("-" * 40)
    
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            sig1 = results[name1].output
            sig2 = results[name2].output
            
            min_len = min(len(sig1), len(sig2))
            sig1 = sig1[:min_len]
            sig2 = sig2[:min_len]
            
            max_diff = np.max(np.abs(sig1 - sig2))
            correlation = np.corrcoef(sig1, sig2)[0, 1]
            
            status = "✓" if max_diff < threshold else "✗"
            lines.append(f"  {name1} vs {name2}:")
            lines.append(f"    Max Diff: {max_diff:.2e} {status}")
            lines.append(f"    Correlation: {correlation:.10f}")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# Import numpy at module level for type hints
import numpy as np
