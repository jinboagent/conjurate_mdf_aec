"""
Layer 5: Reporting — Table Formatting

Markdown table generation for comparison results.
"""

from typing import Dict
from core.runner import TestResult
from core.config import TestConfig


def format_comparison_table(results: Dict[str, TestResult],
                            config: TestConfig = None) -> str:
    """
    Format comparison results as a markdown table.

    Parameters
    ----------
    results : dict
        Algorithm name → TestResult
    config : TestConfig, optional
        For showing targets

    Returns
    -------
    str — Markdown formatted table
    """
    lines = []
    lines.append("| Metric | " + " | ".join(results.keys()) + " |")
    lines.append("|--------|" + "|".join(["-------"] * len(results)) + "|")

    # Signal metrics
    row_erle = "| ERLE (dB) |"
    row_rms = "| Output RMS |"
    for name, r in results.items():
        row_erle += f" {r.erle_db:.2f} |"
        row_rms += f" {r.output_rms:.6f} |"
    lines.extend([row_erle, row_rms])

    # Echo path metrics (if any result has them)
    has_ep = any(r.echo_path_metrics for r in results.values())
    if has_ep:
        for metric_key, metric_label in [
            ('nmse_db', 'NMSE (dB)'),
            ('correlation', 'Correlation'),
            ('coherence', 'Coherence'),
            ('delay_error_samples', 'Delay error (samples)'),
            ('amplitude_error_db', 'Amplitude error (dB)'),
        ]:
            row = f"| {metric_label} |"
            for name, r in results.items():
                if r.echo_path_metrics and metric_key in r.echo_path_metrics:
                    val = r.echo_path_metrics[metric_key]
                    if isinstance(val, float):
                        row += f" {val:.4f} |"
                    else:
                        row += f" {val} |"
                else:
                    row += " N/A |"
            lines.append(row)

    # Pass/Fail
    row_pass = "| RESULT |"
    for name, r in results.items():
        row_pass += f" **{'PASS' if r.passed else 'FAIL'}** |"
    lines.append(row_pass)

    # Targets (if config provided)
    if config:
        lines.append("")
        lines.append("Targets:")
        lines.append(f"  ERLE ≥ {config.erle_target_db} dB, "
                      f"Correlation ≥ {config.correlation_target}, "
                      f"Delay error = {config.delay_error_target}")

    return '\n'.join(lines)


def format_metric_row(result: TestResult) -> str:
    """Single-line metric summary."""
    parts = [f"ERLE={result.erle_db:.2f}dB"]
    if result.echo_path_metrics:
        ep = result.echo_path_metrics
        if 'correlation' in ep:
            parts.append(f"corr={ep['correlation']:.3f}")
        if 'delay_error_samples' in ep:
            parts.append(f"delay_err={ep['delay_error_samples']}")
    parts.append("PASS" if result.passed else "FAIL")
    return " | ".join(parts)
