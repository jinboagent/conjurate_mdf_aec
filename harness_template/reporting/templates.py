"""
Layer 5: Reporting — Templates

Markdown templates for debug reports and status reports.
"""


def debug_report_template(
    title: str,
    problem: str,
    investigation_steps: list,
    root_cause: str,
    fix: str,
    verification: str,
    lessons: list = None,
) -> str:
    """
    Generate a debug report markdown document.

    Parameters
    ----------
    title : str
        Issue title
    problem : str
        Description of what failed
    investigation_steps : list of str
        Steps taken to investigate
    root_cause : str
        The actual bug
    fix : str
        What was changed
    verification : str
        Test results after fix
    lessons : list of str, optional
        Lessons learned

    Returns
    -------
    str — Markdown document
    """
    sections = [
        f"# Debug Report: {title}",
        "",
        "## Problem",
        problem,
        "",
        "## Investigation",
    ]

    for i, step in enumerate(investigation_steps, 1):
        sections.append(f"{i}. {step}")

    sections.extend([
        "",
        "## Root Cause",
        root_cause,
        "",
        "## Fix",
        fix,
        "",
        "## Verification",
        verification,
    ])

    if lessons:
        sections.extend(["", "## Lessons Learned"])
        for lesson in lessons:
            sections.append(f"- {lesson}")

    return '\n'.join(sections)


def status_report_template(
    project: str,
    timestamp: str,
    overview: str,
    completed: list,
    in_progress: list,
    decisions: list = None,
    problems: list = None,
    next_steps: list = None,
    file_index: list = None,
) -> str:
    """
    Generate a status report markdown document (handoff document).

    Parameters
    ----------
    project : str
    timestamp : str
    overview : str
    completed : list of str
    in_progress : list of str
    decisions : list of str
    problems : list of str
    next_steps : list of str
    file_index : list of str

    Returns
    -------
    str — Markdown document
    """
    sections = [
        f"# Status Report: {project}",
        f"**Timestamp:** {timestamp}",
        "",
        "## Overview",
        overview,
        "",
        "## Completed",
    ]

    for item in completed:
        sections.append(f"- [x] {item}")

    sections.extend(["", "## In Progress"])
    for item in in_progress:
        sections.append(f"- [ ] {item}")

    if decisions:
        sections.extend(["", "## Decisions Made"])
        for d in decisions:
            sections.append(f"- {d}")

    if problems:
        sections.extend(["", "## Problems Encountered"])
        for p in problems:
            sections.append(f"- {p}")

    if next_steps:
        sections.extend(["", "## Next Steps"])
        for n in next_steps:
            sections.append(f"1. {n}")

    if file_index:
        sections.extend(["", "## File Index"])
        for f in file_index:
            sections.append(f"- {f}")

    return '\n'.join(sections)
