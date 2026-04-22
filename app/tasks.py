"""
Task configuration utilities for CrossMill-MegaForge.
Exposes Easy / Medium / Hard task metadata.
"""

from app.config import TASK_CONFIG


def list_tasks() -> list:
    """Return list of available task IDs."""
    return list(TASK_CONFIG.keys())


def get_task(task_id: str) -> dict:
    """
    Get task configuration by ID.

    Parameters
    ----------
    task_id : str
        One of 'easy', 'medium', 'hard'.

    Returns
    -------
    dict
        Copy of TASK_CONFIG[task_id].
    """
    if task_id not in TASK_CONFIG:
        raise ValueError(
            f"Unknown task '{task_id}'. Available: {list_tasks()}")
    return dict(TASK_CONFIG[task_id])


def grader_target(task_id: str) -> float:
    """
    Get grader target score (0.0–1.0) for a task.

    Parameters
    ----------
    task_id : str
        One of 'easy', 'medium', 'hard'.

    Returns
    -------
    float
        The 'grader_target' value from TASK_CONFIG.
    """
    return get_task(task_id)['grader_target']


# --------------------------------------------------------------------------- #
# Sanity check                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    print("Available tasks:")
    print()
    for tid in list_tasks():
        cfg = get_task(tid)
        print(f"  {tid:8s}  len={cfg['episode_length']:5d}  "
              f"target={cfg['grader_target']:.2f}")
