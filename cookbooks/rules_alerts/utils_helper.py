"""
Helper utilities for scoring demonstrations
This implements lightweight versions of the scorer utilities from judgeval.
"""

import sys
from contextlib import contextmanager
from typing import List, Any, Optional

class ProgressIndicator:
    """Simple progress indicator for demo purposes"""
    def __init__(self, message):
        self.message = message
        
    def start(self):
        """Start the progress indicator"""
        print(f"⏳ {self.message}")
        
    def stop(self):
        """Stop the progress indicator"""
        print(f"✅ {self.message} - Complete")


@contextmanager
def scorer_progress_meter(
    scorer,
    async_mode: Optional[bool] = None,
    display_meter: bool = True,
    **kwargs
):
    """
    Simplified context manager for scoring progress indication.
    
    Args:
        scorer: The scorer object
        async_mode: Whether async mode is being used
        display_meter: Whether to display progress
    """
    if display_meter:
        scorer_name = getattr(scorer, "__name__", type(scorer).__name__)
        model_name = getattr(scorer, "evaluation_model", "unknown model")
        message = f"Running {scorer_name} scorer (using {model_name})"
        
        indicator = ProgressIndicator(message)
        indicator.start()
        try:
            yield
        finally:
            indicator.stop()
    else:
        yield


def create_verbose_logs(scorer, steps: List[str]) -> str:
    """
    Creates verbose logs string from steps.
    
    Args:
        scorer: The scorer object
        steps: List of step descriptions
        
    Returns:
        Formatted string with all steps
    """
    verbose_logs = "\n\n".join(steps[:-1]) if len(steps) > 1 else ""
    
    # If verbose mode is enabled, print the logs
    if getattr(scorer, "verbose_mode", False):
        scorer_name = getattr(scorer, "__name__", type(scorer).__name__)
        print("=" * 50)
        print(f"{scorer_name} Verbose Logs")
        print("=" * 50)
        print(verbose_logs)
        if steps:
            print(f"\n{steps[-1]}")
        print("=" * 50)
        
    return verbose_logs 