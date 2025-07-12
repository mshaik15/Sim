"""
Logging utilities for benchmark results
"""
import json
import os
from datetime import datetime
from pathlib import Path

def log_results(solver_name, dof, results):
    """
    Log results to JSON file
    
    Args:
        solver_name: Name of the solver
        dof: Degrees of freedom
        results: List of result dictionaries
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Prepare log data
    from utils.metrics import aggregate_metrics
    log_data = {
        "solver": solver_name,
        "dof": dof,
        "timestamp": datetime.now().isoformat(),
        "aggregate_metrics": aggregate_metrics(results),
        "results": results
    }
    
    # Save to file
    filename = log_dir / f"ik_solver_{solver_name}_dof{dof}_results.json"
    with open(filename, 'w') as f:
        json.dump(log_data, f, indent=2)