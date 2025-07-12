"""
Metrics computation for IK solutions
"""
import numpy as np

def compute_metrics(target_pose, actual_pose, solve_time, dof):
    """
    Compute all metrics for IK solution
    
    Args:
        target_pose: [x, y, z, alpha, beta, gamma]
        actual_pose: [x, y, z, alpha, beta, gamma]
        solve_time: Time in milliseconds
        dof: Degrees of freedom
        
    Returns:
        dict: Metrics dictionary
    """
    target_pos = np.array(target_pose[:3])
    actual_pos = np.array(actual_pose[:3])
    target_rot = np.array(target_pose[3:])
    actual_rot = np.array(actual_pose[3:])
    
    # Position error
    position_error = np.linalg.norm(target_pos - actual_pos)
    
    # Orientation error (simple angular distance)
    orientation_error = np.linalg.norm(target_rot - actual_rot)
    
    # Success threshold
    success_threshold = 0.01  # 1cm position error
    success = position_error < success_threshold
    
    # DOF classification
    if dof < 6:
        dof_class = "underactuated"
    elif dof == 6:
        dof_class = "fully-actuated"
    else:
        dof_class = "overactuated"
    
    return {
        "position_error": float(position_error),
        "orientation_error": float(orientation_error),
        "solve_time_ms": float(solve_time),
        "success": success,
        "dof_classification": dof_class
    }

def aggregate_metrics(results):
    """
    Compute aggregate statistics from results
    
    Args:
        results: List of result dictionaries
        
    Returns:
        dict: Aggregated metrics
    """
    if not results:
        return {}
        
    metrics = [r['metrics'] for r in results]
    
    return {
        "mean_position_error": np.mean([m['position_error'] for m in metrics]),
        "mean_orientation_error": np.mean([m['orientation_error'] for m in metrics]),
        "mean_solve_time_ms": np.mean([m['solve_time_ms'] for m in metrics]),
        "success_rate": np.mean([m['success'] for m in metrics]),
        "total_test_cases": len(results)
    }