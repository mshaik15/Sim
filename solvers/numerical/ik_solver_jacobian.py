"""
Jacobian-based IK Solver
"""
import numpy as np
import json
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.kinematics import forward_kinematics
from utils.metrics import compute_metrics
from utils.logging import log_results

class JacobianSolver:
    def __init__(self):
        self.name = "jacobian"
        self.max_iterations = 100
        self.tolerance = 1e-3
        
    def solve(self, target_pose, link_lengths):
        """
        Solve IK using Jacobian pseudo-inverse method
        
        Args:
            target_pose: [x, y, z, alpha, beta, gamma]
            link_lengths: [L0, L1, ..., Ln]
            
        Returns:
            joint_angles: List of joint angles
        """
        dof = len(link_lengths)
        
        # Initialize joint angles
        joint_angles = np.zeros(dof)
        
        # TODO: Implement actual Jacobian IK solver
        # For now, return dummy solution
        for i in range(self.max_iterations):
            # Compute current pose
            current_pose = forward_kinematics(joint_angles, link_lengths)
            
            # Compute error
            error = np.array(target_pose) - np.array(current_pose)
            
            if np.linalg.norm(error[:3]) < self.tolerance:
                break
                
            # Dummy update (replace with actual Jacobian computation)
            joint_angles += 0.01 * np.random.randn(dof)
            
        return joint_angles.tolist()

def run_solver(dof, test_file):
    """CLI runner for the solver"""
    solver = JacobianSolver()
    
    # Load test cases
    with open(test_file, 'r') as f:
        test_cases = json.load(f)
    
    # Filter by DOF
    filtered_cases = [tc for tc in test_cases if tc['dof'] == dof]
    
    results = []
    for i, test_case in enumerate(filtered_cases):
        print(f"Processing test case {i+1}/{len(filtered_cases)}")
        
        start_time = time.time()
        joint_angles = solver.solve(test_case['target_pose'], test_case['link_lengths'])
        solve_time = (time.time() - start_time) * 1000  # ms
        
        # Apply FK and compute metrics
        actual_pose = forward_kinematics(joint_angles, test_case['link_lengths'])
        metrics = compute_metrics(test_case['target_pose'], actual_pose, solve_time, dof)
        
        result = {
            'test_case_id': i,
            'joint_angles': joint_angles,
            'metrics': metrics
        }
        results.append(result)
    
    # Log results
    log_results(solver.name, dof, results)
    print(f"Results saved to logs/ik_solver_{solver.name}_dof{dof}_results.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dof', type=int)
    parser.add_argument('test_file', type=str)
    args = parser.parse_args()
    
    run_solver(args.dof, args.test_file)