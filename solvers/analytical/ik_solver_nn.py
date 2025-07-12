"""
Neural Network-based IK Solver
"""
import numpy as np
import torch
import torch.nn as nn
import json
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.kinematics import forward_kinematics
from utils.metrics import compute_metrics
from utils.logging import log_results

class IKNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class NeuralNetworkSolver:
    def __init__(self):
        self.name = "nn"
        self.model = None
        
    def solve(self, target_pose, link_lengths):
        """
        Solve IK using Neural Network
        
        Args:
            target_pose: [x, y, z, alpha, beta, gamma]
            link_lengths: [L0, L1, ..., Ln]
            
        Returns:
            joint_angles: List of joint angles
        """
        dof = len(link_lengths)
        
        # TODO: Load appropriate model based on DOF
        # For now, return dummy solution
        if self.model is None:
            # Create dummy model
            self.model = IKNet(6 + dof, dof)  # pose + link_lengths -> joint_angles
            
        # Prepare input
        input_tensor = torch.FloatTensor(target_pose + link_lengths)
        
        # Predict
        with torch.no_grad():
            joint_angles = self.model(input_tensor).numpy()
            
        return joint_angles.tolist()

def run_solver(dof, test_file):
    """CLI runner for the solver"""
    solver = NeuralNetworkSolver()
    
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