#!/usr/bin/env python3
"""
IK Benchmarking System - Main Entry Point
"""
import argparse
import json
import sys
from pathlib import Path
from importlib import import_module

def main():
    parser = argparse.ArgumentParser(description='IK Benchmarking System')
    parser.add_argument('--solver', type=str, help='Solver name (e.g., jacobian, nn)')
    parser.add_argument('--dof', type=int, help='Degrees of freedom')
    parser.add_argument('--test-file', type=str, default='data/test_cases.json', 
                        help='Path to test cases JSON')
    parser.add_argument('--visualize', action='store_true', help='Launch visualizer')
    
    args = parser.parse_args()
    
    if args.visualize:
        from visualizer.visualizer import IKVisualizer
        app = IKVisualizer()
        app.run()
    elif args.solver:
        # Run solver from CLI
        solver_module = import_module(f'solvers.ik_solver_{args.solver}')
        solver_module.run_solver(args.dof, args.test_file)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()