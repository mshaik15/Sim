# IK Benchmarking System - User Guide

## 🚀 Quick Start

### Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure
```
ik_benchmark/
├── solvers/               # IK solver implementations
│   ├── numerical/         # Numerical methods (Jacobian)
│   └── analytical/        # ML-based methods (Neural Network)
├── utils/                 # Shared utilities
├── data/                  # Test cases
├── visualizer/            # 3D/2D visualization tool
├── logs/                  # Benchmark results
└── main.py               # Main entry point
```

## 🎮 Running the System

### 1. Interactive Visualizer (Recommended for Testing)
```bash
python main.py --visualize
```

**Features:**
- 3D and 2D robot arm visualization
- Interactive controls for DOF, target pose, and solver selection
- Real-time solving and animation
- End-effector shape selection (sphere/hook)

### 2. Command Line Interface

#### Run Specific Solver
```bash
# Using main.py
python main.py --solver jacobian --dof 6 --test-file data/test_cases.json
python main.py --solver nn --dof 3 --test-file data/test_cases.json

# Direct module execution
python -m solvers.numerical.ik_solver_jacobian <DOF> <test_file>
python -m solvers.analytical.ik_solver_nn <DOF> <test_file>
```

#### Examples by DOF Category
```bash
# Underactuated (< 6 DOF)
python -m solvers.numerical.ik_solver_jacobian 3 data/test_cases.json
python -m solvers.numerical.ik_solver_jacobian 4 data/test_cases.json
python -m solvers.numerical.ik_solver_jacobian 5 data/test_cases.json

# Fully-actuated (6 DOF)
python -m solvers.numerical.ik_solver_jacobian 6 data/test_cases.json

# Overactuated (> 6 DOF)
python -m solvers.numerical.ik_solver_jacobian 7 data/test_cases.json
python -m solvers.numerical.ik_solver_jacobian 8 data/test_cases.json
```

## 📊 Understanding the Output

### Log Files Location
```bash
logs/ik_solver_<solver_name>_dof<n>_results.json
```

### Metrics Collected
- **Position Error**: Euclidean distance between target and actual position (meters)
- **Orientation Error**: Angular distance between target and actual orientation (radians)
- **Solve Time**: Computation time in milliseconds
- **Success Rate**: Percentage of solutions within error threshold (1cm)
- **DOF Classification**: underactuated/fully-actuated/overactuated

### View Results
```bash
# Windows
type logs\ik_solver_jacobian_dof6_results.json

# PowerShell
Get-Content logs\ik_solver_jacobian_dof6_results.json

# Open in editor
notepad logs\ik_solver_jacobian_dof6_results.json
```

## 🧪 Test Case Format

Test cases in `data/test_cases.json`:
```json
{
  "target_pose": [x, y, z, alpha, beta, gamma],  // Position + Euler angles
  "link_lengths": [L0, L1, ..., Ln],              // Link lengths in meters
  "dof": n                                        // Degrees of freedom
}
```

## 🛠️ Development Commands

### Run All Tests for a Solver
```bash
# Test Jacobian solver on all DOFs
for ($i=3; $i -le 8; $i++) {
    python -m solvers.numerical.ik_solver_jacobian $i data/test_cases.json
}
```

### Quick System Test
```bash
# Create and run a quick test
python -c "from solvers.numerical.ik_solver_jacobian import JacobianSolver; print('Import successful!')"
```

### Batch Processing
```bash
# Run both solvers on 6 DOF
python main.py --solver jacobian --dof 6 --test-file data/test_cases.json
python main.py --solver nn --dof 6 --test-file data/test_cases.json
```

## 🎯 Visualizer Controls

When running `python main.py --visualize`:

| Control | Function |
|---------|----------|
| **DOF Input** | Change robot degrees of freedom (1-10) |
| **Pose Inputs** | Set target position (X,Y,Z) and orientation (α,β,γ) |
| **Solver Radio** | Choose between Jacobian or Neural Network solver |
| **EE Shape Radio** | Select end-effector shape (sphere/hook) |
| **Solve & Animate** | Run solver and show animation path |
| **Home View** | Reset camera to default position |
| **Mouse Drag** | Rotate 3D view |
| **Scroll** | Zoom in/out |

## 📈 Analyzing Results

### Compare Solvers
```bash
# Run same test cases with different solvers
python main.py --solver jacobian --dof 6 --test-file data/test_cases.json
python main.py --solver nn --dof 6 --test-file data/test_cases.json

# Compare results
diff logs/ik_solver_jacobian_dof6_results.json logs/ik_solver_nn_dof6_results.json
```

### Extract Key Metrics
```powershell
# PowerShell script to extract success rates
$results = Get-Content logs\ik_solver_jacobian_dof6_results.json | ConvertFrom-Json
Write-Host "Success Rate: $($results.aggregate_metrics.success_rate * 100)%"
Write-Host "Mean Position Error: $($results.aggregate_metrics.mean_position_error)m"
Write-Host "Mean Solve Time: $($results.aggregate_metrics.mean_solve_time_ms)ms"
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the project root
   cd C:\path\to\ik_benchmark
   
   # Verify virtual environment is activated
   # You should see (venv) in your prompt
   ```

2. **No Display for Visualizer**
   ```bash
   # For remote/SSH sessions, use CLI mode instead
   python main.py --solver jacobian --dof 6
   ```

3. **Poor Solver Results**
   - Current implementations are placeholders
   - Need to implement actual IK algorithms in solver files

## 🔧 Extending the System

### Add New Solver
1. Create `solvers/my_solver.py`
2. Implement `solve()` method
3. Add CLI runner function
4. Update `solvers/__init__.py`

### Add New End-Effector
1. Create `visualizer/end_effectors/my_shape.py`
2. Implement `draw_3d()` and `draw_2d()` methods
3. Update `visualizer/end_effectors/__init__.py`
4. Add to visualizer's `ee_handlers` dict

### Add More Test Cases
Edit `data/test_cases.json` with new entries following the format above.

## 📝 Notes

- All angles are in radians
- All distances are in meters
- The system uses revolute joints only
- No joint limits or collision detection
- Forward kinematics alternates rotation axes (Z-Y-X pattern)