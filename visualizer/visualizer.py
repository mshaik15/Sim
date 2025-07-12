"""
3D Visualization Tool for IK Benchmarking
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.kinematics import forward_kinematics
from visualizer.end_effectors import HookEE, SphereEE

class IKVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('IK Benchmarking Visualizer', fontsize=16)
        
        # 3D plot
        self.ax3d = self.fig.add_subplot(121, projection='3d')
        self.ax3d.set_xlabel('X')
        self.ax3d.set_ylabel('Y')
        self.ax3d.set_zlabel('Z')
        self.ax3d.set_title('3D Robot View')
        
        # XY slice plot
        self.ax2d = self.fig.add_subplot(122)
        self.ax2d.set_xlabel('X')
        self.ax2d.set_ylabel('Y')
        self.ax2d.set_title('XY Plane View')
        self.ax2d.grid(True)
        self.ax2d.set_aspect('equal')
        
        # State
        self.dof = 6
        self.link_lengths = [0.1] * self.dof
        self.joint_angles = [0.0] * self.dof
        self.target_pose = [0.3, 0.2, 0.4, 0.0, 0.0, 0.0]
        self.solver_name = 'jacobian'
        self.ee_shape = 'sphere'
        self.home_view = None
        
        # Animation
        self.animation_path = []
        
        # End effector handlers
        self.ee_handlers = {
            'sphere': SphereEE(),
            'hook': HookEE()
        }
        
        self._setup_controls()
        self._update_plot()
        
    def _setup_controls(self):
        """Setup UI controls"""
        # Control panel area
        control_height = 0.35
        
        # Solver selection
        ax_solver = plt.axes([0.05, control_height - 0.05, 0.1, 0.15])
        self.radio_solver = RadioButtons(ax_solver, ('jacobian', 'nn'))
        self.radio_solver.on_clicked(self._on_solver_change)
        
        # DOF input
        ax_dof = plt.axes([0.2, control_height - 0.03, 0.1, 0.03])
        self.text_dof = TextBox(ax_dof, 'DOF:', initial=str(self.dof))
        self.text_dof.on_submit(self._on_dof_change)
        
        # Pose inputs
        pose_labels = ['X', 'Y', 'Z', 'α', 'β', 'γ']
        self.text_pose = []
        for i, label in enumerate(pose_labels):
            ax = plt.axes([0.35 + i*0.08, control_height - 0.03, 0.06, 0.03])
            text = TextBox(ax, f'{label}:', initial=f'{self.target_pose[i]:.2f}')
            text.on_submit(lambda x, idx=i: self._on_pose_change(x, idx))
            self.text_pose.append(text)
        
        # End effector selection
        ax_ee = plt.axes([0.85, control_height - 0.05, 0.1, 0.15])
        self.radio_ee = RadioButtons(ax_ee, ('sphere', 'hook'))
        self.radio_ee.on_clicked(self._on_ee_change)
        
        # Buttons
        ax_solve = plt.axes([0.4, control_height - 0.1, 0.1, 0.04])
        self.btn_solve = Button(ax_solve, 'Solve & Animate')
        self.btn_solve.on_clicked(self._on_solve_click)
        
        ax_home = plt.axes([0.52, control_height - 0.1, 0.08, 0.04])
        self.btn_home = Button(ax_home, 'Home View')
        self.btn_home.on_clicked(self._on_home_click)
        
    def _on_solver_change(self, label):
        self.solver_name = label
        
    def _on_dof_change(self, text):
        try:
            new_dof = int(text)
            if 1 <= new_dof <= 10:
                self.dof = new_dof
                self.link_lengths = [0.1] * self.dof
                self.joint_angles = [0.0] * self.dof
                self._update_plot()
        except ValueError:
            pass
            
    def _on_pose_change(self, text, idx):
        try:
            self.target_pose[idx] = float(text)
        except ValueError:
            pass
            
    def _on_ee_change(self, label):
        self.ee_shape = label
        self._update_plot()
        
    def _on_solve_click(self, event):
        """Run solver and animate"""
        # Import solver
        if self.solver_name == 'jacobian':
            from solvers.numerical.ik_solver_jacobian import JacobianSolver
            solver = JacobianSolver()
        else:
            from solvers.analytical.ik_solver_nn import NeuralNetworkSolver
            solver = NeuralNetworkSolver()
        
        # Solve
        self.joint_angles = solver.solve(self.target_pose, self.link_lengths)
        
        # Create animation path
        self._create_animation_path()
        
        # Update plot
        self._update_plot()
        
    def _on_home_click(self, event):
        """Reset to home view"""
        if self.home_view:
            self.ax3d.view_init(elev=self.home_view[0], azim=self.home_view[1])
            self.ax3d.set_xlim3d(self.home_view[2])
            self.ax3d.set_ylim3d(self.home_view[3])
            self.ax3d.set_zlim3d(self.home_view[4])
        self.fig.canvas.draw()
        
    def _create_animation_path(self):
        """Create smooth animation path"""
        # Simple linear interpolation for demo
        steps = 20
        start_angles = np.zeros(self.dof)
        end_angles = np.array(self.joint_angles)
        
        self.animation_path = []
        for i in range(steps):
            t = i / (steps - 1)
            angles = (1 - t) * start_angles + t * end_angles
            pose = forward_kinematics(angles.tolist(), self.link_lengths)
            self.animation_path.append(pose[:3])  # Just positions
            
    def _update_plot(self):
        """Update the visualization"""
        # Clear plots
        self.ax3d.clear()
        self.ax2d.clear()
        
        # Compute robot configuration
        positions = self._compute_joint_positions()
        
        # Plot 3D robot
        xs, ys, zs = zip(*positions)
        self.ax3d.plot(xs, ys, zs, 'b-o', linewidth=2, markersize=8)
        
        # Plot 2D projection
        self.ax2d.plot(xs, ys, 'b-o', linewidth=2, markersize=8)
        
        # Plot end effector
        ee_pose = forward_kinematics(self.joint_angles, self.link_lengths)
        ee_handler = self.ee_handlers[self.ee_shape]
        ee_handler.draw_3d(self.ax3d, ee_pose)
        ee_handler.draw_2d(self.ax2d, ee_pose)
        
        # Plot target
        tx, ty, tz = self.target_pose[:3]
        self.ax3d.plot([tx], [ty], [tz], 'r*', markersize=15)
        self.ax2d.plot([tx], [ty], 'r*', markersize=15)
        
        # Plot animation path if exists
        if self.animation_path:
            path_array = np.array(self.animation_path)
            self.ax3d.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                          'g--', alpha=0.5, linewidth=1)
            self.ax2d.plot(path_array[:, 0], path_array[:, 1], 
                          'g--', alpha=0.5, linewidth=1)
        
        # Set limits and labels
        max_reach = sum(self.link_lengths)
        self.ax3d.set_xlim([-max_reach, max_reach])
        self.ax3d.set_ylim([-max_reach, max_reach])
        self.ax3d.set_zlim([0, max_reach])
        self.ax3d.set_xlabel('X')
        self.ax3d.set_ylabel('Y')
        self.ax3d.set_zlabel('Z')
        
        self.ax2d.set_xlim([-max_reach, max_reach])
        self.ax2d.set_ylim([-max_reach, max_reach])
        self.ax2d.set_xlabel('X')
        self.ax2d.set_ylabel('Y')
        self.ax2d.grid(True)
        self.ax2d.set_aspect('equal')
        
        # Store home view
        if self.home_view is None:
            self.home_view = (
                self.ax3d.elev, self.ax3d.azim,
                self.ax3d.get_xlim3d(),
                self.ax3d.get_ylim3d(),
                self.ax3d.get_zlim3d()
            )
        
        self.fig.canvas.draw()
        
    def _compute_joint_positions(self):
        """Compute positions of all joints"""
        positions = [(0, 0, 0)]  # Base
        T = np.eye(4)
        
        for i, (angle, length) in enumerate(zip(self.joint_angles, self.link_lengths)):
            # Simple FK for visualization
            if i % 3 == 0:  # Z rotation
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            elif i % 3 == 1:  # Y rotation
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else:  # X rotation
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            
            T_i = np.eye(4)
            T_i[:3, :3] = R
            T_i[:3, 3] = [length, 0, 0]
            
            T = T @ T_i
            positions.append(tuple(T[:3, 3]))
            
        return positions
    
    def run(self):
        """Run the visualizer"""
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = IKVisualizer()
    app.run()