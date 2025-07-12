"""
Hook End Effector
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class HookEE:
    def __init__(self):
        self.size = 0.02
        
    def draw_3d(self, ax, pose):
        """Draw hook in 3D plot"""
        x, y, z = pose[:3]
        
        # Simple hook shape
        hook_points = [
            [0, 0, 0],
            [0, 0, -self.size],
            [self.size/2, 0, -self.size],
            [self.size/2, 0, -self.size/2]
        ]
        
        # Transform points based on pose
        hook_points = np.array(hook_points) + [x, y, z]
        
        # Draw hook
        ax.plot(hook_points[:, 0], hook_points[:, 1], hook_points[:, 2], 
                'k-', linewidth=3)
        
    def draw_2d(self, ax, pose):
        """Draw hook in 2D plot"""
        x, y = pose[:2]
        
        # Simple 2D hook projection
        hook_points = [
            [x, y],
            [x, y - self.size],
            [x + self.size/2, y - self.size]
        ]
        
        hook_points = np.array(hook_points)
        ax.plot(hook_points[:, 0], hook_points[:, 1], 'k-', linewidth=3)