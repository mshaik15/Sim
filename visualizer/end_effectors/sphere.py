"""
Sphere End Effector
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class SphereEE:
    def __init__(self):
        self.radius = 0.02
        
    def draw_3d(self, ax, pose):
        """Draw sphere in 3D plot"""
        x, y, z = pose[:3]
        
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        xs = self.radius * np.outer(np.cos(u), np.sin(v)) + x
        ys = self.radius * np.outer(np.sin(u), np.sin(v)) + y
        zs = self.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        
        # Draw sphere
        ax.plot_surface(xs, ys, zs, color='orange', alpha=0.8)
        
    def draw_2d(self, ax, pose):
        """Draw sphere in 2D plot"""
        x, y = pose[:2]
        
        # Draw as circle in 2D
        circle = plt.Circle((x, y), self.radius, color='orange', alpha=0.8)
        ax.add_patch(circle)

# Import matplotlib for Circle
import matplotlib.pyplot as plt