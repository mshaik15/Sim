"""
Forward Kinematics Implementation
"""
import numpy as np

def forward_kinematics(joint_angles, link_lengths):
    """
    Compute forward kinematics for revolute joint robot
    
    Args:
        joint_angles: List of joint angles in radians
        link_lengths: List of link lengths
        
    Returns:
        pose: [x, y, z, alpha, beta, gamma] - end effector pose
    """
    assert len(joint_angles) == len(link_lengths), "DOF mismatch"
    
    # Initialize transformation matrix
    T = np.eye(4)
    
    # For each joint, apply rotation and translation
    for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
        # Alternate between different rotation axes for variety
        if i % 3 == 0:  # Rotate around Z
            R = rotation_z(angle)
        elif i % 3 == 1:  # Rotate around Y
            R = rotation_y(angle)
        else:  # Rotate around X
            R = rotation_x(angle)
            
        # Translation along link
        t = np.array([length, 0, 0])
        
        # Build transformation matrix
        T_i = np.eye(4)
        T_i[:3, :3] = R
        T_i[:3, 3] = t
        
        # Apply transformation
        T = T @ T_i
    
    # Extract position
    x, y, z = T[:3, 3]
    
    # Extract orientation (ZYX Euler angles)
    R = T[:3, :3]
    gamma = np.arctan2(R[1, 0], R[0, 0])  # yaw
    beta = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))  # pitch
    alpha = np.arctan2(R[2, 1], R[2, 2])  # roll
    
    return [x, y, z, alpha, beta, gamma]

def rotation_x(angle):
    """Rotation matrix around X axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotation_y(angle):
    """Rotation matrix around Y axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rotation_z(angle):
    """Rotation matrix around Z axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])