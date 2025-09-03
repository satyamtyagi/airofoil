#!/usr/bin/env python3
"""
Airfoil Validation Functions

This module provides functions to validate airfoil shapes and ensure they meet
physical constraints such as:
- No self-intersection
- Minimum thickness
- Maximum thickness
- Realistic trailing edge angle
"""

import numpy as np
from shapely.geometry import LineString


def check_self_intersection(x_coords, y_coords):
    """
    Check if an airfoil intersects itself.
    
    Args:
        x_coords (array): x-coordinates of airfoil
        y_coords (array): y-coordinates of airfoil
    
    Returns:
        bool: True if valid (no self-intersection), False otherwise
    """
    # Create a LineString from the coordinates
    airfoil = LineString(zip(x_coords, y_coords))
    
    # A simple LineString is valid if it doesn't self-intersect
    return airfoil.is_simple


def calculate_thickness(x_coords, y_coords):
    """
    Calculate thickness distribution of an airfoil.
    
    Args:
        x_coords (array): x-coordinates of airfoil
        y_coords (array): y-coordinates of airfoil
    
    Returns:
        tuple: (max_thickness, max_thickness_position)
    """
    # Convert to numpy arrays if they aren't already
    x = np.array(x_coords)
    y = np.array(y_coords)
    
    # Find leading edge (minimum x value)
    le_idx = np.argmin(x)
    
    # Separate upper and lower surfaces based on the leading edge
    if le_idx == 0:
        # Airfoil is ordered from leading edge to trailing edge and back
        # Find the point where x starts decreasing again (returning from TE)
        for i in range(1, len(x)):
            if x[i] < x[i-1]:  # x starts decreasing
                split_idx = i
                break
        else:
            # If no split found, assume middle point
            split_idx = len(x) // 2
            
        x_upper = x[:split_idx]
        y_upper = y[:split_idx]
        x_lower = np.flip(x[split_idx:])
        y_lower = np.flip(y[split_idx:])
    else:
        # Airfoil is ordered differently, split at leading edge
        x_upper = np.flip(x[:le_idx+1])
        y_upper = np.flip(y[:le_idx+1])
        x_lower = x[le_idx:]
        y_lower = y[le_idx:]
    
    # Interpolate to match x-coordinates for thickness calculation
    # We'll use 100 points evenly distributed along the chord
    x_points = np.linspace(0, 1, 100)
    
    # Skip interpolation if not enough points
    if len(x_upper) <= 2 or len(x_lower) <= 2:
        return 0, 0
    
    try:
        # Interpolate upper and lower surfaces to matching x points
        y_upper_interp = np.interp(x_points, x_upper, y_upper)
        y_lower_interp = np.interp(x_points, x_lower, y_lower)
        
        # Calculate thickness at each point (ensure positive)
        thickness = np.abs(y_upper_interp - y_lower_interp)
        
        # Find maximum thickness and its position
        max_thickness = np.max(thickness)
        max_thickness_pos = x_points[np.argmax(thickness)]
        
        return max_thickness, max_thickness_pos
    
    except Exception as e:
        print(f"Error calculating thickness: {str(e)}")
        return 0, 0


def check_min_thickness(x_coords, y_coords, min_thickness=0.01):
    """
    Check if airfoil meets minimum thickness requirement.
    
    Args:
        x_coords (array): x-coordinates of airfoil
        y_coords (array): y-coordinates of airfoil
        min_thickness (float): minimum allowed thickness as fraction of chord
    
    Returns:
        bool: True if valid (meets minimum thickness), False otherwise
    """
    max_thickness, _ = calculate_thickness(x_coords, y_coords)
    return max_thickness >= min_thickness


def check_max_thickness(x_coords, y_coords, max_thickness=0.25):
    """
    Check if airfoil doesn't exceed maximum thickness.
    
    Args:
        x_coords (array): x-coordinates of airfoil
        y_coords (array): y-coordinates of airfoil
        max_thickness (float): maximum allowed thickness as fraction of chord
    
    Returns:
        bool: True if valid (doesn't exceed maximum thickness), False otherwise
    """
    thickness, _ = calculate_thickness(x_coords, y_coords)
    return thickness <= max_thickness


def check_trailing_edge_angle(params, min_angle_deg=5, max_angle_deg=30):
    """
    Check if trailing edge angle is within realistic bounds.
    
    Args:
        params (dict): PARSEC parameters including trailing edge angle
        min_angle_deg (float): minimum trailing edge angle in degrees
        max_angle_deg (float): maximum trailing edge angle in degrees
    
    Returns:
        bool: True if valid (realistic trailing edge angle), False otherwise
    """
    # PARSEC parameter for trailing edge angle (in radians)
    te_angle_param = params.get("Δyte''", 0)
    
    # Convert to degrees for comparison
    te_angle_deg = abs(np.degrees(te_angle_param))
    
    return min_angle_deg <= te_angle_deg <= max_angle_deg


def validate_airfoil(x_coords, y_coords, params, min_thickness=0.01, max_thickness=0.25,
                    min_te_angle=5, max_te_angle=30, verbose=False):
    """
    Complete airfoil validation applying all constraints.
    
    Args:
        x_coords (array): x-coordinates of airfoil
        y_coords (array): y-coordinates of airfoil
        params (dict): PARSEC parameters
        min_thickness (float): minimum allowed thickness as fraction of chord
        max_thickness (float): maximum allowed thickness as fraction of chord
        min_te_angle (float): minimum trailing edge angle in degrees
        max_te_angle (float): maximum trailing edge angle in degrees
        verbose (bool): If True, print detailed validation results
    
    Returns:
        tuple: (is_valid, reason) where is_valid is a boolean and reason is a string
              explaining why validation failed (empty if valid)
    """
    # Check for self-intersection
    if not check_self_intersection(x_coords, y_coords):
        if verbose:
            print("Validation failed: Airfoil has self-intersection")
        return False, "Self-intersection"
    
    # Check minimum thickness
    if not check_min_thickness(x_coords, y_coords, min_thickness):
        if verbose:
            print(f"Validation failed: Airfoil thickness below minimum ({min_thickness})")
        return False, f"Thickness below {min_thickness}"
    
    # Check maximum thickness
    if not check_max_thickness(x_coords, y_coords, max_thickness):
        if verbose:
            print(f"Validation failed: Airfoil thickness above maximum ({max_thickness})")
        return False, f"Thickness above {max_thickness}"
    
    # Check trailing edge angle
    if not check_trailing_edge_angle(params, min_te_angle, max_te_angle):
        if verbose:
            print(f"Validation failed: Trailing edge angle outside range ({min_te_angle}°-{max_te_angle}°)")
        return False, f"TE angle outside {min_te_angle}°-{max_te_angle}°"
    
    return True, ""
