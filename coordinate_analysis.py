#!/usr/bin/env python3
"""
NED Coordinate System Analysis for Quadrotor Model

This script analyzes the coordinate system conventions and identifies
issues with thrust and gravity directions in the analytical model.
"""
import torch
import numpy as np

def analyze_coordinate_system():
    """
    Analyze the current coordinate system setup and identify issues.
    """
    print("=== NED COORDINATE SYSTEM ANALYSIS ===")
    print()
    
    print("1. COORDINATE SYSTEM CONVENTIONS:")
    print("-" * 40)
    print("NED (North-East-Down):")
    print("  X = North (forward)")
    print("  Y = East (right)")  
    print("  Z = Down (toward ground)")
    print("  → Gravity vector should be: [0, 0, +9.81] (positive Z)")
    print()
    
    print("ENU (East-North-Up) - Alternative:")
    print("  X = East")
    print("  Y = North") 
    print("  Z = Up (away from ground)")
    print("  → Gravity vector should be: [0, 0, -9.81] (negative Z)")
    print()
    
    print("2. CURRENT MODEL ANALYSIS:")
    print("-" * 40)
    
    # Current gravity vector from model
    current_gravity = [0, 0, 9.81]
    print(f"Current gravity_w = {current_gravity}")
    print("→ This suggests Z-axis points DOWN (NED convention)")
    print()
    
    # Current thrust calculation
    print("Current thrust calculation:")
    print("  total_thrust_z_b = -torch.sum(thrusts, dim=1)")
    print("  total_thrust_vector_b[:, 2] = total_thrust_z_b")
    print("→ Negative sign means: positive PWM → negative body thrust")
    print("→ In NED with Z-down: negative thrust = upward force ✓")
    print()
    
    # Controller thrust commands
    print("Controller thrust commands:")
    print("  U1 = -UAV_mass*9.81 = -1.0*9.81 = -9.81 N")
    print("→ Negative U1 for hover (upward force to oppose gravity)")
    print()
    
    print("3. ISSUE IDENTIFICATION:")
    print("-" * 40)
    
    # Your trajectory analysis
    print("Trajectory analysis:")
    print("  test1(t): returns (0, 0, t, 0.1)")
    print("  → Positive Z trajectory (moving DOWN in NED)")
    print("  test2(t): returns (0, 0, -t, 1.77) for t < up_time")  
    print("  → Negative Z trajectory (moving UP in NED)")
    print()
    
    print("Your control allocation comments:")
    print("  U1: Total thrust (positive = upward)")
    print("  → This contradicts NED convention!")
    print("  → In NED: positive thrust should be DOWNWARD")
    print()
    
    print("4. THE PROBLEM:")
    print("-" * 40)
    print("❌ COORDINATE SYSTEM MISMATCH:")
    print("   • Gravity vector: [0, 0, +9.81] → NED convention (Z down)")
    print("   • Control allocation: 'positive = upward' → ENU convention (Z up)")
    print("   • Controller: U1 = -9.81 → Assumes positive thrust = upward")
    print("   • Analytical model: negative thrust sign → Assumes body Z points up")
    print()
    
    print("5. SOLUTIONS:")
    print("-" * 40)
    print("Option A: Full NED Convention (Recommended)")
    print("  • Keep gravity_w = [0, 0, +9.81]")
    print("  • Change thrust sign: total_thrust_z_b = +torch.sum(thrusts)")
    print("  • Update control allocation: 'positive thrust = downward'")
    print("  • Update controller: U1 = +UAV_mass*9.81 for hover")
    print()
    
    print("Option B: Full ENU Convention")
    print("  • Change gravity_w = [0, 0, -9.81]")
    print("  • Keep current thrust calculation")
    print("  • Keep control allocation: 'positive thrust = upward'")
    print("  • Keep controller: U1 = -UAV_mass*9.81 for hover")
    print()
    
    print("6. PHYSICS CHECK:")
    print("-" * 40)
    print("For hover equilibrium:")
    print("  F_net = F_thrust + F_gravity = 0")
    print("  In NED: F_thrust = [0, 0, -9.81], F_gravity = [0, 0, +9.81]")
    print("  In ENU: F_thrust = [0, 0, +9.81], F_gravity = [0, 0, -9.81]")
    print()
    
    return True

def recommend_fixes():
    """
    Provide specific code fixes for NED convention.
    """
    print("=== RECOMMENDED FIXES FOR NED CONVENTION ===")
    print()
    
    print("1. analytical_model.py - Keep gravity as is:")
    print("   self.gravity_w = torch.tensor([0, 0, 9.81], device=device)")
    print()
    
    print("2. analytical_model.py - Fix thrust direction:")
    print("   OLD: total_thrust_z_b = -torch.sum(thrusts, dim=1)")
    print("   NEW: total_thrust_z_b = torch.sum(thrusts, dim=1)")
    print()
    
    print("3. controller.py - Fix hover thrust:")
    print("   OLD: U1 = -UAV_mass*9.81")
    print("   NEW: U1 = UAV_mass*9.81")
    print()
    
    print("4. control_to_pwm comments - Fix description:")
    print("   OLD: U1: Total thrust (positive = upward)")
    print("   NEW: U1: Total thrust (positive = downward, NED convention)")
    print()
    
    print("5. Trajectory interpretation:")
    print("   • test1(t) = (0, 0, t): Moving DOWN (positive Z)")
    print("   • test2(t) = (0, 0, -t): Moving UP (negative Z)")
    print("   • This is now consistent with NED!")
    print()

if __name__ == "__main__":
    analyze_coordinate_system()
    print()
    recommend_fixes()
