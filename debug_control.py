#!/usr/bin/env python3
"""
Debug the control allocation and thrust calculation
"""
import torch
import math
import config as cfg
from run_att_in_analytical_model import control_to_pwm

def debug_control_allocation():
    """Debug the control to PWM conversion"""
    
    print("=== DEBUGGING CONTROL ALLOCATION ===")
    print()
    
    # Test hover condition: U1 = -9.81 (negative for upward force)
    U1_hover = -cfg.UAV_mass * 9.81  # -9.81 N
    U2, U3, U4 = 0.0, 0.0, 0.0
    
    print(f"Input: U1={U1_hover:.3f}, U2={U2}, U3={U3}, U4={U4}")
    print()
    
    # Calculate PWM
    pwm = control_to_pwm(U1_hover, U2, U3, U4)
    print(f"Output PWM: {pwm.numpy().flatten()}")
    
    # Calculate expected individual thrusts
    base_thrust = (-U1_hover) / 4.0  # Should be 9.81/4 = 2.453
    print(f"Base thrust per motor: {base_thrust:.3f} N")
    print(f"Expected PWM per motor: {base_thrust/cfg.UAV_max_thrust:.3f}")
    print()
    
    # Check what the analytical model will do with this PWM
    print("=== ANALYTICAL MODEL CALCULATION ===")
    total_thrust_from_pwm = torch.sum(pwm * cfg.UAV_max_thrust)
    print(f"Total thrust from PWM: {total_thrust_from_pwm.item():.3f} N")
    
    # In analytical model: total_thrust_z_b = -torch.sum(thrusts)
    total_thrust_z_b = -total_thrust_from_pwm  # This is what goes into dynamics
    print(f"total_thrust_z_b (body frame): {total_thrust_z_b.item():.3f} N")
    
    # Force in world frame (assuming level attitude)
    force_world_z = total_thrust_z_b  # No rotation for level flight
    print(f"Force in world Z: {force_world_z.item():.3f} N")
    
    # Acceleration = (force + gravity) / mass
    gravity_force = 9.81  # Downward in NED
    net_force = force_world_z + gravity_force
    acceleration = net_force / cfg.UAV_mass
    print(f"Net force (thrust + gravity): {net_force:.3f} N")
    print(f"Expected acceleration: {acceleration:.3f} m/s² (should be ≈ 0)")
    print()
    
    print("=== ISSUE DIAGNOSIS ===")
    if abs(acceleration) < 0.01:
        print("✅ Control allocation is working correctly!")
    else:
        print("❌ There's still an issue with the control allocation.")
        print(f"   The hover thrust should result in near-zero acceleration.")
        print(f"   Current error: {acceleration:.3f} m/s²")

if __name__ == "__main__":
    debug_control_allocation()
