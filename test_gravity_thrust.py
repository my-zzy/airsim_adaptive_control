#!/usr/bin/env python3
"""
Test script to verify gravity and thrust directions are consistent
"""
import torch
import numpy as np
from analytical_model import SimpleFlightDynamicsTorch
import config as cfg

def test_gravity_thrust_consistency():
    """Test that thrust and gravity directions are physically consistent"""
    
    print("Testing gravity and thrust consistency...")
    print("=" * 50)
    
    # Initialize model
    model = SimpleFlightDynamicsTorch(num_samples=1, dt=0.005)
    
    # Print physics setup
    print(f"Gravity vector (world frame): {model.gravity_w.numpy()}")
    print(f"UAV mass: {model.mass.numpy()} kg")
    print(f"Max thrust per motor: {model.rotor_max_thrust.numpy()} N")
    print(f"Total max thrust: {4 * model.rotor_max_thrust.numpy()} N")
    print(f"Weight: {model.mass.numpy() * 9.81} N")
    print(f"Coordinate system: NED (Z-axis points DOWN)")
    print()
    
    # Test 1: Hover condition (all motors at equal PWM)
    print("Test 1: Hover condition")
    print("-" * 30)
    
    # Calculate required PWM for hover (NED convention)
    weight = model.mass * 9.81  # Weight in Newtons (downward force)
    required_thrust_command = -weight  # Negative thrust command for upward force
    required_total_thrust = abs(required_thrust_command)  # Actual thrust magnitude
    required_thrust_per_motor = required_total_thrust / 4.0
    hover_pwm = required_thrust_per_motor / model.rotor_max_thrust
    
    print(f"Required thrust command for hover: {required_thrust_command:.3f} N (negative = upward)")
    print(f"Required thrust magnitude: {required_total_thrust:.3f} N")
    print(f"Required thrust per motor: {required_thrust_per_motor:.3f} N")
    print(f"Required PWM per motor: {hover_pwm:.3f}")
    
    # Use control_to_pwm function instead of direct calculation
    from run_att_in_analytical_model import control_to_pwm
    hover_pwm_tensor = control_to_pwm(required_thrust_command, 0.0, 0.0, 0.0)
    print(f"PWM from control_to_pwm: {hover_pwm_tensor.numpy().flatten()}")
    
    # Initial state: hovering at z=0
    initial_state = torch.tensor([[
        0.0, 0.0, 0.0,  # position (x, y, z)
        0.0, 0.0, 0.0,  # velocity (vx, vy, vz)
        1.0, 0.0, 0.0, 0.0,  # quaternion (w, x, y, z) - level attitude
        0.0, 0.0, 0.0   # angular velocity (wx, wy, wz)
    ]], device=cfg.device, dtype=torch.float32)
    
    # Hover PWM (use the correct control allocation function)
    hover_pwm_tensor = control_to_pwm(required_thrust_command, 0.0, 0.0, 0.0)
    
    # Initial motor filter state (pre-charge to steady state for accurate testing)
    initial_filter = hover_pwm_tensor.clone()  # Start at steady state
    
    # Step forward
    next_state, next_filter = model._dynamics_step(
        initial_state, initial_filter, hover_pwm_tensor
    )
    
    # Check vertical acceleration
    z_accel = (next_state[0, 5] - initial_state[0, 5]) / model.dt  # dv_z/dt
    print(f"Vertical acceleration with hover PWM: {z_accel.item():.6f} m/s²")
    print(f"Expected (should be ≈ 0): 0.000000 m/s²")
    print()
    
    # Test 2: No thrust (free fall) 
    print("Test 2: Free fall (no thrust)")
    print("-" * 30)
    
    zero_pwm = torch.zeros((1, 4), device=cfg.device, dtype=torch.float32)
    zero_filter = torch.zeros((1, 4), device=cfg.device, dtype=torch.float32)  # Start from zero
    next_state_freefall, _ = model._dynamics_step(
        initial_state, zero_filter, zero_pwm
    )
    
    z_accel_freefall = (next_state_freefall[0, 5] - initial_state[0, 5]) / model.dt
    print(f"Vertical acceleration with no thrust: {z_accel_freefall.item():.3f} m/s² (positive = downward)")
    print(f"Expected (should be ≈ +9.81): +9.810 m/s² (gravity pulling down)")
    print()
    
    # Test 3: Maximum thrust
    print("Test 3: Maximum thrust")
    print("-" * 30)
    
    max_pwm = torch.ones((1, 4), device=cfg.device, dtype=torch.float32)
    max_filter = torch.ones((1, 4), device=cfg.device, dtype=torch.float32)  # Pre-charge to max
    next_state_max, _ = model._dynamics_step(
        initial_state, max_filter, max_pwm
    )
    
    z_accel_max = (next_state_max[0, 5] - initial_state[0, 5]) / model.dt
    expected_max_accel = -(4 * model.rotor_max_thrust / model.mass) + 9.81  # Max upward acceleration + gravity
    print(f"Vertical acceleration with max thrust: {z_accel_max.item():.3f} m/s² (positive = downward)")
    print(f"Expected: {expected_max_accel.item():.3f} m/s² (upward)")
    print()
    
    # Summary
    print("Summary:")
    print("-" * 30)
    hover_error = abs(z_accel.item())
    freefall_error = abs(z_accel_freefall.item() - 9.81)  # NED: should be +9.81
    max_error = abs(z_accel_max.item() - expected_max_accel.item())
    
    print(f"Hover error: {hover_error:.6f} m/s² (should be < 0.01)")
    print(f"Freefall error: {freefall_error:.6f} m/s² (should be < 0.01)")
    print(f"Max thrust error: {max_error:.6f} m/s² (should be < 0.1)")
    
    if hover_error < 0.01 and freefall_error < 0.01 and max_error < 0.1:
        print("\n✅ All tests PASSED! Gravity and thrust are consistent.")
    else:
        print("\n❌ Some tests FAILED! Check the physics implementation.")
    
    return hover_error < 0.01 and freefall_error < 0.01 and max_error < 0.1

if __name__ == "__main__":
    test_gravity_thrust_consistency()
