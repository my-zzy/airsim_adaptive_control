#!/usr/bin/env python3
"""
Debug the analytical model dynamics step by step
"""
import torch
import config as cfg
from analytical_model import SimpleFlightDynamicsTorch

def debug_dynamics_step():
    """Debug what happens inside _dynamics_step"""
    
    print("=== DEBUGGING DYNAMICS STEP ===")
    print()
    
    # Initialize model
    model = SimpleFlightDynamicsTorch(num_samples=1, dt=0.005)
    
    # Initial state: hovering at z=0
    initial_state = torch.tensor([[
        0.0, 0.0, 0.0,  # position (x, y, z)
        0.0, 0.0, 0.0,  # velocity (vx, vy, vz)
        1.0, 0.0, 0.0, 0.0,  # quaternion (w, x, y, z) - level attitude
        0.0, 0.0, 0.0   # angular velocity (wx, wy, wz)
    ]], device=cfg.device, dtype=torch.float32)
    
    # Hover PWM (should create 9.81 N upward force)
    hover_pwm = 0.587  # Each motor
    hover_pwm_tensor = torch.full((1, 4), hover_pwm, device=cfg.device, dtype=torch.float32)
    
    # Initial motor filter state
    initial_filter = torch.zeros((1, 4), device=cfg.device, dtype=torch.float32)
    
    print(f"Input PWM: {hover_pwm_tensor.numpy().flatten()}")
    print()
    
    # Manually trace through the dynamics calculation
    print("=== MANUAL DYNAMICS TRACE ===")
    
    # 1. Control signal (after filter - assume no filtering for now)
    control_signal = hover_pwm_tensor
    print(f"Control signal: {control_signal.numpy().flatten()}")
    
    # 2. Individual thrusts
    thrusts = control_signal * model.rotor_max_thrust
    print(f"Individual thrusts: {thrusts.numpy().flatten()} N")
    print(f"Total thrust magnitude: {torch.sum(thrusts).item():.3f} N")
    
    # 3. Body frame thrust calculation
    total_thrust_z_b = -torch.sum(thrusts, dim=1)
    print(f"total_thrust_z_b: {total_thrust_z_b.item():.3f} N")
    
    # 4. Body thrust vector
    total_thrust_vector_b = torch.zeros((1, 3), device=cfg.device, dtype=torch.float32)
    total_thrust_vector_b[:, 2] = total_thrust_z_b
    print(f"Body thrust vector: {total_thrust_vector_b.numpy().flatten()}")
    
    # 5. Convert to world frame (quaternion rotation)
    q_bw_wxyz = initial_state[:, 6:10]  # [1, 0, 0, 0] for level
    print(f"Quaternion: {q_bw_wxyz.numpy().flatten()}")
    
    # For level attitude, body frame = world frame
    total_force_w = total_thrust_vector_b  # No rotation needed
    print(f"World thrust vector: {total_force_w.numpy().flatten()}")
    
    # 6. Calculate drag (should be zero for stationary hover)
    vel_w = initial_state[:, 3:6]  # [0, 0, 0]
    q_bw_inv = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=cfg.device)  # Conjugate of level quaternion
    velocity_b = vel_w  # No rotation for level attitude
    force_drag_b = -model.drag_box_body.unsqueeze(0) * torch.abs(velocity_b) * velocity_b
    print(f"Velocity: {vel_w.numpy().flatten()}")
    print(f"Drag force: {force_drag_b.numpy().flatten()}")
    
    # 7. Total force including drag
    total_force_b = total_thrust_vector_b + force_drag_b
    print(f"Body force (thrust + drag): {total_force_b.numpy().flatten()}")
    
    # 8. Convert to world frame and add gravity
    total_force_w = total_force_b  # No rotation for level attitude
    print(f"World force before gravity: {total_force_w.numpy().flatten()}")
    
    gravity_w = model.gravity_w
    print(f"Gravity vector: {gravity_w.numpy()}")
    
    total_force_with_gravity = total_force_w + gravity_w.unsqueeze(0)
    print(f"Total force (thrust + drag + gravity): {total_force_with_gravity.numpy().flatten()}")
    
    # 9. Calculate acceleration
    linear_accel_w = total_force_with_gravity / model.mass
    print(f"Linear acceleration: {linear_accel_w.numpy().flatten()}")
    print()
    
    # 8. Compare with actual model step
    print("=== ACTUAL MODEL STEP ===")
    next_state, next_filter = model._dynamics_step(
        initial_state, initial_filter, hover_pwm_tensor
    )
    
    # Check vertical acceleration
    z_accel = (next_state[0, 5] - initial_state[0, 5]) / model.dt
    print(f"Model Z acceleration: {z_accel.item():.6f} m/s²")
    print(f"Manual Z acceleration: {linear_accel_w[0, 2].item():.6f} m/s²")
    print()
    
    # Check if there's motor filtering affecting the result
    print("=== MOTOR FILTERING CHECK ===")
    print(f"Input PWM: {hover_pwm_tensor.numpy().flatten()}")
    print(f"Filtered PWM: {next_filter.numpy().flatten()}")
    
    if abs(z_accel.item() - linear_accel_w[0, 2].item()) < 1e-6:
        print("✅ Manual calculation matches model!")
    else:
        print("❌ Manual calculation doesn't match model!")
        print("   There might be additional effects (drag, filtering, etc.)")

if __name__ == "__main__":
    debug_dynamics_step()
