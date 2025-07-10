import numpy as np
import torch
import time
import math
import config as cfg
from analytical_model import SimpleFlightDynamicsTorch
from controller import quaternion_to_euler
import matplotlib.pyplot as plt

# Initialize the analytical model
model = SimpleFlightDynamicsTorch(num_samples=1, dt=0.005)

def control_to_pwm(U1, U2, U3, U4):
    """
    Convert control signals (U1, U2, U3, U4) to PWM signals for 4 motors.
    
    Args:
        U1: Total thrust
        U2: Roll torque
        U3: Pitch torque  
        U4: Yaw torque
    
    Returns:
        pwm_signals: torch.Tensor of shape (1, 4) with PWM values [0, 1]
    """
    # Motor configuration: FR, RL, FL, RR (Front-Right, Rear-Left, Front-Left, Rear-Right)
    L = cfg.UAV_arm_length * math.cos(math.pi / 4.0)  # Effective arm length
    
    # Base thrust for each motor (hover condition)
    base_thrust = -U1 / 4.0
    
    # Torque contributions
    roll_contrib = U2 / (4.0 * L)
    pitch_contrib = U3 / (4.0 * L)
    yaw_contrib = U4 / (4.0 * cfg.UAV_max_torque)
    
    # Calculate individual motor thrusts
    T_FR = base_thrust - roll_contrib + pitch_contrib + yaw_contrib
    T_RL = base_thrust + roll_contrib - pitch_contrib - yaw_contrib
    T_FL = base_thrust + roll_contrib + pitch_contrib + yaw_contrib
    T_RR = base_thrust - roll_contrib - pitch_contrib - yaw_contrib
    
    # Convert thrusts to PWM signals (normalized by max thrust)
    pwm_FR = max(0, min(1, T_FR / cfg.UAV_max_thrust))
    pwm_RL = max(0, min(1, T_RL / cfg.UAV_max_thrust))
    pwm_FL = max(0, min(1, T_FL / cfg.UAV_max_thrust))
    pwm_RR = max(0, min(1, T_RR / cfg.UAV_max_thrust))
    
    return torch.tensor([[pwm_FR, pwm_RL, pwm_FL, pwm_RR]], 
                       device=cfg.device, dtype=torch.float32)

def simple_phi_controller(phi_current, phi_desired, phi_dot_current, phi_dot_desired, dt):
    """
    Simple PD controller for roll angle (phi) only.
    
    Args:
        phi_current: Current roll angle
        phi_desired: Desired roll angle
        phi_dot_current: Current roll rate
        phi_dot_desired: Desired roll rate
        dt: Time step
        
    Returns:
        U2: Roll torque command
    """
    # PD gains - conservative values
    kp_phi = 5.0   # Proportional gain
    kd_phi = 2.0   # Derivative gain
    
    # Error calculations
    e_phi = phi_desired - phi_current
    e_phi_dot = phi_dot_desired - phi_dot_current
    
    # PD control law
    U2 = kp_phi * e_phi + kd_phi * e_phi_dot
    
    # Limit output to prevent saturation
    U2 = max(-1.0, min(1.0, U2))
    
    return U2

def adaptive_phi_controller(phi_current, phi_desired, phi_dot_current, phi_dot_desired, 
                           phi_dot2_desired, dhat_phi, integral_phi, dt):
    """
    Adaptive controller for roll angle (phi) only - simplified version.
    
    Args:
        phi_current: Current roll angle
        phi_desired: Desired roll angle  
        phi_dot_current: Current roll rate
        phi_dot_desired: Desired roll rate
        phi_dot2_desired: Desired roll acceleration
        dhat_phi: Disturbance estimate
        integral_phi: Integral term
        dt: Time step
        
    Returns:
        U2: Roll torque command
        dhat_phi_new: Updated disturbance estimate
        integral_phi_new: Updated integral term
    """
    # Control gains - conservative tuning
    c_phi = 2.0        # Damping gain
    lambda_phi = 0.5   # Integral gain
    lambda_star = 0.1  # Adaptation gain
    
    # Error calculations
    e_phi = phi_current - phi_desired
    e_phi_dot = phi_dot_current - phi_dot_desired
    
    # Update integral term with anti-windup
    integral_phi_new = integral_phi + e_phi * dt
    integral_phi_new = max(-0.5, min(0.5, integral_phi_new))  # Anti-windup
    
    # Sliding surface
    s_phi = e_phi_dot + c_phi * e_phi
    
    # Control law
    phi_dot2_command = -c_phi * e_phi_dot - lambda_phi * integral_phi_new - e_phi + phi_dot2_desired
    
    # Adaptation law
    dhat_phi_dot = lambda_star * s_phi
    dhat_phi_new = dhat_phi + dhat_phi_dot * dt
    
    # Final control signal
    U2 = (phi_dot2_command - dhat_phi_new) * cfg.UAV_inertia_diag[0] / cfg.UAV_arm_length
    
    # Limit output
    U2 = max(-2.0, min(2.0, U2))
    
    return U2, dhat_phi_new, integral_phi_new

def generate_phi_reference(t, ref_type="step"):
    """
    Generate reference trajectory for roll angle.
    
    Args:
        t: Current time
        ref_type: Type of reference ("step", "ramp", "sine")
        
    Returns:
        phi_ref, phi_dot_ref, phi_dot2_ref
    """
    if ref_type == "step":
        if t < 2.0:
            phi_ref = 0.0
            phi_dot_ref = 0.0
            phi_dot2_ref = 0.0
        else:
            phi_ref = 0.1  # 0.1 radians ≈ 5.7 degrees
            phi_dot_ref = 0.0
            phi_dot2_ref = 0.0
            
    elif ref_type == "ramp":
        if t < 2.0:
            phi_ref = 0.0
            phi_dot_ref = 0.0
            phi_dot2_ref = 0.0
        elif t < 5.0:
            # Smooth ramp over 3 seconds
            phi_ref = 0.1 * (t - 2.0) / 3.0
            phi_dot_ref = 0.1 / 3.0
            phi_dot2_ref = 0.0
        else:
            phi_ref = 0.1
            phi_dot_ref = 0.0
            phi_dot2_ref = 0.0
            
    elif ref_type == "sine":
        if t < 2.0:
            phi_ref = 0.0
            phi_dot_ref = 0.0
            phi_dot2_ref = 0.0
        else:
            freq = 0.2  # Hz
            amplitude = 0.05  # radians
            phi_ref = amplitude * math.sin(2 * math.pi * freq * (t - 2.0))
            phi_dot_ref = amplitude * 2 * math.pi * freq * math.cos(2 * math.pi * freq * (t - 2.0))
            phi_dot2_ref = -amplitude * (2 * math.pi * freq)**2 * math.sin(2 * math.pi * freq * (t - 2.0))
            
    return phi_ref, phi_dot_ref, phi_dot2_ref

def run_single_channel_simulation(controller_type="adaptive", reference_type="ramp"):
    """
    Run simulation with single channel (phi) control.
    
    Args:
        controller_type: "pd" or "adaptive"
        reference_type: "step", "ramp", or "sine"
    """
    
    # Simulation parameters
    dt = 0.02  # Controller time step (50 Hz)
    total_time = 15.0  # Total simulation time
    num_steps = int(total_time / dt)
    
    # Initial state: [pos_w(3), vel_w(3), quat_wxyz(4), ang_vel_b(3)]
    initial_state = torch.tensor([[
        0.0, 0.0, 0.0,      # position
        0.0, 0.0, 0.0,      # velocity
        1.0, 0.0, 0.0, 0.0, # quaternion [w,x,y,z] (identity)
        0.0, 0.0, 0.0       # angular velocity
    ]], device=cfg.device, dtype=torch.float32)
    
    # Initialize filter outputs (motor PWM filter states)
    current_filter_outputs = torch.zeros((1, 4), device=cfg.device, dtype=torch.float32)
    
    # Controller state variables
    if controller_type == "adaptive":
        dhat_phi = 0.0
        integral_phi = 0.0
        phi_dot_prev = 0.0
    
    # State tracking
    current_state = initial_state.clone()
    
    # Data logging
    time_log = []
    phi_log = []
    phi_ref_log = []
    phi_dot_log = []
    phi_dot_ref_log = []
    control_log = []
    pwm_log = []
    error_log = []
    
    print(f"Starting single channel simulation...")
    print(f"Controller type: {controller_type}")
    print(f"Reference type: {reference_type}")
    print(f"Initial state: {current_state[0, 0:3].cpu().numpy()}")
    
    for step in range(num_steps):
        current_time = step * dt
        time_log.append(current_time)
        
        # Extract current state
        state = current_state[0].cpu().numpy()
        quat_wxyz = state[6:10]  # [w, x, y, z]
        ang_vel_b = state[10:13]
        
        # Convert quaternion to Euler angles
        w, x, y, z = quat_wxyz
        roll, pitch, yaw = quaternion_to_euler(x, y, z, w)
        
        # Current roll angle and rate
        phi_current = roll
        phi_dot_current = ang_vel_b[0]  # Roll rate in body frame
        
        # Generate reference trajectory
        phi_ref, phi_dot_ref, phi_dot2_ref = generate_phi_reference(current_time, reference_type)
        
        # Control calculation
        if controller_type == "pd":
            U2 = simple_phi_controller(phi_current, phi_ref, phi_dot_current, phi_dot_ref, dt)
            
        elif controller_type == "adaptive":
            U2, dhat_phi, integral_phi = adaptive_phi_controller(
                phi_current, phi_ref, phi_dot_current, phi_dot_ref, 
                phi_dot2_ref, dhat_phi, integral_phi, dt
            )
        
        # Set other control channels to zero (hover condition + roll only)
        U1 = -cfg.UAV_mass * 9.81  # Hover thrust
        U3 = 0.0  # No pitch torque
        U4 = 0.0  # No yaw torque
        
        # Convert to PWM signals
        pwm_signals = control_to_pwm(U1, U2, U3, U4)
        
        # Run dynamics for multiple sub-steps
        for _ in range(int(dt / model.dt)):
            current_state, current_filter_outputs = model._dynamics_step(
                current_state, current_filter_outputs, pwm_signals
            )
        
        # Log data
        phi_log.append(phi_current)
        phi_ref_log.append(phi_ref)
        phi_dot_log.append(phi_dot_current)
        phi_dot_ref_log.append(phi_dot_ref)
        control_log.append([U1, U2, U3, U4])
        pwm_log.append(pwm_signals.cpu().numpy().flatten())
        error_log.append(phi_ref - phi_current)
        
        # Print progress
        if step % 250 == 0:  # Every 5 seconds
            print(f"t={current_time:5.1f}s: phi={math.degrees(phi_current):6.1f}°, "
                  f"phi_ref={math.degrees(phi_ref):6.1f}°, "
                  f"error={math.degrees(phi_ref - phi_current):6.1f}°, U2={U2:6.3f}")
    
    print("Simulation completed!")
    
    return {
        'time': np.array(time_log),
        'phi': np.array(phi_log),
        'phi_ref': np.array(phi_ref_log),
        'phi_dot': np.array(phi_dot_log),
        'phi_dot_ref': np.array(phi_dot_ref_log),
        'control': np.array(control_log),
        'pwm': np.array(pwm_log),
        'error': np.array(error_log)
    }

def plot_results(data, controller_type, reference_type):
    """Plot simulation results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Single Channel Roll Control - {controller_type.upper()} Controller, {reference_type.capitalize()} Reference')
    
    # Roll angle tracking
    axes[0, 0].plot(data['time'], np.degrees(data['phi']), 'b-', linewidth=2, label='Actual φ')
    axes[0, 0].plot(data['time'], np.degrees(data['phi_ref']), 'r--', linewidth=2, label='Reference φ')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Roll Angle (°)')
    axes[0, 0].set_title('Roll Angle Tracking')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Roll rate
    axes[0, 1].plot(data['time'], np.degrees(data['phi_dot']), 'b-', linewidth=2, label='Actual φ̇')
    axes[0, 1].plot(data['time'], np.degrees(data['phi_dot_ref']), 'r--', linewidth=2, label='Reference φ̇')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Roll Rate (°/s)')
    axes[0, 1].set_title('Roll Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Tracking error
    axes[0, 2].plot(data['time'], np.degrees(data['error']), 'g-', linewidth=2)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Error (°)')
    axes[0, 2].set_title('Roll Angle Error')
    axes[0, 2].grid(True)
    
    # Control signals
    axes[1, 0].plot(data['time'], data['control'][:, 1], 'r-', linewidth=2, label='U2 (roll)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Torque (N⋅m)')
    axes[1, 0].set_title('Roll Control Signal')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # PWM signals
    axes[1, 1].plot(data['time'], data['pwm'][:, 0], 'r-', label='FR')
    axes[1, 1].plot(data['time'], data['pwm'][:, 1], 'g-', label='RL')
    axes[1, 1].plot(data['time'], data['pwm'][:, 2], 'b-', label='FL')
    axes[1, 1].plot(data['time'], data['pwm'][:, 3], 'k-', label='RR')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('PWM Signal')
    axes[1, 1].set_title('Motor PWM Signals')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Error statistics
    rms_error = np.sqrt(np.mean(data['error']**2))
    max_error = np.max(np.abs(data['error']))
    steady_state_error = np.mean(np.abs(data['error'][-100:]))  # Last 2 seconds
    
    stats_text = f'RMS Error: {np.degrees(rms_error):.2f}°\n'
    stats_text += f'Max Error: {np.degrees(max_error):.2f}°\n'
    stats_text += f'SS Error: {np.degrees(steady_state_error):.2f}°'
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].set_title('Performance Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test different controllers and references
    test_cases = [
        ("pd", "step"),
        ("pd", "ramp"), 
        ("adaptive", "step"),
        ("adaptive", "ramp"),
        ("adaptive", "sine")
    ]
    
    print("Testing single channel roll control...")
    print("=" * 50)
    
    for controller_type, reference_type in test_cases:
        print(f"\nRunning test: {controller_type} controller with {reference_type} reference")
        
        # Run simulation
        data = run_single_channel_simulation(controller_type, reference_type)
        
        # Plot results
        plot_results(data, controller_type, reference_type)
        
        # Calculate performance metrics
        rms_error = np.sqrt(np.mean(data['error']**2))
        max_error = np.max(np.abs(data['error']))
        
        print(f"Performance metrics:")
        print(f"  RMS Error: {np.degrees(rms_error):.2f}°")
        print(f"  Max Error: {np.degrees(max_error):.2f}°")
        print(f"  Final Error: {np.degrees(data['error'][-1]):.2f}°")
        
        input("Press Enter to continue to next test...")
    
    print("\nAll tests completed!")
