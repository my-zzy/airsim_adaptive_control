import numpy as np
import torch
import time
import math
import config as cfg
from analytical_model import SimpleFlightDynamicsTorch
from controller import adaptive_controller, quaternion_to_euler
import traj

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
    # Convert torques to individual motor thrusts
    L = cfg.UAV_arm_length * math.cos(math.pi / 4.0)  # Effective arm length
    
    # Solve the motor mixing equations
    # U1 = -(T_FR + T_RL + T_FL + T_RR)
    # U2 = L * (T_FL + T_RL - T_FR - T_RR)
    # U3 = L * (T_FR + T_FL - T_RL - T_RR)
    # U4 = T_FR - T_RL + T_FL - T_RR (simplified, assuming torque coefficients)
    
    # Base thrust for each motor
    base_thrust = -U1 / 4.0
    
    # Torque contributions
    roll_contrib = U2 / (4.0 * L)
    pitch_contrib = U3 / (4.0 * L)
    yaw_contrib = U4 / (4.0 * cfg.UAV_max_torque)  # Simplified
    
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

def state_to_lists(state_tensor):
    """
    Convert state tensor to the list format expected by the controller.
    
    Args:
        state_tensor: torch.Tensor of shape (1, 13)
        
    Returns:
        pos, att: Lists in the format expected by adaptive_controller
    """
    state = state_tensor[0].cpu().numpy()
    
    # Extract position and quaternion
    pos_w = state[0:3]
    quat_wxyz = state[6:10]  # [w, x, y, z]
    
    # Convert quaternion to Euler angles
    w, x, y, z = quat_wxyz
    roll, pitch, yaw = quaternion_to_euler(x, y, z, w)
    
    # Create lists with history (controller expects at least 2 elements)
    pos = [
        [pos_w[0], pos_w[0]],  # x history
        [pos_w[1], pos_w[1]],  # y history
        [pos_w[2], pos_w[2]]   # z history
    ]
    
    att = [
        [roll, roll],    # roll history
        [pitch, pitch],  # pitch history
        [yaw, yaw]       # yaw history
    ]
    
    return pos, att

def update_history_lists(pos, att, new_pos, new_att):
    """Update position and attitude history lists."""
    pos[0].append(new_pos[0])
    pos[1].append(new_pos[1])
    pos[2].append(new_pos[2])
    
    att[0].append(new_att[0])
    att[1].append(new_att[1])
    att[2].append(new_att[2])
    
    # Keep only last 3 elements for numerical derivatives
    if len(pos[0]) > 3:
        for i in range(3):
            pos[i] = pos[i][-3:]
            att[i] = att[i][-3:]

def run_simulation():
    """Run the step-by-step simulation with adaptive controller."""
    
    # Simulation parameters
    dt = 0.1  # Controller time step
    total_time = 20.0  # Total simulation time
    num_steps = int(total_time / dt)
    
    # Initial state: [pos_w(3), vel_w(3), quat_wxyz(4), ang_vel_b(3)]
    initial_state = torch.tensor([[
        0.0, 0.0, 0.0,      # position
        0.0, 0.0, 0.0,      # velocity
        1.0, 0.0, 0.0, 0.0, # quaternion [w,x,y,z]
        0.0, 0.0, 0.0       # angular velocity
    ]], device=cfg.device, dtype=torch.float32)
    
    # Initialize filter outputs (motor PWM filter states)
    current_filter_outputs = torch.zeros((1, 4), device=cfg.device, dtype=torch.float32)
    
    # Initialize adaptive controller variables
    dhat = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # disturbance estimates
    jifen = [0.0, 0.0, 0.0]  # integral terms
    
    # State tracking
    current_state = initial_state.clone()
    
    # Initialize position and attitude lists
    pos, att = state_to_lists(current_state)
    
    # Data logging
    time_log = []
    position_log = []
    attitude_log = []
    control_log = []
    
    print("Starting simulation...")
    print(f"Initial position: {current_state[0, 0:3].cpu().numpy()}")
    
    for step in range(num_steps):
        current_time = step * dt
        time_log.append(current_time)
        
        # Get desired trajectory
        xd, yd, zd, psid = traj.test1(current_time)
        
        # Create desired trajectory lists (controller expects history)
        if step == 0:
            posd = [[xd, xd], [yd, yd], [zd, zd]]
            attd = [[0.0, 0.0], [0.0, 0.0], [psid, psid]]
        else:
            posd[0].append(xd)
            posd[1].append(yd)
            posd[2].append(zd)
            attd[2].append(psid)  # Only yaw is specified
            
            # Keep history length manageable
            if len(posd[0]) > 3:
                for i in range(3):
                    posd[i] = posd[i][-3:]
                for i in range(3):
                    attd[i] = attd[i][-3:]
        
        # Run adaptive controller
        try:
            U1, U2, U3, U4, phid_new, thetad_new, dhat, jifen = adaptive_controller(
                pos, att, posd, attd, dhat, jifen, dt
            )
            
            # Update desired attitude
            attd[0][-1] = phid_new   # roll
            attd[1][-1] = thetad_new # pitch
            
        except Exception as e:
            print(f"Controller error at step {step}: {e}")
            # Use hover control as fallback
            U1 = cfg.UAV_mass * 9.81
            U2 = U3 = U4 = 0.0
        
        # Convert control signals to PWM
        pwm_signals = control_to_pwm(U1, U2, U3, U4)
        
        # Run dynamics for multiple sub-steps
        for _ in range(int(dt / model.dt)):
            current_state, current_filter_outputs = model._dynamics_step(
                current_state, current_filter_outputs, pwm_signals
            )
        
        # Update position and attitude lists for next iteration
        pos, att = state_to_lists(current_state)
        if step > 0:  # Update history
            state_np = current_state[0].cpu().numpy()
            pos_w = state_np[0:3]
            quat_wxyz = state_np[6:10]
            w, x, y, z = quat_wxyz
            roll, pitch, yaw = quaternion_to_euler(x, y, z, w)
            
            update_history_lists(pos, att, pos_w, [roll, pitch, yaw])
        
        # Log data
        position_log.append(current_state[0, 0:3].cpu().numpy().copy())
        quat = current_state[0, 6:10].cpu().numpy()
        w, x, y, z = quat
        roll, pitch, yaw = quaternion_to_euler(x, y, z, w)
        attitude_log.append([roll, pitch, yaw])
        control_log.append([U1, U2, U3, U4])
        
        # Print progress every 50 steps
        if step % 50 == 0:
            pos_current = current_state[0, 0:3].cpu().numpy()
            print(f"Step {step:3d}, Time: {current_time:5.1f}s, "
                  f"Pos: [{pos_current[0]:6.2f}, {pos_current[1]:6.2f}, {pos_current[2]:6.2f}], "
                  f"Target: [{xd:6.2f}, {yd:6.2f}, {zd:6.2f}]")
    
    print("Simulation completed!")
    
    # Print final results
    final_pos = current_state[0, 0:3].cpu().numpy()
    final_target = traj.test1(total_time)[:3]
    print(f"Final position: [{final_pos[0]:6.2f}, {final_pos[1]:6.2f}, {final_pos[2]:6.2f}]")
    print(f"Final target:   [{final_target[0]:6.2f}, {final_target[1]:6.2f}, {final_target[2]:6.2f}]")
    
    return time_log, position_log, attitude_log, control_log

if __name__ == "__main__":
    # Run the simulation
    time_data, pos_data, att_data, ctrl_data = run_simulation()
    
    # Optional: Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Convert data to numpy arrays
        time_data = np.array(time_data)
        pos_data = np.array(pos_data)
        att_data = np.array(att_data)
        ctrl_data = np.array(ctrl_data)
        
        # Plot position
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(time_data, pos_data[:, 0], 'r-', label='x')
        plt.plot(time_data, pos_data[:, 1], 'g-', label='y')
        plt.plot(time_data, pos_data[:, 2], 'b-', label='z')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Position vs Time')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(time_data, np.degrees(att_data[:, 0]), 'r-', label='roll')
        plt.plot(time_data, np.degrees(att_data[:, 1]), 'g-', label='pitch')
        plt.plot(time_data, np.degrees(att_data[:, 2]), 'b-', label='yaw')
        plt.xlabel('Time (s)')
        plt.ylabel('Attitude (deg)')
        plt.title('Attitude vs Time')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(time_data, ctrl_data[:, 0], 'k-', label='U1 (thrust)')
        plt.xlabel('Time (s)')
        plt.ylabel('Thrust (N)')
        plt.title('Thrust Control')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(time_data, ctrl_data[:, 1], 'r-', label='U2 (roll)')
        plt.plot(time_data, ctrl_data[:, 2], 'g-', label='U3 (pitch)')
        plt.plot(time_data, ctrl_data[:, 3], 'b-', label='U4 (yaw)')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (N⋅m)')
        plt.title('Torque Control')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Plotting skipped.")
        print("Install matplotlib with: pip install matplotlib")

