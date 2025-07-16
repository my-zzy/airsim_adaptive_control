import numpy as np
import torch
import time
import math
import config as cfg
from analytical_model import SimpleFlightDynamicsTorch
from controller_old import adaptive_controller, quaternion_to_euler, adaptive_att_controller, adaptive_single_channel_controller, adaptive_psi_controller
import traj
import matplotlib.pyplot as plt

# Initialize the analytical model
model = SimpleFlightDynamicsTorch(num_samples=1, dt=0.005)

def control_to_pwm(U1, U2, U3, U4):
    """
    Convert control signals (U1, U2, U3, U4) to PWM signals for 4 motors.
    
    Standard X-configuration quadrotor:
    Motor layout (viewed from top):
         X (Front)
    FL       FR
      \     /
       \   /
        \ /
         +  (Center)
        / \
       /   \
      /     \
    RL       RR
         (Rear)
    
    Motor rotation directions:
    - FR: CW  (generates CCW torque)
    - FL: CCW (generates CW torque) 
    - RR: CCW (generates CW torque)
    - RL: CW  (generates CCW torque)
    
    Args:
        U1: Total thrust command (negative = upward force, positive = downward force in NED)
        U2: Roll torque (positive = roll right)
        U3: Pitch torque (positive = pitch up/nose up)
        U4: Yaw torque (positive = yaw right/CW)
    
    Returns:
        pwm_signals: torch.Tensor of shape (1, 4) with PWM values [0, 1]
    """
    # Effective arm length for X-configuration
    L = cfg.UAV_arm_length * math.cos(math.pi / 4.0)
    
    # Motor mixing matrix for X-configuration
    # Standard equations:
    # U1 = T_FR + T_FL + T_RR + T_RL  (total thrust)
    # U2 = L * (T_FR - T_FL + T_RR - T_RL)  (roll torque)
    # U3 = L * (T_FR + T_FL - T_RR - T_RL)  (pitch torque) 
    # U4 = k_M * (T_FR - T_FL - T_RR + T_RL)  (yaw torque)
    # where k_M is the moment coefficient ratio (torque/thrust)
    
    # Torque to thrust ratio (from motor characteristics)
    k_M = cfg.UAV_max_torque / cfg.UAV_max_thrust
    
    # Solve the inverse mixing matrix:
    # [T_FR]   [1   1/(2L)   1/(2L)   1/(2k_M)] [U1]
    # [T_FL] = [1  -1/(2L)   1/(2L)  -1/(2k_M)] [U2]
    # [T_RR]   [1   1/(2L)  -1/(2L)  -1/(2k_M)] [U3] 
    # [T_RL]   [1  -1/(2L)  -1/(2L)   1/(2k_M)] [U4]
    
    base_thrust = (-U1) / 4.0  # Convert: negative U1 command → positive thrust magnitude
    
    # Torque contributions (note: factor of 2 because of X-config geometry)
    roll_contrib = U2 / (2.0 * L)
    pitch_contrib = U3 / (2.0 * L)
    yaw_contrib = U4 / (2.0 * k_M)
    
    # Calculate individual motor thrusts using correct mixing
    T_FR = base_thrust + roll_contrib + pitch_contrib + yaw_contrib
    T_FL = base_thrust - roll_contrib + pitch_contrib - yaw_contrib
    T_RR = base_thrust + roll_contrib - pitch_contrib - yaw_contrib
    T_RL = base_thrust - roll_contrib - pitch_contrib + yaw_contrib
    
    # Convert thrusts to PWM signals (normalized by max thrust)
    pwm_FR = max(0.0, min(1.0, T_FR / cfg.UAV_max_thrust))
    pwm_FL = max(0.0, min(1.0, T_FL / cfg.UAV_max_thrust))
    pwm_RR = max(0.0, min(1.0, T_RR / cfg.UAV_max_thrust))
    pwm_RL = max(0.0, min(1.0, T_RL / cfg.UAV_max_thrust))
    
    # Return in order: FR, RL, FL, RR (matching your original order)
    return torch.tensor([[pwm_FR, pwm_RL, pwm_FL, pwm_RR]], 
                       device=cfg.device, dtype=torch.float32)

def verify_control_mixing():
    """
    Verify that the control mixing is correct by testing known inputs.
    """
    print("Testing control allocation matrix...")
    
    # Test 1: Pure thrust (hover)
    U1, U2, U3, U4 = 10.0, 0.0, 0.0, 0.0
    pwm = control_to_pwm(U1, U2, U3, U4)
    print(f"Pure thrust U1={U1}: PWM = {pwm.numpy().flatten()}")
    print(f"  Expected: all motors equal = {U1/4.0/cfg.UAV_max_thrust:.3f}")
    
    # Test 2: Pure roll 
    U1, U2, U3, U4 = 10.0, 1.0, 0.0, 0.0
    pwm = control_to_pwm(U1, U2, U3, U4)
    print(f"Roll torque U2={U2}: PWM = {pwm.numpy().flatten()}")
    print(f"  FR & RR should be higher, FL & RL should be lower")
    
    # Test 3: Pure pitch
    U1, U2, U3, U4 = 10.0, 0.0, 1.0, 0.0
    pwm = control_to_pwm(U1, U2, U3, U4)
    print(f"Pitch torque U3={U3}: PWM = {pwm.numpy().flatten()}")
    print(f"  Front motors (FR & FL) should be higher, rear motors (RR & RL) should be lower")
    
    # Test 4: Pure yaw
    U1, U2, U3, U4 = 10.0, 0.0, 0.0, 1.0
    pwm = control_to_pwm(U1, U2, U3, U4)
    print(f"Yaw torque U4={U4}: PWM = {pwm.numpy().flatten()}")
    print(f"  CW motors (FR & RL) should be higher, CCW motors (FL & RR) should be lower")
    
    # Verify thrust conservation
    L = cfg.UAV_arm_length * math.cos(math.pi / 4.0)
    k_M = cfg.UAV_max_torque / cfg.UAV_max_thrust
    
    for test_name, (U1, U2, U3, U4) in [
        ("Hover", (10.0, 0.0, 0.0, 0.0)),
        ("Roll", (10.0, 2.0, 0.0, 0.0)),
        ("Pitch", (10.0, 0.0, 2.0, 0.0)),
        ("Yaw", (10.0, 0.0, 0.0, 1.0))
    ]:
        pwm = control_to_pwm(U1, U2, U3, U4)
        T_FR, T_RL, T_FL, T_RR = pwm.numpy().flatten() * cfg.UAV_max_thrust
        
        # Verify forward equations
        U1_check = T_FR + T_FL + T_RR + T_RL
        U2_check = L * (T_FR - T_FL + T_RR - T_RL)
        U3_check = L * (T_FR + T_FL - T_RR - T_RL)
        U4_check = k_M * (T_FR - T_FL - T_RR + T_RL)
        
        print(f"\n{test_name} test verification:")
        print(f"  Input:  U1={U1:.1f}, U2={U2:.1f}, U3={U3:.1f}, U4={U4:.1f}")
        print(f"  Output: U1={U1_check:.1f}, U2={U2_check:.1f}, U3={U3_check:.1f}, U4={U4_check:.1f}")
        print(f"  Error:  dU1={abs(U1-U1_check):.3f}, dU2={abs(U2-U2_check):.3f}, dU3={abs(U3-U3_check):.3f}, dU4={abs(U4-U4_check):.3f}")
    
    print("\nControl allocation verification complete!")
    print("="*50)

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
        [pos_w[0], pos_w[0], pos_w[0]],  # x history
        [pos_w[1], pos_w[1], pos_w[1]],  # y history
        [pos_w[2], pos_w[2], pos_w[2]]   # z history
    ]
    
    att = [
        [roll, roll, roll],    # roll history
        [pitch, pitch, pitch],  # pitch history
        [yaw, yaw, yaw]       # yaw history
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
    dt = 0.01  # Controller time step
    total_time = 5.0  # Total simulation time
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
    pwm_log = []
    
    print("Starting simulation...")
    print(f"Initial position: {current_state[0, 0:3].cpu().numpy()}")
    
    for step in range(num_steps):
        current_time = step * dt
        time_log.append(current_time)
        
        # Get desired trajectory
        xd, yd, zd, psid = traj.test1(current_time)
        
        # Create desired trajectory lists (controller expects history)
        if step == 0:
            posd = [[xd]*3, [yd]*3, [zd]*3]
            attd = [[0.0]*3, [0.0]*3, [psid]*3]
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
        U1, U2, U3, U4, phid_new, thetad_new, dhat, jifen = adaptive_att_controller(
            pos, att, posd, attd, dhat, jifen, dt, current_time
        )
        # Update desired attitude
        # attd[0][-1] = phid_new
        # attd[1][-1] = thetad_new
        attd[0].append(phid_new)   # roll
        attd[1].append(thetad_new) # pitch

        # Use hover control as fallback
        # U1 = cfg.UAV_mass * 9.81
        # U2 = U3 = U4 = 0.0
        
        # Convert control signals to PWM
        pwm_signals = control_to_pwm(U1, U2, U3, U4)
        
        # Run dynamics for multiple sub-steps
        for _ in range(int(dt / model.dt)):
            current_state, current_filter_outputs = model._dynamics_step(
                current_state, current_filter_outputs, pwm_signals
            )
        # print(current_state)
        # Update position and attitude lists for next iteration
        # pos, att = state_to_lists(current_state)
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
        pwm_log.append(pwm_signals.cpu().numpy().flatten())
        
        # Print progress every 50 steps
        if step % 50 == 0:
            pos_current = current_state[0, 0:3].cpu().numpy()
            # print(f"Step {step:3d}, Time: {current_time:5.1f}s, "
            #       f"Pos: [{pos_current[0]:6.2f}, {pos_current[1]:6.2f}, {pos_current[2]:6.2f}], "
            #       f"Target: [{xd:6.2f}, {yd:6.2f}, {zd:6.2f}]")
    
    print("Simulation completed!")
    
    # Print final results
    final_pos = current_state[0, 0:3].cpu().numpy()
    final_target = traj.test1(total_time)[:3]
    print(f"Final position: [{final_pos[0]:6.2f}, {final_pos[1]:6.2f}, {final_pos[2]:6.2f}]")
    print(f"Final target:   [{final_target[0]:6.2f}, {final_target[1]:6.2f}, {final_target[2]:6.2f}]")
    
    return time_log, position_log, attitude_log, control_log, pwm_log


if __name__ == "__main__":
    # First verify the control allocation
    # verify_control_mixing()
    
    # Run the simulation
    time_data, pos_data, att_data, ctrl_data, pwm_data = run_simulation()
    

    # Convert data to numpy arrays
    time_data = np.array(time_data)
    pos_data = np.array(pos_data)
    att_data = np.array(att_data)
    ctrl_data = np.array(ctrl_data)
    pwm_data = np.array(pwm_data)
    print(pwm_data.shape)

    # plt.figure(figsize=(12, 8))
    # plt.plot(time_data, pwm_data[:, 0], 'r-', label='PWM FR')
    # plt.plot(time_data, pwm_data[:, 1], 'g-', label='PWM RL')
    # plt.plot(time_data, pwm_data[:, 2], 'b-', label='PWM FL')
    # plt.plot(time_data, pwm_data[:, 3], 'k-', label='PWM RR')
    # plt.xlabel('Time (s)')
    # plt.ylabel('PWM Signal')
    # plt.title('PWM Signals vs Time')
    # plt.legend()
    # plt.grid(True)
    
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

