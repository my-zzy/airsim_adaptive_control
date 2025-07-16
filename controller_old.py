#!/usr/bin/env python

import math
from config import *
import numpy as np
# import rclpy.logging
# logger = rclpy.logging.get_logger("controller")

# constant
m = UAV_mass
Ixx = Ixx_body + Ixx_motors
Iyy = Iyy_body + Iyy_motors
Izz = Izz_body + Izz_motors
l = UAV_arm_length

def quaternion_to_euler(x, y, z, w):
    """
    Convert a quaternion into roll, pitch, yaw (in radians)
    Roll  = rotation around x-axis
    Pitch = rotation around y-axis
    Yaw   = rotation around z-axis
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    # in radians

    return roll, pitch, yaw

def lowpass_filter(data, alpha=0.5):
    smoothed = [data[0]]
    for i in range(1, len(data)):
        smoothed_val = alpha * data[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(smoothed_val)
    return smoothed

def pd(x, x_dot, xd, xd_dot, kp, kd):
    x_dot2 = - kd*(x_dot-xd_dot) - kp*(x-xd)
    return x_dot2


def pd_controller(pos, att, posd, attd, dt, t):
    
    x, y, z = pos
    phi, theta, psi = att
    xd, yd, zd = posd
    phid, thetad, psid = attd
    g = -9.8    # negative?!

    # calculate pos_dot & att_dot
    x_dot = (pos[0][-1] - pos[0][-2])/dt    # x,y,z
    y_dot = (pos[1][-1] - pos[1][-2])/dt
    z_dot = (pos[2][-1] - pos[2][-2])/dt

    phi_dot = (att[0][-1] - att[0][-2])/dt
    theta_dot = (att[1][-1] - att[1][-2])/dt
    psi_dot = (att[2][-1] - att[2][-2])/dt

    xd_dot = (xd[-1] - xd[-2])/dt
    yd_dot = (yd[-1] - yd[-2])/dt
    zd_dot = (zd[-1] - zd[-2])/dt


    # PD control of position
    # x_dot2 = pd(x, x_dot, xd, xd_dot, kp1, kd1)
    # y_dot2 = pd(y, y_dot, yd, yd_dot, kp2, kd2)
    # z_dot2 = pd(z, z_dot, zd, zd_dot, kp3, kd3)
    x_dot2 = -pd(x[-1], x_dot, xd[-1], xd_dot, kp1, kd1)
    y_dot2 = -pd(y[-1], y_dot, yd[-1], yd_dot, kp2, kd2)
    z_dot2 = pd(z[-1], z_dot, zd[-1], zd_dot, kp3, kd3)
    # print(f"{z[-1]}, {z_dot}, {zd[-1]}, {zd_dot}")
    # print(f"z_dot2: {z_dot2}")
    # print(f"x_dot2: {x_dot2}")

    # !!for testing only
    # x_dot2 = 0
    # y_dot2 = 0

    # Note that U1 may be negative
    if z_dot2+g > 0:
        U1 = math.sqrt((m*x_dot2)**2+(m*y_dot2)**2+(m*z_dot2+m*g)**2)
    else:
        U1 = -math.sqrt((m*x_dot2)**2+(m*y_dot2)**2+(m*z_dot2+m*g)**2)

    # Calculate desired phi & theta from expected translation acceleration
    # !! non-standard operation
    psi = psi[-1]
    # g = 9.8


    tem = (x_dot2*math.sin(psi)-y_dot2*math.cos(psi))**2/(x_dot2**2+y_dot2**2+(z_dot2+g)**2)
    if x_dot2*math.sin(psi)-y_dot2*math.cos(psi) > 0:
        phidd = math.asin(math.sqrt(tem))
    else:
        phidd = -math.asin(math.sqrt(tem))

    tem = (z_dot2+g)**2/((x_dot2*math.cos(psi)+y_dot2*math.sin(psi))**2+(z_dot2+g)**2)
    if x_dot2*math.cos(psi)+y_dot2*math.sin(psi) > 0:
    # if m*x_dot2/U1 - math.sin(phi[-1])*math.sin(psi) > 0:
        thetadd = math.acos(math.sqrt(tem))
    else:
        thetadd = -math.acos(math.sqrt(tem))

    # !!for testing only
    # phidd = 0
    # thetadd = 0
    # if t > 2:
    #     # phidd = 0.2
    #     # thetadd = 0.2
    #     phidd = 0.1*math.sin(2*t)
    #     thetadd = 0.2*math.sin(2*t)
    


    # Calculate derivative of desired phi & theta from previous
    phid_dot = (phidd - phid[-1])/dt
    thetad_dot = (thetadd - thetad[-1])/dt
    psid_dot = (psid[-1] - psid[-2])/dt


    # PD control of attitude
    phi_dot2 = pd(phi[-1], phi_dot, phidd, phid_dot, kp4, kd4)
    theta_dot2 = pd(theta[-1], theta_dot, thetadd, thetad_dot, kp5, kd5)
    psi_dot2 = pd(psi, psi_dot, psid[-1], psid_dot, kp6, kd6)

    # TODO: how to get air friction
    # U2 = phi_dot2 * Ixx + l*k4*phi_dot
    U2 = phi_dot2 * Ixx
    U3 = theta_dot2 * Iyy
    U4 = psi_dot2 * Izz

    return U1, U2, U3, U4, phidd, thetadd, x_dot2, y_dot2

    # Why we need to return phid & thetad?
    # To calculate phid_dot & thetad_dot

def adaptive_single_channel_controller(pos, att, posd, attd, dhat, jifen, dt, t):
    # lowpass filter
    alp = 0.1
    pos = [lowpass_filter(p, alp) for p in pos]
    att = [lowpass_filter(a, alp) for a in att]
    posd = [lowpass_filter(pd, alp) for pd in posd]
    attd = [lowpass_filter(ad, alp) for ad in attd]


    phi = att[0][-1]

    # Use a smoother reference trajectory instead of step input
    if t < 2:
        phid_new = 0
    # elif t < 7:
    #     # Smooth transition over 5 seconds
    #     phid_new = 0.1 * (1 - math.cos(math.pi * (t - 2) / 5)) / 2
    else:
        phid_new = 0.1
        phid_new = 0.1*math.sin(t-2)
    thetad_new = 0

    dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat = dhat
    xphi, xtheta, xpsi = jifen


    phi_dot = (att[0][-1] - att[0][-2])/dt
    theta_dot = 0
    psi_dot = 0

    phid_dot = (attd[0][-1] - attd[0][-2])/dt
    phid_dot2 = ((attd[0][-1] - attd[0][-2])/dt - (attd[0][-2] - attd[0][-3])/dt)/dt
    # print(attd[0][-1], phid_dot, phid_dot2)

    # Adaptive control law with improved stability
    ephi = phi - attd[0][-1]
    ephi_dot = phi_dot - phid_dot
    xphi += ephi * dt  # Proper integration
    
    # Limit integral windup
    # xphi = max(-1.0, min(1.0, xphi))
    
    alpha_phi = phid_dot - cphi*ephi
    beta_phi = phi_dot - alpha_phi + lamphi*xphi
    phi_dot2 = -cp*beta_phi + phid_dot2 - cphi*ephi_dot - lamphi*ephi - 0.1*ephi
    dphi_hat_dot = lamphi_star*beta_phi
    dphi_hat += dphi_hat_dot*dt
    U2 = (phi_dot2 - dphi_hat - theta_dot*psi_dot*(Iyy-Izz)/Ixx)*Ixx/l
    
    # Debug output with reduced frequency
    if int(t * 10) % 1 == 0:  # Print every 1 second
        print(f"t={t:.1f}: phi={phi:.4f}, phid={attd[0][-1]:.4f}, ephi={ephi:.4f}, U2={U2:.4f}")
        components = np.array([-cp*beta_phi, phid_dot2, -cphi*ephi_dot, -lamphi*ephi, -ephi])
        # components = np.array([phi_dot2, -dphi_hat, -theta_dot*psi_dot*(Iyy-Izz)/Ixx])
        print(f"Control components: {components}")
    
    # Limit control output to prevent instability
    # U2 = max(-2.0, min(2.0, U2))

    dhat_old = [dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat]
    jifen_old = [xphi, xtheta, xpsi]

    # NED convention: negative thrust value creates upward force (opposes gravity)
    U1 = -UAV_mass*9.81  # Hover thrust (negative for upward force in NED)
    U3 = 0
    U4 = 0

    return U1, U2, U3, U4, phid_new, thetad_new, dhat_old, jifen_old

def adaptive_psi_controller(pos, att, posd, attd, dhat, jifen, dt, t):
    # lowpass filter
    alp = 0.1
    pos = [lowpass_filter(p, alp) for p in pos]
    att = [lowpass_filter(a, alp) for a in att]
    posd = [lowpass_filter(pd, alp) for pd in posd]
    attd = [lowpass_filter(ad, alp) for ad in attd]

    phi = att[0][-1]
    theta = att[1][-1]
    psi = att[2][-1]

    phid_new = 0.0
    thetad_new = 0.0
    psid = attd[2][-1]  # Use the last value of psid from attd

    dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat = dhat
    xphi, xtheta, xpsi = jifen


    phi_dot = (att[0][-1] - att[0][-2])/dt
    theta_dot = (att[1][-1] - att[1][-2])/dt
    psi_dot = (att[2][-1] - att[2][-2])/dt

    phid_dot = (attd[0][-1] - attd[0][-2])/dt
    thetad_dot = (attd[1][-1] - attd[1][-2])/dt
    psid_dot = (attd[2][-1] - attd[2][-2])/dt

    phid_dot2 = ((attd[0][-1] - attd[0][-2])/dt - (attd[0][-2] - attd[0][-3])/dt)/dt
    thetad_dot2 = ((attd[1][-1] - attd[1][-2])/dt - (attd[1][-2] - attd[1][-3])/dt)/dt
    psid_dot2 = ((attd[2][-1] - attd[2][-2])/dt - (attd[2][-2] - attd[2][-3])/dt)/dt

    epsi = psi - psid
    epsi_dot = psi_dot - psid_dot
    xpsi += epsi*dt
    alpha_psi = psid_dot - cpsi*epsi
    beta_psi = psi_dot - alpha_psi + lampsi*xpsi
    psi_dot2 = -cr*beta_psi + psid_dot2 - cpsi*epsi_dot - lampsi*epsi - epsi
    dpsi_hat_dot = lampsi_star*beta_psi
    dpsi_hat += dpsi_hat_dot*dt
    U4 = (psi_dot2 - dpsi_hat - theta_dot*phi_dot*(Ixx-Iyy)/Izz)*Izz/l

    dhat_old = [dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat]
    jifen_old = [xphi, xtheta, xpsi]

    U1 = -UAV_mass*9.81  # Hover thrust (negative for upward force in NED)
    U2 = 0.0
    U3 = 0.0
    # U4 = 1e-6

    return U1, U2, U3, U4, phid_new, thetad_new, dhat_old, jifen_old


def adaptive_att_controller(pos, att, posd, attd, dhat, jifen, dt, t):
    # lowpass filter
    alp = 0.1
    pos = [lowpass_filter(p, alp) for p in pos]
    att = [lowpass_filter(a, alp) for a in att]
    posd = [lowpass_filter(pd, alp) for pd in posd]
    attd = [lowpass_filter(ad, alp) for ad in attd]

    phi = att[0][-1]
    theta = att[1][-1]
    psi = att[2][-1]

    phid_new = 0.1*math.sin(t)
    phid_new = 0.2
    thetad_new = 0.0
    # thetad_new = -0.1*math.sin(t)
    psid = attd[2][-1]  # Use the last value of psid from attd

    dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat = dhat
    xphi, xtheta, xpsi = jifen


    phi_dot = (att[0][-1] - att[0][-2])/dt
    theta_dot = (att[1][-1] - att[1][-2])/dt
    psi_dot = (att[2][-1] - att[2][-2])/dt

    phid_dot = (attd[0][-1] - attd[0][-2])/dt
    thetad_dot = (attd[1][-1] - attd[1][-2])/dt
    psid_dot = (attd[2][-1] - attd[2][-2])/dt

    phid_dot2 = ((attd[0][-1] - attd[0][-2])/dt - (attd[0][-2] - attd[0][-3])/dt)/dt
    thetad_dot2 = ((attd[1][-1] - attd[1][-2])/dt - (attd[1][-2] - attd[1][-3])/dt)/dt
    psid_dot2 = ((attd[2][-1] - attd[2][-2])/dt - (attd[2][-2] - attd[2][-3])/dt)/dt

    epsi = psi - psid
    epsi_dot = psi_dot - psid_dot
    xpsi += epsi*dt
    alpha_psi = psid_dot - cpsi*epsi
    beta_psi = psi_dot - alpha_psi + lampsi*xpsi
    psi_dot2 = -cr*beta_psi + psid_dot2 - cpsi*epsi_dot - lampsi*epsi - epsi
    dpsi_hat_dot = lampsi_star*beta_psi
    dpsi_hat += dpsi_hat_dot*dt
    U4 = (psi_dot2 - dpsi_hat - theta_dot*phi_dot*(Ixx-Iyy)/Izz)*Izz/l

    ephi = phi - phid_new
    ephi_dot = phi_dot - phid_dot
    xphi += ephi*dt
    alpha_phi = phid_dot - cphi*ephi
    beta_phi = phi_dot - alpha_phi + lamphi*xphi
    phi_dot2 = -cp*beta_phi + phid_dot2 - cphi*ephi_dot - lamphi*ephi - ephi
    dphi_hat_dot = lamphi_star*beta_phi
    dphi_hat += dphi_hat_dot*dt
    U2 = (phi_dot2 - dphi_hat - theta_dot*psi_dot*(Iyy-Izz)/Ixx)*Ixx/l

    ethata = theta - thetad_new
    etheta_dot = theta_dot - thetad_dot
    xtheta += ethata*dt
    alpha_theta = thetad_dot - cthe*ethata
    beta_theta = theta_dot - alpha_theta + lamthe*xtheta
    theta_dot2 = -cq*beta_theta + thetad_dot2 - cthe*etheta_dot - lamthe*ethata - ethata
    dtheta_hat_dot = lamthe_star*beta_theta
    dtheta_hat += dtheta_hat_dot*dt
    U3 = (theta_dot2 - dtheta_hat - phi_dot*psi_dot*(Izz-Ixx)/Iyy)*Iyy/l

    dhat_old = [dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat]
    jifen_old = [xphi, xtheta, xpsi]

    # NED convention: negative thrust value creates upward force (opposes gravity)  
    U1 = -UAV_mass*12  # Hover thrust (negative for upward force in NED)

    return U1, -U2, U3, U4, phid_new, thetad_new, dhat_old, jifen_old

def adaptive_controller(pos, att, posd, attd, dhat, jifen, dt, t):
    # lowpass filter
    alp = 0.1
    pos = [lowpass_filter(p, alp) for p in pos]
    att = [lowpass_filter(a, alp) for a in att]
    posd = [lowpass_filter(pd, alp) for pd in posd]
    attd = [lowpass_filter(ad, alp) for ad in attd]
    
    x = pos[0][-1]
    y = pos[1][-1]
    z = pos[2][-1]
    phi = att[0][-1]
    theta = att[1][-1]
    psi = att[2][-1]

    xd = posd[0][-1]
    yd = posd[1][-1]
    zd = posd[2][-1]
    phid = attd[0][-1]
    thetad = attd[1][-1]
    psid = attd[2][-1]

    dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat = dhat
    xphi, xtheta, xpsi = jifen
    g = -9.8

    # calculate pos_dot & att_dot
    u = (pos[0][-1] - pos[0][-2])/dt    # x,y,z_dot
    v = (pos[1][-1] - pos[1][-2])/dt
    w = (pos[2][-1] - pos[2][-2])/dt

    phi_dot = (att[0][-1] - att[0][-2])/dt
    theta_dot = (att[1][-1] - att[1][-2])/dt
    psi_dot = (att[2][-1] - att[2][-2])/dt

    xd_dot = (posd[0][-1] - posd[0][-2])/dt
    yd_dot = (posd[1][-1] - posd[1][-2])/dt
    zd_dot = (posd[2][-1] - posd[2][-2])/dt

    xd_dot2 = ((posd[0][-1] - posd[0][-2])/dt - (posd[0][-2] - posd[0][-3])/dt)/dt
    yd_dot2 = ((posd[1][-1] - posd[1][-2])/dt - (posd[1][-2] - posd[1][-3])/dt)/dt
    zd_dot2 = ((posd[2][-1] - posd[2][-2])/dt - (posd[2][-2] - posd[2][-3])/dt)/dt

    # wrong! use new phid & thetad
    phid_dot2 = ((attd[0][-1] - attd[0][-2])/dt - (attd[0][-2] - attd[0][-3])/dt)/dt
    thetad_dot2 = ((attd[1][-1] - attd[1][-2])/dt - (attd[1][-2] - attd[1][-3])/dt)/dt
    psid_dot2 = ((attd[2][-1] - attd[2][-2])/dt - (attd[2][-2] - attd[2][-3])/dt)/dt

    # wrong! use new phid & thetad
    phid_dot = (attd[0][-1] - attd[0][-2])/dt
    thetad_dot = (attd[1][-1] - attd[1][-2])/dt
    psid_dot = (attd[2][-1] - attd[2][-2])/dt

    # position control
    ez = z - zd
    ew = w - zd_dot + cz*ez
    ez_dot = ew - cz*ez
    w_dot = -cw*ew - ez + zd_dot2 - cz*ez_dot
    dz_hat_dot = lamz*ew
    dz_hat += dz_hat_dot*dt
    U1 = (w_dot - dz_hat +g)*m/(math.cos(phi)*math.cos(theta))
    print(f"U1: {U1}")

    ex = x - xd
    eu = u - xd_dot + cu*ex
    ex_dot = eu - cx*ex
    u_dot = -cu*eu - ex + xd_dot2 - cx*ex_dot
    # u_dot = 0   # for testing
    dx_hat_dot = lamx*eu
    dx_hat += dx_hat_dot*dt
    Ux = (u_dot - dx_hat)*m/U1
    print(f"Ux: {Ux}")

    ey = y - yd
    ev = v - yd_dot + cv*ey
    ey_dot = ev - cy*ey
    v_dot = -cv*ev - ey + yd_dot2 - cy*ey_dot
    # v_dot = 0   # for testing
    dy_hat_dot = lamy*ev
    dy_hat += dy_hat_dot*dt
    Uy = (v_dot - dy_hat)*m/U1
    print(f"Uy: {Uy}")


    # attitude control
    # print(f"{Ux*math.sin(psi) - Uy*math.cos(psi)}")
    phid_new = math.asin(Ux*math.sin(psi) - Uy*math.cos(psi))
    # print(f"{(Ux*math.cos(psi) + Uy*math.sin(psi))/math.cos(phid_new), Ux, Uy, psi, phid_new}")
    thetad_new = math.asin((Ux*math.cos(psi) + Uy*math.sin(psi))/math.cos(phid_new))

    components = np.array([(Ux*math.cos(psi) + Uy*math.sin(psi))/math.cos(phid_new), Ux, Uy, psi, phid_new])
    print(f"Control components: {components}\n")

    epsi = psi - psid
    epsi_dot = psi_dot - psid_dot
    xpsi += epsi    # TODO:initialize xpsi
    alpha_psi = psid_dot - cpsi*epsi
    beta_psi = psi_dot - alpha_psi + lampsi*xpsi
    psi_dot2 = -cr*beta_psi + psid_dot2 - cpsi*epsi_dot - lampsi*epsi - epsi
    dpsi_hat_dot = lampsi_star*beta_psi
    dpsi_hat += dpsi_hat_dot*dt
    U4 = (psi_dot2 - dpsi_hat - theta_dot*phi_dot*(Ixx-Iyy)/Izz)*Izz/l

    ephi = phi - phid_new
    ephi_dot = phi_dot - phid_dot
    xphi += ephi
    alpha_phi = phid_dot - cphi*ephi
    beta_phi = phi_dot - alpha_phi + lamphi*xphi
    phi_dot2 = -cp*beta_phi + phid_dot2 - cphi*ephi_dot - lamphi*ephi - ephi
    dphi_hat_dot = lamphi_star*beta_phi
    dphi_hat += dphi_hat_dot*dt
    U2 = (phi_dot2 - dphi_hat - theta_dot*psi_dot*(Iyy-Izz)/Ixx)*Ixx/l

    ethata = theta - thetad_new
    etheta_dot = theta_dot - thetad_dot
    xtheta += ethata
    alpha_theta = thetad_dot - cthe*ethata
    beta_theta = theta_dot - alpha_theta + lamthe*xtheta
    theta_dot2 = -cq*beta_theta + thetad_dot2 - cthe*etheta_dot - lamthe*ethata - ethata
    dtheta_hat_dot = lamthe_star*beta_theta
    dtheta_hat += dtheta_hat_dot*dt
    U3 = (theta_dot2 - dtheta_hat - phi_dot*psi_dot*(Izz-Ixx)/Iyy)*Iyy/l

    dhat_old = [dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat]
    jifen_old = [xphi, xtheta, xpsi]

    return U1, -U2, U3, U4, phid_new, thetad_new, dhat_old, jifen_old