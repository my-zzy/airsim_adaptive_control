import numpy as np

def lowpass_filter(data, alpha):
    """指数加权移动平均滤波"""
    smoothed = [data[0]]
    for i in range(1, len(data)):
        smoothed_val = alpha * data[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(smoothed_val)
    return np.array(smoothed)

def calculate_velocity(positions, dt):
    """基于位置数据计算速度"""
    velocities = np.diff(positions) / dt
    # 添加第一个速度为0（假设初始速度为0）
    velocities = np.insert(velocities, 0, 0)
    return velocities

def calculate_acceleration(velocities, dt):
    """基于速度数据计算加速度"""
    accelerations = np.diff(velocities) / dt
    # 添加第一个加速度为0（假设初始加速度为0）
    accelerations = np.insert(accelerations, 0, 0)
    return accelerations

import matplotlib.pyplot as plt

# 假设的原始位置数据
positions = np.array([0,0,0,0,0 ,1, 2, 3, 4, 5, 6, 7, 8, 9,9,9,9,9,9,9,9,9,9])  # 包含跳变的数据
dt = 0.001  # 时间间隔
alpha = 0.5  # EWMA的平滑系数

# 平滑位置数据
smoothed_positions = lowpass_filter(positions, alpha)

# 时间轴
time = np.arange(len(positions)) * dt

# 绘制原始数据和平滑后的数据
plt.figure(figsize=(10, 5))
plt.plot(time, positions, label='原始数据', marker='o')
plt.plot(time, smoothed_positions, label='平滑后数据', marker='x')
plt.show()

# 计算速度和加速度
velocities = calculate_velocity(smoothed_positions, dt)
accelerations = calculate_acceleration(velocities, dt)

# 输出结果
print("原始位置数据:", positions)
print("平滑位置数据:", smoothed_positions)
print("计算的速度:", velocities)
print("计算的加速度:", accelerations)
