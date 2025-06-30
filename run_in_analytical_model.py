import numpy as np
import torch
import time
import math
import config as cfg
from analytical_model_gpu import SimpleFlightDynamicsTorch

analytical_model_instance = SimpleFlightDynamicsTorch(
    n_samples_cem, dt=0.1, dtype=torch.float32       # dt是动力学模型内部积分步长
)
examiner_instance = SimpleFlightDynamicsTorch(       # 检验并行化之后模型可靠性
    1, dt=0.05, dtype=torch.float32
)

predicted_trajectory_batch = analytical_model_instance.simulate_horizon(
        current_true_state_gpu, # 初始状态
        sampled_controls_gpu.permute(1, 0, 2), # 转置成(H, K, 4)
        dt_mpc)