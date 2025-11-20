# config.py

D_REAL = 32        # 真实用户偏好的维度（你可以理解为“世界真维度”）
INIT_K = 2         # 系统一开始认为的有效维度
MAX_K = 10         # 最多长到多少维（防炸）
NOISE_STD = 0.15   # 用户反馈的噪声
LAMBDA_RIDGE = 1.0 # 子空间内 ridge 回归正则
WINDOW = 6         # 每隔多少步检查一次“要不要升维”
ERR_THRESH = 0.12  # 均方误差大于这个阈值 → 尝试升维
T_STEPS = 400      # 模拟交互轮数
SEED = 42          # 随机种子方便复现
EXPLORE_PROB = 0.3 # 采样行为时用于随机探索的概率
RESIDUAL_NORM_THRESH = 0.03  # 升维时允许的最小正交残差
BAD_MEAN_THRESH = -0.2       # 最近 reward 均值低于该值才认为整体体验差
MIN_BAD_SPAN = 10            # 计算平均 reward 的最小窗口
EXPAND_COOLDOWN = 10         # 升维冷却步数
