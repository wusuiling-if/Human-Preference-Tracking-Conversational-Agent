# user_env.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class UserEnv:
    """
    真实用户环境：
    - 用户有一个隐藏偏好向量 w_true ∈ R^D
    - 系统给一个行为向量 a ∈ R^D
    - 用户返回一个模糊满意度 r = <w_true, a> + 噪声
    """

    w_true: np.ndarray
    noise_std: float = 0.1

    @classmethod
    def random(cls, dim: int, noise_std: float = 0.1, rng: np.random.Generator = None) -> "UserEnv":
        rng = rng or np.random.default_rng()
        w = rng.normal(0, 1, size=dim)
        w /= np.linalg.norm(w) + 1e-9
        return cls(w_true=w, noise_std=noise_std)

    def step(self, action: np.ndarray) -> float:
        """给一个行为向量 action，返回用户的模糊反馈 r"""
        action = action.astype(float)
        action /= np.linalg.norm(action) + 1e-9
        base = float(np.dot(self.w_true, action))
        noise = float(np.random.normal(0, self.noise_std))
        return base + noise

    def true_pref(self) -> np.ndarray:
        return self.w_true.copy()
