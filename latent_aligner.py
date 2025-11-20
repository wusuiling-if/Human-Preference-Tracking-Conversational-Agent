# latent_aligner.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class LatentAligner:
    """
    在高维空间中对齐一个隐藏用户偏好的系统：
    - 一开始只有 k 维子空间
    - 不断观察 (a_t, r_t)
    - 在子空间内用 ridge 回归拟合用户
    - 如果误差长期解释不了，就从误差方向中挖出新维度，扩展子空间
    """

    D: int                  # 真实空间维度
    k_init: int             # 初始子空间维度
    k_max: int              # 子空间最大维度
    lam: float              # ridge 正则

    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    explore_prob: float = 0.3  # 探索概率，越大越偏随机

    B: np.ndarray = field(init=False)   # D x k, 列正交的基
    k: int = field(init=False)         # 当前维度
    A: np.ndarray = field(init=False)  # k x k, 特征协方差矩阵
    b: np.ndarray = field(init=False)  # k, 响应向量
    theta: np.ndarray = field(init=False)  # k, 回归系数
    grad_residual: np.ndarray = field(init=False)  # D, 残差方向累积

    def __post_init__(self):
        # 初始化子空间基底
        B0 = self.rng.normal(0, 1, size=(self.D, self.k_init))
        B0, _ = np.linalg.qr(B0)  # QR 分解得到列正交基
        self.B = B0
        self.k = self.k_init

        self.A = self.lam * np.eye(self.k)
        self.b = np.zeros(self.k)
        self.theta = np.zeros(self.k)
        self.grad_residual = np.zeros(self.D)

    def sample_action(self, alpha: float = 0.3) -> np.ndarray:
    # 1. 先在当前子空间里采一个方向（老逻辑）
     z = self.rng.normal(0, 1, size=self.k)
     z /= np.linalg.norm(z) + 1e-9
     a_in = self.B @ z

    # 2. 再加一点“子空间外”的探索噪声
     u = self.rng.normal(0, 1, size=self.D)
    # 去掉在当前子空间里的部分，留下正交分量
     u_orth = u - self.B @ (self.B.T @ u)
     nrm = np.linalg.norm(u_orth)
     if nrm > 1e-6:
        u_orth /= nrm
        a = a_in + alpha * u_orth
     else:
        a = a_in

     a /= np.linalg.norm(a) + 1e-9
     return a


    def predict(self, action: np.ndarray) -> float:
        """在当前子空间内预测用户反馈"""
        x = self.B.T @ action  # k 维特征
        return float(self.theta @ x)

    def update_with_sample(self, action: np.ndarray, reward: float):
        """
        用一次 (a_t, r_t) 样本更新：
        - 子空间内 ridge 回归参数
        - 残差累积向量 grad_residual
        """
        action = action.astype(float)
        action /= np.linalg.norm(action) + 1e-9

        x = self.B.T @ action
        r_hat = float(self.theta @ x)

        # 对齐信号直接取 reward（或优势），不再依赖预测误差
        e = reward

        # 在线更新 ridge 回归: A += x x^T, b += x r
        self.A += np.outer(x, x)
        self.b += x * reward
        self.theta = np.linalg.solve(self.A, self.b)

        # 残差方向累积: sum signal_t * a_t
        self.grad_residual += e * action

        return e, r_hat

    def maybe_expand(self, recent_errors: List[float]) -> bool:
        """
        根据近期误差决定是否升维：
        - 如果误差均值还挺大，尝试从残差方向中挖新维度
        - 成功扩展时返回 True
        """
        if self.k >= self.k_max:
            return False
        if len(recent_errors) == 0:
            return False

        mean_err = float(np.mean(np.square(recent_errors)))
        # 由调用方控制阈值
        return False  # 默认这里不直接触发，由外层控制
        # （在 run_simulation 里根据 mean_err 和阈值再调用 expand_subspace）

    def expand_subspace(self, min_norm: float = 1e-6) -> bool:
        """
        真正的升维操作：
        - 用 grad_residual 作为“新信息”的来源
        - 从当前子空间里投影出去，取正交部分
        - 如果正交部分长度够大，则加入为新基向量
        """
        g = self.grad_residual.copy()
        if np.allclose(g, 0):
            return False

        # 投影到当前子空间: proj = B(B^T g)
        proj = self.B @ (self.B.T @ g)
        g_orth = g - proj
        nrm = float(np.linalg.norm(g_orth))
        if nrm < min_norm:
            return False

        g_orth /= nrm
        # 扩展 B
        self.B = np.column_stack([self.B, g_orth])
        self.k += 1

        # 扩维时保留已有的协方差与参数，避免“升维=重置”
        A_old = self.A.copy()
        theta_old = self.theta.copy()
        b_old = self.b.copy()

        A_new = np.zeros((self.k, self.k))
        A_new[:-1, :-1] = A_old
        A_new[-1, -1] = self.lam
        self.A = A_new

        theta_new = np.zeros(self.k)
        theta_new[:-1] = theta_old
        self.theta = theta_new

        b_new = np.zeros(self.k)
        b_new[:-1] = b_old
        self.b = b_new
        self.grad_residual = np.zeros(self.D)

        return True

    def current_approx_pref(self) -> np.ndarray:
        """
        当前系统在 D 维空间里对用户偏好的近似：
        w_hat = B θ
        """
        return self.B @ self.theta
